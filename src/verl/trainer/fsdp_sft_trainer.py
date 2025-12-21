# -*- coding: utf-8 -*- 
# 完整可覆盖版本，适配不支持 `ignored_parameters` 的 PyTorch FSDP，
# 改用 `ignored_modules=[model.cls_head0, model.cls_head1]`。

import os
import logging
from contextlib import nullcontext
from typing import Tuple, List, Optional, Dict

import hydra
import torch
import torch.nn.functional as F
import torch.distributed
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AddedToken, PreTrainedModel
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from tqdm import tqdm

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset.sft_dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

try:
    from verl.utils.ulysses import (
        gather_outputs_and_unpad,
        get_ulysses_sequence_parallel_world_size,
        ulysses_pad_and_slice_inputs,
    )
except ImportError:
    from verl.utils.ulysses import (
        gather_outpus_and_unpad as gather_outputs_and_unpad,
        get_ulysses_sequence_parallel_world_size,
        ulysses_pad_and_slice_inputs,
    )

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
# 设置NCCL超时时间（默认10小时太长，改为30分钟）
if "NCCL_TIMEOUT" not in os.environ:
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30分钟（秒）

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def _to_bool(x):
    try:
        return bool(x()) if callable(x) else bool(x)
    except Exception:
        return False


_IS_CUDA = _to_bool(is_cuda_available)
_IS_NPU = _to_bool(is_npu_available)

# ===================== 常量/超参 =====================
PSEP_TOKEN = "<|psep|>"
CSTART_TOKEN = "<|cstart|>"
CEND_TOKEN = "<|cend|>"

PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END = "\n</POTENTIAL_SPANS>\n"

CLS_NO = 0
CLS_COMMENT = 1
CLS_REPOST = 2

ALPHA_HEAD0 = [0.02, 0.68, 0.3]    # TL=0：三分类 0/1/2
ALPHA_HEAD1 = [0.01, 0.99]             # TL=1：二分类 0/1（1 = 有互动 = 原 1/2）
FOCAL_GAMMA = 2.0
TL_TAIL_WINDOW = 16
# 默认开启：打印每个 micro-batch 的 head loss
LOG_MICRO_LOSSES = True


# ===================== 冻结工具 =====================
def _freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_by_prefix(model: nn.Module, prefixes: Tuple[str, ...]):
    for n, p in model.named_parameters():
        if any(n.startswith(px) for px in prefixes):
            p.requires_grad_(True)


def _apply_freezing_from_cfg(model: nn.Module, cfg_model, rank0: bool = False):
    freeze_cfg = getattr(cfg_model, "freeze", None)
    enable = True
    if freeze_cfg is not None:
        try:
            enable = bool(freeze_cfg.get("enable", True))
        except Exception:
            enable = bool(getattr(freeze_cfg, "enable", True))
    if getattr(cfg_model, "only_train_cls_heads", False):
        enable = True

    try:
        prefixes = tuple(getattr(cfg_model, "trainable_name_prefixes", ["cls_head0.", "cls_head1."]))
        if not prefixes:
            prefixes = ("cls_head0.", "cls_head1.")
    except Exception:
        prefixes = ("cls_head0.", "cls_head1.")

    if enable:
        _freeze_all_params(model)
        _unfreeze_by_prefix(model, prefixes)

    if rank0:
        tot = sum(p.numel() for p in model.parameters())
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[freeze] trainable={trn/1e6:.2f}M / total={tot/1e6:.2f}M ({(trn/max(1,tot)):.2%})")
        print(f"[freeze] prefixes={list(prefixes)}")
        print(f"[freeze] trainable names: {names}")


# ===================== tokenizer 特殊符号 =====================
def _ensure_special_tokens(tokenizer: AutoTokenizer, new_tokens: List[str]) -> AutoTokenizer:
    cur_list = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    normalized: List[str] = []
    for t in cur_list:
        normalized.append(t.content if isinstance(t, AddedToken) else str(t))
    to_add = [
        AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False)
        for t in new_tokens
        if t not in normalized
    ]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": cur_list + to_add})
    return tokenizer


# ===================== 一些小工具 =====================
def _find_subseq(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return -1
    L = needle.numel()
    for i in range(haystack.numel() - L + 1):
        if torch.equal(haystack[i:i + L], needle):
            return int(i)
    return -1


def _assistant_content_start(row_ids: torch.Tensor, start_col: int, encode_fn) -> int:
    device, dtype = row_ids.device, row_ids.dtype
    T = row_ids.numel()
    prefix_variants = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
    pos = start_col
    matched = False
    for delta in (0, 1, 2, 3):
        p = start_col + delta
        if p >= T:
            break
        for pref in prefix_variants:
            pref_ids = encode_fn(pref, device, dtype)
            L = pref_ids.numel()
            if p + L <= T and torch.equal(row_ids[p:p + L], pref_ids):
                pos = p + L
                matched = True
                break
        if matched:
            break
    if not matched:
        nl_ids = encode_fn("\n", device, dtype)
        end = min(T, start_col + 64)
        window = row_ids[start_col:end]
        if nl_ids.numel() == 1:
            idx = torch.nonzero(window == nl_ids.item(), as_tuple=False).flatten()
            if idx.numel() > 0:
                pos = start_col + int(idx[0].item()) + 1
        else:
            L = nl_ids.numel()
            for i in range(max(0, window.numel() - L + 1)):
                if torch.equal(window[i:i + L], nl_ids):
                    pos = start_col + i + L
                    break
    return int(max(0, min(pos, T - 1)))


def _find_spans_block_in_user(
    row_ids: torch.Tensor,
    user_l: int,
    user_r: int,
    start_ids: torch.Tensor,
    end_ids: torch.Tensor,
    start_ids_alt: torch.Tensor,
    end_ids_alt: torch.Tensor,
    with_label_ids: torch.Tensor,
) -> Tuple[int, int]:
    window = row_ids[user_l:user_r]
    if window.numel() == 0:
        return -1, -1
    candidates = [
        (start_ids, end_ids),
        (start_ids_alt, end_ids_alt),
        (with_label_ids, end_ids),
    ]
    for s_ids, e_ids in candidates:
        s = _find_subseq(window, s_ids.to(device=window.device, dtype=window.dtype))
        e = _find_subseq(window, e_ids.to(device=window.device, dtype=window.dtype))
        if s >= 0 and e >= 0:
            s_abs = user_l + s + s_ids.numel()
            e_abs = user_l + e
            if e_abs > s_abs:
                return int(s_abs), int(e_abs)
    return -1, -1


def _decode_json_types(ids: torch.Tensor, start_col: int, tokenizer) -> Tuple[Optional[List[int]], int]:
    txt_full = tokenizer.decode(ids[start_col:].tolist(), skip_special_tokens=False).strip()
    import json as _json, re as _re
    try:
        arr = _json.loads(txt_full)
        if isinstance(arr, list):
            return [int(x.get("type", 0)) if isinstance(x, dict) else 0 for x in arr], 0
    except Exception:
        pass
    lpos = txt_full.find("[")
    rpos = txt_full.rfind("]")
    if 0 <= lpos < rpos:
        mid = txt_full[lpos:rpos + 1]
        try:
            arr = _json.loads(mid)
            if isinstance(arr, list):
                return [int(x.get("type", 0)) if isinstance(x, dict) else 0 for x in arr], 1
        except Exception:
            pass
    hits = _re.findall(r'"type"\s*:\s*(-?\d+)', txt_full)
    if hits:
        out = []
        for h in hits:
            try:
                v = int(h)
                if v not in (0, 1, 2):
                    v = 0
            except Exception:
                v = 0
            out.append(v)
        return out, 2
    return None, -1


def _find_tl_in_user_slice_ex(row_ids: torch.Tensor, user_l: int, user_r: int, encode_fn):
    t0 = encode_fn("<TL0>", row_ids.device, row_ids.dtype)
    t1 = encode_fn("<TL1>", row_ids.device, row_ids.dtype)
    window = row_ids[user_l:user_r]

    def _find_pos(needle: torch.Tensor):
        L = needle.numel()
        if L == 0 or window.numel() < L:
            return -1
        for i in range(window.numel() - L + 1):
            if torch.equal(window[i:i + L], needle):
                return i
        return -1

    pos1 = _find_pos(t1)
    if pos1 >= 0:
        return 1, "direct", int(user_l + pos1)
    pos0 = _find_pos(t0)
    if pos0 >= 0:
        return 0, "direct", int(user_l + pos0)
    return 0, "none", -1


# ===================== Trainer =====================
class FSDPSFTTrainer:
    def __init__(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # 纯LLM训练模式：不再使用分类头

        self._normalize_config_bsz()
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        
        # 检查是否需要从 checkpoint 恢复
        resume_from_checkpoint = getattr(self.config.trainer, "resume_from_checkpoint", None)
        self._build_model_optimizer(resume_from_checkpoint=resume_from_checkpoint)

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    @staticmethod
    def _normalize_alpha(alpha: List[float]) -> List[float]:
        v = [max(float(a), 1e-3) for a in alpha]
        s = sum(v)
        return [a / s for a in v]

    def _focal_loss_multiclass(self, logits: torch.Tensor, targets: torch.Tensor, alpha_vec: List[float]) -> Tuple[torch.Tensor, dict]:
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        pt = p[torch.arange(p.size(0), device=p.device), targets]
        ce_plain = F.nll_loss(logp, targets, reduction="none").float()
        alpha = logp.new_tensor(alpha_vec, dtype=torch.float32)[targets]
        focal_factor = (1.0 - pt).pow(self.focal_gamma)
        loss_vec = alpha * focal_factor * ce_plain
        loss = loss_vec.mean()
        dbg = {
            "ce_mean": float(ce_plain.mean().item()),
            "pt_mean": float(pt.mean().item()),
            "n_samples": int(targets.numel()),
        }
        return loss, dbg

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")
        assert self.config.data.train_batch_size % dp_size == 0
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _get_attr_from_fsdp(self, name: str):
        obj = getattr(self.fsdp_model, name, None)
        if obj is None and hasattr(self.fsdp_model, "module"):
            obj = getattr(self.fsdp_model.module, name, None)
        return obj

    def _build_model_optimizer(self, resume_from_checkpoint=None):
        """
        构建模型和优化器
        
        Args:
            resume_from_checkpoint: 如果提供，从此 checkpoint 加载模型（而不是从 partial_pretrain）
        """
        # 如果指定了 resume_from_checkpoint，优先使用它；否则使用 partial_pretrain
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            if self.device_mesh.get_rank() == 0:
                print(f"[INFO] Loading model from checkpoint: {resume_from_checkpoint}")
            local_model_path = copy_to_local(src=resume_from_checkpoint, verbose=True)
        else:
            local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32"))
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel requires remove_padding=True"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )
        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=self.config.model.get("attn_implementation", "flash_attention_2"),
                trust_remote_code=trust_remote_code,
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.output_hidden_states = False  # 不再需要hidden_states
            self.model.config.return_dict = True

            # 可选 LoRA
            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # 纯LLM训练：根据配置决定是否冻结部分参数
        # 如果配置了freeze，则应用冻结策略；否则训练全部参数
        freeze_cfg = getattr(self.config.model, "freeze", None)
        if freeze_cfg is not None:
            try:
                enable = bool(freeze_cfg.get("enable", False))
            except Exception:
                enable = bool(getattr(freeze_cfg, "enable", False))
            if enable:
                # 纯LLM模式下，如果配置了trainable_name_prefixes，则只训练这些前缀的参数
                # 否则训练全部参数
                try:
                    prefixes = tuple(getattr(self.config.model, "trainable_name_prefixes", []))
                except Exception:
                    prefixes = tuple()
                if prefixes:
                    _freeze_all_params(self.model)
                    _unfreeze_by_prefix(self.model, prefixes)
                    if self.device_mesh.get_rank() == 0:
                        tot = sum(p.numel() for p in self.model.parameters())
                        trn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
                        print(f"[freeze] trainable={trn/1e6:.2f}M / total={tot/1e6:.2f}M ({(trn/max(1,tot)):.2%})")
                        print(f"[freeze] prefixes={list(prefixes)}")
                        print(f"[freeze] trainable names: {names}")
                # 如果没有指定prefixes，则不冻结任何参数（训练全部参数）

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        cpu_offload = (
            None
            if not self.config.model.fsdp_config.cpu_offload
            else CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
        )

        # 纯LLM训练：不再需要ignored_modules
        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 required for FSDP2"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            (p for p in self.fsdp_model.parameters() if p.requires_grad),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, "
                f"epochs {self.config.trainer.total_epochs}, total steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps, self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                self.optimizer, num_warmup_steps, self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")
    
    def _load_optimizer_and_scheduler(self, checkpoint_path: str):
        """
        从 checkpoint 加载 optimizer 和 lr_scheduler 状态
        
        Args:
            checkpoint_path: checkpoint 目录路径
        """
        checkpoint_state = self.load_checkpoint(checkpoint_path)
        if checkpoint_state is None:
            if self.device_mesh.get_rank() == 0:
                print("[WARN] Cannot load optimizer and scheduler state, will start from scratch")
            return
        
        fsdp_strategy = self.config.model.strategy
        
        try:
            # 加载 optimizer 状态
            if fsdp_strategy == "fsdp":
                optimizer_state = checkpoint_state.get("optimizer")
                if optimizer_state is not None:
                    try:
                        # FSDP 需要将 full optimizer state dict 转换为 sharded state dict
                        # 先获取 sharded optimizer state dict
                        from torch.distributed.fsdp import ShardedOptimStateDictConfig, StateDictType
                        
                        opt_cfg = ShardedOptimStateDictConfig(offload_to_cpu=False)
                        with FSDP.state_dict_type(
                            self.fsdp_model, 
                            StateDictType.SHARDED_STATE_DICT,
                            optim_state_dict_config=opt_cfg
                        ):
                            # 将 full optimizer state dict 转换为 sharded
                            sharded_optim_state = FSDP.shard_full_optim_state_dict(
                                optimizer_state, self.fsdp_model
                            )
                            # 加载 sharded optimizer state
                            self.optimizer.load_state_dict(sharded_optim_state)
                            if self.device_mesh.get_rank() == 0:
                                print("[Checkpoint] Loaded optimizer state")
                    except Exception as opt_e:
                        if self.device_mesh.get_rank() == 0:
                            print(f"[WARN] Failed to load FSDP optimizer state: {opt_e}")
                            print("[WARN] Will continue with fresh optimizer state")
            
            elif fsdp_strategy == "fsdp2":
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    set_optimizer_state_dict,
                )
                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                optimizer_state = checkpoint_state.get("optimizer")
                if optimizer_state is not None:
                    set_optimizer_state_dict(
                        self.fsdp_model, self.optimizer, optimizer_state, options=options
                    )
                    if self.device_mesh.get_rank() == 0:
                        print("[Checkpoint] Loaded optimizer state")
            
            # 加载 lr_scheduler 状态
            lr_scheduler_state = checkpoint_state.get("lr_scheduler")
            if lr_scheduler_state is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
                if self.device_mesh.get_rank() == 0:
                    print("[Checkpoint] Loaded lr_scheduler state")
                    
        except Exception as e:
            if self.device_mesh.get_rank() == 0:
                print(f"[WARN] Failed to load optimizer/scheduler state: {e}")
                import traceback
                traceback.print_exc()
                print("[WARN] Will continue training with fresh optimizer/scheduler state")

    # --------- 分类头安全前向 ---------
    def _safe_cls_forward(
        self,
        head_name: str,
        pooled_list: List[torch.Tensor],
        alpha_now: List[float],
        targets_list: List[int],
        zero_anchor: torch.Tensor,
        compute_dtype=torch.bfloat16,
    ):
        """
        head0: 3 类（0/1/2）
        head1: 2 类（0/1，其中 1 表示“有互动”，融合原来的 1/2）
        num_classes 由 alpha_now 的长度自动决定。
        """
        if len(pooled_list) == 0:
            return zero_anchor + 0.0, {"ce_mean": 0.0, "pt_mean": 0.0}

        feats = torch.stack(pooled_list, dim=0)  # [N, 2H]
        if feats.ndim != 2:
            feats = feats.view(feats.shape[0], -1)

        cls_head = self._get_attr_from_fsdp(head_name)
        assert hasattr(cls_head, "weight") and hasattr(cls_head, "bias"), f"{head_name} not a Linear-like head"
        head_w = cls_head.weight
        head_b = cls_head.bias

        if head_w.numel() == 0:
            raise RuntimeError(
                f"{head_name}.weight has numel=0 at runtime. "
                f"This indicates the head was not excluded from FSDP. "
                f"Ensure FSDP(..., ignored_modules=[model.cls_head0, model.cls_head1])."
            )

        feats = feats.to(device=head_w.device, dtype=head_w.dtype, non_blocking=True)
        expected_in = feats.size(-1)
        expected_out = len(alpha_now)

        need_reshape = (
            (head_w.ndim != 2)
            or (head_w.shape[0] != expected_out)
            or (head_w.shape[1] != expected_in)
        )
        if need_reshape:
            if head_w.numel() == expected_out * expected_in:
                W2 = head_w.view(expected_out, expected_in)
            else:
                twoH = expected_in
                if head_w.numel() == expected_out * twoH:
                    W2 = head_w.view(expected_out, twoH)
                else:
                    raise RuntimeError(
                        f"{head_name}.weight malformed: shape={tuple(head_w.shape)} numel={head_w.numel()}, "
                        f"expected ({expected_out},{expected_in}) or numel={expected_out*expected_in}. "
                        f"Likely a bad checkpoint overwrote the head."
                    )
        else:
            W2 = head_w

        bias2 = head_b.to(device=W2.device, dtype=W2.dtype) if head_b is not None else None
        logits_cls = F.linear(feats, W2, bias2)
        targets_tensor = torch.as_tensor(targets_list, device=W2.device, dtype=torch.long)

        seg_cls_loss, dbg = self._focal_loss_multiclass(logits_cls, targets_tensor, alpha_now)
        return seg_cls_loss, dbg

    # ----------------- 核心前向/损失：纯LLM训练模式 -----------------
    def _compute_loss_and_backward(self, batch, do_backward: bool = False):
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask_2d = batch.pop("loss_mask").to(self.device_name)  # [B, T]

        # 长度裁剪（左截断）
        L = input_ids.size(1)
        Lmax_cfg = int(getattr(self.config.data, "max_length", 100000))
        Lmax_model = int(getattr(self.model.config, "max_position_embeddings", Lmax_cfg))
        Lmax = min(Lmax_cfg, Lmax_model)
        if L > Lmax:
            start = L - Lmax
            input_ids = input_ids[:, start:]
            attention_mask = attention_mask[:, start:]
            position_ids = position_ids[:, start:]
            loss_mask_2d = loss_mask_2d[:, start:]

        ce_per_token = nn.CrossEntropyLoss(reduction="none")

        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # ===== 模型前向 =====
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits  # [B, T, V]

            zero_anchor = logits[..., :1, :1].sum() * 0.0

            # ===== 构造 LM 的 shift logits / labels =====
            labels_for_lm = input_ids[:, 1:].contiguous()       # 目标 token
            shift_logits = logits[..., :-1, :].contiguous()     # 预测这些 token
            shift_labels = labels_for_lm.contiguous()

            B, T = input_ids.shape
            V = shift_logits.size(-1)

            shift_logits_flat = shift_logits.view(-1, V)
            shift_labels_flat = shift_labels.view(-1).to(shift_logits_flat.device)
            per_token_loss_flat = ce_per_token(shift_logits_flat, shift_labels_flat).float()   # [B*(T-1)]
            per_token_loss = per_token_loss_flat.view(B, T - 1)                                # [B, T-1]

            # loss_mask_2d 与 input_ids 对齐，长度 T，shift 后用前 T-1 个
            lm_mask = loss_mask_2d[:, :-1].to(
                device=per_token_loss.device,
                dtype=per_token_loss.dtype,
            )  # [B, T-1]

            # ===== LM loss：标准 SFT，对所有 loss_mask=1 的 token 做 LM =====
            lm_num_total = (per_token_loss * lm_mask).sum()
            lm_den_total = lm_mask.sum()

            # 分布式聚合
            with torch.no_grad():
                torch.distributed.all_reduce(lm_num_total)
                torch.distributed.all_reduce(lm_den_total)

            loss = zero_anchor
            if float(lm_den_total.detach().item()) > 0.0:
                loss = lm_num_total / (lm_den_total + 1e-8)

        if do_backward:
            loss.backward()

        out = {
            "train/loss": float(loss.detach().item()),
            "train/lm_loss": float(loss.detach().item()),
        }
        return loss, out

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss_val = 0.0

        sum_metrics: Dict[str, float] = {}
        cnt_metrics: Dict[str, int] = {}

        rank0 = self.device_mesh.get_rank() == 0
        log_micro = LOG_MICRO_LOSSES

        for mb_idx, micro_batch in enumerate(micro_batches):
            loss, extras = self._compute_loss_and_backward(batch=micro_batch, do_backward=False)
            (loss / n_micro_batches).backward()
            step_loss_val += float(loss.detach().item())

            for k, v in extras.items():
                if v is None:
                    continue
                sum_metrics[k] = sum_metrics.get(k, 0.0) + float(v)
                cnt_metrics[k] = cnt_metrics.get(k, 0) + 1

            if log_micro and rank0:
                lm_loss = extras.get("train/lm_loss", 0.0)
                print(
                    f"[micro] {mb_idx+1}/{n_micro_batches} | "
                    f"lm_loss={lm_loss:.4f}"
                )

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.step()
        self.lr_scheduler.step()

        merged_metrics: Dict[str, float] = {}
        for k, total in sum_metrics.items():
            merged_metrics[k] = total / max(1, cnt_metrics.get(k, 1))

        merged_metrics["train/loss"] = step_loss_val / max(1, n_micro_batches)

        lr = self.lr_scheduler.get_last_lr()[0]
        merged_metrics["train/lr(1e-3)"] = lr * 1e3
        return merged_metrics

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, _ = self._compute_loss_and_backward(batch, do_backward=False)
            if _IS_CUDA:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif _IS_NPU:
                torch.distributed.all_reduce(loss)
            loss /= max(1, self.device_mesh.size(0))
        return loss

    def save_checkpoint(self, step, epoch=None):
        """
        保存 checkpoint，包括模型、optimizer、lr_scheduler 和训练状态
        
        Args:
            step: 当前训练步数
            epoch: 当前 epoch（可选）
        """
        rank = self.device_mesh.get_rank()
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        fsdp_strategy = self.config.model.strategy

        try:
            # 在开始保存前，确保所有rank都准备好
            torch.distributed.barrier()
            
            if rank == 0:
                print(f"[Checkpoint] Starting checkpoint save at step {step}...")

            if fsdp_strategy == "fsdp":
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                # 先收集模型状态（需要所有rank参与）
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                    full_state_dict = self.fsdp_model.state_dict()
                
                # 在rank 0上执行I/O操作
                if rank == 0:
                    os.makedirs(path, exist_ok=True)
                    self.model.save_pretrained(path, state_dict=full_state_dict)
                    self.tokenizer.save_pretrained(path)
                
                # 再次barrier，确保模型保存完成后再收集optimizer状态
                torch.distributed.barrier()
                
                # 收集optimizer状态（需要所有rank参与，这是最容易超时的操作）
                # 可以通过配置跳过optimizer状态保存以加快checkpoint速度
                save_optimizer_state = getattr(self.config.trainer, "save_optimizer_state", True)
                optimizer_state = None
                
                if save_optimizer_state:
                    if rank == 0:
                        print(f"[Checkpoint] Collecting optimizer state...")
                    try:
                        optimizer_state = FSDP.full_optim_state_dict(self.fsdp_model, self.optimizer)
                    except Exception as opt_e:
                        if rank == 0:
                            print(f"[WARN] Failed to collect optimizer state: {opt_e}")
                            print(f"[WARN] Will save checkpoint without optimizer state")
                        optimizer_state = None
                else:
                    if rank == 0:
                        print(f"[Checkpoint] Skipping optimizer state (configured to skip)")
                
                # 在rank 0上保存optimizer和scheduler状态
                if rank == 0:
                    checkpoint_state = {
                        "step": step,
                        "epoch": epoch if epoch is not None else (step // self.steps_per_epoch),
                        "optimizer": optimizer_state,
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    }
                    try:
                        torch.save(checkpoint_state, os.path.join(path, "training_state.pt"))
                        print(f"[Checkpoint] Saved training state to {path}")
                    except Exception as save_e:
                        print(f"[ERROR] Failed to save training state: {save_e}")

            elif fsdp_strategy == "fsdp2":
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_model_state_dict,
                )

                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                full_state_dict = get_model_state_dict(self.fsdp_model, options=options)
                
                # 在rank 0上执行I/O操作
                if rank == 0:
                    os.makedirs(path, exist_ok=True)
                    self.model.save_pretrained(path, state_dict=full_state_dict)
                    self.model_config.save_pretrained(path)
                    self.tokenizer.save_pretrained(path)
                
                # 再次barrier，确保模型保存完成后再收集optimizer状态
                torch.distributed.barrier()
                
                # 收集optimizer状态（FSDP2）
                # 可以通过配置跳过optimizer状态保存以加快checkpoint速度
                save_optimizer_state = getattr(self.config.trainer, "save_optimizer_state", True)
                optimizer_state = None
                
                if save_optimizer_state:
                    if rank == 0:
                        print(f"[Checkpoint] Collecting optimizer state (FSDP2)...")
                    try:
                        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
                        optimizer_state = get_optimizer_state_dict(self.fsdp_model, self.optimizer, options=options)
                    except Exception as opt_e:
                        if rank == 0:
                            print(f"[WARN] Failed to collect optimizer state: {opt_e}")
                            print(f"[WARN] Will save checkpoint without optimizer state")
                        optimizer_state = None
                else:
                    if rank == 0:
                        print(f"[Checkpoint] Skipping optimizer state (configured to skip)")
                
                # 在rank 0上保存optimizer和scheduler状态
                if rank == 0:
                    checkpoint_state = {
                        "step": step,
                        "epoch": epoch if epoch is not None else (step // self.steps_per_epoch),
                        "optimizer": optimizer_state,
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    }
                    try:
                        torch.save(checkpoint_state, os.path.join(path, "training_state.pt"))
                        print(f"[Checkpoint] Saved training state to {path}")
                    except Exception as save_e:
                        print(f"[ERROR] Failed to save training state: {save_e}")
            else:
                raise NotImplementedError(f"not implement {fsdp_strategy}")

            # 上传到HDFS（如果配置了）
            if rank == 0 and self.config.trainer.default_hdfs_dir:
                try:
                    hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                    hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
                    print(f"[Checkpoint] Uploaded checkpoint to HDFS")
                except Exception as hdfs_e:
                    print(f"[WARN] Failed to upload checkpoint to HDFS: {hdfs_e}")
                    print(f"[WARN] Checkpoint saved locally at {path}")
            
            # 最后barrier，确保所有rank都完成
            torch.distributed.barrier()
            
            if rank == 0:
                print(f"[Checkpoint] Successfully saved checkpoint at step {step} to {path}")
                
        except Exception as e:
            # 如果保存失败，记录错误但继续训练
            if rank == 0:
                print(f"[ERROR] Failed to save checkpoint at step {step}: {e}")
                import traceback
                traceback.print_exc()
                print(f"[WARN] Training will continue without saving checkpoint")
            # 尝试barrier，但如果通信已经失败，这可能会失败
            try:
                torch.distributed.barrier()
            except Exception:
                pass

    def load_checkpoint(self, checkpoint_path: str):
        """
        从 checkpoint 加载训练状态
        
        Args:
            checkpoint_path: checkpoint 目录路径（包含模型文件和 training_state.pt）
            
        Returns:
            dict: 包含 step, epoch, optimizer_state, lr_scheduler_state 的字典
        """
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if not os.path.exists(training_state_path):
            if self.device_mesh.get_rank() == 0:
                print(f"[WARN] training_state.pt not found in {checkpoint_path}, "
                      f"will only load model weights (cannot resume training state)")
            return None
        
        checkpoint_state = None
        if self.device_mesh.get_rank() == 0:
            checkpoint_state = torch.load(training_state_path, map_location="cpu")
            print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}")
            print(f"[Checkpoint] Resuming from step {checkpoint_state.get('step', 0)}, "
                  f"epoch {checkpoint_state.get('epoch', 0)}")
        
        # 广播 checkpoint_state 到所有进程
        if torch.distributed.is_initialized():
            obj_list = [checkpoint_state]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            checkpoint_state = obj_list[0]
        
        return checkpoint_state

    def fit(self):
        rank = self.device_mesh.get_rank()
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        # 检查是否需要从 checkpoint 恢复
        resume_from_checkpoint = getattr(self.config.trainer, "resume_from_checkpoint", None)
        global_step = 0
        start_epoch = 0
        last_valid_metric = None
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            checkpoint_state = self.load_checkpoint(resume_from_checkpoint)
            if checkpoint_state is not None:
                global_step = checkpoint_state.get("step", 0)
                start_epoch = checkpoint_state.get("epoch", 0)
                if rank == 0:
                    print(f"[Resume] Resuming training from step {global_step}, epoch {start_epoch}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        if rank == 0:
            print(f"Total training steps: {self.total_training_steps}")

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            # 计算当前 epoch 需要跳过的步数（仅在恢复训练的第一个 epoch）
            skip_steps = 0
            if epoch == start_epoch and resume_from_checkpoint:
                skip_steps = global_step % self.steps_per_epoch
            
            step_in_epoch = 0
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                # 跳过已经训练过的步数
                if step_in_epoch < skip_steps:
                    step_in_epoch += 1
                    continue
                
                global_step += 1
                step_in_epoch += 1
                data = TensorDict(
                    data, batch_size=self.config.data.train_batch_size
                ).to(get_device_name())
                metric = self.training_step(data)
                if rank == 0:
                    Tracking.log(tracking, data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and (
                    global_step % self.config.trainer.test_freq == 0
                )
                is_save_step = self.config.trainer.save_freq > 0 and (
                    global_step % self.config.trainer.save_freq == 0
                )

                if is_last_step or is_valid_step:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(
                            val_data,
                            batch_size=self.config.data.micro_batch_size_per_gpu,
                        ).to(get_device_name())
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        Tracking.log(
                            tracking,
                            data={"val/loss": float(val_loss.detach().item())},
                            step=global_step,
                        )
                        last_valid_metric = {
                            "val/loss": float(val_loss.detach().item())
                        }
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step, epoch=epoch)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


# ===================== 入口 =====================
def create_sft_dataset(data_paths, data_config, tokenizer):
    if data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    else:
        dataset_cls = SFTDataset
    return dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path, trust_remote_code=config.model.trust_remote_code
    )

    # 纯LLM训练：不再需要特殊分割符token
    # tokenizer = _ensure_special_tokens(
    #     tokenizer, [PSEP_TOKEN, CSTART_TOKEN, CEND_TOKEN, "<TL0>", "<TL1>"]
    # )
    tok_save_dir = os.path.join(
        config.trainer.default_local_dir, "tokenizer_with_spans"
    )
    if torch.distributed.get_rank() == 0:
        os.makedirs(tok_save_dir, exist_ok=True)
        tokenizer.save_pretrained(tok_save_dir)
    torch.distributed.barrier()
    tokenizer = AutoTokenizer.from_pretrained(
        tok_save_dir, trust_remote_code=config.model.trust_remote_code
    )
    try:
        tokenizer.truncation_side = "left"
    except Exception:
        pass
    try:
        tokenizer.model_max_length = int(getattr(config.data, "max_length", 100000))
    except Exception:
        pass

    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


if __name__ == "__main__":
    main()
