#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO 训练脚本 (v32.2-Resumable) — 支持 LoRA 续训 + 跳过已训数据
修改内容：
1. 增加了 --resume_path 参数
2. 支持加载 saved adapter 权重
3. 自动解析 step 数并跳过 DataLoader 前面的数据
"""

import os
import time
import json
import argparse
import re
import ast
from contextlib import nullcontext
from collections import Counter

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, PeftModel  # <=== 修改点：引入 PeftModel

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

SCRIPT_VERSION = "v32.2-Resumable"
ROOT_PARENT_KEY = "__ROOT__"
EPS = 1e-6

# ==================== reward switch (hard-coded) ====================
# 通过硬编码切换奖励函数：
# - "avg": 使用当前奖励函数（REWARD_CONFIG 加权后的 avg_reward）
# - "f1" : 使用 F1 作为奖励（范围 0~1）
REWARD_MODE = "f1"  # change to "f1" to use F1 reward

PSEP_TOKEN = "<|psep|>"
PSEP_BLOCK_START = "\n<POTENTIAL_SPANS>\n"
PSEP_BLOCK_END = "\n</POTENTIAL_SPANS>\n"

REWARD_CONFIG = {
    0: {
        "correct_rejection": 0.0,
        "true_positive": 1.2,
        "false_negative": -0.6,
        "false_positive": -0.8,
        "wrong_type": -0.3,
    },
    1: {
        "correct_rejection": 0.0,
        "true_positive": 4.0,
        "false_negative": -1.5,
        "false_positive": -2.0,
        "wrong_type": -1.0,
    },
}

def _ts():
    return time.strftime("%m-%d %H:%M:%S")

def _cuda_mem_str(device: torch.device):
    try:
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserv = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return f"{alloc:.1f}/{reserv:.1f}/{total:.1f}GB"
    except Exception:
        return "0.0/0.0/0.0GB"

def log_status(rank: int, msg: str, *, pbar=None, force_print=False):
    if rank != 0:
        return
    line = f"[{_ts()}] {msg}"
    if (pbar is not None) and (tqdm is not None) and (not force_print):
        try:
            pbar.write(line)
            return
        except Exception:
            pass
    print(line, flush=True)

def rank0_append_jsonl(path: str, obj: dict):
    if dist.get_rank() != 0:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def get_trainable_param(ddp_model: torch.nn.Module):
    for p in ddp_model.parameters():
        if p.requires_grad:
            return p
    return None

# ==================== potentials / psep ====================

def get_root_potential_objs(obj: dict) -> list:
    if not isinstance(obj, dict):
        return []
    rm = obj.get("reward_model")
    if isinstance(rm, dict):
        rp = rm.get("root_potential") or {}
    else:
        rp = obj.get("root_potential") or {}

    pots = []
    if isinstance(rp, dict):
        full = rp.get("full")
        if isinstance(full, list) and full:
            for it in full:
                if isinstance(it, dict):
                    name = (it.get("user_name") or "").strip()
                    if not name:
                        continue
                    pots.append({
                        "user_name": name,
                        "interests": it.get("interests") or [],
                        "interaction_count": it.get("interaction_count", it.get("interaction_cnt", it.get("interaction", 0))),
                    })
        elif isinstance(rp.get("user_names"), list):
            for name in rp.get("user_names") or []:
                name = (name or "").strip()
                if name:
                    pots.append({"user_name": name, "interests": [], "interaction_count": 0})
    return pots

_PSEP_BLOCK_RE = re.compile(r"(<POTENTIAL_SPANS>\s*)(?P<body>.*?)(\s*</POTENTIAL_SPANS>)", re.DOTALL)
_USER_NAME_RE = re.compile(r'"user_name"\s*:\s*"([^"]+)"')
_DEPTH_FIELD_RE = re.compile(r'("depth"\s*:\s*)(\d+)')

def extract_psep_block(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    m = _PSEP_BLOCK_RE.search(text)
    return m.group(0) if m else ""

def extract_cand_order_from_content(text: str) -> list:
    blk = extract_psep_block(text)
    if not blk:
        return []
    names = []
    for m in _USER_NAME_RE.finditer(blk):
        n = (m.group(1) or "").strip()
        if n:
            names.append(n)
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def bump_psep_depths(block: str, new_depth: int) -> str:
    if not block:
        return block
    dval = int(new_depth)
    def _sub(m: re.Match) -> str:
        return f"{m.group(1)}{dval}"
    return _DEPTH_FIELD_RE.sub(_sub, block)

def render_psep_block_from_list(pots: list, depth: int) -> str:
    parts = [PSEP_BLOCK_START]
    dval = int(depth)
    for p in pots or []:
        uname = (p.get("user_name") or "").strip()
        if not uname:
            continue
        blk = {
            "user_name": uname,
            "interests": p.get("interests") or [],
            "depth": dval,
            "interaction_count": p.get("interaction_count", 0),
        }
        parts.append(PSEP_TOKEN)
        parts.append(json.dumps(blk, ensure_ascii=False, separators=(",", ":")))
        parts.append(PSEP_TOKEN)
    parts.append(PSEP_BLOCK_END)
    return "".join(parts)

def parse_psep_candidates(block: str) -> dict:
    if not block:
        return {}
    pieces = block.split(PSEP_TOKEN)
    mp = {}
    for seg in pieces:
        seg = seg.strip()
        if not (seg.startswith("{") and seg.endswith("}")):
            continue
        try:
            obj = json.loads(seg)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        uname = (obj.get("user_name") or "").strip()
        if not uname:
            continue
        mp[uname] = {
            "interests": obj.get("interests") or [],
            "interaction_count": obj.get("interaction_count", 0),
        }
    return mp

_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

def extract_username_from_content(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = _USERNAME_LINE_RE.search(s)
    return (m.group("name").strip() if m else "")

def build_child_content(
    *,
    child_username: str,
    child_text: str,
    child_interests: list,
    historical_names: list,
    psep_block: str,
    depth: int,
) -> str:
    u = (child_username or "").strip()
    c = child_text if isinstance(child_text, str) else ("" if child_text is None else str(child_text))
    ui = child_interests or []
    hist = historical_names or []
    prefix = (
        f"username: {u}\n"
        f"content:\n{c}\n\n"
        f"userinterest: {json.dumps(ui, ensure_ascii=False)}\n"
        f"historicalinteractors: {json.dumps(hist, ensure_ascii=False)}\n"
        f"potentialspan:"
    )
    blk = bump_psep_depths(psep_block, depth)
    return prefix + blk

# ==================== GT ====================

def get_view_filtered_gold_types(depth, parent_name, cond_gt, cands):
    pk = (parent_name or "").strip() or ROOT_PARENT_KEY
    golds = []
    if isinstance(cond_gt, list) and depth < len(cond_gt):
        layer = cond_gt[depth] or []
        if isinstance(layer, dict):
            layer = [layer]
        for m in layer:
            if isinstance(m, dict):
                for k, v in m.items():
                    if (k or "").strip() == pk:
                        golds = [str(x) for x in (v if isinstance(v, list) else [v])]
                        break
    gset = set(golds)
    return [1 if c in gset else 0 for c in cands]

# ==================== parse ====================

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_MODEL_PREFIX_RE = re.compile(r"^\s*[\w\-]+:\s*", re.UNICODE)
_FORBIDDEN_KEYS = {"interests", "depth", "interaction_count", "interaction_cnt", "idx"}

def strip_model_prefix_and_code_fence(text: str) -> str:
    s = (text or "").strip()
    s = _MODEL_PREFIX_RE.sub("", s, count=1)
    m = _CODE_FENCE_RE.search(s)
    return m.group("body").strip() if m else s

def parse_generated_json_array(gen_text: str, cand_order: list):
    if not cand_order:
        return [], [], {}, False, "no_cands"

    cleaned = strip_model_prefix_and_code_fence(gen_text)
    l, r = cleaned.find("["), cleaned.rfind("]")
    if l != -1 and r != -1 and r > l:
        cleaned = cleaned[l:r+1]

    parsed_obj = None
    try:
        parsed_obj = json.loads(cleaned)
    except Exception:
        try:
            parsed_obj = ast.literal_eval(cleaned)
        except Exception:
            parsed_obj = None

    if parsed_obj is None or not isinstance(parsed_obj, list):
        return [0]*len(cand_order), [], {u: "" for u in cand_order}, False, "json_not_list"

    data_map = {}
    for item in parsed_obj:
        if not isinstance(item, dict):
            continue
        u = (item.get("user_name") or "").strip()
        if not u:
            continue
        if any(k in item for k in _FORBIDDEN_KEYS):
            return [0]*len(cand_order), [], {u: "" for u in cand_order}, False, "forbidden_keys_copy_input"
        data_map[u] = item

    pred_types, pred_names = [], []
    content_map = {u: "" for u in cand_order}

    for uname in cand_order:
        it = data_map.get(uname)
        t = 0
        c = ""
        if it:
            try:
                t = int(it.get("type", 0))
            except Exception:
                t = 0
            c = it.get("content", "")
            if c is None:
                c = ""
            c = str(c)
        pred_types.append(t)
        content_map[uname] = c
        if t in (1, 2):
            pred_names.append(uname)

    return pred_types, pred_names, content_map, True, "ok"

# ==================== reward ====================

def score_step(depth: int, pred_types: list, gt_types: list):
    cfg = REWARD_CONFIG.get(depth, REWARD_CONFIG[0])

    CR = FP = FN = TP = TYPE_ERR = WRONG_TYPE = 0
    tp = fp = fn = 0
    sum_reward = 0.0
    n = max(len(gt_types), 1)

    for p, g in zip(pred_types, gt_types):
        if g == 0 and p == 0:
            r = cfg["correct_rejection"]; CR += 1
        elif g == 1 and p == 1:
            r = cfg["true_positive"]; TP += 1
        elif g == 1 and p == 0:
            r = cfg["false_negative"]; FN += 1
        elif g == 0 and p > 0:
            r = cfg["false_positive"]; FP += 1
        else:
            r = cfg["wrong_type"]; WRONG_TYPE += 1
            if g == 1 and p > 0 and p != 1:
                TYPE_ERR += 1

        sum_reward += float(r)

        ppos = (p > 0)
        gpos = (g > 0)
        if ppos and gpos:
            tp += 1
        elif ppos and (not gpos):
            fp += 1
        elif (not ppos) and gpos:
            fn += 1

    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    avg_reward = sum_reward / n
    conf = {"CR": CR, "FP": FP, "FN": FN, "TP": TP, "TYPE_ERR": TYPE_ERR, "WRONG_TYPE": WRONG_TYPE}
    f1_stats = {"tp": tp, "fp": fp, "fn": fn, "f1": float(f1)}
    return float(avg_reward), float(sum_reward), conf, f1_stats

# ==================== rollout ====================

def rollout_single_step(
    model, tokenizer,
    chat_content: str, system_prompt: str,
    cand_order: list,
    gt_types: list,
    max_input_tokens: int,
    gen_params: dict,
    device: torch.device,
    depth: int,
    debug_file: str = None,
    global_step: int = 0,
    tag: str = "",
    parent_key: str = "",
):
    base_model = getattr(model, "module", model)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": chat_content}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer([chat_text], return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
    enc = {k: v.to(device) for k, v in enc.items()}

    actual_gen_params = gen_params.copy()
    actual_gen_params["pad_token_id"] = tokenizer.pad_token_id
    actual_gen_params["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        out = base_model.generate(**enc, **actual_gen_params, use_cache=True)

    new_ids = out[0, enc["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    pred_types, pred_names, content_map, success, reason = parse_generated_json_array(gen_text, cand_order)

    if (not gt_types) or (len(gt_types) != len(cand_order)):
        success = False
        reason = "gt_empty_or_mismatch"

    if not success:
        avg_reward = 0.0
        sum_reward = 0.0
        conf = {"CR": 0, "FP": 0, "FN": 0, "TP": 0, "TYPE_ERR": 0, "WRONG_TYPE": 0}
        f1_stats = {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0}
        status = "❌ FAIL"
        reward_used = 0.0
    else:
        avg_reward, sum_reward, conf, f1_stats = score_step(depth, pred_types, gt_types)
        status = "✅ OK"
        if str(REWARD_MODE).lower() == "f1":
            reward_used = float(f1_stats.get("f1", 0.0))
        else:
            reward_used = float(avg_reward)

    type1_names = []
    if success and cand_order and pred_types:
        for uname, t in zip(cand_order, pred_types):
            if t == 1:
                type1_names.append(uname)

    if dist.get_rank() == 0 and debug_file:
        rank0_append_jsonl(debug_file, {
            "step": int(global_step),
            "depth": int(depth),
            "tag": tag,
            "parent": parent_key,
            "status": status,
            "reason": reason,
            "reward_mode": str(REWARD_MODE),
            "reward_used": float(reward_used),
            "reward_avg": float(avg_reward),
            "reward_sum": float(sum_reward),
            "n_cands": int(len(cand_order)),
            "conf": conf,
            "f1": f1_stats,
            "gen_text": gen_text,
        })

    return {
        "chat_text": chat_text,
        "gen_text": gen_text,
        "reward": float(reward_used),
        "reward_sum": float(sum_reward),
        "success": bool(success),
        "pred_type1_names": type1_names,
        "content_map": content_map,
    }

# ==================== logprob (always grad) ====================

def compute_log_prob_mean(
    model, tokenizer,
    chat_text: str,
    gen_text: str,
    max_tokens: int,
    device,
    *,
    max_logprob_tokens: int = 1536,
):
    with torch.enable_grad():
        enc = tokenizer([chat_text], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        inp = enc["input_ids"].to(device)
        inp_len = inp.shape[1]

        gen_ids = tokenizer(
            [gen_text],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"].to(device)

        if gen_ids.shape[1] == 0:
            return None, {"prompt_len": int(inp_len), "gen_len": 0, "full_len": int(inp_len)}

        max_gen = max_tokens - inp_len
        if max_gen <= 0:
            return None, {"prompt_len": int(inp_len), "gen_len": 0, "full_len": int(inp_len)}
        if gen_ids.shape[1] > max_gen:
            gen_ids = gen_ids[:, :max_gen]
        gen_len = gen_ids.shape[1]

        total_cap = int(max_logprob_tokens)
        if total_cap < 64:
            total_cap = 64

        keep_prompt = total_cap - gen_len
        if keep_prompt < 1:
            gen_ids = gen_ids[:, :max(total_cap - 1, 1)]
            gen_len = gen_ids.shape[1]
            keep_prompt = total_cap - gen_len
            if keep_prompt < 1:
                keep_prompt = 1

        prompt_tail = inp[0, max(0, inp_len - keep_prompt):]
        new_inp_len = prompt_tail.shape[0]
        full = torch.cat([prompt_tail, gen_ids[0]], dim=0)

        out = model(input_ids=full.unsqueeze(0), use_cache=False)
        logits = out.logits[0, new_inp_len - 1: new_inp_len - 1 + gen_len]
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        token_logp = logp.gather(dim=-1, index=gen_ids[0].unsqueeze(-1)).squeeze(-1)

        info = {"prompt_len": int(new_inp_len), "gen_len": int(gen_len), "full_len": int(full.shape[0])}
        return token_logp.mean(), info

# ==================== reward stats sync ====================

def sync_reward_stats(reward_value: float, valid_for_adv: bool, device: torch.device):
    r = torch.tensor([float(reward_value)], device=device, dtype=torch.float32)
    v = torch.tensor([1.0 if valid_for_adv else 0.0], device=device, dtype=torch.float32)

    local_sum = r * v
    local_sumsq = (r * r) * v
    local_cnt = v

    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)

    g_cnt = local_cnt.clamp_min(1.0)
    g_mean = local_sum / g_cnt
    g_var = (local_sumsq / g_cnt) - g_mean * g_mean
    g_std = torch.sqrt(g_var.clamp_min(EPS))
    return g_mean, g_std, local_sum, local_cnt

# ==================== dataset ====================

class GrpoDataset(Dataset):
    def __init__(self, path):
        if dist.get_rank() == 0:
            try:
                self.df = pd.read_parquet(path)
            except Exception:
                self.df = pd.read_json(path, lines=True)
            self.data = self.df.to_dict("records")
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# ==================== parent select ====================

def select_depth1_parents_vote(type1_lists_world, *, world_size: int, max_parents: int, min_votes: int = 2):
    counter = Counter()
    for lst in type1_lists_world:
        for n in lst or []:
            counter[n] += 1

    cand_total = len(counter)

    # 入队规则：票数 >= min_votes
    # 注：如果 world_size < min_votes，则这个阈值不可能满足；此时自动降到 world_size，避免“永远选不到”
    thr = int(min_votes) if int(min_votes) > 0 else 1
    if world_size > 0 and thr > world_size:
        thr = int(world_size)

    picked = [n for n, c in counter.items() if c >= thr]
    mode = f"votes>={thr}"

    picked = sorted(picked, key=lambda n: (-counter[n], n))[:max_parents]
    info = {
        "mode": mode,
        "cand_total": cand_total,
        "picked": len(picked),
        "picked_list": picked,
        "max_parents": max_parents,
        "min_votes": int(min_votes),
        "world_size": int(world_size),
    }
    return picked, info

# ==================== main ====================

def main_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--roots_per_update", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--save_steps", type=int, default=10)

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen_max_new_tokens", type=int, default=1024)
    parser.add_argument("--gen_top_p", type=float, default=0.9)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)

    parser.add_argument("--depth_limit", type=int, default=1)
    parser.add_argument("--max_depth1_parents", type=int, default=2)
    parser.add_argument("--min_parent_votes", type=int, default=2, help="Depth1 parent enters queue if votes >= this threshold (default: 2)")

    # <=== 修改点：新增参数，指定续训的 checkpoint 路径 ===
    parser.add_argument("--resume_path", type=str, default=None, help="Path to LoRA checkpoint (adapter) to resume training") 

    parser.add_argument("--max_logprob_tokens", type=int, default=1536)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--no_grad_ckpt", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(42 + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    debug_log_path = os.path.join(args.output_dir, "debug_io.jsonl")
    
    # 续训时，如果是追加模式，不要覆盖原来的 debug 日志
    write_mode = "a" if args.resume_path else "w"
    if rank == 0:
        with open(debug_log_path, write_mode, encoding="utf-8") as f:
            if not args.resume_path:
                f.write("")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    DUMMY_CHAT_TEXT = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Return an empty JSON array []"}],
        tokenize=False, add_generation_prompt=True
    )
    DUMMY_GEN_TEXT = "[]"

    use_ckpt = True
    if args.no_grad_ckpt:
        use_ckpt = False
    if args.grad_ckpt:
        use_ckpt = True

    if args.use_lora:
        # <=== 修改点：Base Model 加载保持不变
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        
        # 梯度检查点
        if use_ckpt:
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()

        # <=== 修改点：支持续训的 LoRA 逻辑
        if args.resume_path:
            log_status(rank, f"Resuming LoRA adapter from: {args.resume_path}")
            # 加载现有 Adapter 并设为可训练
            base = PeftModel.from_pretrained(base, args.resume_path, is_trainable=True)
        else:
            log_status(rank, "Initializing new LoRA adapter...")
            cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            base = get_peft_model(base, cfg)
        
        # 确保梯度检查点配置没有被覆盖
        if use_ckpt and hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
            
        base = base.to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        if use_ckpt:
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()

    model = DDP(base, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    trainable_param = get_trainable_param(model)
    if trainable_param is None:
        raise RuntimeError("No trainable parameters found (all requires_grad=False). Check LoRA/freeze settings.")

    optimizer_name = "torch.AdamW"
    try:
        import bitsandbytes as bnb
        if hasattr(bnb.optim, "PagedAdamW8bit"):
            optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)
            optimizer_name = "bnb.PagedAdamW8bit"
        else:
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
            optimizer_name = "bnb.AdamW8bit"
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_status(rank, f"start {SCRIPT_VERSION} world_size={world_size} use_lora={bool(args.use_lora)} "
                     f"grad_ckpt={int(use_ckpt)} optimizer={optimizer_name} mem={_cuda_mem_str(device)}")

    ds = GrpoDataset(args.data) if rank == 0 else []
    # shuffle=True 配合 set_seed 保证每次运行的顺序是一样的，这样才能按步数跳过
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda x: x) if rank == 0 else []

    total_roots = len(ds) if rank == 0 else 0
    c_ts = torch.tensor([total_roots], device=device)
    dist.broadcast(c_ts, src=0)
    total_roots = int(c_ts.item())
    iter_loader = iter(loader) if rank == 0 else None

    # === [新增] 解析需要跳过的步数 ===
    start_global_step = 0
    if args.resume_path:
        import re
        # 尝试匹配 ckpt-step100-upd10 中的 100
        match = re.search(r"step(\d+)", args.resume_path)
        if match:
            start_global_step = int(match.group(1))
            log_status(rank, f"Found resume step: {start_global_step}. Skipping first {start_global_step} samples.")
        else:
            log_status(rank, "WARN: Could not parse step count from resume_path (e.g. 'step100'). Starting from 0.")
    # ==============================

    gen_cfg = {
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.gen_top_p),
        "max_new_tokens": int(args.gen_max_new_tokens),
    }
    if not args.do_sample:
        gen_cfg.pop("temperature", None)
        gen_cfg.pop("top_p", None)

    # update stats (rank0)
    upd_loss_sum = 0.0
    upd_valid_lp = 0
    upd_oom_f = 0
    upd_oom_b = 0
    upd_roots = 0

    for epoch in range(args.epochs):
        if rank == 0:
            iter_loader = iter(loader)

        pbar = None
        if rank == 0 and tqdm is not None:
            pbar = tqdm(total=total_roots, desc=f"epoch {epoch+1}/{args.epochs}",
                        dynamic_ncols=True, leave=True, mininterval=1.0)

        for root_i in range(total_roots):
            global_root_step = epoch * total_roots + (root_i + 1)

            # === [新增] 跳过已经训练过的步数 ===
            if global_root_step <= start_global_step:
                # 即使是跳过，rank0 也要消耗一下迭代器，保证随机数序列/数据顺序一致
                if rank == 0:
                    try:
                        next(iter_loader)
                        if pbar is not None:
                            pbar.update(1)
                    except StopIteration:
                        pass
                continue
            # =================================

            is_last = (global_root_step % args.roots_per_update == 0)

            obj = [None]
            if rank == 0:
                try:
                    obj = [next(iter_loader)[0]]
                except Exception:
                    obj = [None]
            dist.broadcast_object_list(obj, src=0)
            root_sample = obj[0]

            # ---- parse shared ----
            parsed_ok = False
            root_json = None
            rm = None
            cond_gt = []
            sys_prompt = ""
            pots = []
            root_content = ""
            psep_block = ""
            cand_order = []
            psep_user_info = {}
            root_author = ""

            if root_sample:
                try:
                    root_json = json.loads(root_sample["root_user_json"])
                    rm_str = root_sample["reward_model"]
                    rm = json.loads(rm_str) if isinstance(rm_str, str) else rm_str
                    cond_gt = rm.get("ground_truth", {}).get("cond_gt_by_turn") or []
                    sys_prompt = root_sample.get("system_prompt", "") or ""
                    pots = get_root_potential_objs(rm) or get_root_potential_objs(root_json)
                    root_content = root_json.get("content", "") or ""
                    psep_block = extract_psep_block(root_content)
                    if not psep_block and pots:
                        psep_block = render_psep_block_from_list(pots, depth=0)
                    cand_order = [p["user_name"] for p in pots] if pots else extract_cand_order_from_content(root_content)
                    if ("<POTENTIAL_SPANS>" not in root_content) and psep_block:
                        root_content = root_content.rstrip() + psep_block
                    psep_user_info = parse_psep_candidates(psep_block)
                    root_author = extract_username_from_content(root_content) or ""
                    parsed_ok = True
                except Exception:
                    parsed_ok = False

            sync_ctx = (model.no_sync() if not is_last else nullcontext())
            with sync_ctx:
                per_root_loss_detached = 0.0
                per_root_valid_lp = 0
                per_root_oom_f = 0
                per_root_oom_b = 0
                per_root_succ = 0

                # ========== depth0 rollout ==========
                if parsed_ok and cand_order:
                    gt0 = get_view_filtered_gold_types(0, ROOT_PARENT_KEY, cond_gt, cand_order)
                    model.eval()
                    res0 = rollout_single_step(
                        model, tokenizer,
                        chat_content=root_content,
                        system_prompt=sys_prompt,
                        cand_order=cand_order,
                        gt_types=gt0,
                        max_input_tokens=args.max_input_tokens,
                        gen_params=gen_cfg,
                        device=device,
                        depth=0,
                        debug_file=debug_log_path,
                        global_step=global_root_step,
                        tag=f"d0_r{rank}",
                        parent_key=ROOT_PARENT_KEY,
                    )
                else:
                    res0 = {"chat_text": "", "gen_text": "", "reward": 0.0, "reward_sum": 0.0,
                            "success": False, "pred_type1_names": [], "content_map": {}}

                # gather type1
                local_type1 = res0.get("pred_type1_names") or []
                type1_lists_world = [None for _ in range(world_size)]
                dist.all_gather_object(type1_lists_world, local_type1)

                picked_parents = []
                select_info = {"mode": "none", "cand_total": 0, "picked": 0, "picked_list": [], "max_parents": args.max_depth1_parents}
                if rank == 0 and args.depth_limit >= 1:
                    picked_parents, select_info = select_depth1_parents_vote(
                        type1_lists_world,
                        world_size=world_size,
                        max_parents=args.max_depth1_parents,
                        min_votes=args.min_parent_votes,
                    )
                    log_status(rank, f"depth1 parent_select: {select_info}", pbar=pbar)
                obj2 = [picked_parents]
                dist.broadcast_object_list(obj2, src=0)
                picked_parents = obj2[0] or []

                groups_per_root = 1 + (len(picked_parents) if args.depth_limit >= 1 else 0)

                # ========== depth0 logprob/backward（所有 rank 都做一次） ==========
                model.train()
                chat0 = res0["chat_text"] if res0["chat_text"] else DUMMY_CHAT_TEXT
                gen0 = res0["gen_text"].strip() if isinstance(res0["gen_text"], str) else ""
                if not gen0:
                    gen0 = DUMMY_GEN_TEXT

                mask0 = 1.0 if res0.get("success", False) else 0.0

                try:
                    lp0, _ = compute_log_prob_mean(
                        model, tokenizer,
                        chat0, gen0,
                        args.max_input_tokens, device,
                        max_logprob_tokens=args.max_logprob_tokens,
                    )
                    if lp0 is None:
                        # 极端情况下 max_gen<=0；此时所有 rank 都会 None（输入一致）
                        lp0 = trainable_param.sum() * 0.0
                except torch.cuda.OutOfMemoryError:
                    per_root_oom_f += 1
                    torch.cuda.empty_cache()
                    # 仍然要保持一致路径：用 dummy（注意：这里可能导致少一次 forward，但 OOM 通常同批次各 rank 一致）
                    lp0 = trainable_param.sum() * 0.0

                g_mean0, g_std0, g_sum0, g_cnt0 = sync_reward_stats(res0["reward"], bool(mask0), device)

                if dist.get_rank() == 0:
                    rank0_append_jsonl(debug_log_path, {
                        "step": int(global_root_step),
                        "depth": 0,
                        "tag": "d0_reward_total_world",
                        "parent": ROOT_PARENT_KEY,
                        "reward_total_world": float(g_sum0.item()),
                        "valid_cnt_world": float(g_cnt0.item()),
                        "reward_mean_valid": float(g_mean0.item()),
                        "reward_std_valid": float(g_std0.item()),
                    })

                try:
                    if mask0 > 0:
                        adv0 = (torch.tensor(res0["reward"], device=device, dtype=torch.float32) - g_mean0) / g_std0
                    else:
                        adv0 = torch.tensor(0.0, device=device, dtype=torch.float32)
                    coef0 = (torch.tensor(mask0, device=device) * adv0).detach()
                    loss0 = -(coef0 * lp0) / float(args.roots_per_update) / float(groups_per_root)
                    if not loss0.requires_grad:
                        loss0 = trainable_param.sum() * 0.0
                    loss0.backward()
                    per_root_loss_detached += float(loss0.detach().item())
                    if mask0 > 0:
                        per_root_valid_lp += 1
                        per_root_succ += 1
                except torch.cuda.OutOfMemoryError:
                    per_root_oom_b += 1
                    torch.cuda.empty_cache()
                    try:
                        (trainable_param.sum() * 0.0).backward()
                    except Exception:
                        pass

                # ========== depth1 groups ==========
                if args.depth_limit >= 1 and picked_parents and parsed_ok and cand_order:
                    hist = [root_author] if root_author else []

                    for pi, parent_name in enumerate(picked_parents):
                        local_parent_text = (res0.get("content_map") or {}).get(parent_name, "") or ""
                        parent_text_world = [None for _ in range(world_size)]
                        dist.all_gather_object(parent_text_world, local_parent_text)

                        best_text = ""
                        if rank == 0:
                            best_text = max([t or "" for t in parent_text_world], key=lambda x: len(x))
                        objt = [best_text]
                        dist.broadcast_object_list(objt, src=0)
                        best_text = objt[0] or ""

                        ui = (psep_user_info.get(parent_name) or {}).get("interests") or []
                        child_content = build_child_content(
                            child_username=parent_name,
                            child_text=best_text,
                            child_interests=ui,
                            historical_names=hist + ([parent_name] if parent_name else []),
                            psep_block=psep_block,
                            depth=1,
                        )
                        gt1 = get_view_filtered_gold_types(1, parent_name, cond_gt, cand_order)

                        model.eval()
                        res1 = rollout_single_step(
                            model, tokenizer,
                            chat_content=child_content,
                            system_prompt=sys_prompt,
                            cand_order=cand_order,
                            gt_types=gt1,
                            max_input_tokens=args.max_input_tokens,
                            gen_params=gen_cfg,
                            device=device,
                            depth=1,
                            debug_file=debug_log_path,
                            global_step=global_root_step,
                            tag=f"d1_p{pi}_r{rank}",
                            parent_key=parent_name,
                        )

                        # 所有 rank 都做一次 logprob/backward
                        model.train()
                        chat1 = res1["chat_text"] if res1["chat_text"] else DUMMY_CHAT_TEXT
                        gen1 = res1["gen_text"].strip() if isinstance(res1["gen_text"], str) else ""
                        if not gen1:
                            gen1 = DUMMY_GEN_TEXT
                        mask1 = 1.0 if res1.get("success", False) else 0.0

                        try:
                            lp1, _ = compute_log_prob_mean(
                                model, tokenizer,
                                chat1, gen1,
                                args.max_input_tokens, device,
                                max_logprob_tokens=args.max_logprob_tokens,
                            )
                            if lp1 is None:
                                lp1 = trainable_param.sum() * 0.0
                        except torch.cuda.OutOfMemoryError:
                            per_root_oom_f += 1
                            torch.cuda.empty_cache()
                            lp1 = trainable_param.sum() * 0.0

                        g_mean1, g_std1, g_sum1, g_cnt1 = sync_reward_stats(res1["reward"], bool(mask1), device)

                        r_list_world = [None for _ in range(world_size)]
                        dist.all_gather_object(r_list_world, float(res1["reward"]))
                        if dist.get_rank() == 0:
                            rank0_append_jsonl(debug_log_path, {
                                "step": int(global_root_step),
                                "depth": 1,
                                "tag": "d1_parent_reward_total_world",
                                "parent": parent_name,
                                "reward_total_world": float(g_sum1.item()),
                                "reward_list_world": r_list_world,
                                "valid_cnt_world": float(g_cnt1.item()),
                                "reward_mean_valid": float(g_mean1.item()),
                                "reward_std_valid": float(g_std1.item()),
                                "parent_select": select_info,
                            })

                        try:
                            if mask1 > 0:
                                adv1 = (torch.tensor(res1["reward"], device=device, dtype=torch.float32) - g_mean1) / g_std1
                            else:
                                adv1 = torch.tensor(0.0, device=device, dtype=torch.float32)
                            coef1 = (torch.tensor(mask1, device=device) * adv1).detach()
                            loss1 = -(coef1 * lp1) / float(args.roots_per_update) / float(groups_per_root)
                            if not loss1.requires_grad:
                                loss1 = trainable_param.sum() * 0.0
                            loss1.backward()
                            per_root_loss_detached += float(loss1.detach().item())
                            if mask1 > 0:
                                per_root_valid_lp += 1
                                per_root_succ += 1
                        except torch.cuda.OutOfMemoryError:
                            per_root_oom_b += 1
                            torch.cuda.empty_cache()
                            try:
                                (trainable_param.sum() * 0.0).backward()
                            except Exception:
                                pass

                # update stats (rank0)
                if rank == 0:
                    upd_loss_sum += float(per_root_loss_detached)
                    upd_valid_lp += int(per_root_valid_lp)
                    upd_oom_f += int(per_root_oom_f)
                    upd_oom_b += int(per_root_oom_b)
                    upd_roots += 1

            # pbar
            if rank == 0 and pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "root": f"{root_i+1}/{total_roots}",
                    "step": int(global_root_step),
                    "upd": 1 if is_last else 0,
                    "loss": f"{per_root_loss_detached:.4f}",
                    "succ": int(per_root_succ),
                    "v_lp": int(per_root_valid_lp),
                    "oom_f": int(per_root_oom_f),
                    "oom_b": int(per_root_oom_b),
                    "mem": _cuda_mem_str(device),
                }, refresh=True)

            # optimizer step
            if is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                oom_in_step = False
                try:
                    optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    oom_in_step = True
                    torch.cuda.empty_cache()
                finally:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except TypeError:
                        optimizer.zero_grad()

                update_idx = global_root_step // args.roots_per_update
                if rank == 0:
                    log_status(
                        rank,
                        f"UPDATE {update_idx} @global_root_step={global_root_step} "
                        f"(roots_in_update={upd_roots}) "
                        f"loss_update_sum={upd_loss_sum:.6f} valid_lp={upd_valid_lp} "
                        f"oom_f={upd_oom_f} oom_b={upd_oom_b} optimizer_step_oom={int(oom_in_step)} "
                        f"mem={_cuda_mem_str(device)}",
                        pbar=pbar
                    )

                    if update_idx > 0 and (update_idx % args.save_steps == 0) and (not oom_in_step):
                        ckpt_dir = os.path.join(args.output_dir, f"ckpt-step{global_root_step}-upd{update_idx}")
                        try:
                            model.module.save_pretrained(ckpt_dir)
                            log_status(rank, f"saved checkpoint: {ckpt_dir}", pbar=pbar)
                        except Exception as e:
                            log_status(rank, f"WARN: save_pretrained failed: {e}", pbar=pbar)

                    upd_loss_sum = 0.0
                    upd_valid_lp = 0
                    upd_oom_f = 0
                    upd_oom_b = 0
                    upd_roots = 0

        if rank == 0 and pbar is not None:
            pbar.close()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main_loop()