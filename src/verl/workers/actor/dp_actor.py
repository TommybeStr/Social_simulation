# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor  (DP/FSDP)

【改动说明】
- 在 update_policy() 中加入“全局样本数对齐 + 空占位陪跑”：
  1) 用 all_reduce(SUM) 得到 global_n 并据此计算全局 K=ceil(global_n/ppo_mini_batch_size)；
  2) 每个 mini-batch 轮次中，用 all_reduce(MAX) 对齐 micro-batch 的累计步数，并对不足的 rank 用 zero-loss 占位补齐；
  3) 保证所有 rank 在每个轮次进入相同次数的梯度 AllReduce，避免 NCCL 超时。
"""

import itertools
import logging
import math
import os
from typing import Tuple, List

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    # -------------------- Comm helpers -------------------- #
    @staticmethod
    def _allreduce_int_sum(x: int) -> int:
        t = torch.tensor([x], device=get_device_id(), dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return int(t.item())

    @staticmethod
    def _allreduce_int_max(x: int) -> int:
        t = torch.tensor([x], device=get_device_id(), dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return int(t.item())

    @staticmethod
    def _bcast_int(x: int, src: int = 0) -> int:
        t = torch.tensor([x], device=get_device_id(), dtype=torch.int64)
        torch.distributed.broadcast(t, src=src)
        return int(t.item())

    @staticmethod
    def _dummy_zero_loss(module: nn.Module) -> torch.Tensor:
        """
        构造一个与所有可训练参数相关联、值恒为 0 的 loss，
        用于“占位陪跑”触发与其它 rank 完全一致的 FSDP/DDP AllReduce。
        """
        loss = None
        for p in module.parameters():
            if p.requires_grad:
                z = (p.sum() * 0.0)
                loss = z if loss is None else (loss + z)
        if loss is None:
            # 极端情况：没有可训练参数
            loss = torch.zeros((), device=get_device_id(), requires_grad=True)
        return loss

    # -------------------- Core fwd utils -------------------- #
    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                # MiniCPM-o 特例
                if (key == "pixel_values" and isinstance(micro_batch["multi_modal_inputs"][0]["pixel_values"], list)) or key == "image_bound" or key == "tgt_sizes":
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (1, total_nnz)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # rotary 对齐
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if "multi_modal_inputs" in micro_batch.keys():
                    if "image_bound" in multi_modal_inputs:
                        left_padding_length = torch.argmax(attention_mask, dim=1)
                        image_bounds = []
                        for i in range(len(multi_modal_inputs["image_bound"])):
                            image_bound = multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
                            image_bounds.append(image_bound)
                        multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
                        pixel_values = []
                        for i in range(len(multi_modal_inputs["pixel_values"])):
                            pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])
                        multi_modal_inputs["pixel_values"] = [pixel_values]
                    if "tgt_sizes" in multi_modal_inputs:
                        multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                # ulysses sp
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad = logits_rmpad.clone()
                    logits_rmpad = logits_rmpad / temperature

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:  # no rmpad
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids"""
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # -------------------- 全局样本数对齐：计算 global_n & 全局 K -------------------- #
        local_n = int(batch.batch_size[0])  # 本 rank 样本条数（rollout 端已保证有效）
        global_n = self._allreduce_int_sum(local_n)

        if global_n == 0:
            # 没有任何样本：所有 rank 一致跳过
            torch.distributed.barrier()
            self.actor_optimizer.zero_grad()
            return {"actor/skip": 1}

        # 用全局样本数计算“全局一致”的 mini-batch 轮次（每轮大小按 ppo_mini_batch_size 设计）
        ppo_mini = int(self.config.ppo_mini_batch_size)
        K = int(math.ceil(global_n / max(1, ppo_mini)))
        K = self._bcast_int(K, src=0)  # 再保险，确保完全一致

        # -------------------- 构建“本地” mini-batch 迭代器 -------------------- #
        # 说明：has_multi_modal_inputs 的路径用 chunk(ceil(...))，确保尾部不足也会被包含；
        #       非多模态路径用 TensorDict.split(size)（最后一块可小于 size）。
        if has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
            num_mini_batches_local = int(math.ceil(local_n / max(1, ppo_mini)))
            dataloader_local = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches_local) if num_mini_batches_local > 0 else []
        else:
            dataloader_local = batch.split(max(1, ppo_mini)) if local_n > 0 else []

        # 把可迭代对象转成 list 以便多次遍历（不会太大；K 决定轮次）
        dataloader_local = list(dataloader_local)
        local_iter_index = 0  # 指向当前本地 mini-batch

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for k in range(K):
                # 取本地第 k 个 mini-batch；若没有，记为 None（后续用 dummy 占位）
                mini_batch = None
                if local_iter_index < len(dataloader_local):
                    mini_batch = dataloader_local[local_iter_index]
                    local_iter_index += 1

                # ----------- 依据 mini_batch 生成“本地 micro-batches”，并统计本地累积步数 ----------- #
                micro_batches: List = []
                local_accum = 0

                if mini_batch is not None:
                    if has_multi_modal_inputs:
                        if self.config.use_dynamic_bsz:
                            all_multi_modal_inputs_list = mini_batch.non_tensor_batch["multi_modal_inputs"]
                            batch_tensordict_for_rearrange = mini_batch.batch
                            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                            rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                                batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                            )
                            for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                                current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                                mb_dict = {k: v for k, v in text_mb_td.items()}
                                mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                                micro_batches.append(mb_dict)
                        else:
                            # 固定 micro size；按全局 mini size 推导每卡累积步数
                            self.gradient_accumulation = max(
                                1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                            )
                            num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                            micro_batches = mini_batch.select(select_keys, ["multi_modal_inputs"]).chunk(num_micro_batches)
                            micro_batches = list(micro_batches)
                    elif self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                        micro_batches = list(micro_batches)
                    else:
                        self.gradient_accumulation = max(
                            1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = list(mini_batch.split(self.config.ppo_micro_batch_size_per_gpu))

                    local_accum = len(micro_batches)
                else:
                    # 没有本地 mini-batch：本地累积步数为 0（后面会用全局 MAX 对齐）
                    local_accum = 0
                    micro_batches = []

                # ----------- 用全局 MAX 对齐“每轮的微批累计步数” -----------
                accum_global = self._allreduce_int_max(local_accum)

                # ----------- 开始该轮的优化：所有 rank 必须做“恰好 accum_global 次 backward” -----------
                self.actor_optimizer.zero_grad()

                for step_i in range(accum_global):
                    if step_i < local_accum:
                        data_mb = micro_batches[step_i]
                        # 统一搬运到设备
                        if isinstance(data_mb, DataProto):
                            data_mb = {**data_mb.batch.to(get_device_id()), **data_mb.non_tensor_batch}
                        elif isinstance(data_mb, dict):
                            for k2, v2 in data_mb.items():
                                if isinstance(v2, torch.Tensor):
                                    data_mb[k2] = v2.to(get_device_id())
                                elif k2 == "multi_modal_inputs" and v2 is not None:
                                    data_mb[k2] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v2]
                                else:
                                    data_mb[k2] = v2
                        else:
                            data_mb = data_mb.to(get_device_id())  # actor device is cpu when using offload

                        responses = data_mb["responses"]
                        response_length = responses.size(1)
                        attention_mask = data_mb["attention_mask"]
                        if multi_turn:
                            response_mask = data_mb["loss_mask"][:, -response_length:]
                        else:
                            response_mask = attention_mask[:, -response_length:]

                        old_log_prob = data_mb["old_log_probs"]
                        advantages = data_mb["advantages"]

                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        calculate_entropy = (entropy_coeff != 0)
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data_mb, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        if self.config.policy_loss.loss_mode == "vanilla":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config
                            )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = data_mb["ref_log_prob"]
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data_mb) / max(1, self.config.ppo_mini_batch_size))
                        else:
                            # 梯度累积缩放
                            loss = policy_loss / max(1, getattr(self, "gradient_accumulation", 1))

                        loss.backward()

                        data_log = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, data_log)
                    else:
                        # 本地没有足够 micro-batch，用占位 0-loss 补齐一次 backward
                        dummy_loss = self._dummy_zero_loss(self.actor_module)
                        dummy_loss.backward()

                grad_norm = self._optimizer_step()
                data_log = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_log)

        self.actor_optimizer.zero_grad()
        return metrics
