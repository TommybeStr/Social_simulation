#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于文本生成的 GRPO 训练脚本（OOM 修复版）：
- 修复：增加显式垃圾回收 (GC) 和显存清理，防止 OOM。
- 修复：完善梯度累积 (Gradient Accumulation) 逻辑。
- 修复：增加 Token 长度硬截断，防止长文本撑爆显存。
- 逻辑：Dataset 优先复用 Reward Model，解析对齐 Evaluation 脚本。
"""

import os
import json
import argparse
import re
import gc
import time
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ==================== 常量 & 工具 ====================

ROOT_PARENT_KEY = "__ROOT__"
EPS = 1e-6

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_MODEL_PREFIX_RE = re.compile(r"^\s*[\w\-]+:\s*", re.UNICODE)

def get_root_potential_objs(user_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    pots = []
    rm = user_json.get("reward_model") or {}
    rp = rm.get("root_potential") or {}
    
    if isinstance(rp, dict):
        full = rp.get("full")
        if isinstance(full, list) and full:
            for it in full:
                name = (it.get("user_name") or "").strip()
                if name:
                    pots.append({
                        "user_name": name,
                        "interests": it.get("interests") or [],
                        "depth": int(it.get("depth", 0)),
                        "interaction_count": int(it.get("interaction_count", 0))
                    })
        elif isinstance(rp.get("user_names"), list):
            for name in rp.get("user_names") or []:
                name = (name or "").strip()
                if name:
                    pots.append({"user_name": name, "interests": [], "interaction_count": 0, "depth": 0})
    return pots

def rebuild_user_view_and_order_with_root_potential_sft(user_json: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    user_view = dict(user_json)
    cur_depth = int(user_view.get("depth", user_view.get("_step_depth", 0)) or 0)
    pots = get_root_potential_objs(user_json)
    
    content_raw = user_view.get("content", "") or ""
    lines = content_raw.split("\n")
    new_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("potentials:") or line_stripped.startswith("potentialspan:"):
            break
        new_lines.append(line)
    
    base_content = "\n".join(new_lines).rstrip()
    
    if pots:
        blocks = []
        for p in pots:
            blk = {
                "user_name": (p.get("user_name") or "").strip(),
                "interests": p.get("interests") or [],
                "depth": cur_depth,
                "interaction_count": p.get("interaction_count", 0)
            }
            if blk["user_name"]:
                blocks.append(blk)
        potentials_json = json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))
        new_content = base_content + "\npotentials: " + potentials_json
    else:
        new_content = base_content + "\npotentials: []"
    
    user_view["content"] = new_content
    user_view.pop("reward_model", None)
    user_view.pop("ground_truth", None)
    user_view.pop("gold", None)
    
    cand_order = [p["user_name"] for p in pots] if pots else []
    return user_view, cand_order

def strip_model_output(text: str) -> str:
    if not isinstance(text, str): return ""
    s = text.strip()
    s = _MODEL_PREFIX_RE.sub("", s, count=1)
    m = _CODE_FENCE_RE.search(s)
    if m: s = m.group("body").strip()
    return s

def parse_generated_json_array(gen_text: str, cand_order: List[str]) -> Tuple[List[int], List[str], bool]:
    clean_text = strip_model_output(gen_text)
    l = clean_text.find("[")
    r = clean_text.rfind("]")
    json_str = clean_text[l:r+1] if (l != -1 and r != -1 and r > l) else clean_text
    
    try:
        data = json.loads(json_str)
    except:
        return [0] * len(cand_order), [], False

    if not isinstance(data, list):
        return [0] * len(cand_order), [], False

    data_map = {}
    for item in data:
        if isinstance(item, dict):
            u = item.get("user_name")
            if u: data_map[u] = item

    pred_types = []
    pred_user_names = []

    for i, uname in enumerate(cand_order):
        pred_item = None
        if i < len(data) and isinstance(data[i], dict):
            if data[i].get("user_name") == uname or not data[i].get("user_name"):
                pred_item = data[i]
        
        if pred_item is None:
            pred_item = data_map.get(uname)
            
        t = 0
        if pred_item:
            try: t = int(pred_item.get("type", 0))
            except: t = 0
        
        pred_types.append(t)
        if t in (1, 2):
            pred_user_names.append(uname)

    return pred_types, pred_user_names, True

def set_f1(pred: List[str], gold: List[str]) -> float:
    ps = set(p for p in pred if p)
    gs = set(g for g in gold if g)
    if not ps and not gs: return 1.0
    if not ps or not gs: return 0.0
    tp = len(ps & gs)
    if tp == 0: return 0.0
    prec = tp / len(ps)
    rec = tp / len(gs)
    return 2 * prec * rec / (prec + rec)

def strict_gold_for_parent(step_depth, parent_key, cond_gt_by_turn):
    parent_key = (parent_key or "").strip()
    if not parent_key: parent_key = ROOT_PARENT_KEY
    if not isinstance(cond_gt_by_turn, list): return [], parent_key
    if step_depth < 0 or step_depth >= len(cond_gt_by_turn): return [], parent_key
    
    layer = cond_gt_by_turn[step_depth] or []
    for mapping in layer:
        if not isinstance(mapping, dict): continue
        for k, v in mapping.items():
            if (k or "").strip() == parent_key:
                return [str(x) for x in (v if isinstance(v, list) else [v])], parent_key
    return [], parent_key

def normalize_prompt(cell):
    if cell is None: return []
    if isinstance(cell, str):
        try: return json.loads(cell) if isinstance(json.loads(cell), list) else []
        except: return []
    try: return list(cell)
    except: return []

# ==================== Dataset ====================

class GrpoGenDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int | None = None):
        super().__init__()
        print(f"[Dataset] Loading data from {data_path} ...")
        try:
            if data_path.endswith(".jsonl"):
                self.df = pd.read_json(data_path, lines=True)
            elif data_path.endswith(".parquet"):
                self.df = pd.read_parquet(data_path)
            else:
                try: self.df = pd.read_parquet(data_path)
                except: self.df = pd.read_json(data_path, lines=True)
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

        # 内存优化：如果数据集过大，先切片再处理
        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)
        
        self.samples = []
        for idx, row in self.df.iterrows():
            raw_msgs = row.get("messages")
            if raw_msgs is None: raw_msgs = row.get("prompt")
            messages = normalize_prompt(raw_msgs)
            if not messages: continue
            
            system_prompt = ""
            user_msg = None
            for m in messages:
                if isinstance(m, dict):
                    role = m.get("role", "")
                    if role == "system": system_prompt = m.get("content", "")
                    elif role == "user": user_msg = m
            
            if user_msg is None: continue
            user_content = user_msg.get("content", "")
            
            try: root_user_json = json.loads(user_content)
            except: root_user_json = {"content": user_content}
            
            sft_ci = row.get("sft_chunk_info", {})
            if isinstance(sft_ci, str):
                try: sft_ci = json.loads(sft_ci)
                except: sft_ci = {}
            if not isinstance(sft_ci, dict): sft_ci = {}

            node_depth = sft_ci.get("node_depth")
            if node_depth is None: node_depth = row.get("node_depth", 0)
            try: node_depth = int(node_depth)
            except: node_depth = 0
            
            if node_depth != 0: continue
            
            if "depth" not in root_user_json: root_user_json["depth"] = 0
            root_user_json["_step_depth"] = 0
            
            # 优先检查 reward_model 是否完整
            has_valid_rm = False
            rm = root_user_json.get("reward_model")
            if isinstance(rm, dict):
                rp = rm.get("root_potential")
                if isinstance(rp, dict) and (rp.get("user_names") or rp.get("full")):
                    has_valid_rm = True
            
            if not has_valid_rm:
                cand_order = []
                root_pots_full = []
                for prefix in ["potentials:", "potentialspan:"]:
                    idx_pos = user_content.find(prefix)
                    if idx_pos >= 0:
                        span_text = user_content[idx_pos + len(prefix):].strip()
                        try:
                            arr = json.loads(span_text)
                            if isinstance(arr, list):
                                for blk in arr:
                                    if isinstance(blk, dict):
                                        n = str(blk.get("user_name") or "").strip()
                                        if n:
                                            cand_order.append(n)
                                            root_pots_full.append({
                                                "user_name": n,
                                                "interests": blk.get("interests") or [],
                                                "depth": int(blk.get("depth", 0)),
                                                "interaction_count": int(blk.get("interaction_count", 0))
                                            })
                                break
                        except: pass

                cond_gt_by_turn = [[{ROOT_PARENT_KEY: []}]] # 仅作占位，防止 crash
                root_user_json["reward_model"] = {
                    "ground_truth": {"cond_gt_by_turn": cond_gt_by_turn},
                    "root_potential": {
                        "user_names": cand_order,
                        "full": root_pots_full if root_pots_full else None,
                    },
                    "root_parent_key": ROOT_PARENT_KEY,
                }
            
            self.samples.append({
                "system_prompt": system_prompt,
                "root_user_json": root_user_json,
                "sft_chunk_info": sft_ci,
            })
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]: return self.samples[idx]

def _identity_collate(batch): return batch

# ==================== Policy Model ====================

def build_base_model(model_name_or_path: str, *, use_lora: bool, lora_r: int, lora_alpha: int = 16, lora_dropout: float = 0.05) -> nn.Module:
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if not use_lora: return base
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
    base = get_peft_model(base, lora_config)
    base.print_trainable_parameters()
    return base

# ==================== Rollout (带 Hard Cutoff) ====================

def rollout_one_trajectory_inference(model, tokenizer, sample, depth_limit, temperature, max_input_tokens, gen_max_new_tokens, gen_top_p, device):
    base_model = getattr(model, "module", model)
    system_prompt = sample["system_prompt"]
    root_user_json = json.loads(json.dumps(sample["root_user_json"], ensure_ascii=False))
    
    user_view, cand_order = rebuild_user_view_and_order_with_root_potential_sft(root_user_json)
    
    if depth_limit != 0: return {}
    
    steps = []
    step_rewards = []
    
    with torch.no_grad():
        user_blob = json.dumps(user_view, ensure_ascii=False)
        messages = [{"role": "system", "content": system_prompt or ""}, {"role": "user", "content": user_blob}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 【重点】生成前的 Hard Truncation
        gen_enc = tokenizer([chat_text], return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
        gen_enc = {k: v.to(device) for k, v in gen_enc.items()}
        
        gen_out = base_model.generate(**gen_enc, do_sample=True, top_p=float(gen_top_p), temperature=float(temperature), max_new_tokens=int(gen_max_new_tokens), eos_token_id=tokenizer.eos_token_id, use_cache=True, pad_token_id=(tokenizer.eos_token_id or tokenizer.pad_token_id))
        
        inp_len = gen_enc["input_ids"].shape[1]
        new_ids = gen_out[0, inp_len:]
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        pred_types, pred_user_names, parse_success = parse_generated_json_array(gen_text, cand_order)
        
        rm = root_user_json.get("reward_model") or {}
        cond_gt_by_turn = rm.get("ground_truth", {}).get("cond_gt_by_turn") or []
        gold_names, _ = strict_gold_for_parent(0, ROOT_PARENT_KEY, cond_gt_by_turn)
        
        step_reward = -1.0 if not parse_success else float(set_f1(pred_user_names, gold_names))
        step_rewards.append(step_reward)
        
        # 释放显存
        del gen_enc, gen_out, new_ids
        
        steps.append({
            "depth": 0, "parent": ROOT_PARENT_KEY, "input_user": root_user_json,
            "candidate_names": cand_order, "pred_types": pred_types, "pred_user_names": pred_user_names,
            "gen_text": gen_text, "chat_text": chat_text, "step_reward": step_reward, "parse_success": parse_success
        })
    
    total_reward = float(sum(step_rewards)/len(step_rewards)) if step_rewards else 0.0
    return {"responses": [], "reward": total_reward, "root_user_json": root_user_json, "system_prompt": system_prompt, "steps": steps}

# ==================== LogProb (带 Hard Cutoff) ====================

def compute_log_prob_for_steps(model, tokenizer, rollout_result, max_input_tokens, device, max_gen_tokens=None):
    is_ddp = hasattr(model, "module")
    base_model = model.module if is_ddp else model
    base_model.train()
    
    steps = rollout_result.get("steps", [])
    if not steps: return []
    logprobs = []
    
    for step in steps:
        if step.get("depth", -1) != 0: continue
        chat_text = step["chat_text"]
        gen_text = step["gen_text"]
        
        # 【重点】LogProb 计算时的 Hard Truncation
        # 无论 Chat Template 出来多长，强制截断
        enc = tokenizer([chat_text], return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        gen_enc = tokenizer([gen_text], return_tensors="pt", padding=True, truncation=False)
        gen_ids = gen_enc["input_ids"].to(device)
        gen_len = gen_ids.shape[1]
        
        # 如果生成为空，跳过
        if gen_len == 0: 
            del enc, input_ids, attention_mask, gen_enc, gen_ids
            continue
        
        # 限制生成的最大长度
        if max_gen_tokens and gen_len > max_gen_tokens:
            gen_ids = gen_ids[:, :max_gen_tokens]
            gen_len = max_gen_tokens
            
        full_ids = torch.cat([input_ids[0], gen_ids[0]], dim=0)
        full_attention_mask = torch.ones_like(full_ids)
        
        if is_ddp: model.train()
        base_model.train()
        
        # Checkpointing
        cp_enabled = hasattr(base_model, 'gradient_checkpointing') and base_model.gradient_checkpointing
        ir_enabled = False
        if hasattr(base_model, 'enable_input_require_grads'):
            try: base_model.enable_input_require_grads(); ir_enabled = True
            except: pass
        
        cp_disabled = False
        if not ir_enabled and cp_enabled:
            if hasattr(base_model, 'gradient_checkpointing_disable'): base_model.gradient_checkpointing_disable()
            cp_disabled = True
        
        # 内存碎片整理
        if torch.cuda.is_available() and (torch.cuda.memory_reserved(device)/1024**3 > 0.8): 
            torch.cuda.empty_cache()
        
        try:
            out = base_model(input_ids=full_ids.unsqueeze(0), attention_mask=full_attention_mask.unsqueeze(0), use_cache=False)
            full_logits = out.logits[0]
            del out # 立即释放
        finally:
            if cp_disabled and hasattr(base_model, 'gradient_checkpointing_enable'): base_model.gradient_checkpointing_enable()
            if ir_enabled and hasattr(base_model, 'disable_input_require_grads'):
                try: base_model.disable_input_require_grads()
                except: pass
        
        input_len = input_ids.shape[1]
        # 对齐 logits 和 targets
        gen_logits = full_logits[input_len-1 : input_len-1+gen_len]
        gen_targets = gen_ids[0, :gen_len]
        
        del input_ids, attention_mask, gen_ids, full_ids, full_attention_mask, full_logits
        
        # 分块计算 LogSoftmax 防止显存尖峰
        chunk_size = 32 # 减小 Chunk Size
        t_lps_list = []
        for i in range(0, gen_len, chunk_size):
            end = min(i+chunk_size, gen_len)
            c_logits = gen_logits[i:end]
            c_targets = gen_targets[i:end]
            c_lps = []
            for t in range(c_logits.shape[0]):
                sl = c_logits[t:t+1]
                st = c_targets[t:t+1]
                sm = torch.nn.functional.log_softmax(sl, dim=-1)
                lp = sm.gather(dim=1, index=st.unsqueeze(0)).squeeze(0)
                c_lps.append(lp)
            t_lps_list.append(torch.stack(c_lps))
            
        logprobs.append(torch.cat(t_lps_list, dim=0).sum())
        
        # 彻底清理中间变量
        del gen_logits, gen_targets, t_lps_list
        torch.cuda.empty_cache()
        
    return logprobs

def compute_group_loss(logprobs, rewards, clip_adv=5.0, global_baseline=None):
    baseline = global_baseline if global_baseline is not None else rewards.mean()
    adv = (rewards - baseline).clamp(-clip_adv, clip_adv) if clip_adv else (rewards - baseline)
    return -(adv.detach() * logprobs).mean(), rewards.mean(), adv.detach()

# ==================== Main ====================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5) # 调小LR
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--depth_limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen_max_new_tokens", type=int, default=2048) # 调小
    parser.add_argument("--gen_top_p", type=float, default=0.9)
    parser.add_argument("--num_traj_per_root", type=int, default=2) # 建议减小
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--stage", choices=["rollout", "train", "online"], default="online")
    parser.add_argument("--rollout_prefix", default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=1) # 此参数保留给 Dataset Batch Size, GRPO有自己的 roots_per_update
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--roots_per_update", type=int, default=1) # 建议设为 1 或 2，防止梯度堆积
    args = parser.parse_args()
    
    args.depth_limit = 0
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    base_model = build_base_model(args.model, use_lora=args.use_lora, lora_r=args.lora_r).to(device)
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        if rank == 0: print("[INFO] Enabled gradient checkpointing", flush=True)
    try: base_model.resize_token_embeddings(len(tokenizer))
    except: pass
    
    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world_size > 1 else base_model
    
    if args.stage == "online":
        # 限制 Dataset 加载数量以测试内存
        dataset = GrpoGenDataset(args.data)
        import random
        rng = random.Random(42)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        
        try:
            import bitsandbytes as bnb
            if rank == 0: print("[online] Using bitsandbytes PagedAdamW8bit", flush=True)
            optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)
        except:
            if rank == 0: print("[online] Using torch.optim.AdamW", flush=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            
        model.train()
        global_step = 0
        roots_since_update = 0
        acc_f1 = []
        
        # 梯度累积逻辑：每 roots_per_update 次循环 step 一次
        optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(args.epochs):
            rng = random.Random(42 + epoch)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            if rank == 0: print(f"[online] ===== Epoch {epoch + 1}/{args.epochs} =====", flush=True)
            root_iter = tqdm(indices, desc=f"[online] epoch {epoch+1}", disable=(rank!=0))
            
            for root_idx, idx in enumerate(root_iter):
                sample = dataset[idx]
                
                # 1. Rollout (Inference Mode)
                model.eval()
                local_trajs = []
                with torch.no_grad():
                    for _ in range(args.num_traj_per_root):
                        local_trajs.append(rollout_one_trajectory_inference(model, tokenizer, sample, args.depth_limit, args.temperature, args.max_input_tokens, args.gen_max_new_tokens, args.gen_top_p, device))
                
                # Rollout 结束，强制回收
                torch.cuda.empty_cache()
                
                # 2. Forward (Train Mode)
                model.train()
                all_lps, all_rwds = [], []
                for traj in local_trajs:
                    all_lps.extend(compute_log_prob_for_steps(model, tokenizer, traj, args.max_input_tokens, device, args.gen_max_new_tokens))
                    for step in traj.get("steps", []):
                        if step.get("depth", -1) == 0: all_rwds.append(float(step.get("step_reward", 0.0)))
                
                if not all_lps: 
                    # 清理本次循环的垃圾
                    del local_trajs
                    continue
                
                lps_t = torch.stack(all_lps)
                rwds_t = torch.tensor(all_rwds, device=device, dtype=torch.bfloat16)
                loss, mean_r, adv = compute_group_loss(lps_t, rwds_t)
                loss_val = loss.item()
                acc_f1.append(float(mean_r))
                
                # 3. Backward (梯度累积)
                # 注意：这里 loss 除以 roots_per_update
                (loss / max(1, args.roots_per_update)).backward()
                
                # 4. 暴力清理显存 (防止计算图残留)
                del loss, mean_r, adv, lps_t, rwds_t, all_lps, local_trajs
                gc.collect()
                torch.cuda.empty_cache()
                
                roots_since_update += 1
                if roots_since_update % args.roots_per_update == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True) # set_to_none 更省显存
                    global_step += 1
                    
                    if rank == 0:
                        avg_f1 = sum(acc_f1)/len(acc_f1) if acc_f1 else 0.0
                        print(f"[online] Step {global_step}, Loss: {loss_val:.4f}, F1: {avg_f1:.4f}", flush=True)
                        acc_f1 = []
                    
                    if rank == 0 and global_step % args.save_steps == 0:
                        os.makedirs(args.output_dir, exist_ok=True)
                        to_save = model.module if isinstance(model, DDP) else model
                        torch.save(to_save.state_dict(), os.path.join(args.output_dir, f"online_step{global_step}.pt"))
            
            if rank == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                to_save = model.module if isinstance(model, DDP) else model
                torch.save(to_save.state_dict(), os.path.join(args.output_dir, f"online_epoch{epoch + 1}.pt"))
        
        cleanup_distributed()

if __name__ == "__main__":
    main()