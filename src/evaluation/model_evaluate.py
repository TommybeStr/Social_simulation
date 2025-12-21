# -*- coding: utf-8 -*-
"""
纯 LLM 评估脚本 (多卡并行版 v9.1 - 支持加载 .pt 权重)
- 基础功能: V9 (ID锚定聚合 + SFT适配 + 分片评估)
- 新增功能: 支持通过 --checkpoint_pt 加载 RL 训练后的 state_dict
"""

import os
import re
import json
import argparse
import time
import torch.distributed as dist
from collections import deque, defaultdict
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# 1. 基础工具 & 正则
# ===========================
ROOT_FALLBACK_KEY = "__ROOT__"
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MODEL_PREFIX_RE = re.compile(r"^\s*[\w\-]+:\s*", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)

def extract_username_from_sft_format(text: str) -> str:
    if not isinstance(text, str): return ""
    m = _USERNAME_LINE_RE.search(text)
    if m: return m.group("name").strip()
    return ""

def extract_pure_content_from_sft_format(text: str) -> str:
    if not isinstance(text, str): return ""
    start_marker = "content:\n"
    idx_start = text.find(start_marker)
    if idx_start == -1: 
        start_marker = "content:"
        idx_start = text.find(start_marker)
    if idx_start == -1: return text 
    
    idx_end = text.find("\nuserinterest:", idx_start)
    if idx_end == -1: idx_end = text.find("userinterest:", idx_start)
        
    if idx_end != -1:
        pure_content = text[idx_start + len(start_marker):idx_end].strip()
    else:
        idx_end_pot = text.find("\npotentials:", idx_start)
        if idx_end_pot != -1:
            pure_content = text[idx_start + len(start_marker):idx_end_pot].strip()
        else:
            pure_content = text[idx_start + len(start_marker):].strip()
    return pure_content

def extract_potentialspan_from_text(user_content_str: str) -> List[Dict[str, Any]]:
    if not isinstance(user_content_str, str): return []
    for prefix in ["potentials:", "potentialspan:"]:
        idx = user_content_str.find(prefix)
        if idx >= 0:
            span_text = user_content_str[idx + len(prefix):].strip()
            try:
                arr = json.loads(span_text)
                if isinstance(arr, list): return arr
            except: pass
    return []

def render_potentialspan_json(pots: List[Dict[str, Any]], depth: int) -> str:
    dval = int(depth)
    blocks = []
    for p in (pots or []):
        blk = {
            "user_name": (p.get("user_name") or "").strip(),
            "interests": p.get("interests") or [],
            "depth": dval,
            "interaction_count": int(p.get("interaction_count", 0))
        }
        if blk["user_name"]:
            blocks.append(blk)
    return json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))

def strip_model_output(text: str) -> str:
    if not isinstance(text, str): return ""
    s = text.strip()
    s = _MODEL_PREFIX_RE.sub("", s, count=1)
    m = _CODE_FENCE_RE.search(s)
    if m: s = m.group("body").strip()
    return s

def parse_model_output(gen_text: str, cand_order: List[str]) -> tuple[List[Dict[str, Any]], bool]:
    clean_text = strip_model_output(gen_text)
    l = clean_text.find("[")
    r = clean_text.rfind("]")
    json_str = clean_text[l:r+1] if (l != -1 and r != -1 and r > l) else clean_text
    
    try: data = json.loads(json_str)
    except: return [], False

    if not isinstance(data, list): return [], False

    data_map = {}
    for item in data:
        if isinstance(item, dict):
            u = item.get("user_name")
            if u: data_map[u] = item

    results = []
    for i, uname in enumerate(cand_order):
        pred_item = None
        if i < len(data) and isinstance(data[i], dict):
            if data[i].get("user_name") == uname or not data[i].get("user_name"):
                pred_item = data[i]
        if pred_item is None:
            pred_item = data_map.get(uname)
            
        res = {"user_name": uname, "type": 0, "content": ""}
        if pred_item:
            try: t = int(pred_item.get("type", 0))
            except: t = 0
            res["type"] = t
            res["content"] = str(pred_item.get("content", "")).strip()
        results.append(res)
    return results, True

def smart_extract_potentials(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rm = json_data.get("reward_model", {})
    rp = rm.get("root_potential", {})
    if rp.get("full") and isinstance(rp["full"], list): return rp["full"]
    raw_content = json_data.get("content", "")
    pots_from_text = extract_potentialspan_from_text(raw_content)
    if pots_from_text: return pots_from_text
    if rp.get("user_names") and isinstance(rp["user_names"], list):
        return [{"user_name": n, "interests": [], "interaction_count": 0} for n in rp["user_names"] if n]
    return []

# ===========================
# 2. 全局池管理
# ===========================

class GlobalPoolManager:
    def __init__(self):
        self.pools: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    
    def preload(self, rows: List[Dict[str, Any]], rank: int):
        iterator = tqdm(rows, desc=f"Rank {rank} Preloading", disable=(rank != 0))
        for row in iterator:
            sft_info = row.get("sft_chunk_info") or {}
            record_id = sft_info.get("record_id")
            if not record_id: continue
            
            user_msgs = [m for m in row.get("prompt", []) if m.get("role") == "user"]
            if not user_msgs: continue
            content_str = user_msgs[0].get("content", "")
            
            pots = []
            try:
                data = json.loads(content_str)
                if isinstance(data, dict):
                    pots = smart_extract_potentials(data)
            except: pass
            if not pots:
                pots = extract_potentialspan_from_text(content_str)
            
            for p in pots:
                uname = p.get("user_name")
                if uname:
                    old_p = self.pools[record_id].get(uname)
                    if not old_p or (not old_p.get("interests") and p.get("interests")):
                        self.pools[record_id][uname] = p
        
        if rank == 0:
            valid_trees = len(self.pools)
            total_cands = sum(len(v) for v in self.pools.values())
            print(f"[GlobalPool] Aggregated {valid_trees} trees. Total candidates: {total_cands}")

    def get_candidates(self, record_id: str) -> List[Dict[str, Any]]:
        return list(self.pools.get(record_id, {}).values())

    def find_interests(self, record_id: str, username: str) -> List[str]:
        u = self.pools.get(record_id, {}).get(username)
        return u.get("interests") or [] if u else []

# ===========================
# 3. 推理逻辑
# ===========================

@torch.no_grad()
def step_rollout(model, tokenizer, device, system_prompt: str, base_user_content: str, 
                 candidates: List[Dict[str, Any]], depth: int, gen_params: Dict[str, Any]):
    pot_json = render_potentialspan_json(candidates, depth)
    clean_base = base_user_content.split("\npotentials:")[0].split("\npotentialspan:")[0].strip()
    if clean_base.startswith("{") and clean_base.endswith("}"):
        try:
            js = json.loads(clean_base)
            if "content" in js: clean_base = js["content"]
        except: pass
    
    final_content = clean_base + "\npotentials: " + pot_json
    cand_order = [p["user_name"] for p in candidates]

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": final_content}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    
    if inputs["input_ids"].shape[1] > gen_params.get("max_input_tokens", 8192):
         inputs["input_ids"] = inputs["input_ids"][:, -gen_params["max_input_tokens"]:]
         if "attention_mask" in inputs:
             inputs["attention_mask"] = inputs["attention_mask"][:, -gen_params["max_input_tokens"]:]

    try:
        out = model.generate(
            **inputs, do_sample=True, temperature=gen_params.get("temperature", 0.7),
            top_p=gen_params.get("top_p", 0.9), max_new_tokens=gen_params.get("max_new_tokens", 4096),
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
        )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        raw_gen = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except Exception as e:
        raw_gen = "[]"

    if not cand_order: return [], chat_text, raw_gen
    preds, _ = parse_model_output(raw_gen, cand_order)
    return preds, chat_text, raw_gen

# ===========================
# 4. 主流程
# ===========================

def setup_distributed():
    from datetime import timedelta
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 增加超时时间以防止长尾效应导致 crash
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=5))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def evaluate_jsonl(data_path, model_path, checkpoint_pt, tokenizer_path, jsonl_detail, jsonl_io, max_samples=10000, max_turns=3, depth_limit=1):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"[Init] World Size: {world_size}, Base Model: {model_path}")

    if tokenizer_path is None: tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 1. 加载 Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map={"": local_rank}, 
        torch_dtype="auto", 
        trust_remote_code=True
    )

    # 2. 【核心新增】加载 .pt 权重 (如果指定)
    if checkpoint_pt and os.path.exists(checkpoint_pt):
        if rank == 0:
            print(f"[Init] Loading Checkpoint from .pt file: {checkpoint_pt}")
        
        # 加载到 CPU 避免显存碎片，load_state_dict 会自动处理 device
        state_dict = torch.load(checkpoint_pt, map_location="cpu")
        
        # 处理可能的封装 key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            
        # 清理 Key 前缀 (兼容 DDP 或 torch.compile 产生的 prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            # 去除 DDP 前缀
            if k.startswith("module."): k = k[7:]
            # 去除 torch.compile 前缀
            if k.startswith("_orig_mod."): k = k[10:]
            new_state_dict[k] = v
        
        # 加载权重 (strict=False 允许忽略不匹配的 key，如 RL 的 value head)
        load_result = model.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0:
            print(f"[Init] Weights Loaded. Missing keys: {len(load_result.missing_keys)}, Unexpected keys: {len(load_result.unexpected_keys)}")
            if len(load_result.unexpected_keys) > 0:
                print(f"[Init] Sample unexpected keys (e.g. value head): {load_result.unexpected_keys[:3]}")

    model.eval()
    
    rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    
    if rank == 0: print(f"[Data] Loaded {len(rows)} raw lines.")
    
    pool_mgr = GlobalPoolManager()
    pool_mgr.preload(rows, rank)
    
    detail_file_rank = f"{jsonl_detail}.rank{rank}"
    io_file_rank = f"{jsonl_io}.rank{rank}" if jsonl_io else None
    ensure_dir(detail_file_rank)
    f_det = open(detail_file_rank, "w", encoding="utf-8")
    f_io = open(io_file_rank, "w", encoding="utf-8") if io_file_rank else None
    
    my_max_samples = max_samples // world_size
    my_indices = list(range(rank, len(rows), world_size))
    
    gen_params = {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 4096, "max_input_tokens": 8192}
    DYNAMIC_CHUNK_SIZE = 50 
    
    count = 0
    iterator = tqdm(my_indices, desc=f"Rank {rank} Eval", disable=(rank != 0))
    
    for ridx in iterator:
        if count >= my_max_samples: break
        
        row = rows[ridx]
        prompt = row.get("prompt") or []
        sft_info = row.get("sft_chunk_info") or {}
        record_id = sft_info.get("record_id") or f"row{ridx}"
        
        sys_prompt = ""
        root_user_json = None
        for m in prompt:
            if m.get("role") == "system": sys_prompt = m.get("content", "")
            if m.get("role") == "user" and root_user_json is None:
                try: root_user_json = json.loads(m.get("content", ""))
                except: root_user_json = {"content": m.get("content", "")}
        
        if not root_user_json: continue
        
        raw_content_str = root_user_json.get("content", "")
        pure_root_content = extract_pure_content_from_sft_format(raw_content_str)
        
        root_user_json["depth"] = 0
        root_user_json["_step_depth"] = 0
        if not root_user_json.get("user_name"):
            root_user_json["user_name"] = extract_username_from_sft_format(raw_content_str)
        
        queue = deque([root_user_json])
        root_username = root_user_json.get("user_name", "")
        
        steps_done = 0
        while queue and steps_done < max_turns:
            curr_node = queue.popleft()
            curr_depth = int(curr_node.get("_step_depth", 0))
            
            rm = curr_node.get("reward_model") or {}
            cond_gt = rm.get("ground_truth", {}).get("cond_gt_by_turn") or []
            curr_gold = []
            
            if curr_depth < len(cond_gt):
                layer_gt = cond_gt[curr_depth]
                parent_name = curr_node.get("user_name") or ROOT_FALLBACK_KEY
                found = False
                for item in layer_gt:
                    if parent_name in item:
                        curr_gold = item[parent_name]; found = True; break
                if not found and curr_depth > 0:
                    for item in layer_gt:
                        if ROOT_FALLBACK_KEY in item: pass

            all_preds = []
            
            if curr_depth == 0:
                raw_pots = smart_extract_potentials(curr_node)
                preds, chat_text, raw_gen = step_rollout(
                    model, tokenizer, device, sys_prompt, 
                    curr_node.get("content", ""), 
                    raw_pots, curr_depth, gen_params
                )
                all_preds = preds
                if f_io:
                    f_io.write(json.dumps({
                        "ts": int(time.time()*1000), "group_id": record_id,
                        "depth": curr_depth, "input_text": chat_text, "output_text": raw_gen
                    }, ensure_ascii=False) + "\n")
            
            else:
                all_candidates = pool_mgr.get_candidates(record_id)
                if not all_candidates: 
                    all_candidates = extract_potentialspan_from_text(curr_node.get("content", ""))
                
                chunks = [all_candidates[i:i + DYNAMIC_CHUNK_SIZE] for i in range(0, len(all_candidates), DYNAMIC_CHUNK_SIZE)]
                for chunk_idx, chunk_pots in enumerate(chunks):
                    chunk_preds, chat_text, raw_gen = step_rollout(
                        model, tokenizer, device, sys_prompt,
                        curr_node.get("content", ""), 
                        chunk_pots, curr_depth, gen_params
                    )
                    all_preds.extend(chunk_preds)
                    if f_io:
                        f_io.write(json.dumps({
                            "ts": int(time.time()*1000), "group_id": record_id,
                            "depth": curr_depth, "chunk_idx": chunk_idx,
                            "input_text": chat_text, "output_text": raw_gen
                        }, ensure_ascii=False) + "\n")
            
            positive_preds = [p for p in all_preds if p["type"] in (1, 2)]
            
            out_list = [{"user_name": p["user_name"], "type": p["type"], "pred_type": p["type"]} for p in all_preds]
            f_det.write(json.dumps({
                "ts": int(time.time()*1000), "group_id": record_id,
                "input_user": {"user_name": curr_node.get("user_name")},
                "depth": curr_depth, 
                "output_text": out_list, 
                "gold": curr_gold
            }, ensure_ascii=False) + "\n")
            
            if curr_depth < depth_limit:
                next_depth = curr_depth + 1
                for p in positive_preds:
                    if p["type"] == 1: 
                        child_name = p["user_name"]
                        if not child_name: continue
                        child_interests = pool_mgr.find_interests(record_id, child_name)
                        root_ctx = json.dumps({"root_username": root_username, "root_content": pure_root_content}, ensure_ascii=False, separators=(',', ':'))
                        new_content_base = (
                            f"username: {child_name}\ncontent:\n{p['content']}\n"
                            f"userinterest: {json.dumps(child_interests, ensure_ascii=False)}\n"
                            f"root_context: {root_ctx}"
                        )
                        queue.append({
                            "user_name": child_name, "content": new_content_base, 
                            "depth": next_depth, "_step_depth": next_depth,
                            "reward_model": curr_node.get("reward_model")
                        })
            steps_done += 1
        count += 1
        if count % 10 == 0:
            f_det.flush(); 
            if f_io: f_io.flush()

    f_det.close()
    if f_io: f_io.close()
    dist.barrier()
    if rank == 0:
        print("[Done] All GPUs finished evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input .jsonl file")
    parser.add_argument("--model", required=True, help="Base model path")
    # 新增参数
    parser.add_argument("--checkpoint_pt", default=None, help="Optional: Path to .pt checkpoint file to load state_dict from")
    
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--jsonl_detail", required=True, help="Output prefix")
    parser.add_argument("--jsonl_io", default=None, help="Output prefix")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--depth_limit", type=int, default=1)
    args = parser.parse_args()
    
    evaluate_jsonl(args.data, args.model, args.checkpoint_pt, args.tokenizer, args.jsonl_detail, args.jsonl_io, max_samples=args.max_samples, depth_limit=args.depth_limit)