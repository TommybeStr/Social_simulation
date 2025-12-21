# -*- coding: utf-8 -*-
"""
GRPO 训练数据构建脚本 (扁平化版 - 修正 Depth 逻辑)
功能：
1. 模型输入(Prompt)保留真实的 depth (如 1, 2)。
2. 训练器元数据欺骗为 depth=0 (通过过滤器)。
3. 真值(GT)扁平化到第 0 层 (适配 Reward 计算)。
"""

import json
import pandas as pd
from tqdm import tqdm
import argparse
import os
from hashlib import blake2b
import re
import numpy as np

# ==========================================
# 复用工具函数 (保持不变)
# ==========================================
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_META_ROLE_KEY = "meta"
ROOT_PARENT_KEY = "__ROOT__" 

def _normalize_messages(cell):
    if cell is None: return []
    if isinstance(cell, str):
        try: return json.loads(cell)
        except: return []
    try: return list(cell)
    except: return []

def _normalize_system(msg):
    if not msg or msg.get("role") != "system": return None
    content = msg.get("content", "")
    try:
        j = json.loads(content)
        if isinstance(j, str): content = j
    except: pass
    return {"role": "system", "content": content}

def _hash_prompt(system_content: str, root_user_str: str) -> str:
    h = blake2b(digest_size=16)
    h.update((system_content or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((root_user_str or "").encode("utf-8"))
    return h.hexdigest()

# ==========================================
# 内容解析工具
# ==========================================
def _extract_potentialspan_from_user_content(user_content_str: str) -> list[dict]:
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

def _extract_node_depth_from_user_content(user_content_str: str) -> int:
    pots = _extract_potentialspan_from_user_content(user_content_str)
    if pots:
        try: return int(pots[0].get("depth", 0))
        except: return 0
    return 0

def _candidate_order_from_user_content(user_content_str: str) -> list[str]:
    pots = _extract_potentialspan_from_user_content(user_content_str)
    order = []
    for blk in pots:
        if isinstance(blk, dict):
            n = str(blk.get("user_name") or "").strip()
            if n: order.append(n)
    return order

def _extract_root_potential_full(user_content_str: str) -> list[dict]:
    pots = _extract_potentialspan_from_user_content(user_content_str)
    result = []
    for blk in pots:
        if isinstance(blk, dict):
            obj = {
                "user_name": str(blk.get("user_name") or "").strip(),
                "interests": blk.get("interests") or [],
                "depth": int(blk.get("depth", 0)) if isinstance(blk.get("depth", 0), (int, float, str)) else 0,
                "interaction_count": int(blk.get("interaction_count", 0)) if isinstance(blk.get("interaction_count", 0), (int, float, str)) else 0,
            }
            if obj["user_name"]: result.append(obj)
    return result

# ==========================================
# Assistant 解析工具
# ==========================================
def _extract_gold_names(assistant_content: str, candidate_order: list) -> list[str]:
    if not assistant_content: return []
    try:
        data = json.loads(assistant_content)
        if not isinstance(data, list): return []
        
        golds = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict): continue
            try: t = int(item.get("type", 0))
            except: t = 0
            if t in (1, 2):
                uname = (item.get("user_name") or "").strip()
                if not uname and idx < len(candidate_order):
                    uname = candidate_order[idx]
                if uname:
                    golds.append(uname)
        return list(set(golds))
    except:
        return []

def _extract_meta_from_messages(messages: list[dict]) -> dict | None:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == _META_ROLE_KEY:
            try: return json.loads(m.get("content", ""))
            except: return None
    return None

# ==========================================
# 核心构建逻辑 (扁平化 + 真实 Depth)
# ==========================================
def _build_grpo_flat(
    sft_parquet_path: str,
    out_jsonl: str,
    *,
    data_source: str = "social_grpo_flat",
    dedup_by_prompt: bool = True,
    embed_root_potential_full: bool = True,
):
    print(f"[load] reading sft parquet: {sft_parquet_path}")
    df = pd.read_parquet(sft_parquet_path)
    print(f"[load] rows: {len(df)}")

    samples = []
    seen = set()
    
    for ridx, row in tqdm(df.iterrows(), total=len(df), desc="Build Flat"):
        msgs = _normalize_messages(row.get("messages"))
        if len(msgs) < 2: continue

        # 1. 提取内容
        sys_content = ""
        user_content = ""
        asst_content = ""
        
        for m in msgs:
            role = m.get("role")
            if role == "system":
                sys_content = _normalize_system(m).get("content", "")
            elif role == "user":
                user_content = m.get("content", "")
            elif role == "assistant":
                asst_content = m.get("content", "")

        if not user_content: continue

        # 2. 提取信息
        candidate_order = _candidate_order_from_user_content(user_content)
        chunk_pots_full = _extract_root_potential_full(user_content)
        
        # 【重要】获取真实的 depth (例如 1 或 2)
        real_depth = _extract_node_depth_from_user_content(user_content)
        
        # 3. 提取 Gold
        gold_users = _extract_gold_names(asst_content, candidate_order)
        
        # 4. 构造 Prompt JSON
        try:
            root_user_json = json.loads(user_content)
        except:
            root_user_json = {"content": user_content}
            
        # 【修改点 A】: JSON 里的 depth 设为真实值
        # 这样训练时 rebuild_user_view 就会把真实的 depth 写进 Prompt 文本里
        root_user_json["depth"] = int(real_depth) 
        
        # _step_depth 设为 0，因为我们的 GT 只有一层，防止 lookup 越界
        root_user_json["_step_depth"] = 0 
        
        # 5. 构造 Reward Model
        # 将 Gold 放在 cond_gt_by_turn 的第 0 层
        cond_gt = []
        if gold_users:
            cond_gt = [[{ROOT_PARENT_KEY: gold_users}]]
        else:
            cond_gt = [[]]

        root_user_json["reward_model"] = {
            "ground_truth": {
                "cond_gt_by_turn": cond_gt,
                "edge_types_by_turn": [], 
            },
            "root_potential": {
                "user_names": candidate_order,
                **({"full": chunk_pots_full} if embed_root_potential_full else {}),
            },
            "root_parent_key": ROOT_PARENT_KEY,
        }

        root_user_str = json.dumps(root_user_json, ensure_ascii=False)
        
        if dedup_by_prompt:
            key = _hash_prompt(sys_content, root_user_str)
            if key in seen: continue
            seen.add(key)

        # 6. 构造 Metadata
        meta_msg = _extract_meta_from_messages(msgs) or {}
        sft_chunk_info = {}
        try:
            raw_ci = row.get("sft_chunk_info")
            if isinstance(raw_ci, str): sft_chunk_info = json.loads(raw_ci)
            elif isinstance(raw_ci, dict): sft_chunk_info = raw_ci
        except: pass
        
        # 【修改点 B】: sft_chunk_info 里的 node_depth 强制设为 0
        # 这是为了通过 train_grpo_cls_bfs.py 中 Dataset 类的 `if node_depth != 0: continue` 检查
        sft_info = {
            **sft_chunk_info, 
            "record_id": meta_msg.get("record_id") or f"row{ridx}",
            "real_node_depth": real_depth, # 留个底，万一需要查
            "node_depth": 0,               # <--- 骗过 Dataset 过滤器
            "gold_count": len(gold_users)
        }

        samples.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": root_user_str}
            ],
            "ability": "social_prediction",
            "sft_chunk_info": sft_info
        })

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    print(f"[write] Writing {len(samples)} lines to {out_jsonl} ...")
    
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="Writing JSONL"):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print("[done] Generation complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_sft_parquet", required=True, help="Input SFT Parquet path")
    ap.add_argument("--val_output", required=True, help="Output JSONL path")
    ap.add_argument("--no_dedup", action="store_true")
    ap.add_argument("--no_embed_root_potential_full", action="store_true", 
                    help="Disable embedding full potential objects to save space.")
    
    args = ap.parse_args()
    
    _build_grpo_flat(
        sft_parquet_path=args.val_sft_parquet,
        out_jsonl=args.val_output,
        dedup_by_prompt=(not args.no_dedup),
        embed_root_potential_full=(not args.no_embed_root_potential_full)
    )