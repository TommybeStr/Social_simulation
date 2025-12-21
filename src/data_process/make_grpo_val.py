# -*- coding: utf-8 -*-
"""
GRPO 数据构建脚本 (JSONL 最终版)
- 修复 Argparse 逻辑
- 增加写入进度条 (解决"卡住"的错觉)
- 明确区分 Prompt 数据(保留)与 Meta 数据(可选)
"""

import json
import pandas as pd
from tqdm import tqdm
import argparse
import os
from hashlib import blake2b
import re
from collections import defaultdict
import numpy as np

# ... (正则和基础工具部分保持不变，为了节省篇幅，只列出修改的核心逻辑) ...
# ==========================================
# 基础配置与正则
# ==========================================
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_META_ROLE_KEY = "meta"
_ID_RE = re.compile(r"^(?P<rec>.+?)_root_(?P<root>.+?)_flchunk_(?P<idx>\d+)$")

def _extract_parent_name_from_user_content(user_content_str: str) -> str:
    if not isinstance(user_content_str, str): return ""
    m = _USERNAME_LINE_RE.search(user_content_str)
    return (m.group("name") or "").strip() if m else ""

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
def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _parse_children_names_from_assistant_old(gt_text: str) -> list[str]:
    s = (gt_text or "").strip()
    if not s or s == NO_INTERACTION_STR or s == "[]": return []
    try: j = json.loads(s)
    except: j = None
    out = []
    if isinstance(j, list):
        for it in j:
            if isinstance(it, dict):
                name = (it.get("user_name") or "").strip()
                if name: out.append(name)
            elif isinstance(it, str) and it.strip(): out.append(it.strip())
        return _dedup_keep_order(out)
    if isinstance(j, dict):
        name = (j.get("user_name") or "").strip()
        return [name] if name else []
    return [s]

def _parse_positive_children_from_new_assistant(asst_text: str, candidate_order: list[str]) -> list[str]:
    s = (asst_text or "").strip()
    try: arr = json.loads(s)
    except: arr = None
    
    if isinstance(arr, list) and (len(arr) == 0 or (isinstance(arr[0], dict) and "type" in arr[0])):
        seen, out = set(), []
        for idx, it in enumerate(arr):
            if not isinstance(it, dict): continue
            try: t = int(it.get("type", 0))
            except: t = 0
            nm = (it.get("user_name") or "").strip()
            if not nm and idx < len(candidate_order): nm = candidate_order[idx]
            if nm and (t in (1, 2)) and nm not in seen:
                seen.add(nm)
                out.append(nm)
        return out
    return _dedup_keep_order(_parse_children_names_from_assistant_old(s))

# ==========================================
# Meta Info 解析
# ==========================================
def _extract_meta_from_messages(messages: list[dict]) -> dict | None:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == _META_ROLE_KEY:
            try: return json.loads(m.get("content", ""))
            except: return None
    return None

def _compose_sft_chunk_info(row: pd.Series, messages: list[dict]) -> dict:
    meta_msg = _extract_meta_from_messages(messages) or {}
    record_id = meta_msg.get("record_id")
    if not record_id:
        try:
            raw_ci = row.get("sft_chunk_info")
            if isinstance(raw_ci, str): raw_ci = json.loads(raw_ci)
            if isinstance(raw_ci, dict): record_id = raw_ci.get("record_id")
        except: pass
    
    # 尝试从 id 解析 chunk index
    chunk_index = None
    try:
        sample_id = row.get("id")
        m = _ID_RE.match(str(sample_id))
        if m: chunk_index = int(m.group("idx"))
    except: pass

    return {"record_id": record_id, "chunk_index": chunk_index}

# ==========================================
# GT 构建
# ==========================================
def _build_strict_parent_cond_gt_with_types(
    pairs: list[tuple[dict, dict, list[int]]],
    *,
    enforce_children_in_potential: bool = False,
) -> tuple[list[list[dict]], list[list[dict]]]:
    depth_parent2children = defaultdict(dict)
    depth_parent2edge_types = defaultdict(dict)
    max_depth = 0
    observed_depths = set()

    for (u, a, targets_types) in pairs:
        u_content = u.get("content", "")
        parent_name = _extract_parent_name_from_user_content(u_content) or "__ROOT__"
        d = _extract_node_depth_from_user_content(u_content)
        if d < 0: d = 0
        observed_depths.add(d)
        max_depth = max(max_depth, d)

        cand_order = _candidate_order_from_user_content(u_content)
        child_names = _parse_positive_children_from_new_assistant(
            a.get("content", "") or "", candidate_order=cand_order
        )
        if enforce_children_in_potential and cand_order and child_names:
            cand_set = set(cand_order)
            child_names = [c for c in child_names if c in cand_set]

        # GT 结构
        if parent_name not in depth_parent2children[d]:
            depth_parent2children[d][parent_name] = []
        existed = depth_parent2children[d][parent_name]
        exist_set = set(existed)
        for c in child_names:
            if c not in exist_set:
                existed.append(c)
                exist_set.add(c)

        # GT Types
        if parent_name not in depth_parent2edge_types[d]:
            depth_parent2edge_types[d][parent_name] = []
        n = min(len(cand_order), len(targets_types))
        for idx in range(n):
            uname = cand_order[idx]
            if not uname: continue
            try: gtype = int(targets_types[idx])
            except: gtype = 0
            depth_parent2edge_types[d][parent_name].append(
                {"user_name": uname, "gold_type": gtype}
            )

    if not observed_depths: return [], []
    L = max_depth + 1
    cond_gt = [[] for _ in range(L)]
    edge_types = [[] for _ in range(L)]
    for d in range(L):
        if d in depth_parent2children:
            cond_gt[d] = [{p: list(c)} for p, c in depth_parent2children[d].items()]
        if d in depth_parent2edge_types:
            edge_types[d] = [{p: list(e)} for p, e in depth_parent2edge_types[d].items()]
            
    return cond_gt, edge_types

# ==========================================
# 主逻辑
# ==========================================
def _build_grpo_for_one_file(
    sft_parquet_path: str,
    out_jsonl: str,
    *,
    data_source: str = "social_f1",
    seed: int = 42,
    dedup_by_prompt: bool = True,
    max_prompt_len: int = 4096,
    keep_all_no_interact: bool = True,
    allow_system_anywhere: bool = True,
    enforce_children_in_potential: bool = False,
    embed_root_potential_full: bool = True, # 默认为 True
):
    print(f"[load] reading sft parquet: {sft_parquet_path}")
    df = pd.read_parquet(sft_parquet_path)
    print(f"[load] rows: {len(df)}")

    record_map = {}
    
    # Step 1: Scan and Aggregate
    for ridx, row in tqdm(df.iterrows(), total=len(df), desc="Scan"):
        msgs = _normalize_messages(row.get("messages"))
        if len(msgs) < 2: continue

        sys_content = ""
        root_idx = None
        for i, m in enumerate(msgs):
            if m.get("role") == "system" and not sys_content: 
                sys_content = _normalize_system(m).get("content", "")
            if m.get("role") == "user" and root_idx is None:
                root_idx = i
        
        if root_idx is None: continue
        
        root_user_raw = msgs[root_idx].get("content", "") or ""
        node_depth = _extract_node_depth_from_user_content(root_user_raw)
        meta_ci = _compose_sft_chunk_info(row, msgs)
        record_id = meta_ci.get("record_id") or f"row{ridx}"

        rec = record_map.setdefault(record_id, {
            "record_id": record_id,
            "sys_content": sys_content,
            "roots": [], "pairs": []
        })
        if not rec["sys_content"] and sys_content: rec["sys_content"] = sys_content

        # Collect Pairs
        i = root_idx
        raw_tt = row.get("targets_per_potential_types")
        
        # 修复 NumPy Ambiguity Error
        targets_types = []
        if raw_tt is None:
            targets_types = []
        elif hasattr(raw_tt, "tolist"):
            targets_types = raw_tt.tolist()
        elif isinstance(raw_tt, (list, tuple)):
            targets_types = list(raw_tt)
        elif pd.isna(raw_tt):
            targets_types = []
        else:
            try: targets_types = [int(raw_tt)]
            except: targets_types = []
        
        while i + 1 < len(msgs):
            u, a = msgs[i], msgs[i+1]
            i += 2
            if u.get("role") == "user" and a.get("role") == "assistant":
                rec["pairs"].append((u, a, targets_types))

        # Collect Roots
        if node_depth == 0:
            rec["roots"].append({
                "messages": msgs,
                "root_idx": root_idx,
                "node_depth": node_depth,
                "meta_ci": meta_ci
            })

    # Step 2: Build Samples
    samples = []
    seen = set()
    
    print("Building GRPO samples (Chunk-level)...")
    for rec_id, rec in tqdm(record_map.items(), desc="Build Samples"):
        roots = rec.get("roots", [])
        pairs = rec.get("pairs", [])
        if not roots or not pairs: continue

        cond_gt, edge_types = _build_strict_parent_cond_gt_with_types(
            pairs, enforce_children_in_potential=enforce_children_in_potential
        )
        
        has_gold = any(any(k for d in layer for k, v in d.items() if v) for layer in cond_gt)
        if (not keep_all_no_interact) and (not has_gold): continue

        for root_info in roots:
            msgs = root_info["messages"]
            root_idx = root_info["root_idx"]
            node_depth = root_info["node_depth"]
            
            root_user_raw = msgs[root_idx].get("content", "") or ""
            try: root_user_json = json.loads(root_user_raw)
            except: root_user_json = {"content": root_user_raw}
            
            root_user_json["depth"] = int(node_depth)
            root_user_json["_step_depth"] = int(node_depth)
            
            # 【重点】这里从 SFT 原文解析出了 potentials，这里面包含兴趣！
            # 这部分数据是写在 prompt 里的，绝对会被保留。
            root_content_text = root_user_json.get("content", "") or ""
            chunk_pots_full = _extract_root_potential_full(root_content_text)
            chunk_pots_names = [p["user_name"] for p in chunk_pots_full]
            root_parent_key = _extract_parent_name_from_user_content(root_content_text) or "__ROOT__"

            # 构造 Reward Model 元数据 (这里是导致体积大的原因)
            # embed_root_potential_full=True: 把兴趣存入 metadata (用于 depth 1 展开)
            # embed_root_potential_full=False: 只存名字 (省空间，但 depth 1 展开时会缺兴趣)
            root_user_json["reward_model"] = {
                "ground_truth": {
                    "cond_gt_by_turn": cond_gt,
                    "edge_types_by_turn": edge_types,
                },
                "root_potential": {
                    "user_names": chunk_pots_names,
                    **({"full": chunk_pots_full} if embed_root_potential_full else {}),
                },
                "root_parent_key": root_parent_key,
            }

            root_user_str = json.dumps(root_user_json, ensure_ascii=False)
            prompt_msgs = [
                {"role": "system", "content": rec.get("sys_content", "")},
                {"role": "user", "content": root_user_str}
            ]
            
            sft_info = dict(root_info["meta_ci"] or {})
            sft_info.update({
                "record_id": rec_id,
                "root_potential_count": len(chunk_pots_full),
                "node_depth": int(node_depth)
            })

            if dedup_by_prompt:
                key = _hash_prompt(rec.get("sys_content", ""), root_user_str)
                if key in seen: continue
                seen.add(key)

            samples.append({
                "data_source": data_source,
                "prompt": prompt_msgs,
                "ability": "social_prediction",
                "sft_chunk_info": sft_info
            })

    # Output
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    print(f"[write] Writing {len(samples)} lines to {out_jsonl} ...")
    
    # 增加写入进度条，让你看到它在动
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in tqdm(samples, desc="Writing JSONL"):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print("[done] JSONL generation complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_sft_parquet", required=True)
    ap.add_argument("--val_output", required=True, help="Output JSONL file path")
    ap.add_argument("--data_source", default="social_f1")
    ap.add_argument("--no_dedup", action="store_true")
    ap.add_argument("--drop_all_no_interact", action="store_true")
    ap.add_argument("--enforce_children_in_potential", action="store_true")
    
    # 修复：提供一个 --no_embed_root_potential_full 参数来显式关闭
    # 如果不传，默认 embed_root_potential_full = True (保留完整信息)
    ap.add_argument("--no_embed_root_potential_full", action="store_true", 
                    help="Disable embedding full potential objects in metadata to save space.")
    
    args = ap.parse_args()
    
    _build_grpo_for_one_file(
        sft_parquet_path=args.val_sft_parquet,
        out_jsonl=args.val_output,
        data_source=args.data_source,
        dedup_by_prompt=(not args.no_dedup),
        keep_all_no_interact=(not args.drop_all_no_interact),
        enforce_children_in_potential=args.enforce_children_in_potential,
        embed_root_potential_full=(not args.no_embed_root_potential_full) # 取反
    )