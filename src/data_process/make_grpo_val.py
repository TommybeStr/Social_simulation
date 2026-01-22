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
import random
from hashlib import blake2b
import re
from collections import defaultdict
import numpy as np
import math

from typing import Any, Dict, List, Tuple

# ... (正则和基础工具部分保持不变，为了节省篇幅，只列出修改的核心逻辑) ...
# ==========================================
# 基础配置与正则
# ==========================================
NO_INTERACTION_STR = "以上用户都不感兴趣，没有发生任何交互"
_USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_META_ROLE_KEY = "meta"
_ID_RE = re.compile(r"^(?P<rec>.+?)_root_(?P<root>.+?)_flchunk_(?P<idx>\d+)$")

# 与 make_sft.py / make_grpo_train.py 保持一致的 system prompt（供 evaluate.py 使用）
SYSTEM_PROMPT = """你是社交媒体互动预测专家。请严格依据 user 消息中的标注字段进行判断，并输出一个覆盖全部候选的 JSON 数组（顺序必须严格与候选顺序一致）。
【【输入字段（单样本 JSON）】
- username：作者
- interests：作者兴趣（数组）
- content：正文文本。
- root_context：根帖子的上下文信息（JSON 格式）
  * 格式：`root_context: {"root_username": "根帖子作者名", "root_content": "根帖子内容"}`
  * 如果当前节点就是根帖子（depth=0），则 root_username 和 root_content 均为空字符串
  * 如果当前节点是回复根帖子的节点（depth=1），则包含根帖子的作者名和内容
  * root_context 用于帮助理解当前回复的上下文，特别是当回复是针对根帖子的评论时
- 末尾会追加一个特殊段落 `<POTENTIAL_SPANS>`，用于提供候选人信息。
【关于 potentials】
- `potentials:` 紧跟在 content 之后，包含所有候选人的 JSON 数组。
- 格式为：`potentials: [{"user_name": 候选人, "interests": 候选人兴趣, "depth": 层级, "interaction_count": 互动次数}, ...]`
- 这些候选的先后顺序即为评分类与输出顺序的唯一依据；禁止重排、丢失或增添。
【判断原则（务必遵守）】
1. 注意interests和interaction_count
   - 当候选人与作者的 interests 存在明显交集，或候选 interests 与正文 content 中的主题高度相关时，更倾向预测其会发生互动（type=1 或 type=2）。
   - interaction_count 表示候选人与该作者的历史互动次数，数值越高表示历史互动越频繁，更可能产生新的互动。
2. 正确理解 depth 的作用：
   - depth 表示候选在整棵互动/转发树中的层级
   - depth 越小（越接近根节点），通常代表信息距离更近，更有可能产生直接互动。
   - depth 较大时仍可能发生互动，但除非 interests 显著匹配或历史互动强信号支持，否则应更谨慎地预测互动。
3. type 选择和 content 生成：
   - 当候选人更可能围绕正文内容进行讨论、补充、提问或表达看法时，选择 type=1（评论）。
   - 当候选人更可能以“转发 + 简短态度/评语”的形式传播时，选择 type=2（转发微博）。
   - 生成 type=1 或 type=2 的 content 时，应结合作者 content 与双方 interests，生成简短、自然且与话题相关的文本；避免无意义模板句。
   - 每个候选在当前样本中只能对应一个 type（0 或 1 或 2），不得重复预测或多种类型共存。
【唯一输出（严格格式）】
- 输出一个 JSON 数组，长度等于候选数量，顺序与 <POTENTIAL_SPANS> 中候选顺序一致。
- 数组元素结构：
  {"user_name":"...", "content":"...", "type":0/1/2}
  - type：0=无互动；1=评论；2=转发微博
  - content：type=1/2 时输出评论或转发的内容文本（可为空字符串）；type=0 时输出空字符串。
- 仅输出该 JSON 数组，不得包含解释或多余文本。
""".strip()

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
    pairs: list[tuple[dict, dict, list[int], list[str]]],
    *,
    enforce_children_in_potential: bool = False,
) -> tuple[list[list[dict]], list[list[dict]]]:
    depth_parent2children = defaultdict(dict)
    depth_parent2edge_types = defaultdict(dict)
    max_depth = 0
    observed_depths = set()

    for (u, a, targets_types, targets_texts) in pairs:
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

        # GT Types (+ Gold Text)
        if parent_name not in depth_parent2edge_types[d]:
            depth_parent2edge_types[d][parent_name] = []
        if not isinstance(targets_texts, list):
            targets_texts = []
        n = min(len(cand_order), len(targets_types), len(targets_texts))
        for idx in range(n):
            uname = cand_order[idx]
            if not uname: continue
            try: gtype = int(targets_types[idx])
            except: gtype = 0
            try:
                gtext = str(targets_texts[idx] or "").strip()
            except Exception:
                gtext = ""
            depth_parent2edge_types[d][parent_name].append(
                {"user_name": uname, "gold_type": gtype, "gold_text": gtext}
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
        raw_tc = row.get("targets_comment_texts")
        
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

        # targets_comment_texts: list[str] aligned with candidates
        targets_texts = []
        if raw_tc is None:
            targets_texts = []
        elif hasattr(raw_tc, "tolist"):
            targets_texts = raw_tc.tolist()
        elif isinstance(raw_tc, (list, tuple)):
            targets_texts = list(raw_tc)
        elif pd.isna(raw_tc):
            targets_texts = []
        else:
            try:
                targets_texts = [str(raw_tc)]
            except Exception:
                targets_texts = []
        
        while i + 1 < len(msgs):
            u, a = msgs[i], msgs[i+1]
            i += 2
            if u.get("role") == "user" and a.get("role") == "assistant":
                rec["pairs"].append((u, a, targets_types, targets_texts))

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


# ==========================================
# New: 从 rebuild_data.py 输出（树结构 JSON）直接构造 GRPO val JSONL（供 evaluate.py 使用）
# ==========================================

ROOT_FALLBACK_KEY = "__ROOT__"

def _iter_tree_nodes(root: Dict[str, Any]):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in (node.get("replies") or []):
            stack.append(child)

def _safe_depth(node: Dict[str, Any]) -> int:
    try:
        return int(node.get("depth", 0))
    except Exception:
        return 0

def _strip_retweet_tail(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    idx = text.find("//@")
    if idx == -1:
        return text.strip()
    return text[:idx].rstrip()

def _sanitize_content(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()

def _get_comment_or_repost(node: Dict[str, Any]) -> Tuple[int, str]:
    raw_content = _strip_retweet_tail(node.get("content") or "")
    raw_type = str(node.get("type") or "评论")
    mapped_type = 2 if raw_type == "转发微博" else 1
    if mapped_type == 1 and ("//@") in (node.get("content") or ""):
        mapped_type = 2
    safe = _sanitize_content(raw_content)
    if mapped_type == 2 and safe.strip() in ("转发微博", "快转微博"):
        safe = ""
    return mapped_type, safe

def _build_maps_from_rebuild(records: List[Dict[str, Any]]):
    """
    从 rebuild 输出构建：
    - user_interest_map: username -> interests
    - user_interaction_count_map: (root_username, username) -> interaction_count
    - root_user_to_records: root_username -> records
    """
    user_interest_map: Dict[str, Any] = {}
    user_interaction_count_map: Dict[Tuple[str, str], int] = {}
    root_user_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in records:
        root_user = str(rec.get("user") or "").strip()
        if root_user:
            root_user_to_records[root_user].append(rec)
        for node in _iter_tree_nodes(rec):
            uname = str(node.get("user") or "").strip()
            ints = node.get("interests", [])
            try:
                cnt = int(node.get("interaction_count", 0) or 0)
            except Exception:
                cnt = 0
            if uname and ints and uname not in user_interest_map:
                user_interest_map[uname] = ints
            if root_user and uname:
                key = (root_user, uname)
                if key not in user_interaction_count_map:
                    user_interaction_count_map[key] = cnt
                else:
                    user_interaction_count_map[key] = max(user_interaction_count_map[key], cnt)

    return user_interest_map, user_interaction_count_map, root_user_to_records

def _collect_candidate_pool_for_root(records_of_root: List[Dict[str, Any]], user_interest_map: Dict[str, Any]) -> List[str]:
    users = set()
    for rec in records_of_root:
        for node in _iter_tree_nodes(rec):
            uname = str(node.get("user") or "").strip()
            if uname and (uname in user_interest_map) and user_interest_map.get(uname):
                users.add(uname)
    return list(users)

def _make_potential_objs(candidates: List[str], user_interest_map: Dict[str, Any], root_user: str, inter_cnt_map: Dict[Tuple[str, str], int], depth: int = 0):
    blocks = []
    for u in candidates:
        blocks.append({
            "user_name": u,
            "interests": user_interest_map.get(u, []) or [],
            "depth": int(depth),
            "interaction_count": int(inter_cnt_map.get((root_user, u), 0)),
        })
    return blocks

def _format_root_user_content(root_node: Dict[str, Any], root_user: str, root_interests: List[Any], potentials: List[Dict[str, Any]]) -> str:
    content = _strip_retweet_tail(root_node.get("content") or "")
    pot_str = json.dumps(potentials, ensure_ascii=False, separators=(",", ":"))
    root_ctx = {"root_username": "", "root_content": ""}
    return (
        f"username: {root_user}\n"
        f"content:\n{content}\n"
        f"userinterest: {json.dumps(root_interests or [], ensure_ascii=False)}\n"
        f"root_context: {json.dumps(root_ctx, ensure_ascii=False, separators=(',', ':'))}\n"
        f"potentials: {pot_str}"
    )

def _build_ground_truth_from_tree(root_record: Dict[str, Any], *, root_user: str):
    """
    构造 evaluate.py 需要的：
    - cond_gt_by_turn: depth -> [{parent: [gold_children...]} ...]
    - edge_types_by_turn: depth -> [{parent: [{"user_name", "gold_type", "gold_text"}...]} ...]
    这里 edge_types 仅存 gold 边（节省体积），evaluate.py 会对未出现的候选默认 gold_type=0。
    """
    cond_gt0 = []
    edge0 = []
    cond_gt1 = []
    edge1 = []

    # depth 0: root -> L1
    l1_nodes = root_record.get("replies") or []
    l1_map: Dict[str, Tuple[int, str, Dict[str, Any]]] = {}
    for n in l1_nodes:
        uname = str(n.get("user") or "").strip()
        if not uname:
            continue
        t, txt = _get_comment_or_repost(n)
        l1_map[uname] = (t, txt, n)
    l1_users = list(l1_map.keys())
    if l1_users:
        cond_gt0 = [{root_user: l1_users}]
        edge0 = [{root_user: [{"user_name": u, "gold_type": int(l1_map[u][0]), "gold_text": str(l1_map[u][1] or "")} for u in l1_users]}]

    # depth 1: each L1 user -> L2 (from its replies)
    for l1_user, (_t, _txt, l1_node) in l1_map.items():
        l2_nodes = l1_node.get("replies") or []
        l2_map: Dict[str, Tuple[int, str]] = {}
        for n2 in l2_nodes:
            uname2 = str(n2.get("user") or "").strip()
            if not uname2:
                continue
            t2, txt2 = _get_comment_or_repost(n2)
            l2_map[uname2] = (t2, txt2)
        l2_users = list(l2_map.keys())
        if l2_users:
            cond_gt1.append({l1_user: l2_users})
            edge1.append({l1_user: [{"user_name": u2, "gold_type": int(l2_map[u2][0]), "gold_text": str(l2_map[u2][1] or "")} for u2 in l2_users]})

    cond_gt_by_turn = [cond_gt0, cond_gt1]
    edge_types_by_turn = [edge0, edge1]
    return cond_gt_by_turn, edge_types_by_turn

def build_grpo_val_from_rebuild_json(
    *,
    input_json: str,
    output_jsonl: str,
    data_source: str = "social_f1",
    seed: int = 42,
    candidate_chunk_size: int = 50,
    large_pool_threshold: int = 1000,
):
    with open(input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("输入 JSON 顶层必须是 list（rebuild_data.py 输出）")

    user_interest_map, user_inter_cnt_map, root_map = _build_maps_from_rebuild(records)

    # build pools per root_user
    root_user_to_pool: Dict[str, List[str]] = {}
    for root_user, recs in root_map.items():
        pool = _collect_candidate_pool_for_root(recs, user_interest_map)
        rng_local = random.Random(int(seed) ^ (hash(root_user) & 0xffffffff))
        rng_local.shuffle(pool)
        root_user_to_pool[root_user] = pool

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    n_lines = 0

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for root_user, recs in tqdm(root_map.items(), desc="Build GRPO val (from rebuild)"):
            pool = root_user_to_pool.get(root_user, [])
            if not pool:
                continue
            if len(pool) > int(large_pool_threshold):
                continue
            # chunk candidates to allow evaluate.py preload aggregation
            if candidate_chunk_size <= 0:
                chunks = [pool]
            else:
                chunks = [pool[i:i + candidate_chunk_size] for i in range(0, len(pool), candidate_chunk_size)]

            for rec in recs:
                record_id = str(rec.get("id") or "").strip()
                if not record_id:
                    continue

                root_interests = rec.get("interests", []) or []
                cond_gt_by_turn, edge_types_by_turn = _build_ground_truth_from_tree(rec, root_user=root_user)

                for ch_idx, cand_chunk in enumerate(chunks):
                    pot_objs = _make_potential_objs(cand_chunk, user_interest_map, root_user, user_inter_cnt_map, depth=0)
                    root_content = _format_root_user_content(rec, root_user, root_interests, pot_objs)

                    root_user_json = {
                        "user_name": root_user,
                        "content": root_content,
                        "depth": 0,
                        "_step_depth": 0,
                        "reward_model": {
                            "ground_truth": {
                                "cond_gt_by_turn": cond_gt_by_turn,
                                "edge_types_by_turn": edge_types_by_turn,
                            },
                            "root_potential": {
                                "full": pot_objs,
                                "user_names": [p.get("user_name") for p in pot_objs if p.get("user_name")],
                            },
                            "root_parent_key": root_user or ROOT_FALLBACK_KEY,
                        },
                    }

                    row = {
                        "data_source": data_source,
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(root_user_json, ensure_ascii=False)},
                        ],
                        "ability": "social_prediction",
                        "sft_chunk_info": {
                            "record_id": record_id,
                            "root_user": root_user,
                            "chunk_index": ch_idx,
                            "format": "grpo_jsonl_from_rebuild",
                        },
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_lines += 1

    print(f"[done] wrote {n_lines} lines -> {output_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="rebuild_data.py 输出的树结构 JSON（顶层 list）")
    mode.add_argument("--val_sft_parquet", help="兼容旧用法：输入 SFT val.parquet")

    ap.add_argument("--output", required=True, help="输出 JSONL（单文件，供 dataprocess/evaluate.py 使用）")
    ap.add_argument("--data_source", default="social_f1")
    ap.add_argument("--seed", type=int, default=42)
    # 对齐 make_sft.py 的 val 构造默认 chunk 大小（VAL_K_PER_CHUNK=40）
    ap.add_argument("--candidate_chunk_size", type=int, default=40, help="root 候选池分片大小（用于 pool preload 聚合）")
    ap.add_argument("--large_pool_threshold", type=int, default=1000, help="候选池过大则跳过该 root")

    # 旧模式参数（保留）
    ap.add_argument("--no_dedup", action="store_true")
    ap.add_argument("--drop_all_no_interact", action="store_true")
    ap.add_argument("--enforce_children_in_potential", action="store_true")
    ap.add_argument("--no_embed_root_potential_full", action="store_true",
                    help="Disable embedding full potential objects in metadata to save space.")

    args = ap.parse_args()

    if args.input:
        build_grpo_val_from_rebuild_json(
            input_json=args.input,
            output_jsonl=args.output,
            data_source=args.data_source,
            seed=args.seed,
            candidate_chunk_size=args.candidate_chunk_size,
            large_pool_threshold=args.large_pool_threshold,
        )
    else:
        # 兼容旧用法：SFT parquet -> GRPO JSONL
        _build_grpo_for_one_file(
            sft_parquet_path=args.val_sft_parquet,
            out_jsonl=args.output,
            data_source=args.data_source,
            dedup_by_prompt=(not args.no_dedup),
            keep_all_no_interact=(not args.drop_all_no_interact),
            enforce_children_in_potential=args.enforce_children_in_potential,
            embed_root_potential_full=(not args.no_embed_root_potential_full) # 取反
        )