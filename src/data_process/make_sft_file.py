#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单轮 SFT 构造脚本（构造 root-第一层回复 + 第一层-第二层回复的数据集）
- 对 depth 0（root）节点生成样本，预测第一层回复
  * depth=0 → target_layer=0 → 用分类头0（预测第一层）
- 对 depth 1（第一层）节点生成样本，预测第二层回复
  * depth=1 → target_layer=1 → 用分类头1（预测第二层）
- 每条样本的候选人由两部分构成：
  1. 真实互动用户（真实答案）
  2. k倍于真实互动者人数的噪声候选人，k是可配置的随机数
     * 噪声候选人从候选池中随机抽取（不包含本条真实答案）
     * depth=0 和 depth=1 的噪音比例可以分开调控
- 每条样本 messages：system + user + assistant
  * assistant 为覆盖候选数组的唯一 JSON 输出
  * user 消息中包含 root_context（如果是 root 节点则置空）
- 统计&过滤：
  1) 候选池规模（均值、min/max、阈值计数）
  2) 样本构造时跳过候选池人数 > 阈值(默认1000)的根作者
  3) 报告：平均每条样本从多少候选中选出多少 gold（gold=type!=0）

当前版本的切分输出：
- 按 record_id 划分为 train / val 两部分（约 85% / 15%）
- 只输出 train.parquet 和 val.parquet
- 原先 "val + test" 的样本都归入 val
"""

import os
import json
import argparse
import random
import re
import math
from collections import defaultdict, deque
from typing import List, Dict, Any, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------- 常量 ----------

# depth=0（root-第一层）的噪声候选人倍数范围 [min_k, max_k]（浮点数）
NOISE_K_MIN_DEPTH0 = 0.0
NOISE_K_MAX_DEPTH0 = 3.0

# depth=1（第一层-第二层）的噪声候选人倍数范围 [min_k, max_k]（浮点数）
NOISE_K_MIN_DEPTH1 = 5.0
NOISE_K_MAX_DEPTH1 = 11.0

# train部分每条样本的最大gold人数（深度可独立控制）
MAX_GOLD_PER_SAMPLE_DEPTH0 = 10
MAX_GOLD_PER_SAMPLE_DEPTH1 = 4

# val部分每个chunk的候选人数（原方式）
VAL_K_PER_CHUNK = 40

# ---------- System Prompt ----------
SYSTEM_PROMPT = f'''你是社交媒体互动预测专家。请严格依据 user 消息中的标注字段进行判断，并输出一个覆盖全部候选的 JSON 数组（顺序必须严格与候选顺序一致）。
【【输入字段（单样本 JSON）】
- username：作者
- interests：作者兴趣（数组）
- content：正文文本。
- root_context：根帖子的上下文信息（JSON 格式）
  * 格式：`root_context: {{"root_username": "根帖子作者名", "root_content": "根帖子内容"}}`
  * 如果当前节点就是根帖子（depth=0），则 root_username 和 root_content 均为空字符串
  * 如果当前节点是回复根帖子的节点（depth=1），则包含根帖子的作者名和内容
  * root_context 用于帮助理解当前回复的上下文，特别是当回复是针对根帖子的评论时
- 末尾会追加一个特殊段落 `<POTENTIAL_SPANS>`，用于提供候选人信息。
【关于 potentials】
- `potentials:` 紧跟在 content 之后，包含所有候选人的 JSON 数组。
- 格式为：`potentials: [{{"user_name": 候选人, "interests": 候选人兴趣, "depth": 层级, "interaction_count": 互动次数}}, ...]`
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
  {{"user_name":"...", "content":"...", "type":0/1/2}}
  - type：0=无互动；1=评论；2=转发微博
  - content：type=1/2 时输出评论或转发的内容文本（可为空字符串）；type=0 时输出空字符串。
- 仅输出该 JSON 数组，不得包含解释或多余文本。
'''

# ---------- Arrow schema ----------
MSG_STRUCT = pa.struct([
    pa.field("role", pa.string()),
    pa.field("content", pa.string()),
    pa.field("loss", pa.int64()).with_nullable(True),
])
TABLE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("messages", pa.list_(MSG_STRUCT)),
    pa.field("targets_per_potential_types", pa.list_(pa.int32())),
    pa.field("targets_comment_texts", pa.list_(pa.string())),
    pa.field("node_depth", pa.int32()),   # 0/1，与 <TL?> 一致
    pa.field("sft_chunk_info", pa.large_string()),
])

def rows_to_arrow_table(rows: List[Dict[str, Any]]) -> pa.Table:
    ids = pa.array([r['id'] for r in rows], type=pa.string())
    msgs = pa.array(
        [[{"role": m.get("role"), "content": m.get("content"), "loss": m.get("loss")} for m in r['messages']]
         for r in rows],
        type=pa.list_(MSG_STRUCT)
    )
    t_types  = pa.array([r.get("targets_per_potential_types", []) for r in rows], type=pa.list_(pa.int32()))
    t_comms  = pa.array([r.get("targets_comment_texts", []) for r in rows], type=pa.list_(pa.string()))
    node_d   = pa.array([int(r.get("node_depth", 0)) for r in rows], type=pa.int32())
    infos = pa.array([json.dumps(r.get('sft_chunk_info', {}), ensure_ascii=False) for r in rows], type=pa.large_string())
    return pa.Table.from_arrays([ids, msgs, t_types, t_comms, node_d, infos], schema=TABLE_SCHEMA)

def save_parquet_rows(rows: List[Dict[str, Any]], path: str, *, batch_size: int = 4000, desc: str = "写 Parquet", compression: str = "zstd"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        empty = pa.Table.from_arrays(
            [pa.array([], type=pa.string()),
             pa.array([], type=pa.list_(MSG_STRUCT)),
             pa.array([], type=pa.list_(pa.int32())),
             pa.array([], type=pa.list_(pa.string())),
             pa.array([], type=pa.int32()),
             pa.array([], type=pa.large_string())],
            schema=TABLE_SCHEMA
        )
        pq.write_table(empty, path, compression=compression)
        return
    writer = None
    total_batches = (len(rows) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc=desc) as pbar:
        for i in range(0, len(rows), batch_size):
            table = rows_to_arrow_table(rows[i:i+batch_size])
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression, use_dictionary=True)
            writer.write_table(table)
            pbar.update(1)
    if writer is not None:
        writer.close()

def iter_tree_nodes(root: Dict[str, Any]):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in (node.get('replies') or []):
            stack.append(child)

def _safe_depth(node: Dict[str, Any]) -> int:
    try:
        return int(node.get('depth', 0))
    except Exception:
        return 0

# ---------- 查找 root 节点 ----------
def find_root_node(record: Dict[str, Any], target_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    在 record 树中查找 target_node 对应的 root 节点（depth=0）
    如果 target_node 本身就是 root，返回它自己
    """
    # 如果目标节点本身就是 root，直接返回
    if _safe_depth(target_node) == 0:
        return target_node
    
    # 否则，从 record 开始遍历，找到 depth=0 的节点
    for node in iter_tree_nodes(record):
        if _safe_depth(node) == 0:
            return node
    
    # 如果找不到，返回 record 本身（作为 fallback）
    return record

# ---------- 候选池 ----------
def build_user_interest_map_and_group_by_root(records: List[Dict[str, Any]]):
    user_interest_map: Dict[str, Any] = {}
    # (root_user, interactor_user) -> interaction_count
    # 表示在最近一个月内，interactor_user 与 root_user 的总互动次数
    user_interaction_count_map: Dict[Tuple[str, str], int] = {}
    root_user_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in tqdm(records, desc="扫描记录并构建画像/根作者分组"):
        root_user = str(record.get('user') or "")
        if root_user:
            root_user_to_records[root_user].append(record)
        for node in iter_tree_nodes(record):
            u = str(node.get('user') or "")
            ints = node.get('interests', [])
            interaction_count = int(node.get('interaction_count', 0) or 0)
            if u and ints and (u not in user_interest_map):
                user_interest_map[u] = ints
            # 收集互动次数：针对每个 (root_user, interactor_user) 对
            # interaction_count 已经在 rebuild.py 中计算为最近一个月内该互动者与该作者的总互动次数
            if u and root_user:
                key = (root_user, u)
                # 同一个 (root_user, interactor_user) 对在不同节点中的 interaction_count 应该相同
                # 但为了安全，我们取最大值
                if key not in user_interaction_count_map:
                    user_interaction_count_map[key] = interaction_count
                else:
                    user_interaction_count_map[key] = max(user_interaction_count_map[key], interaction_count)
    return user_interest_map, root_user_to_records, user_interaction_count_map

def collect_candidate_pool_for_root(records_of_root: List[Dict[str, Any]], user_interest_map: Dict[str, Any], root_user: str = None) -> List[str]:
    """
    收集候选池，包含 root_user 自己（博主可以回复自己的帖子）
    """
    users: Set[str] = set()
    for rec in records_of_root:
        for node in iter_tree_nodes(rec):
            u = str(node.get('user') or "")
            if u and (u in user_interest_map) and user_interest_map[u]:
                users.add(u)
    return list(users)

# ---------- 文本处理 ----------
def _strip_retweet_tail(text: Any) -> Any:
    """
    移除文本中第一个 "//@" 及其后面的所有内容。
    只处理 "//@"，不处理 "/@"。
    """
    if not isinstance(text, str):
        return text
    idx = text.find("//@")
    if idx == -1:
        return text.strip()
    return text[:idx].rstrip()

def _sanitize_content_for_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()

# ---------- <POTENTIAL_SPANS> ----------
def make_potential_span_text(potentials: List[str], user_interest_map: Dict[str, Any], target_layer: int, root_user: str, user_interaction_count_map: Dict[Tuple[str, str], int] = None) -> str:
    if user_interaction_count_map is None:
        user_interaction_count_map = {}
    dval = int(target_layer)
    blocks = []
    for pid in potentials:
        # 查找 (root_user, pid) 的互动次数
        interaction_count = user_interaction_count_map.get((root_user, pid), 0)
        block = {
            "user_name": pid, 
            "interests": user_interest_map.get(pid, []) or [], 
            "depth": dval,
            "interaction_count": interaction_count
        }
        blocks.append(block)
    # 将候选数组序列化为 JSON，不使用换行
    potentials_json = json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))
    return "potentials: " + potentials_json

def make_user_plain_text_with_cached_spans(node: Dict[str, Any], ancestors: List[str], cached_spans_text: str, target_layer: int, root_node: Dict[str, Any] = None) -> str:
    base_content = _strip_retweet_tail(node.get('content') or "")
    username = str(node.get('user') or "")
    userinterest = node.get('interests', [])
    
    # 构建 root_context（JSON 格式）
    # 如果当前节点是 root（depth=0），root_context 字段存在但内容为空
    # 否则，root_context 包含 root 的用户名和内容
    if root_node is not None and _safe_depth(node) != 0:
        # 非 root 节点：包含 root 的用户名和内容
        root_username = str(root_node.get('user') or "")
        root_content = _strip_retweet_tail(root_node.get('content') or "")
        root_context_obj = {
            "root_username": root_username,
            "root_content": root_content
        }
    else:
        # root 节点：root_context 字段存在但内容为空
        root_context_obj = {
            "root_username": "",
            "root_content": ""
        }
    
    # 将 root_context 序列化为 JSON 字符串（紧凑格式，不使用换行）
    root_context_json = json.dumps(root_context_obj, ensure_ascii=False, separators=(',', ':'))
    root_context = f"root_context: {root_context_json}\n"
    
    user_plain_text = (
        "username: " + username + "\n" +
        "content:\n" + base_content + "\n" +
        "userinterest: " + json.dumps(userinterest, ensure_ascii=False) + "\n" +
        root_context +
        cached_spans_text
    )
    return user_plain_text

# ---------- 标签构造 ----------
def get_comment_or_repost(child: Dict[str, Any]) -> Tuple[int, str]:
    raw_content = _strip_retweet_tail(child.get('content') or "")
    raw_type = str(child.get('type') or "评论")
    mapped_type = 2 if raw_type == "转发微博" else 1
    if mapped_type == 1 and ("//@") in (child.get('content') or ""):
        mapped_type = 2
    safe = _sanitize_content_for_markers(raw_content)
    # 如果 type=2 且内容为"转发微博"或"快转微博"，则清空 content
    if mapped_type == 2:
        safe_stripped = safe.strip()
        if safe_stripped in ("转发微博", "快转微博"):
            safe = ""
    return mapped_type, safe

def extract_children_map(root: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    node_children: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in iter_tree_nodes(root):
        for child in (node.get('replies') or []):
            node_children[id(node)].append(child)
    return node_children

def build_child_map(children: List[Dict[str, Any]]) -> Dict[str, Tuple[int, str]]:
    m = {}
    for c in (children or []):
        u = str(c.get('user') or "").strip()
        if not u:
            continue
        t, txt = get_comment_or_repost(c)
        # type=1（评论）和 type=2（转发微博）都需要保留内容
        m[u] = (t, txt if t in (1, 2) else "")
    return m

# ---------- 分块 ----------
def chunk_list_no_overlap(lst: List[str], k: int) -> List[List[str]]:
    if k <= 0:
        return [lst] if lst else []
    return [lst[i:i+k] for i in range(0, len(lst), k)]

# ---------- 生成样本（train方式：真实互动者+噪声，限制gold最多10人） ----------
def generate_train_rows_for_node(
    record: Dict[str, Any],
    node: Dict[str, Any],
    root_user: str,
    root_potential_full: List[str],
    user_interest_map: Dict[str, Any],
    shuffle_seed: int,
    noise_k_min: float,
    noise_k_max: float,
    user_interaction_count_map: Dict[str, int] = None,
) -> List[Dict[str, Any]]:
    """
    为train部分生成样本，处理 depth=0 或 depth=1 的节点
    - depth=0: 预测第一层回复（target_layer=0）
    - depth=1: 预测第二层回复（target_layer=1）
    - 如果真实互动者 <= MAX_GOLD_PER_SAMPLE，生成一个样本
    - 如果真实互动者 > MAX_GOLD_PER_SAMPLE，分chunk，每个chunk最多MAX_GOLD_PER_SAMPLE个真实互动者
    - 每个chunk的候选列表 = 真实互动者 + k倍噪声候选人（k为可配置的随机数）
    """
    rows: List[Dict[str, Any]] = []
    
    d = _safe_depth(node)
    
    # 只处理 depth=0 或 depth=1 的节点
    if d not in (0, 1):
        return rows
    
    # target_layer = depth（depth=0 预测第一层，depth=1 预测第二层）
    target_layer = d
    
    # 获取该节点的子节点（真实互动者）
    node_children = extract_children_map(record)
    node_children_list = node_children.get(id(node), [])
    cm = build_child_map(node_children_list)
    
    # 真实互动用户列表
    real_interactors = list(cm.keys())
    n_real = len(real_interactors)
    
    if n_real == 0:
        # 如果没有真实互动者，跳过这个节点
        return rows
    
    # 从候选池中排除真实答案，作为噪声候选池
    noise_pool = [u for u in root_potential_full if u not in real_interactors]
    
    # 如果噪声候选池为空，跳过这个节点（因为无法添加噪声）
    if not noise_pool:
        return rows
    
    # 初始化随机数生成器（使用 node 的 id 或内容作为种子的一部分）
    node_id_str = str(node.get('id') or '') + str(id(node))
    rng_node = random.Random(shuffle_seed ^ (hash(node_id_str) & 0xffffffff))
    
    # 根据 depth 选择对应的 MAX_GOLD_PER_SAMPLE
    if d == 0:
        max_gold_per_sample = MAX_GOLD_PER_SAMPLE_DEPTH0
    else:
        max_gold_per_sample = MAX_GOLD_PER_SAMPLE_DEPTH1

    # 如果真实互动者超过对应 depth 的 max_gold_per_sample，需要分 chunk
    if max_gold_per_sample is not None and max_gold_per_sample > 0 and n_real > max_gold_per_sample:
        # 将真实互动者分成多个chunk
        real_chunks = chunk_list_no_overlap(real_interactors, max_gold_per_sample)
    else:
        # 只有一个chunk
        real_chunks = [real_interactors]
    
    record_id = str(record.get('id') or '')
    
    # 找到 root 节点（用于 root_context）
    root_node = find_root_node(record, node)
    
    # 为每个chunk生成一个样本
    for ch_idx, real_chunk in enumerate(real_chunks):
        n_real_chunk = len(real_chunk)
        
        # 为这个chunk随机选择 k（浮点数，范围由参数指定）
        k = rng_node.uniform(noise_k_min, noise_k_max)
        # 计算噪声人数，如果不是整数则向上取整
        n_noise = math.ceil(k * n_real_chunk)
        
        # 确保至少添加1个噪声候选人
        if n_noise < 1:
            n_noise = 1
        
        # 从噪声候选池中随机抽取
        if len(noise_pool) >= n_noise:
            noise_candidates = rng_node.sample(noise_pool, n_noise)
        else:
            # 如果噪声候选池不够，重复采样
            noise_candidates = rng_node.choices(noise_pool, k=n_noise)
        
        # 合并真实互动者和噪声候选人
        candidates = real_chunk + noise_candidates
        
        # 打乱候选列表顺序
        rng_node.shuffle(candidates)
        
        # 生成 potential span 文本
        spans_text = make_potential_span_text(candidates, user_interest_map, target_layer, root_user, user_interaction_count_map)
        
        # user 文本（传入 root_node 用于生成 root_context）
        user_plain = make_user_plain_text_with_cached_spans(node, [], spans_text, target_layer, root_node)
        
        # 构造 assistant JSON（长度与 candidates 一致；顺序严格一致）
        types, comms = [], []
        out_arr = []
        for pid in candidates:
            if pid in cm:
                t, txt = cm[pid]
            else:
                t, txt = 0, ""
            types.append(t)
            comms.append(txt)
            # content：type=1/2 时输出评论或转发的内容文本；type=0 时输出空字符串
            content = txt if t in (1, 2) else ""
            out_arr.append({"user_name": pid, "content": content, "type": int(t)})
        
        assistant_text = json.dumps(out_arr, ensure_ascii=False)
        
        messages = [
            {'role': 'system',    'content': SYSTEM_PROMPT, 'loss': 0},
            {'role': 'user',      'content': user_plain,    'loss': 0},
            {'role': 'assistant', 'content': assistant_text,'loss': 1},
        ]
        
        # 生成样本ID
        if d == 0:
            prefix = "root"
        else:
            prefix = f"depth{d}"
        
        if len(real_chunks) > 1:
            sample_id = "{}_{}_tL{}_chunk{}".format(record_id, prefix, target_layer, ch_idx)
        else:
            sample_id = "{}_{}_tL{}".format(record_id, prefix, target_layer)
        
        rows.append({
            'id': sample_id,
            'messages': messages,
            'targets_per_potential_types': types,
            'targets_comment_texts': comms,
            'node_depth': int(target_layer),
            'sft_chunk_info': {
                "record_id": record_id,
                "root_user": root_user,
                "node_user": str(node.get('user') or ''),
                "orig_node_depth": d,
                "target_layer": target_layer,
                "n_real_interactors": n_real_chunk,
                "n_noise": len(candidates) - n_real_chunk,
                "n_total_candidates": len(candidates),
                "chunk_index": ch_idx if len(real_chunks) > 1 else None,
                "format": "single_turn_messages_with_targets_and_assistant_json",
            }
        })
    
    return rows

# ---------- 生成样本（val方式：原方式，分批送入整个候选池） ----------
def generate_val_rows_for_node(
    record: Dict[str, Any],
    node: Dict[str, Any],
    root_user: str,
    root_potential_full: List[str],
    user_interest_map: Dict[str, Any],
    k_per_chunk: int,
    spans_cache: Dict[Tuple[str, int, int, int], str],  # (root_user, node_id_hash, chunk_idx, target_layer)
    user_interaction_count_map: Dict[str, int] = None,
) -> List[Dict[str, Any]]:
    """
    为val部分生成样本，使用原方式：将整个候选池分批送入
    支持 depth=0 和 depth=1 的节点
    """
    rows: List[Dict[str, Any]] = []
    
    if not root_potential_full:
        return rows
    
    d = _safe_depth(node)
    
    # 只处理 depth=0 或 depth=1 的节点
    if d not in (0, 1):
        return rows
    
    # target_layer = depth（depth=0 预测第一层，depth=1 预测第二层）
    target_layer = d
    
    # 将候选池分成多个chunk
    root_chunks = chunk_list_no_overlap(root_potential_full, k_per_chunk)
    node_children = extract_children_map(record)
    node_children_list = node_children.get(id(node), [])
    cm = build_child_map(node_children_list)
    
    # 找到 root 节点（用于 root_context）
    root_node = find_root_node(record, node)
    
    # 为每个chunk生成一个样本
    node_id_hash = hash(id(node)) & 0xffffffff
    for ch_idx, pot_chunk in enumerate(root_chunks):
        cache_key = (root_user, node_id_hash, ch_idx, target_layer)
        if cache_key in spans_cache:
            spans_text = spans_cache[cache_key]
        else:
            spans_text = make_potential_span_text(pot_chunk, user_interest_map, target_layer, root_user, user_interaction_count_map)
        spans_cache[cache_key] = spans_text
        
        # user 文本（传入 root_node 用于生成 root_context）
        user_plain = make_user_plain_text_with_cached_spans(node, [], spans_text, target_layer, root_node)
        
        # 构造 assistant JSON（长度与 pot_chunk 一致；顺序严格一致）
        types, comms = [], []
        out_arr = []
        for pid in pot_chunk:
            if pid in cm:
                t, txt = cm[pid]
            else:
                t, txt = 0, ""
            types.append(t)
            comms.append(txt)
            # content：type=1/2 时输出评论或转发的内容文本；type=0 时输出空字符串
            content = txt if t in (1, 2) else ""
            out_arr.append({"user_name": pid, "content": content, "type": int(t)})
        
        assistant_text = json.dumps(out_arr, ensure_ascii=False)
        
        messages = [
            {'role': 'system',    'content': SYSTEM_PROMPT, 'loss': 0},
            {'role': 'user',      'content': user_plain,    'loss': 0},
            {'role': 'assistant', 'content': assistant_text,'loss': 1},
        ]
        
        record_id = str(record.get('id') or '')
        if d == 0:
            prefix = "root"
        else:
            prefix = f"depth{d}"
        
        rows.append({
            'id': "{}_{}_node_{}_potchunk_{}_tL{}".format(record_id, prefix, node_id_hash, ch_idx, target_layer),
            'messages': messages,
            'targets_per_potential_types': types,
            'targets_comment_texts': comms,
            'node_depth': int(target_layer),
            'sft_chunk_info': {
                "record_id": record_id,
                "root_user": root_user,
                "node_user": str(node.get('user') or ''),
                "orig_node_depth": d,
                "target_layer": target_layer,
                "potential_chunk_index": ch_idx,
                "k_per_chunk": k_per_chunk,
                "format": "single_turn_messages_with_targets_and_assistant_json",
            }
        })
    
    return rows

# ---------- 统计 ----------
def report_candidate_gold_stats(rows: List[Dict[str, Any]]):
    # gold 定义：type != 0（评论/转发都是正例）
    total_samples = 0
    total_cand = 0
    total_gold = 0
    
    for r in rows:
        types = r.get("targets_per_potential_types", []) or []
        cand = len(types)
        gold = sum(1 for t in types if int(t) != 0)
        total_samples += 1
        total_cand += cand
        total_gold += gold
    
    if total_samples > 0:
        avg_cand = total_cand / total_samples
        avg_gold = total_gold / total_samples
        ratio = (total_gold / total_cand) if total_cand > 0 else 0.0
        print("[stats] 候选与 gold 平均统计（gold=type!=0）：")
        print(f"        样本数={total_samples}, 平均候选={avg_cand:.2f}, 平均gold={avg_gold:.2f}, gold/候选={ratio:.4f}")
    else:
        print("[stats] 样本数=0 (无统计)")

# ---------- 入口 ----------
_ID_RE = re.compile(r"^(?P<rec>.+?)_root_tL(?P<layer>\d+)$")
def parse_record_id(sample_id: str) -> str:
    m = _ID_RE.match(str(sample_id))
    return m.group("rec") if m else ""

def main(input_file: str,
         output_dir: str,
         shuffle_seed: int = 42,
         parquet_batch_size: int = 4000,
         parquet_compression: str = "zstd",
         large_pool_threshold: int = 1000,
         val_depth2_head: int = 10,
         val_k_per_chunk: int = None,
         noise_k_min_depth0: float = None,
         noise_k_max_depth0: float = None,
         noise_k_min_depth1: float = None,
         noise_k_max_depth1: float = None):
    # 如果未指定 val_k_per_chunk，使用默认值
    if val_k_per_chunk is None:
        val_k_per_chunk = VAL_K_PER_CHUNK
    
    # 如果未指定噪音比例，使用默认值
    if noise_k_min_depth0 is None:
        noise_k_min_depth0 = NOISE_K_MIN_DEPTH0
    if noise_k_max_depth0 is None:
        noise_k_max_depth0 = NOISE_K_MAX_DEPTH0
    if noise_k_min_depth1 is None:
        noise_k_min_depth1 = NOISE_K_MIN_DEPTH1
    if noise_k_max_depth1 is None:
        noise_k_max_depth1 = NOISE_K_MAX_DEPTH1
    
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin:
        records = json.load(fin)
    if not isinstance(records, list):
        raise ValueError("输入 JSON 顶层必须是 list")

    user_interest_map, root_user_to_records, user_interaction_count_map = build_user_interest_map_and_group_by_root(records)

    root_user_to_candidate_pool: Dict[str, List[str]] = {}
    for root_user, recs in tqdm(root_user_to_records.items(), total=len(root_user_to_records), desc="按根作者构建候选池"):
        pool = collect_candidate_pool_for_root(recs, user_interest_map, root_user=root_user)
        rng_local = random.Random(shuffle_seed ^ (hash(root_user) & 0xffffffff))
        rng_local.shuffle(pool)
        root_user_to_candidate_pool[root_user] = pool

    # === 候选池规模统计（按根作者） ===
    pool_sizes = [len(v) for v in root_user_to_candidate_pool.values()]
    if pool_sizes:
        avg_sz = sum(pool_sizes) / len(pool_sizes)
        n_large = sum(1 for s in pool_sizes if s > large_pool_threshold)
        print("[stats] 候选池（按根作者）规模统计：")
        print(f"        根作者数 = {len(pool_sizes)}")
        print(f"        平均每池人数 = {avg_sz:.2f}")
        print(f"        min/max = {min(pool_sizes)}/{max(pool_sizes)}")
        print(f"        候选总人数（各根作者池大小之和，非全局去重）= {sum(pool_sizes)}")
        print(f"        候选池 > {large_pool_threshold} 的根作者数 = {n_large}")
    else:
        print("[stats] 未构建到任何候选池（root_user_to_candidate_pool 为空）。")

    # === 跳过候选池人数 > 阈值 的根作者 ===
    skipped_roots = [ru for ru, pool in root_user_to_candidate_pool.items() if len(pool) > large_pool_threshold]
    skipped_records = sum(len(root_user_to_records.get(ru, [])) for ru in skipped_roots)
    if skipped_roots:
        print(f"[filter] 将在样本构造中跳过候选池 > {large_pool_threshold} 的根作者：{len(skipped_roots)} 个（涉及记录 {skipped_records} 条）")
    else:
        print("[filter] 无需跳过任何根作者。")

    rng = random.Random(shuffle_seed)

    # -------- 先按 record_id 分割：train / val --------
    unique_recs = []
    for root_user, recs in root_user_to_records.items():
        for rec in recs:
            record_id = str(rec.get('id') or '')
            if record_id and record_id not in unique_recs:
                unique_recs.append(record_id)
    
    rng.shuffle(unique_recs)
    n_rec = len(unique_recs)
    
    # 约 85% 记录作为 train，剩余 15% 作为 val
    n_train_rec = int(round(n_rec * 0.85))
    n_train_rec = min(max(n_train_rec, 0), n_rec)  # 防止极端 rounding 问题
    
    recs_train = set(unique_recs[:n_train_rec])
    recs_val   = set(unique_recs[n_train_rec:])
    
    print("[split] record_id 级：train/val 记录数 = {}/{}".format(len(recs_train), len(recs_val)))

    # -------- 生成 train 样本（新方式：真实互动者+噪声，限制gold最多10人） --------
    rows_train: List[Dict[str, Any]] = []
    train_rec_count = 0
    train_node_count = 0
    
    # 计算总节点数（用于进度条）
    total_train_nodes = 0
    for root_user, recs in root_user_to_records.items():
        root_pool_full = root_user_to_candidate_pool.get(root_user, [])
        if len(root_pool_full) > large_pool_threshold:
            continue
        for rec in recs:
            record_id = str(rec.get('id') or '')
            if record_id in recs_train:
                # 统计该 record 中 depth=0 和 depth=1 的节点数
                for node in iter_tree_nodes(rec):
                    d = _safe_depth(node)
                    if d in (0, 1):
                        total_train_nodes += 1
    
    with tqdm(total=total_train_nodes, desc="生成train样本（新方式）") as pbar:
        for root_user, recs in root_user_to_records.items():
            root_pool_full = root_user_to_candidate_pool.get(root_user, [])
            # 跳过大池根作者
            if len(root_pool_full) > large_pool_threshold:
                continue
            for rec in recs:
                record_id = str(rec.get('id') or '')
                if record_id in recs_train:
                    # 遍历该 record 中的所有节点，处理 depth=0 和 depth=1 的节点
                    for node in iter_tree_nodes(rec):
                        d = _safe_depth(node)
                        if d in (0, 1):
                            # 根据 depth 选择对应的噪音比例
                            if d == 0:
                                noise_k_min = noise_k_min_depth0
                                noise_k_max = noise_k_max_depth0
                            else:  # d == 1
                                noise_k_min = noise_k_min_depth1
                                noise_k_max = noise_k_max_depth1
                            
                            node_rows = generate_train_rows_for_node(
                                record=rec,
                                node=node,
                                root_user=root_user,
                                root_potential_full=root_pool_full,
                                user_interest_map=user_interest_map,
                                shuffle_seed=shuffle_seed,
                                noise_k_min=noise_k_min,
                                noise_k_max=noise_k_max,
                                user_interaction_count_map=user_interaction_count_map,
                            )
                            rows_train.extend(node_rows)
                            if node_rows:
                                train_node_count += 1
                            pbar.update(1)
                    train_rec_count += 1
    
    print(f"[train] 处理了 {train_rec_count} 条记录，{train_node_count} 个节点，共生成 {len(rows_train)} 个样本")

    # -------- 生成 val 样本（原方式：分批送入整个候选池） --------
    rows_val: List[Dict[str, Any]] = []
    spans_cache: Dict[Tuple[str, int, int, int], str] = {}  # (root_user, node_id_hash, chunk_idx, target_layer)
    val_rec_count = 0
    val_node_count = 0
    
    # 计算总节点数（用于进度条）
    total_val_nodes = 0
    for root_user, recs in root_user_to_records.items():
        root_pool_full = root_user_to_candidate_pool.get(root_user, [])
        if len(root_pool_full) > large_pool_threshold:
            continue
        for rec in recs:
            record_id = str(rec.get('id') or '')
            if record_id in recs_val:
                # 统计该 record 中 depth=0 和 depth=1 的节点数
                for node in iter_tree_nodes(rec):
                    d = _safe_depth(node)
                    if d in (0, 1):
                        total_val_nodes += 1
    
    with tqdm(total=total_val_nodes, desc="生成val样本（原方式）") as pbar:
        for root_user, recs in root_user_to_records.items():
            root_pool_full = root_user_to_candidate_pool.get(root_user, [])
            # 跳过大池根作者
            if len(root_pool_full) > large_pool_threshold:
                continue
            for rec in recs:
                record_id = str(rec.get('id') or '')
                if record_id in recs_val:
                    # 遍历该 record 中的所有节点，处理 depth=0 和 depth=1 的节点
                    for node in iter_tree_nodes(rec):
                        d = _safe_depth(node)
                        if d in (0, 1):
                            rec_rows = generate_val_rows_for_node(
                                record=rec,
                                node=node,
                                root_user=root_user,
                                root_potential_full=root_pool_full,
                                user_interest_map=user_interest_map,
                                k_per_chunk=val_k_per_chunk,
                                spans_cache=spans_cache,
                                user_interaction_count_map=user_interaction_count_map,
                            )
                            rows_val.extend(rec_rows)
                            if rec_rows:
                                val_node_count += 1
                            pbar.update(1)
                    val_rec_count += 1
    
    print(f"[val] 处理了 {val_rec_count} 条记录，{val_node_count} 个节点，共生成 {len(rows_val)} 个样本")

    # === 报告每条样本平均候选/平均 gold ===
    print("\n[train] 统计：")
    report_candidate_gold_stats(rows_train)
    print("\n[val] 统计：")
    report_candidate_gold_stats(rows_val)

    # === 随机打乱训练集 ===
    rng.shuffle(rows_train)

    # === 验证集示例：输出有 gold 的样本 ID（最多 N 条） ===
    val_sample_ids = []
    for r in rows_val:
        types = r.get("targets_per_potential_types", []) or []
        if any(int(t) != 0 for t in types):
            val_sample_ids.append(str(r.get("id")))
    # 去重并截取
    val_sample_ids = list(dict.fromkeys(val_sample_ids))[:max(0, int(val_depth2_head))]
    if val_sample_ids:
        print(f"[val] 有 gold 的样本ID（最多 {val_depth2_head} 条）：")
        for i, sid in enumerate(val_sample_ids, 1):
            print(f"       {i:02d}. {sid}")
    else:
        print("[val] 未找到有 gold 的验证集样本。")

    print("\n[split] 样本条数：train/val = {}/{}".format(len(rows_train), len(rows_val)))

    train_out = os.path.join(output_dir, "train.parquet")
    val_out   = os.path.join(output_dir, "val.parquet")

    save_parquet_rows(rows_train, train_out, batch_size=parquet_batch_size, desc="写 Parquet(train)", compression=parquet_compression)
    save_parquet_rows(rows_val,   val_out,   batch_size=parquet_batch_size, desc="写 Parquet(val)",   compression=parquet_compression)

    print("[done] SFT 构造完成（root-第一层 + 第一层-第二层，已输出 2 个 Parquet）")
    print("train : {}\nval   : {}".format(train_out, val_out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SFT 构造（root-第一层 + 第一层-第二层；record_id 级切分 + 输出 train/val 两个 Parquet，val≈15%）'
    )
    parser.add_argument('--input', required=True, help='输入 JSON 文件路径（原始树形数据，顶层为 list）')
    parser.add_argument('--output_dir', required=True, help='输出文件夹（将生成 train.parquet / val.parquet）')
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--parquet_batch_size', type=int, default=4000)
    parser.add_argument('--parquet_compression', type=str, default="zstd", help='Parquet 压缩算法（zstd/snappy/uncompressed 等）')
    parser.add_argument('--large_pool_threshold', type=int, default=1000, help='阈值：候选池人数 > 阈值 的根作者将被跳过')
    parser.add_argument('--val_depth2_head', type=int, default=30, help='在验证集中输出的有 gold 的样本ID数量上限')
    parser.add_argument('--val_k_per_chunk', type=int, default=None, help='验证集每个chunk的候选人数（默认：30）')
    parser.add_argument('--noise_k_min_depth0', type=float, default=None, help='depth=0 的噪声倍数最小值（默认：0.0）')
    parser.add_argument('--noise_k_max_depth0', type=float, default=None, help='depth=0 的噪声倍数最大值（默认：3.0）')
    parser.add_argument('--noise_k_min_depth1', type=float, default=None, help='depth=1 的噪声倍数最小值（默认：0.0）')
    parser.add_argument('--noise_k_max_depth1', type=float, default=None, help='depth=1 的噪声倍数最大值（默认：3.0）')
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_dir=args.output_dir,
        shuffle_seed=args.shuffle_seed,
        parquet_batch_size=args.parquet_batch_size,
        parquet_compression=args.parquet_compression,
        large_pool_threshold=args.large_pool_threshold,
        val_depth2_head=args.val_depth2_head,
        val_k_per_chunk=args.val_k_per_chunk,
        noise_k_min_depth0=args.noise_k_min_depth0,
        noise_k_max_depth0=args.noise_k_max_depth0,
        noise_k_min_depth1=args.noise_k_min_depth1,
        noise_k_max_depth1=args.noise_k_max_depth1,
    )
