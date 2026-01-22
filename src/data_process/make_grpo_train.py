#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO 训练数据构造脚本 v2.1_clean (修复噪声池污染问题 + 移除死代码)
逻辑行为：全量处理输入数据，不进行 SFT 拆分。
修复点：在采样噪声时，严格排除该 Root 下的所有 L1 真实回复者。
"""

import os
import json
import argparse
import random
import math
from collections import defaultdict
from typing import List, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

# ==================== 配置 ====================
SHUFFLE_SEED = 42
LARGE_POOL_THRESHOLD = 1000
NOISE_K_MIN = 2.0
NOISE_K_MAX = 5.0
MAX_GOLD_PER_SAMPLE = 10 
ROOT_PARENT_KEY = "__ROOT__"

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


TABLE_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("root_user_json", pa.large_string()),
    pa.field("reward_model", pa.large_string()),
    pa.field("system_prompt", pa.large_string()),
    pa.field("sft_chunk_info", pa.large_string()),
])

# ==================== 工具函数 ====================

def _strip_retweet_tail(text: Any) -> str:
    if not isinstance(text, str): return ""
    idx = text.find("//@")
    if idx == -1: return text.strip()
    return text[:idx].rstrip()

def iter_tree_nodes(root: Dict[str, Any]):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in (node.get('replies') or []):
            stack.append(child)

def _safe_depth(node: Dict[str, Any]) -> int:
    try: return int(node.get('depth', 0))
    except: return 0

def build_child_map(children: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = {}
    for c in (children or []):
        u = str(c.get('user') or "").strip()
        if u: m[u] = c
    return m

def extract_node_children_map(root_record: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    mapping = defaultdict(list)
    for node in iter_tree_nodes(root_record):
        for child in (node.get('replies') or []):
            mapping[id(node)].append(child)
    return mapping

# ==================== 全局 Map 构建 ====================

def build_global_maps(records: List[Dict[str, Any]]):
    interest_map = {}
    interact_map = {}
    root_user_to_records = defaultdict(list)
    root_user_to_pool = {}

    print("Scanning raw data for global maps...")
    for rec in tqdm(records, desc="Scanning"):
        root_user = str(rec.get('user') or "")
        if root_user:
            root_user_to_records[root_user].append(rec)
        
        local_pool = set()
        for node in iter_tree_nodes(rec):
            u = str(node.get('user') or "").strip()
            ints = node.get('interests', [])
            count = int(node.get('interaction_count', 0))
            if u and ints:
                if u not in interest_map: interest_map[u] = ints
                if root_user:
                    key = (root_user, u)
                    interact_map[key] = max(interact_map.get(key, 0), count)
                local_pool.add(u)
        
        if root_user:
            if root_user not in root_user_to_pool: root_user_to_pool[root_user] = set()
            root_user_to_pool[root_user].update(local_pool)

    final_pool = {k: list(v) for k, v in root_user_to_pool.items()}
    return interest_map, interact_map, root_user_to_records, final_pool

# ==================== Prompt 构造 ====================

def make_potential_json_obj(candidates: List[str], interest_map: Dict, root_user: str, interact_map: Dict) -> List[Dict]:
    blocks = []
    for uid in candidates:
        blocks.append({
            "user_name": uid,
            "interests": interest_map.get(uid, []) or [],
            "depth": 0,
            "interaction_count": interact_map.get((root_user, uid), 0)
        })
    return blocks

def make_root_user_json_str(node: Dict, potentials: List[Dict]) -> str:
    content = _strip_retweet_tail(node.get('content') or "")
    pot_str = json.dumps(potentials, ensure_ascii=False, separators=(',', ':'))
    root_ctx = {"root_username": "", "root_content": ""}
    
    full_text = (
        f"username: {str(node.get('user') or '')}\n"
        f"content:\n{content}\n"
        f"userinterest: {json.dumps(node.get('interests', []), ensure_ascii=False)}\n"
        f"root_context: {json.dumps(root_ctx, ensure_ascii=False, separators=(',', ':'))}\n"
        f"potentials: {pot_str}"
    )
    
    obj = {
        "user": str(node.get('user') or ''),
        "content": full_text,
        "depth": 0,
        "_step_depth": 0
    }
    return json.dumps(obj, ensure_ascii=False)

# ==================== 样本生成 ====================

def generate_rows(
    record: Dict[str, Any],
    record_id: str,
    node_children_map: Dict[int, List[Dict]],
    global_pool: List[str],
    interest_map: Dict,
    interact_map: Dict,
    seed: int,
    noise_min: float,
    noise_max: float
) -> List[Dict]:
    
    root_node = record
    if _safe_depth(root_node) != 0: return []
    root_user = str(root_node.get('user') or "")
    
    l1_children_nodes = node_children_map.get(id(root_node), [])
    l1_map = build_child_map(l1_children_nodes) 
    l1_names = list(l1_map.keys()) # 【关键】：这是该Root下所有真实的L1回复者
    
    if not l1_names: return []
    
    gt_layer0 = {ROOT_PARENT_KEY: l1_names}
    gt_layer1 = {}
    
    all_l2_users = set()
    
    for l1_name, l1_node in l1_map.items():
        l2_nodes = node_children_map.get(id(l1_node), [])
        l2_names = [str(n.get('user') or '').strip() for n in l2_nodes if str(n.get('user') or '').strip()]
        if l2_names:
            gt_layer1[l1_name] = l2_names
            for u2 in l2_names: all_l2_users.add(u2)
            
    cond_gt_by_turn = [gt_layer0, gt_layer1]
    
    rng = random.Random(seed ^ (hash(root_user) & 0xffffffff))
    rng.shuffle(l1_names)
    chunks = [l1_names[i:i+MAX_GOLD_PER_SAMPLE] for i in range(0, len(l1_names), MAX_GOLD_PER_SAMPLE)]
    
    samples = []
    
    for ch_idx, l1_chunk in enumerate(chunks):
        k = rng.uniform(noise_min, noise_max)
        total_slots = max(1, math.ceil(k * len(l1_chunk)))
        
        # Look-ahead Noise (L2 Users)
        # 必须排除掉 l1_names 中的人（因为如果某人既是L1又是L2，他在Root视角下是L1真值，不能算作L2噪声）
        priority_noise = [u for u in all_l2_users if u in interest_map and u not in l1_names and u != root_user]
        rng.shuffle(priority_noise)
        
        selected_noise = []
        take_n = min(len(priority_noise), total_slots)
        selected_noise.extend(priority_noise[:take_n])
        
        remaining = total_slots - len(selected_noise)
        if remaining > 0:
            # 【重要修复】：这里必须使用 set(l1_names)，彻底排除所有真实回复者
            # 旧逻辑：exclude = set(l1_chunk) -> 导致漏掉的其他L1被选进来当噪声 -> 变成真值 -> 比例失调
            exclude = set(l1_names) | set(selected_noise) | {root_user}
            
            pool_cands = [u for u in global_pool if u not in exclude]
            if len(pool_cands) >= remaining:
                selected_noise.extend(rng.sample(pool_cands, remaining))
            else:
                selected_noise.extend(pool_cands)
                
        final_cands = l1_chunk + selected_noise
        rng.shuffle(final_cands)
        
        pot_objs = make_potential_json_obj(final_cands, interest_map, root_user, interact_map)
        root_user_json_str = make_root_user_json_str(root_node, pot_objs)
        
        reward_model_obj = {
            "ground_truth": {"cond_gt_by_turn": cond_gt_by_turn},
            "root_potential": {"full": pot_objs}
        }
        
        samples.append({
            # 加上 record_id 便于追溯与泄漏检查；不会影响训练逻辑
            "id": f"{record_id}_{root_user}_{ch_idx}",
            "root_user_json": root_user_json_str,
            "reward_model": json.dumps(reward_model_obj, ensure_ascii=False),
            "system_prompt": SYSTEM_PROMPT,
            "sft_chunk_info": json.dumps(
                {"record_id": record_id, "root": root_user, "chunk": ch_idx, "n_l2": take_n},
                ensure_ascii=False
            )
        })
        
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rebuild_data.py 输出的树结构 JSON（顶层 list）")
    parser.add_argument("--output", required=True, help="输出 GRPO 训练 Parquet（单文件）")
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--large_pool_threshold", type=int, default=LARGE_POOL_THRESHOLD)
    args = parser.parse_args()
    
    print(f"Loading raw data: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    interest_map, interact_map, root_map, pool_map = build_global_maps(records)
    
    all_rows = []
    print("Generating GRPO samples (single input -> single output)...")
    for root_user, recs in tqdm(root_map.items()):
        global_pool = pool_map.get(root_user, [])
        if len(global_pool) > args.large_pool_threshold: continue
        
        for rec in recs:
            rid = str(rec.get('id') or "")
            node_children_map = extract_node_children_map(rec)
            rows = generate_rows(rec, rid, node_children_map, global_pool, interest_map, interact_map, args.seed, NOISE_K_MIN, NOISE_K_MAX)
            all_rows.extend(rows)
            
    print(f"Saving {len(all_rows)} rows to {args.output}...")
    df = pd.DataFrame(all_rows)
    
    if df.empty:
        print("Error: No data generated!")
        return

    arrays = [
        pa.array(df["id"], type=pa.string()),
        pa.array(df["root_user_json"], type=pa.large_string()),
        pa.array(df["reward_model"], type=pa.large_string()),
        pa.array(df["system_prompt"], type=pa.large_string()),
        pa.array(df["sft_chunk_info"], type=pa.large_string()),
    ]
    table = pa.Table.from_arrays(arrays, schema=TABLE_SCHEMA)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pq.write_table(table, args.output, compression='zstd')
    print("Done.")

if __name__ == "__main__":
    main()