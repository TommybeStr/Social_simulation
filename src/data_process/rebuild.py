#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import random
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Set

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ======================
# 时间解析与辅助函数
# ======================
DATE_FMT_CANDIDATES = [
    "%a %b %d %H:%M:%S %z %Y",   # Wed Dec 20 10:35:09 +0800 2023
    "%a %b %e %H:%M:%S %z %Y",   # 部分平台支持%e
]

def parse_created_at(s):
    if not isinstance(s, str):
        return None
    for fmt in DATE_FMT_CANDIDATES:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        s2 = " ".join(s.split())
        return datetime.strptime(s2, DATE_FMT_CANDIDATES[0])
    except Exception:
        return None

def load_user_profile_map(interest_file_path, mapping_file_path=None):
    interest_data = {}
    try:
        with open(interest_file_path, 'r', encoding='utf-8') as f1:
            first_line = f1.readline().strip()
            f1.seek(0)
            if not first_line:
                return {}
            
            is_jsonl = False
            try:
                first_obj = json.loads(first_line)
                if isinstance(first_obj, dict):
                    second_line = f1.readline()
                    f1.seek(0)
                    if second_line and second_line.strip():
                        is_jsonl = True
            except json.JSONDecodeError:
                pass
            
            if is_jsonl:
                for line in f1:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            user_id_key = str(obj.get('user_id', ''))
                            if user_id_key: interest_data[user_id_key] = obj
                    except json.JSONDecodeError: continue
            else:
                interest_data = json.load(f1)
    except Exception as e:
        print(f"[警告] 无法读取 profile 文件: {e}")
        return {}
    
    mapping_data = {}
    if mapping_file_path:
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f2:
                mapping_data = json.load(f2) or {}
        except Exception:
            mapping_data = {}
    
    profile_map = {}
    if not mapping_data:
        if isinstance(interest_data, dict):
            for userid_str, user_entry in interest_data.items():
                if isinstance(user_entry, dict):
                    interests = user_entry.get('user_interests', user_entry.get('interests', []))
                    if not isinstance(interests, list): interests = []
                    profile_map[str(userid_str)] = {"interests": interests}
    else:
        for real_id, anon_id in mapping_data.items():
            anon_entry = interest_data.get(str(anon_id), {})
            if isinstance(anon_entry, dict):
                interests = anon_entry.get('user_interests', anon_entry.get('interests', []))
                if not isinstance(interests, list): interests = []
                profile_map[str(real_id)] = {"interests": interests}
    
    return profile_map

def is_image_comment(text):
    return isinstance(text, str) and text.strip().startswith("图片评论")

def build_comment_hierarchy(comments, profile_map, interaction_counts=None, root_blogger_id=None):
    if interaction_counts is None: interaction_counts = {}
    if root_blogger_id is None: root_blogger_id = ""
    
    nodes = {}
    children = defaultdict(list)

    for c in comments:
        text = c.get('text_raw', '')
        if is_image_comment(text): continue
        cid = c['id']
        uid = str(c['user']['id'])
        interactor_id = str(c['user']['id'])
        interaction_counts[(root_blogger_id, interactor_id)] = interaction_counts.get((root_blogger_id, interactor_id), 0)
        
        nodes[cid] = {
            "id": cid, "user_id": c['user']['id'], "user": c['user']['screen_name'],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": c['text_raw'], "type": "评论", "depth": None, "replies": [],
            "interaction_count": interaction_counts.get((root_blogger_id, interactor_id), 0)
        }
        for sub in c.get('comments', []):
            sub_text = sub.get('text_raw', '')
            if is_image_comment(sub_text): continue
            sid = sub['id']
            suid = str(sub['user']['id'])
            sub_interactor_id = str(sub['user']['id'])
            # 注意：这里逻辑稍微简化，子评论也算作与root的互动（根据原逻辑）
            interaction_counts[(root_blogger_id, sub_interactor_id)] = interaction_counts.get((root_blogger_id, sub_interactor_id), 0)
            
            nodes[sid] = {
                "id": sid, "user_id": c['user']['id'], "user": sub['user']['screen_name'],
                "interests": profile_map.get(suid, {}).get("interests", []),
                "content": sub['text_raw'], "type": "评论", "depth": None, "replies": [],
                "interaction_count": interaction_counts.get((root_blogger_id, sub_interactor_id), 0)
            }
            parent_id = sub.get('reply_comment', {}).get('id', cid)
            children[parent_id].append(sid)

    def attach(parent_id, parent_node):
        for child_id in children.get(parent_id, []):
            child = nodes[child_id]
            child["depth"] = parent_node["depth"] + 1
            parent_node["replies"].append(child)
            attach(child_id, child)

    roots = []
    for c in comments:
        if is_image_comment(c.get('text_raw', '')): continue
        root = nodes[c['id']]
        root["depth"] = 1
        roots.append(root)
        attach(c['id'], root)
    return roots

def process_reposts(reposts, profile_map, interaction_counts=None, root_blogger_id=None):
    if interaction_counts is None: interaction_counts = {}
    if root_blogger_id is None: root_blogger_id = ""
    out = []
    for r in reposts:
        text = r.get("text_law", r.get("text_raw", ""))
        if is_image_comment(text): continue
        uid = str(r['user']['id'])
        interactor_id = str(r['user']['id'])
        interaction_counts[(root_blogger_id, interactor_id)] = interaction_counts.get((root_blogger_id, interactor_id), 0)
        
        out.append({
            "id": r["id"], "user_id": r['user']['id'], "user": r["user"]["screen_name"],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": text, "type": "转发微博", "depth": 1, "replies": [],
            "interaction_count": interaction_counts.get((root_blogger_id, interactor_id), 0)
        })
    return out

def process_single_post(post, profile_map, interaction_counts=None):
    uid = str(post['user']['id'])
    root_blogger_id = str(post['user']['id'])
    post_node = {
        "id": post["id"], "user_id": post['user']['id'], "user": post["user"]["screen_name"],
        "interests": profile_map.get(uid, {}).get("interests", []),
        "content": post.get("text_raw", ""), "type": "原始博文", "depth": 0, "replies": []
    }
    comments = post.get("comments", [])
    if comments:
        post_node["replies"].extend(build_comment_hierarchy(comments, profile_map, interaction_counts, root_blogger_id))
    reposts = post.get("reposts", [])
    if reposts:
        post_node["replies"].extend(process_reposts(reposts, profile_map, interaction_counts, root_blogger_id))
    return post_node

def _count_interactions_in_post(post, root_blogger_id, interaction_counts):
    def count_comments_recursive(comment_list):
        for c in comment_list:
            if is_image_comment(c.get('text_raw', '')): continue
            interactor_id = str(c['user']['id'])
            interaction_counts[(root_blogger_id, interactor_id)] += 1
            sub_comments = c.get('comments', [])
            if sub_comments: count_comments_recursive(sub_comments)
    comments = post.get("comments", [])
    if comments: count_comments_recursive(comments)
    reposts = post.get("reposts", [])
    for r in reposts:
        if is_image_comment(r.get("text_law", r.get("text_raw", ""))): continue
        interactor_id = str(r['user']['id'])
        interaction_counts[(root_blogger_id, interactor_id)] += 1

def _write_nodes_json(records, output_path, profile_map, interaction_counts, *, desc: str):
    all_users_in_records = set()
    users_with_interests = set()

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        for idx, record in enumerate(tqdm(records, desc=desc)):
            node = process_single_post(record, profile_map, interaction_counts)
            
            def collect_users_from_node(n):
                user_id = str(n.get('user_id', ''))
                if user_id:
                    all_users_in_records.add(user_id)
                    interests = n.get('interests', [])
                    if interests and isinstance(interests, list) and len(interests) > 0:
                        users_with_interests.add(user_id)
                for child in n.get('replies', []): collect_users_from_node(child)
            
            collect_users_from_node(node)
            if idx > 0: fout.write(",\n")
            json.dump(node, fout, ensure_ascii=False, indent=2)
        fout.write("\n]")
    
    n_total = len(all_users_in_records)
    n_has_int = len(users_with_interests)
    print(f"[统计 {desc.strip()}] 文件: {output_path}")
    print(f"       记录数: {len(records)} | 用户总数: {n_total} | 有兴趣画像用户: {n_has_int}")
    if n_total > 0 and n_has_int < n_total * 0.1:
        print(f"       [警告] 画像匹配率低: {n_has_int/n_total*100:.1f}%")

def main():
    # ==========================
    # 参数解析定义
    # ==========================
    parser = argparse.ArgumentParser(description="微博数据划分与节点构建工具")
    
    # 输入与配置
    parser.add_argument('--inputs', nargs='+', required=True, help='输入的一个或多个 JSONL 文件路径')
    parser.add_argument('--profile', required=True, help='用户画像文件路径')
    parser.add_argument('--mapping', default=None, help='用户ID映射文件路径 (可选)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 输出文件路径
    parser.add_argument('--val-out', required=True, help='验证集输出路径')
    parser.add_argument('--test-out', required=True, help='测试集输出路径')
    parser.add_argument('--train-out', required=True, help='训练集输出路径')

    # 时间窗口控制 (Start Day, End Day) - 相对于全局最新时间
    # 0 代表最新时间，7 代表7天前
    parser.add_argument('--val-days', nargs=2, type=float, default=[0, 0], help='验证集时间窗口 (开始天数 结束天数), 例如: 0 3')
    parser.add_argument('--test-days', nargs=2, type=float, default=[0, 0], help='测试集时间窗口 (开始天数 结束天数), 例如: 3 7')
    parser.add_argument('--train-days', nargs=2, type=float, default=[0, 30], help='训练集时间窗口 (开始天数 结束天数), 例如: 7 10000')

    # 采样率控制 (0.0 - 1.0)
    parser.add_argument('--val-sample', type=float, default=0.05, help='验证集采样比例 (0-1), 默认0.05')
    parser.add_argument('--test-sample', type=float, default=0.05, help='测试集采样比例 (0-1), 默认0.05')
    parser.add_argument('--train-sample', type=float, default=1.0, help='训练集采样比例 (0-1), 默认1.0 (全量)')

    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 加载用户画像
    print("[1/5] 加载用户画像...")
    profile_map = load_user_profile_map(args.profile, args.mapping)
    print(f"      已加载 {len(profile_map)} 个用户画像")

    # 2. 计算全局最新时间
    print("[2/5] 扫描数据计算全局最新时间...")
    latest_dt_global = None
    input_paths = args.inputs
    
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"      扫描时间 ({input_path})"):
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    created_dt = parse_created_at(record.get("created_at"))
                    if created_dt:
                        if (latest_dt_global is None) or (created_dt > latest_dt_global):
                            latest_dt_global = created_dt
                except:
                    continue
    
    if latest_dt_global is None:
        print("[错误] 无法解析任何时间。")
        sys.exit(2)
    print(f"      全局最新时间: {latest_dt_global}")

    # 3. 数据分桶与统计
    print("[3/5] 根据时间窗口分发数据...")
    
    # 准备容器
    # 格式: (list_of_records, interaction_counts_dict)
    val_data = {"records": [], "counts": defaultdict(int)}
    test_data = {"records": [], "counts": defaultdict(int)}
    train_data = {"records": [], "counts": defaultdict(int)}

    # 解析时间窗口范围
    def in_range(days, range_args):
        start, end = range_args[0], range_args[1]
        return start <= days < end

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"      分发数据 ({input_path})"):
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    created_dt = parse_created_at(record.get("created_at"))
                    if created_dt is None: continue

                    delta = latest_dt_global - created_dt
                    days_diff = delta.total_seconds() / 86400.0  # 转为浮点天数
                    
                    root_blogger_id = str(record.get("user", {}).get("id", ""))
                    
                    # 优先级分发逻辑：Val -> Test -> Train (避免重复，如果时间重叠，优先归入前面的集合)
                    # 如果需要随机分发重叠时间段，需要在参数中避免重叠，或者在这里修改逻辑
                    target = None
                    
                    if in_range(days_diff, args.val_days):
                        target = val_data
                    elif in_range(days_diff, args.test_days):
                        target = test_data
                    elif in_range(days_diff, args.train_days):
                        target = train_data
                    
                    if target:
                        # 采样判断 (提前采样以节省内存，或者也可以全存下来后面sample)
                        # 为了统计互动准确性，通常建议全量统计互动，最后输出时采样？
                        # 原代码逻辑是：Week数据全量统计互动，然后采样输出。
                        # 这里我们保持：只要进了时间窗口，就统计互动，但是否加入records列表取决于采样。
                        
                        _count_interactions_in_post(record, root_blogger_id, target["counts"])
                        
                        # 随机采样决定是否保留内容
                        if random.random() < [args.val_sample, args.test_sample, args.train_sample][
                            0 if target is val_data else (1 if target is test_data else 2)
                        ]:
                            target["records"].append(record)

                except Exception:
                    continue

    print(f"      [Val]   时间: {args.val_days}天 | 采样率: {args.val_sample} | 数量: {len(val_data['records'])}")
    print(f"      [Test]  时间: {args.test_days}天 | 采样率: {args.test_sample} | 数量: {len(test_data['records'])}")
    print(f"      [Train] 时间: {args.train_days}天 | 采样率: {args.train_sample} | 数量: {len(train_data['records'])}")

    # 4. 输出文件
    print("[4/5] 写入结果文件...")
    
    _write_nodes_json(val_data["records"], args.val_out, profile_map, val_data["counts"], desc="Val Set")
    _write_nodes_json(test_data["records"], args.test_out, profile_map, test_data["counts"], desc="Test Set")
    _write_nodes_json(train_data["records"], args.train_out, profile_map, train_data["counts"], desc="Train Set")
    
    print("[完成] 所有任务结束。")

if __name__ == "__main__":
    main()