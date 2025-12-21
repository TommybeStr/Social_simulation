#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Set

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ======================
# 时间解析与一周窗口筛选
# ======================
DATE_FMT_CANDIDATES = [
    "%a %b %d %H:%M:%S %z %Y",   # Wed Dec 20 10:35:09 +0800 2023
    "%a %b %e %H:%M:%S %z %Y",   # 部分平台支持%e（空格补齐的日期）
]

def parse_created_at(s):
    """
    尝试解析类似 'Wed Dec 20 10:35:09 +0800 2023' 的时间字符串。
    成功返回带tzinfo的datetime；失败返回None（不改变原有构造逻辑的稳定性）。
    """
    if not isinstance(s, str):
        return None
    for fmt in DATE_FMT_CANDIDATES:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # 宽松兜底：把连续空格压成单空格再试一次主格式
    try:
        s2 = " ".join(s.split())
        return datetime.strptime(s2, DATE_FMT_CANDIDATES[0])
    except Exception:
        return None


def load_user_profile_map(interest_file_path, mapping_file_path=None):
    """
    载入用户兴趣映射：real_id -> {"interests": [...]}
    如果 mapping_file_path 为 None、空或文件为空，则将 profile 中的 userid 视作真实 id，不用映射
    
    支持两种格式：
    1. JSONL 格式：每行一个 JSON 对象
    2. JSON 对象格式：整个文件是一个 JSON 对象（key 为 user_id）
    
    如果没有 user_interests 字段，则返回空列表
    """
    interest_data = {}
    
    # 尝试读取文件，判断是 JSONL 还是 JSON 对象格式
    try:
        with open(interest_file_path, 'r', encoding='utf-8') as f1:
            # 读取第一行来判断格式
            first_line = f1.readline().strip()
            f1.seek(0)
            
            if not first_line:
                print(f"[警告] profile 文件 {interest_file_path} 为空")
                return {}
            
            # 判断是否为 JSONL 格式（每行一个 JSON 对象）
            # JSONL 格式：第一行应该是一个完整的 JSON 对象，且不以 { 开头（或者整个文件只有一行且是对象）
            is_jsonl = False
            try:
                # 尝试解析第一行
                first_obj = json.loads(first_line)
                # 如果第一行是对象，且文件有多行，可能是 JSONL
                if isinstance(first_obj, dict):
                    # 检查是否还有第二行
                    second_line = f1.readline()
                    f1.seek(0)
                    if second_line and second_line.strip():
                        is_jsonl = True
            except json.JSONDecodeError:
                pass
            
            if is_jsonl:
                # JSONL 格式：每行一个 JSON 对象
                for line in f1:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # 使用 user_id 作为 key（如果存在）
                            user_id_key = str(obj.get('user_id', ''))
                            if user_id_key:
                                interest_data[user_id_key] = obj
                    except json.JSONDecodeError:
                        continue
            else:
                # JSON 对象格式：整个文件是一个 JSON 对象
                interest_data = json.load(f1)
    except Exception as e:
        print(f"[警告] 无法读取 profile 文件 {interest_file_path}: {e}")
        return {}
    
    # 尝试加载映射文件
    mapping_data = {}
    if mapping_file_path:
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f2:
                mapping_data = json.load(f2)
                # 如果映射数据为空字典或 None，视为空
                if not mapping_data:
                    mapping_data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            # 文件不存在或解析失败，视为空
            mapping_data = {}
    
    profile_map = {}
    
    # 如果映射文件为空，直接将 profile 中的 userid 视作真实 id
    if not mapping_data:
        # 遍历 interest_data，使用其 key 作为真实 id
        if isinstance(interest_data, dict):
            for userid_str, user_entry in interest_data.items():
                if isinstance(user_entry, dict):
                    # 支持两种字段名：user_interests 或 interests
                    interests = user_entry.get('user_interests', user_entry.get('interests', []))
                    # 如果没有 interests 字段，返回空列表
                    if not isinstance(interests, list):
                        interests = []
                    profile_map[str(userid_str)] = {
                        "interests": interests
                    }
    else:
        # 使用映射逻辑
        for real_id, anon_id in mapping_data.items():
            anon_entry = interest_data.get(str(anon_id), {})
            if isinstance(anon_entry, dict):
                # 支持两种字段名：user_interests 或 interests
                interests = anon_entry.get('user_interests', anon_entry.get('interests', []))
                # 如果没有 interests 字段，返回空列表
                if not isinstance(interests, list):
                    interests = []
                profile_map[str(real_id)] = {
                    "interests": interests
                }
    
    return profile_map

def is_image_comment(text):
    """
    判断是否为图片评论
    """
    return isinstance(text, str) and text.strip().startswith("图片评论")


def build_comment_hierarchy(comments, profile_map, interaction_counts=None, root_blogger_id=None):
    """
    扁平化创建所有评论节点，并根据 reply_comment 关系构建多层嵌套
    interaction_counts: dict, (root_blogger_id, interactor_user_id) -> count
    root_blogger_id: 当前博文的root博主ID
    """
    if interaction_counts is None:
        interaction_counts = {}
    if root_blogger_id is None:
        root_blogger_id = ""
    
    nodes = {}
    children = defaultdict(list)

    # 1) 创建节点
    for c in comments:
        text = c.get('text_raw', '')
        if is_image_comment(text):
            continue
        cid = c['id']
        uid = str(c['user']['id'])
        interactor_id = str(c['user']['id'])
        interaction_count = interaction_counts.get((root_blogger_id, interactor_id), 0)
        nodes[cid] = {
            "id": cid,
            "user_id": c['user']['id'],
            "user": c['user']['screen_name'],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": c['text_raw'],
            "type": "评论",
            "depth": None,
            "replies": [],
            "interaction_count": interaction_count
        }
        # 二级及以上评论也先扁平记录
        for sub in c.get('comments', []):
            sub_text = sub.get('text_raw', '')
            if is_image_comment(sub_text):
                continue
            sid = sub['id']
            suid = str(sub['user']['id'])
            sub_interactor_id = str(sub['user']['id'])
            sub_interaction_count = interaction_counts.get((root_blogger_id, sub_interactor_id), 0)
            nodes[sid] = {
                "id": sid,
                "user_id": c['user']['id'],  # 保持原逻辑不改动
                "user": sub['user']['screen_name'],
                "interests": profile_map.get(suid, {}).get("interests", []),
                "content": sub['text_raw'],
                "type": "评论",
                "depth": None,
                "replies": [],
                "interaction_count": sub_interaction_count
            }
            # reply_comment.id 指向真正的父评论，否则归到当前一级评论
            parent_id = sub.get('reply_comment', {}).get('id', cid)
            children[parent_id].append(sid)

    # 2) 递归挂载并设置 depth
    def attach(parent_id, parent_node):
        for child_id in children.get(parent_id, []):
            child = nodes[child_id]
            child["depth"] = parent_node["depth"] + 1
            parent_node["replies"].append(child)
            attach(child_id, child)

    # 3) 处理所有一级评论
    roots = []
    for c in comments:
        if is_image_comment(c.get('text_raw', '')):
            continue
        root = nodes[c['id']]
        root["depth"] = 1
        roots.append(root)
        attach(c['id'], root)

    return roots

def process_reposts(reposts, profile_map, interaction_counts=None, root_blogger_id=None):
    """
    将转发扁平化为 depth=1 的列表
    interaction_counts: dict, (root_blogger_id, interactor_user_id) -> count
    root_blogger_id: 当前博文的root博主ID
    """
    if interaction_counts is None:
        interaction_counts = {}
    if root_blogger_id is None:
        root_blogger_id = ""
    
    out = []
    for r in reposts:
        text = r.get("text_law", r.get("text_raw", ""))
        if is_image_comment(text):
            continue
        uid = str(r['user']['id'])
        interactor_id = str(r['user']['id'])
        interaction_count = interaction_counts.get((root_blogger_id, interactor_id), 0)
        out.append({
            "id": r["id"],
            "user_id": r['user']['id'],
            "user": r["user"]["screen_name"],
            "interests": profile_map.get(uid, {}).get("interests", []),
            "content": r.get("text_law", r.get("text_raw", "")),
            "type": "转发微博",
            "depth": 1,
            "replies": [],
            "interaction_count": interaction_count
        })
    return out

def process_single_post(post, profile_map, interaction_counts=None):
    """
    构建单条博文及其下所有评论、转发的嵌套结构
    interaction_counts: dict, (root_blogger_id, interactor_user_id) -> count
    """
    uid = str(post['user']['id'])
    root_blogger_id = str(post['user']['id'])
    post_node = {
        "id": post["id"],
        "user_id": post['user']['id'],
        "user": post["user"]["screen_name"],
        "interests": profile_map.get(uid, {}).get("interests", []),
        "content": post.get("text_raw", ""),
        "type": "原始博文",
        "depth": 0,
        "replies": []
    }

    # 评论部分（key 名为 'comments'）
    comments = post.get("comments", [])
    if comments:
        post_node["replies"].extend(build_comment_hierarchy(comments, profile_map, interaction_counts, root_blogger_id))

    # 转发部分
    reposts = post.get("reposts", [])
    if reposts:
        post_node["replies"].extend(process_reposts(reposts, profile_map, interaction_counts, root_blogger_id))

    return post_node

def main(input_paths, output_path):
    print("[1/4] 加载用户画像...")
    profile_map = load_user_profile_map(
        '/home/zss/Social_Behavior_Simulation/toothbrush/electric_toothbrush_profile.graph.anon',
        '/home/zss/Social_Behavior_Simulation/data_preprocess/sft_data_raw/id_dict.json'
    )
    print(f"      已加载 {len(profile_map)} 个用户画像")

    # ===========
    # 第一遍：读入全部记录，收集 (user_id, created_at_dt)，并计算每个博主的最新时间
    # ===========
    print("[2/4] 读取数据并计算最新时间...")
    latest_by_user = {}  # user_id_str -> latest_dt
    total_lines = 0
    
    # 先统计总行数（用于进度条）
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as fin:
            for _ in fin:
                total_lines += 1
    
    # 读取并计算最新时间
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"      读取记录 ({input_path})"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    user = record.get("user", {}) or {}
                    uid_str = str(user.get("id"))
                    created_dt = parse_created_at(record.get("created_at"))

                    if created_dt is not None and uid_str != "None":
                        prev = latest_by_user.get(uid_str)
                        if (prev is None) or (created_dt > prev):
                            latest_by_user[uid_str] = created_dt
                except json.JSONDecodeError as e:
                    # JSON 解析错误，跳过这一行
                    continue
                except Exception as e:
                    # 其他错误（如缺少字段等），也跳过
                    continue

    print(f"      找到 {len(latest_by_user)} 个博主")

    # ===========
    # 第二遍：筛选记录并统计互动次数（一周时间窗口）
    # ===========
    print("[3/4] 筛选记录并统计互动次数（一周时间窗口）...")
    one_week = timedelta(days=7)
    filtered_records = []  # 保存筛选后的记录
    interaction_counts = defaultdict(int)  # (root_blogger_id, interactor_user_id) -> count
    
    def count_interactions_in_post(post, root_blogger_id):
        """递归统计单条博文下的所有互动"""
        def count_comments_recursive(comment_list):
            """递归统计所有层级的评论"""
            for c in comment_list:
                if is_image_comment(c.get('text_raw', '')):
                    continue
                interactor_id = str(c['user']['id'])
                interaction_counts[(root_blogger_id, interactor_id)] += 1
                
                # 递归处理嵌套评论
                sub_comments = c.get('comments', [])
                if sub_comments:
                    count_comments_recursive(sub_comments)
        
        # 统计评论（包括所有嵌套层级）
        comments = post.get("comments", [])
        if comments:
            count_comments_recursive(comments)
        
        # 统计转发
        reposts = post.get("reposts", [])
        for r in reposts:
            if is_image_comment(r.get("text_law", r.get("text_raw", ""))):
                continue
            interactor_id = str(r['user']['id'])
            interaction_counts[(root_blogger_id, interactor_id)] += 1

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"      筛选和统计 ({input_path})"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    user = record.get("user", {}) or {}
                    uid_str = str(user.get("id"))
                    created_dt = parse_created_at(record.get("created_at"))
                    
                    # 时间窗口筛选（一周）
                    latest_dt = latest_by_user.get(uid_str)
                    keep = True
                    if (latest_dt is not None) and (created_dt is not None):
                        keep = (created_dt >= (latest_dt - one_week))
                    
                    if not keep:
                        continue
                    
                    # 统计互动次数（只统计一周内的记录）
                    root_blogger_id = str(record.get("user", {}).get("id", ""))
                    count_interactions_in_post(record, root_blogger_id)
                    
                    filtered_records.append((record, uid_str))
                except json.JSONDecodeError as e:
                    # JSON 解析错误，跳过这一行
                    continue
                except Exception as e:
                    # 其他错误（如缺少字段等），也跳过
                    continue

    print(f"      筛选后保留 {len(filtered_records)} 条记录（一周时间窗口）")

    # ===========
    # 第四遍：构建节点并写入文件（流式写入，减少内存占用）
    # ===========
    print("[4/4] 构建节点并写入文件...")
    
    # 统计 interest 匹配情况
    all_users_in_records: Set[str] = set()
    users_with_interests: Set[str] = set()
    
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        for idx, (record, uid_str) in enumerate(tqdm(filtered_records, desc="      构建节点")):
            node = process_single_post(record, profile_map, interaction_counts)
            
            # 统计用户和 interest 匹配情况
            def collect_users_from_node(n):
                user_id = str(n.get('user_id', ''))
                if user_id:
                    all_users_in_records.add(user_id)
                    interests = n.get('interests', [])
                    if interests and isinstance(interests, list) and len(interests) > 0:
                        users_with_interests.add(user_id)
                for child in n.get('replies', []):
                    collect_users_from_node(child)
            
            collect_users_from_node(node)
            
            if idx > 0:
                fout.write(",\n")
            json.dump(node, fout, ensure_ascii=False, indent=2)
        fout.write("\n]")
    
    # 检查 interest 匹配情况并输出警告
    n_total_users = len(all_users_in_records)
    n_users_with_interests = len(users_with_interests)
    n_users_without_interests = n_total_users - n_users_with_interests
    
    print(f"[完成] 已生成 {len(filtered_records)} 条博文数据（一周时间窗口）")
    print(f"[统计] 总用户数: {n_total_users}, 有 interests 的用户数: {n_users_with_interests}, 无 interests 的用户数: {n_users_without_interests}")
    
    if n_total_users > 0 and n_users_with_interests == 0:
        print("\n[警告] ⚠️  所有用户都没有匹配到 interest！")
        print("可能的原因：")
        print("  1. profile_map 为空或未正确加载")
        print(f"     - profile_map 大小: {len(profile_map)}")
        if len(profile_map) == 0:
            print("     - 原因：profile 文件为空或格式不正确")
        else:
            print("  2. 记录中的 user_id 与 profile_map 中的 key 不匹配")
            # 检查几个样本 user_id
            sample_user_ids = list(all_users_in_records)[:5]
            print(f"     - 样本 user_id（记录中）: {sample_user_ids}")
            sample_profile_keys = list(profile_map.keys())[:5]
            print(f"     - 样本 key（profile_map 中）: {sample_profile_keys}")
            # 检查是否有匹配
            matched_count = sum(1 for uid in sample_user_ids if uid in profile_map)
            print(f"     - 样本匹配数: {matched_count}/{len(sample_user_ids)}")
        print("  3. profile 文件中没有 interests 字段或 interests 为空")
        if len(profile_map) > 0:
            sample_profile = list(profile_map.values())[0]
            sample_interests = sample_profile.get('interests', [])
            print(f"     - 样本 profile 的 interests: {sample_interests}")
            if not sample_interests or len(sample_interests) == 0:
                print("     - 原因：profile 中的 interests 字段为空或不存在")
        print("\n建议检查：")
        print("  - profile 文件路径是否正确")
        print("  - profile 文件格式是否符合要求（JSONL 或 JSON 对象）")
        print("  - profile 文件中是否包含 user_interests 或 interests 字段")
        print("  - 记录中的 user_id 是否与 profile 中的 user_id 匹配")
        print("  - 如果使用映射文件，检查映射关系是否正确")
    elif n_total_users > 0 and n_users_with_interests < n_total_users * 0.1:
        # 如果匹配率低于 10%，也给出警告
        match_rate = n_users_with_interests / n_total_users * 100
        print(f"\n[警告] ⚠️  interest 匹配率较低: {match_rate:.1f}% ({n_users_with_interests}/{n_total_users})")
        print("建议检查 profile 文件和 user_id 匹配情况")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} input1.jsonl [input2.jsonl ...] output.json")
        sys.exit(1)
    # 最后一个参数是输出文件，其余都是输入文件
    input_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    main(input_paths, output_path)
