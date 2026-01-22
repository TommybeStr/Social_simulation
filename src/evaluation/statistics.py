# -*- coding: utf-8 -*-
"""
stats_final_v23.py — Strict Intersection & Context-Injected Edition

核心逻辑 (Strict Correspondence):
1. 【严格匹配】：仅计算 Intersection (交集)。
   - 遍历模型预测列表 (Preds)。
   - 只有当 Pred_User 在 Gold_List 中存在时，才视为命中 (Hit)。
   - 仅对命中的边计算文本分数。幻觉(False Positive)和漏召回(False Negative)不参与文本打分。

2. 【高分复现 (Context Injection)】：
   - 使用包含 <src_text> 的模板。
   - Pred 和 Gold 填充完全相同的 src_text (原帖)，确保分数与参考代码一致。
   
3. 【配置对齐】：
   - BERTScore: lang='en', rescale_with_baseline=True
   - ROUGE-L: use_stemmer=True
"""

import json
import argparse
from collections import defaultdict
import numpy as np
import sys
import os

# ================== 1. 环境与警告 ==================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import transformers
    transformers.logging.set_verbosity_error() 
except ImportError:
    pass

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_func
    HAS_BERT_SCORE = True
except ImportError:
    HAS_BERT_SCORE = False

EPS = 1e-8

# ==========================================
# 2. 定制模板 (无时间戳，含上下文)
# ==========================================
CUSTOM_TEMPLATE = """
<label>{label}</label>
<src_text>{src_text}</src_text>
<dst_text>{dst_text}</dst_text>"""

# ==========================================
# 工具函数
# ==========================================

def calc_rouge_l_single_pair(scorer, pred_str, gold_str):
    if not pred_str or not gold_str: return 0.0
    score = scorer.score(gold_str, pred_str)
    return score['rougeL'].fmeasure

def apply_template(label, src_text, dst_text):
    """
    将 Label + Context + Reply 拼接到一起
    """
    try:
        return CUSTOM_TEMPLATE.format(
            label=str(label),
            src_text=str(src_text),  # 关键点：Pred和Gold这里完全一样
            dst_text=str(dst_text)
        )
    except Exception as e:
        return ""

def compute_recall_at_k(pred_list, gold_set, k=None):
    if not gold_set: return 0.0
    candidates = pred_list[:k] if k is not None else pred_list
    cut_preds_set = set(candidates)
    tp = len(cut_preds_set & gold_set)
    return tp / len(gold_set)

def calculate_matched_type_accuracy(type_pairs):
    matched_pairs = [(p, g) for p, g in type_pairs if p > 0 and g > 0]
    total = len(matched_pairs)
    if total == 0:
        return 0.0, 0
    correct = sum(1 for p, g in matched_pairs if p == g)
    return correct / total, total

# ==========================================
# MMD Utils
# ==========================================
def gaussian_kernel(x1, x2, sigma=1.0):
    dist = np.sum(np.abs(x1 - x2)) / 2
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))

def compute_mmd(dist1, dist2, sigma=1.0):
    d1 = dist1 / (np.sum(dist1, axis=1, keepdims=True) + EPS)
    d2 = dist2 / (np.sum(dist2, axis=1, keepdims=True) + EPS)
    def kernel_mean(X, Y):
        s = 0.0
        for x in X:
            for y in Y:
                s += gaussian_kernel(x, y, sigma)
        return s / (len(X) * len(Y))
    term1 = kernel_mean(d1, d1)
    term2 = kernel_mean(d2, d2)
    term3 = kernel_mean(d1, d2)
    return np.sqrt(max(0, term1 + term2 - 2 * term3))

def make_histogram(values, bins):
    hist, _ = np.histogram(values, bins=bins)
    return hist / (np.sum(hist) + EPS)

def compute_graph_mmd_metrics(pred_edges, gt_edges):
    if not HAS_NX: return {}
    G_pred = nx.DiGraph(); G_pred.add_edges_from(pred_edges)
    G_gt = nx.DiGraph(); G_gt.add_edges_from(gt_edges)
    metrics = {}
    
    deg_pred = [d for n, d in G_pred.degree()]
    deg_gt = [d for n, d in G_gt.degree()]
    if deg_pred and deg_gt:
        max_deg = max(max(deg_pred), max(deg_gt))
        bins = np.arange(0, max_deg + 2)
        h_pred = make_histogram(deg_pred, bins); h_gt = make_histogram(deg_gt, bins)
        metrics['degree_mmd'] = compute_mmd(h_pred[None, :], h_gt[None, :], sigma=1.0)
    else: metrics['degree_mmd'] = np.nan

    clus_pred = list(nx.clustering(G_pred.to_undirected()).values())
    clus_gt = list(nx.clustering(G_gt.to_undirected()).values())
    if clus_pred and clus_gt:
        bins = np.linspace(0, 1, 101)
        h_pred = make_histogram(clus_pred, bins)
        h_gt = make_histogram(clus_gt, bins)
        metrics['cluster_mmd'] = compute_mmd(h_pred[None, :], h_gt[None, :], sigma=0.1)
    else: metrics['cluster_mmd'] = np.nan

    if 0 < len(G_pred) < 5000 and 0 < len(G_gt) < 5000:
        try:
            L_pred = nx.normalized_laplacian_matrix(G_pred.to_undirected())
            L_gt = nx.normalized_laplacian_matrix(G_gt.to_undirected())
            evals_pred = np.linalg.eigvalsh(L_pred.toarray())
            evals_gt = np.linalg.eigvalsh(L_gt.toarray())
            bins = np.linspace(0, 2, 201)
            h_pred = make_histogram(evals_pred, bins)
            h_gt = make_histogram(evals_gt, bins)
            metrics['spectra_mmd'] = compute_mmd(h_pred[None, :], h_gt[None, :], sigma=1.0)
        except: metrics['spectra_mmd'] = np.nan
    else: metrics['spectra_mmd'] = np.nan
    return metrics

# ==========================================
# Data Loading (Gold Content)
# ==========================================
def extract_username_from_user_content(user_content: str) -> str:
    if not isinstance(user_content, str): return ""
    import re
    _USERNAME_LINE_RE = re.compile(r"^\s*username:\s*(?P<name>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    m = _USERNAME_LINE_RE.search(user_content)
    if m: return m.group("name").strip()
    return ""

def load_gold_content_from_grpo_data(grpo_data_path: str):
    gold_content_map = {}
    if grpo_data_path and grpo_data_path.endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(grpo_data_path)
            print(f"[Gold Content] Loading from SFT Parquet: {grpo_data_path}")
            for idx, row in df.iterrows():
                messages = row.get("messages", [])
                if not isinstance(messages, list): continue
                sft_info = row.get("sft_chunk_info", {})
                if isinstance(sft_info, str):
                    try: sft_info = json.loads(sft_info)
                    except: sft_info = {}
                record_id = str(sft_info.get("record_id") or f"row{idx}")
                assistant_msg = next((m for m in messages if m.get("role")=="assistant"), None)
                if not assistant_msg: continue
                try:
                    asst_arr = json.loads(assistant_msg.get("content", ""))
                    if isinstance(asst_arr, list):
                        for item in asst_arr:
                            if isinstance(item, dict):
                                uname = (item.get("user_name") or "").strip()
                                gcontent = str(item.get("content", "")).strip()
                                if uname and gcontent:
                                    gold_content_map[(record_id, uname)] = gcontent
                except: pass
        except Exception as e:
            print(f"[WARN] Failed to load gold from Parquet: {e}")
            
    elif grpo_data_path and grpo_data_path.endswith('.jsonl'):
        print(f"[Gold Content] Loading from GRPO JSONL: {grpo_data_path}")
        pass
        
    print(f"[Gold Content] Loaded {len(gold_content_map)} gold content entries")
    return gold_content_map

# ==========================================
# 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input detail jsonl")
    parser.add_argument("--grpo_data", default=None, help="GRPO data source")
    args = parser.parse_args()

    print(f"[Stats] Loading {args.input} ...")
    
    gold_content_map = load_gold_content_from_grpo_data(args.grpo_data) if args.grpo_data else {}
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if HAS_ROUGE else None

    # 数据结构初始化
    nodes_map = defaultdict(lambda: {
        "preds": [], "golds": set(), "type_pairs": [], 
        "pred_edges": [], "gold_edges": [], "is_root": False,
        "row_context": {} 
    })
    
    panel_stats = defaultdict(lambda: {"cands":0, "g0":0, "g_pos":0, "p0":0, "p1":0, "p2":0})
    graph_all_pred_edges = set()
    graph_all_gt_edges = set()
    
    line_count = 0
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try: row = json.loads(line)
            except: continue
            line_count += 1
            
            rec_id = str(row.get("group_id", "unknown"))
            depth = int(row.get("depth", 0))
            input_user = row.get("input_user", {}).get("user_name")
            if not input_user: continue 
            
            key = (rec_id, input_user, depth)
            node = nodes_map[key]
            if depth == 0: node["is_root"] = True
            
            # --- 1. 获取上下文 (src_text) ---
            # 这部分是 "送分项"，必须提取
            src_text = ""
            if "input_user" in row and isinstance(row["input_user"], dict):
                src_text = row["input_user"].get("text") or row["input_user"].get("content") or ""
            if not src_text: 
                src_text = row.get("input_text") or row.get("src_text") or ""
            
            node["row_context"]["src_text"] = str(src_text)

            output_list = row.get("output_text", [])
            gold_list = row.get("gold", [])
            
            # --- 2. 处理 Gold 数据 ---
            gold_text_from_output = {} 
            for item in output_list: 
                if item.get("user_name") and "gold_text" in item:
                    gold_text_from_output[item["user_name"]] = str(item["gold_text"]).strip()

            gold_user_set = set()
            for g in gold_list:
                gname = g.get("user_name") if isinstance(g, dict) else str(g)
                node["golds"].add(gname)
                gold_user_set.add(gname)
                
                if isinstance(g, dict):
                    gcontent = str(g.get("content", "")).strip()
                    gtype = g.get("type") or g.get("gold_type", 1)
                else:
                    gcontent = gold_text_from_output.get(gname) or \
                               gold_content_map.get((rec_id, gname)) or ""
                    gtype = 1
                
                graph_all_gt_edges.add((input_user, gname))
                node["gold_edges"].append({"user_name": gname, "type": gtype, "content": gcontent})

            # --- 3. 处理 Pred 数据 (保留所有预测，用于图结构计算) ---
            n_cands = len(output_list)
            n_g_pos = len(gold_user_set & set([x['user_name'] for x in output_list if x.get('user_name')]))
            n_g0 = n_cands - n_g_pos
            
            panel_stats[depth]["cands"] += n_cands
            panel_stats[depth]["g0"] += n_g0
            panel_stats[depth]["g_pos"] += n_g_pos
            
            p0 = p1 = p2 = 0
            for item in output_list:
                t = int(item.get("type", 0))
                if t == 0 and "pred_type" in item: t = int(item["pred_type"])
                uname = item.get("user_name")
                
                if t == 0: p0 += 1
                elif t == 1: p1 += 1
                elif t == 2: p2 += 1
                
                chunk_gold_type = 0
                if "gold_type" in item: chunk_gold_type = int(item["gold_type"])
                elif uname and uname in gold_user_set: chunk_gold_type = 1
                
                node["type_pairs"].append((t, chunk_gold_type))
                
                if uname:
                    pred_edge = {"user_name": uname, "type": t, "content": str(item.get("content", "")).strip()}
                    node["pred_edges"].append(pred_edge)
                
                if uname and t in (1, 2):
                    node["preds"].append(uname)
                    graph_all_pred_edges.add((input_user, uname))
            
            panel_stats[depth]["p0"] += p0
            panel_stats[depth]["p1"] += p1
            panel_stats[depth]["p2"] += p2

    print(f"[Stats] Processed {line_count} chunks. Nodes: {len(nodes_map)}")

    # -------------------------------------------------
    # Valid Tree Filter (用于 F1/Recall 计算, 保证图结构指标的准确性)
    # -------------------------------------------------
    print("[Stats] Building Ground Truth Tree...")
    gt_graph = defaultdict(lambda: defaultdict(set))
    roots = defaultdict(set)
    for (rid, uname, d), data in nodes_map.items():
        if data["is_root"]: roots[rid].add(uname)
        for child in data["golds"]: gt_graph[rid][uname].add(child)
    
    valid_nodes_set = set()
    for rid, root_users in roots.items():
        queue = list(root_users)
        for r in root_users: valid_nodes_set.add((rid, r))
        visited = set(root_users)
        while queue:
            curr = queue.pop(0)
            for c in gt_graph[rid].get(curr, set()):
                if c not in visited:
                    visited.add(c)
                    valid_nodes_set.add((rid, c))
                    queue.append(c)

    # -------------------------------------------------
    # Metrics Calculation
    # -------------------------------------------------
    node_metrics_valid = defaultdict(list)
    acc_metrics_filtered = defaultdict(list)
    recall_at_100_list = []
    
    template_filled_pairs = []

    for (rid, uname, d), data in nodes_map.items():
        # 1. 基础指标过滤：必须是有效节点
        if (rid, uname) not in valid_nodes_set: 
            continue

        # 2. ACC 计算 (Pred>0 & Gold>0)
        acc, count = calculate_matched_type_accuracy(data["type_pairs"])
        if count > 0: 
            acc_metrics_filtered[d].append(acc)
        
        # 3. F1 计算 (仅计算 Gold>0 的节点)
        p_list = data["preds"]; g_set = data["golds"]
        p_set = set(p_list)
        
        if len(g_set) > 0:
            tp = len(p_set & g_set)
            fp = len(p_set - g_set); fn = len(g_set - p_set)
            p = tp / (tp+fp+EPS); r = tp / (tp+fn+EPS); f1 = 2*p*r/(p+r+EPS)
            node_metrics_valid[d].append({"p":p, "r":r, "f1":f1})
            recall_at_100_list.append(compute_recall_at_k(p_list, g_set, k=100))
            
        # 4. 【文本指标核心逻辑】严格对应 (Strict Intersection)
        gold_edges_map = {e["user_name"]: e for e in data["gold_edges"]}
        src_text = data["row_context"].get("src_text", "")
        
        for pred_item in data["pred_edges"]:
            name = pred_item["user_name"]
            
            # --- 严格判断：只有 Pred 的人在 Gold 中也存在时，才算分 ---
            if name in gold_edges_map:
                gold_item = gold_edges_map[name]
                
                # 应用上下文注入模板
                p_str = apply_template(pred_item["type"], src_text, pred_item["content"])
                g_str = apply_template(gold_item["type"], src_text, gold_item["content"])
                
                template_filled_pairs.append((p_str, g_str))

    # --- 批量计算 ROUGE-L (Template) ---
    rouge_l_scores_all = []
    if HAS_ROUGE and template_filled_pairs:
        rouge_scores = [calc_rouge_l_single_pair(rouge_scorer_obj, p, g) for p, g in template_filled_pairs]
        rouge_l_scores_all = rouge_scores

    # --- 批量计算 BERTScore (Template) ---
    bert_score_f1_all = []
    if HAS_BERT_SCORE and template_filled_pairs:
        print(f"[BERT_SCORE] Batch processing {len(template_filled_pairs)} templated pairs...")
        try:
            preds_all = [p[0] for p in template_filled_pairs]
            golds_all = [p[1] for p in template_filled_pairs]
            # 保持 Reference Config: lang='en', rescale=True
            P, R, F1 = bert_score_func(preds_all, golds_all, lang='en', verbose=True, batch_size=32, rescale_with_baseline=True)
            bert_score_f1_all = F1.tolist()
        except Exception as e:
            print(f"[ERROR] BERT_SCORE failed: {e}")

    # =================================================
    # Output Sections
    # =================================================
    print("\n" + "="*80)
    print(" (I) 面板统计 (Panel Statistics - Raw Data)")
    print("="*80)
    print(f"{'Depth':<6} {'Cands':<8} {'Gold=0':<8} {'Gold>0':<8} {'Pred=0':<8} {'Pred=1':<8} {'Pred=2':<8}")
    for d in sorted(panel_stats.keys()):
        s = panel_stats[d]
        print(f"{d:<6} {s['cands']:<8} {s['g0']:<8} {s['g_pos']:<8} {s['p0']:<8} {s['p1']:<8} {s['p2']:<8}")

    print("\n" + "="*80)
    print(" (V-B) 节点级指标 (Valid Nodes Only)")
    print(" * ACC: Type Accuracy on Matched Edges")
    print(" * F1:  Averaged over nodes with Gold > 0")
    print("="*80)
    print(f"{'Depth':<6} {'ValidN':<8} {'F1':<8} | {'ACC':<8}")
    all_f1s, all_accs = [], []
    for d in sorted(set(node_metrics_valid.keys()) | set(acc_metrics_filtered.keys())):
        ms = node_metrics_valid.get(d, []); accs = acc_metrics_filtered.get(d, [])
        avg_f1 = np.mean([x['f1'] for x in ms]) if ms else 0.0
        avg_acc = np.mean(accs) if accs else 0.0
        all_f1s.extend(ms); all_accs.extend(accs)
        print(f"{d:<6} {len(ms):<8} {avg_f1:.4f}   | {avg_acc:.4f}")
    
    if all_f1s or all_accs:
        print("-" * 40)
        of1 = np.mean([x['f1'] for x in all_f1s]) if all_f1s else 0.0
        oacc = np.mean(all_accs) if all_accs else 0.0
        print(f"{'ALL':<6} {len(all_f1s):<8} {of1:.4f}   | {oacc:.4f}")

    print("\n" + "="*80)
    print(" (VII) 图级与文本生成指标")
    print(" * ROUGE/BERT: Strict Intersection (TP) + Context-Injected Template")
    print(" * Recall: Top-100 Truncation")
    print("="*80)
    print(f"Recall@100:         {np.mean(recall_at_100_list) if recall_at_100_list else 0.0:.4f}")
    
    if HAS_ROUGE:
        print(f"ROUGE-L (Tmpl):     {np.mean(rouge_l_scores_all) if rouge_l_scores_all else 0.0:.4f} (n={len(rouge_l_scores_all)})")
    if HAS_BERT_SCORE:
        print(f"BERT_SCORE F1 (Tmpl):{np.mean(bert_score_f1_all) if bert_score_f1_all else 0.0:.4f}")
    
    if len(graph_all_gt_edges) > 0:
        eo = len(graph_all_pred_edges & graph_all_gt_edges) / len(graph_all_gt_edges)
        print(f"Edge Overlap:       {eo:.4f}")
    
    if HAS_NX:
        mmd = compute_graph_mmd_metrics(graph_all_pred_edges, graph_all_gt_edges)
        print(f"Degree MMD:         {mmd.get('degree_mmd', 'N/A')}")
        print(f"Cluster MMD:        {mmd.get('cluster_mmd', 'N/A')}") 
        print(f"Spectra MMD:        {mmd.get('spectra_mmd', 'N/A')}")

if __name__ == "__main__":
    main()