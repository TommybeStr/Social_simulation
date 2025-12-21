# -*- coding: utf-8 -*-
"""
stats_final_v5.py — 综合评估统计脚本 (ACC 逻辑修正版)

修正内容：
1. 【ACC 重构】：仅计算模型预测出的边 (Pred > 0)。
   - Pred=0: 不参与计算 (跳过)。
   - Pred>0, Gold=0: 计为错误 (0分)。
   - Pred>0, Gold>0: 需类型一致才计为正确 (Pred==Gold)。
   - 包含幻觉节点：如果幻觉节点预测了边，ACC=0，拉低平均分。
2. 【F1/Recall】: 保持仅在 Valid Parent 上计算。
3. 【图指标】: 保持全局计算。
"""

import json
import argparse
from collections import defaultdict, Counter
import numpy as np
import sys

# 尝试导入 networkx
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not found. Graph MMD metrics will be skipped.")

# ==========================================
# 1. 基础工具函数
# ==========================================

EPS = 1e-8

def compute_recall_at_k(pred_list, gold_set, k=None):
    if not gold_set: return 0.0
    if k is not None:
        candidates = pred_list[:k]
    else:
        candidates = pred_list
    cut_preds_set = set(candidates)
    tp = len(cut_preds_set & gold_set)
    return tp / len(gold_set)

def calculate_positive_prediction_accuracy(type_pairs):
    """
    计算正向预测准确率 (ACC)
    type_pairs: list of (pred_type, gold_type)
    逻辑: 
      - 只看 pred_type > 0 的样本
      - 正确: pred_type == gold_type
      - 错误: pred_type != gold_type (包括 gold=0 或 类型不匹配)
    返回: (acc, count) - count用于判断是否跳过该节点
    """
    # 筛选出模型预测有互动的样本
    predicted_positives = [(p, g) for p, g in type_pairs if p > 0]
    
    total = len(predicted_positives)
    if total == 0:
        return 0.0, 0  # 分母为0，该节点不参与 ACC 统计
    
    # 计算完全匹配的数量
    correct = sum(1 for p, g in predicted_positives if p == g)
    
    return correct / total, total

# ==========================================
# 2. MMD / Graph Metrics 工具
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
    
    G_pred = nx.DiGraph()
    G_pred.add_edges_from(pred_edges)
    G_gt = nx.DiGraph()
    G_gt.add_edges_from(gt_edges)
    
    metrics = {}
    
    # 1. Degree
    deg_pred = [d for n, d in G_pred.degree()]
    deg_gt = [d for n, d in G_gt.degree()]
    if deg_pred and deg_gt:
        max_deg = max(max(deg_pred), max(deg_gt))
        bins = np.arange(0, max_deg + 2)
        h_pred = make_histogram(deg_pred, bins)
        h_gt = make_histogram(deg_gt, bins)
        metrics['degree_mmd'] = compute_mmd(h_pred[None, :], h_gt[None, :], sigma=1.0)
    else: metrics['degree_mmd'] = np.nan

    # 2. Clustering (Undirected)
    clus_pred = list(nx.clustering(G_pred.to_undirected()).values())
    clus_gt = list(nx.clustering(G_gt.to_undirected()).values())
    if clus_pred and clus_gt:
        bins = np.linspace(0, 1, 101)
        h_pred = make_histogram(clus_pred, bins)
        h_gt = make_histogram(clus_gt, bins)
        metrics['cluster_mmd'] = compute_mmd(h_pred[None, :], h_gt[None, :], sigma=0.1)
    else: metrics['cluster_mmd'] = np.nan

    # 3. Spectra
    if len(G_pred) < 5000 and len(G_gt) < 5000 and len(G_pred) > 0 and len(G_gt) > 0:
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
# 3. 主逻辑
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input detail jsonl")
    args = parser.parse_args()

    print(f"[Stats] Loading {args.input} ...")
    
    # -------------------------------------------------
    # Step 1: 聚合 (Aggregation)
    # -------------------------------------------------
    nodes_map = defaultdict(lambda: {
        "preds": [],        # list of predicted names (ordered)
        "golds": set(),     # set of gold names
        "type_pairs": [],   # list of (pred_type, gold_type) for ACC
        "is_root": False
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
            
            rec_id = row.get("group_id", "unknown")
            depth = int(row.get("depth", 0))
            input_user = row.get("input_user", {}).get("user_name")
            if not input_user: continue 
            
            key = (rec_id, input_user, depth)
            node = nodes_map[key]
            if depth == 0: node["is_root"] = True
            
            output_list = row.get("output_text", [])
            gold_list = row.get("gold", [])
            
            node["golds"].update(gold_list)
            
            # Panel & Preds Collection
            n_cands = len(output_list)
            n_g_pos = len(set(gold_list) & set([x['user_name'] for x in output_list if x.get('user_name')]))
            n_g0 = n_cands - n_g_pos
            
            p0 = p1 = p2 = 0
            for item in output_list:
                # 1. Pred Type
                t = int(item.get("type", 0))
                if t == 0 and "pred_type" in item: t = int(item["pred_type"])
                
                # 2. Gold Type (Strict for ACC)
                uname = item.get("user_name")
                chunk_gold_type = 0
                if "gold_type" in item:
                    try: chunk_gold_type = int(item["gold_type"])
                    except: chunk_gold_type = 0
                elif uname and uname in gold_list:
                    # Fallback: if in gold list, assume type 1 (or 2 is ambiguous, so beware)
                    # Ideally rollout.py provides gold_type. If not, this is a best guess.
                    chunk_gold_type = 1 
                
                node["type_pairs"].append((t, chunk_gold_type))
                
                if t == 0: p0 += 1
                elif t == 1: p1 += 1
                elif t == 2: p2 += 1
                
                if uname and t in (1, 2):
                    node["preds"].append(uname)
                    graph_all_pred_edges.add((input_user, uname))
            
            for g in gold_list:
                graph_all_gt_edges.add((input_user, g))

            panel_stats[depth]["cands"] += n_cands
            panel_stats[depth]["g0"] += n_g0
            panel_stats[depth]["g_pos"] += n_g_pos
            panel_stats[depth]["p0"] += p0
            panel_stats[depth]["p1"] += p1
            panel_stats[depth]["p2"] += p2

    print(f"[Stats] Processed {line_count} chunks. Aggregated into {len(nodes_map)} unique nodes.")

    # -------------------------------------------------
    # Step 2: 真值树构建与过滤 (Valid Tree Filtering)
    # -------------------------------------------------
    print("[Stats] Building Ground Truth Tree...")
    gt_graph = defaultdict(lambda: defaultdict(set)) 
    roots = defaultdict(set)
    
    for (rid, uname, d), data in nodes_map.items():
        if data["is_root"]:
            roots[rid].add(uname)
        for child in data["golds"]:
            gt_graph[rid][uname].add(child)
            
    valid_nodes_set = set()
    for rid, root_users in roots.items():
        queue = list(root_users)
        for r in root_users: valid_nodes_set.add((rid, r))
        visited = set(root_users)
        while queue:
            curr = queue.pop(0)
            children = gt_graph[rid].get(curr, set())
            for c in children:
                if c not in visited:
                    visited.add(c)
                    valid_nodes_set.add((rid, c))
                    queue.append(c)
                    
    print(f"[Stats] Valid nodes in GT Tree: {len(valid_nodes_set)}")

    # -------------------------------------------------
    # Step 3: 指标计算
    # -------------------------------------------------
    
    node_metrics_valid = defaultdict(list) # F1/P/R (Valid Only)
    acc_metrics_all = defaultdict(list)    # ACC (All nodes with predictions)
    
    recall_at_100_list = []
    recall_at_all_list = []
    
    skipped_hallucinations = 0
    
    for (rid, uname, d), data in nodes_map.items():
        
        # --- 1. ACC 计算 (全量，包含幻觉) ---
        # 逻辑：只要模型预测了正例 (t>0)，就考核其准确性
        acc, count = calculate_positive_prediction_accuracy(data["type_pairs"])
        if count > 0:
            # 只有当模型真正做出了正向预测时，ACC 才有意义并计入平均
            # 如果模型全预测 0，count=0，该节点ACC不计入 (跳过)
            acc_metrics_all[d].append(acc)
        
        # --- 2. 幻觉过滤 ---
        if (rid, uname) not in valid_nodes_set:
            skipped_hallucinations += 1
            continue
            
        # --- 3. F1/P/R 计算 (仅 Valid) ---
        pred_list = data["preds"]
        pred_set = set(pred_list)
        gold_set = data["golds"]
        
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        if len(gold_set) == 0:
            if len(pred_set) == 0: p, r, f1 = 1.0, 1.0, 1.0
            else: p, r, f1 = 0.0, 0.0, 0.0
        else:
            p = tp / (tp + fp + EPS)
            r = tp / (tp + fn + EPS)
            f1 = 2 * p * r / (p + r + EPS)
            
        node_metrics_valid[d].append({"p": p, "r": r, "f1": f1})
        
        # --- 4. Recall@K (仅 Valid & 有Gold) ---
        if len(gold_set) > 0:
            r100 = compute_recall_at_k(pred_list, gold_set, k=100)
            rall = compute_recall_at_k(pred_list, gold_set, k=None)
            recall_at_100_list.append(r100)
            recall_at_all_list.append(rall)

    # -------------------------------------------------
    # Step 4: 输出报告
    # -------------------------------------------------
    
    print("\n" + "="*80)
    print(" (I) 面板统计 (Panel Statistics - Raw Data)")
    print("="*80)
    print(f"{'Depth':<6} {'Cands':<8} {'Gold=0':<8} {'Gold>0':<8} {'Pred=0':<8} {'Pred=1':<8} {'Pred=2':<8}")
    for d in sorted(panel_stats.keys()):
        s = panel_stats[d]
        print(f"{d:<6} {s['cands']:<8} {s['g0']:<8} {s['g_pos']:<8} {s['p0']:<8} {s['p1']:<8} {s['p2']:<8}")
        
    print("\n" + "="*80)
    print(" (V-B) 节点级指标 (Node-level Metrics)")
    print(" * P/R/F1: Valid Parents Only (Filtered)")
    print(" * ACC:    All Parents where model predicted interaction (Strict Type Match)")
    print("="*80)
    print(f"{'Depth':<6} {'ValidN':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} | {'PredN':<8} {'ACC':<8}")
    
    all_f1s = []
    all_accs = []
    
    depths = sorted(set(node_metrics_valid.keys()) | set(acc_metrics_all.keys()))
    for d in depths:
        ms = node_metrics_valid.get(d, [])
        avg_p = np.mean([x['p'] for x in ms]) if ms else 0.0
        avg_r = np.mean([x['r'] for x in ms]) if ms else 0.0
        avg_f1 = np.mean([x['f1'] for x in ms]) if ms else 0.0
        all_f1s.extend(ms)
        
        accs = acc_metrics_all.get(d, [])
        avg_acc = np.mean(accs) if accs else 0.0
        all_accs.extend(accs)
        
        print(f"{d:<6} {len(ms):<8} {avg_p:.4f}   {avg_r:.4f}   {avg_f1:.4f}   | {len(accs):<8} {avg_acc:.4f}")
        
    if all_f1s or all_accs:
        op = np.mean([x['p'] for x in all_f1s]) if all_f1s else 0.0
        or_ = np.mean([x['r'] for x in all_f1s]) if all_f1s else 0.0
        of1 = np.mean([x['f1'] for x in all_f1s]) if all_f1s else 0.0
        oacc = np.mean(all_accs) if all_accs else 0.0
        print(f"{'ALL':<6} {len(all_f1s):<8} {op:.4f}   {or_:.4f}   {of1:.4f}   | {len(all_accs):<8} {oacc:.4f}")
        
    print("\n" + "="*80)
    print(" (VII) 图级与检索指标 (Graph & Retrieval Metrics)")
    print(" * Note: EO & MMD include ALL edges (hallucinations included).")
    print("="*80)
    
    # Recall
    mean_r100 = np.mean(recall_at_100_list) if recall_at_100_list else 0.0
    mean_rall = np.mean(recall_at_all_list) if recall_at_all_list else 0.0
    print(f"Recall@100 (Truncated): {mean_r100:.4f}")
    print(f"Recall@All (Full):      {mean_rall:.4f}")
    
    # Edge Overlap
    if len(graph_all_gt_edges) > 0:
        eo = len(graph_all_pred_edges & graph_all_gt_edges) / len(graph_all_gt_edges)
        print(f"Edge Overlap:           {eo:.4f} (|Gt|={len(graph_all_gt_edges)}, |Pred|={len(graph_all_pred_edges)})")
    else:
        print(f"Edge Overlap:           N/A")
        
    # MMD
    if HAS_NX:
        print("Calculating MMDs...")
        mmd_res = compute_graph_mmd_metrics(graph_all_pred_edges, graph_all_gt_edges)
        print(f"Degree MMD:             {mmd_res.get('degree_mmd', 'N/A')}")
        print(f"Cluster MMD:            {mmd_res.get('cluster_mmd', 'N/A')}")
        print(f"Spectra MMD:            {mmd_res.get('spectra_mmd', 'N/A')}")
    else:
        print("MMD Metrics:            Skipped")

if __name__ == "__main__":
    main()