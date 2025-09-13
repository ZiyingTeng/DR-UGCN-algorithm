from dealpickle import process_pickle_file
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from baseline import compute_dc, compute_bc, compute_hdc, compute_sc
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dealtxt import create_incidence_matrix_from_file
from connectivity import hypergraph_natural_connectivity
from rwiec import RWIECalculator
from rwhc import RWHCCalculator
from nonlinear import HypergraphContagion
import pickle
import os


def compute_motif_coefficient(incidence_matrix):
    num_nodes = incidence_matrix.shape[0]
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    adj = adj.tocsr()
    motif_coeff = np.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = adj[i].nonzero()[1]
        if len(neighbors) < 2:
            continue
        triangles = 0
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                if adj[neighbors[j], neighbors[k]] > 0:
                    triangles += 1
        possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        motif_coeff[i] = triangles / possible_triangles if possible_triangles > 0 else 0
    return motif_coeff

def compute_overlap_degree(incidence_matrix):
    num_nodes = incidence_matrix.shape[0]
    overlap_degree = np.zeros(num_nodes)
    for i in range(num_nodes):
        edges_i = incidence_matrix[i].nonzero()[1]
        if len(edges_i) == 0:
            continue
        total_overlap = 0
        count = 0
        for edge_idx, edge in enumerate(edges_i):
            nodes_in_edge = incidence_matrix[:, edge].nonzero()[0]
            for j in nodes_in_edge:
                if j != i:
                    edges_j = incidence_matrix[j].nonzero()[1]
                    common_edges = len(set(edges_i) & set(edges_j))
                    union_edges = len(set(edges_i) | set(edges_j))
                    if union_edges > 0:
                        overlap = common_edges / union_edges
                        total_overlap += overlap
                        count += 1
        overlap_degree[i] = total_overlap / count if count > 0 else 0
    return overlap_degree

# def compute_edge_size_feature(incidence_matrix):
#     edge_sizes = np.sum(incidence_matrix, axis=0).A1
#     node_edge_sizes = np.zeros(incidence_matrix.shape[0])
#     for i in range(incidence_matrix.shape[0]):
#         edges = incidence_matrix[i].nonzero()[1]
#         if len(edges) > 0:
#             node_edge_sizes[i] = np.mean(edge_sizes[edges])
#     return node_edge_sizes

# 动态计算rwhc和rwiec

def compute_features(incidence_matrix, nu_values=None):
    if nu_values is None:
        nu_values = [1.0]  # 默认值
    features_list = []
    for nu in nu_values:
        # 动态调整 RWHC 和 RWIEC 参数
        theta = 1.0 + 0.5 * (nu - 1.0) ** 1.5  # 非线性调整，nu=1.8 时 theta≈1.45
        gamma = 0.8 + 0.3 * (nu - 1.0) ** 1.5  # 非线性调整，nu=1.8 时 gamma≈1.08
        features = [
            compute_hdc(incidence_matrix),
            # compute_dc(incidence_matrix),
            # compute_bc(incidence_matrix),
            compute_sc(incidence_matrix),
            RWHCCalculator(incidence_matrix).calculate_rwhc(theta=theta),
            RWIECalculator(incidence_matrix).calculate_rwiec(gamma=gamma),
            compute_motif_coefficient(incidence_matrix),
            compute_overlap_degree(incidence_matrix),
            # compute_coreness(incidence_matrix),
            # compute_edge_size_feature(incidence_matrix)
        ]
        features_list.append(np.stack(features, axis=1))
    # 如果只有一个 nu，返回单个特征矩阵；否则返回 nu-specific 特征列表
    return features_list[0] if len(nu_values) == 1 else features_list

# def create_seed_set_simulation_labels(incidence_matrix, lambda_val, nu_values, top_k_base=0.05, num_runs=50, n_workers=8):
#     num_nodes = incidence_matrix.shape[0]
#     baseline_scores = {
#         'HDC': compute_hdc(incidence_matrix),
#         'DC': compute_dc(incidence_matrix),
#         'BC': compute_bc(incidence_matrix),
#         'SC': compute_sc(incidence_matrix),
#         'RWHC': RWHCCalculator(incidence_matrix).calculate_rwhc(),
#         'RWIEC': RWIECalculator(incidence_matrix).calculate_rwiec()
#     }
#     all_seed_sets = []
#     all_nu_values = []
#     all_infected_fracs = []
#     print(f"Generating seed set simulation labels for {len(nu_values)} ν values...")
#
#     for nu in nu_values:
#         print(f"Processing ν={nu:.1f}")
#         top_k_ratio = max(0.02, top_k_base - 0.03 * (nu - 1.0))
#         top_k = int(num_nodes * top_k_ratio)
#         for method_name, scores in baseline_scores.items():
#             top_nodes = np.argsort(scores)[-top_k:]
#             all_seed_sets.append(top_nodes)
#             all_nu_values.append(nu)
#             infected_frac = compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=1.0, num_runs=num_runs)
#             all_infected_fracs.append(infected_frac)
#             print(f"  {method_name}: infected_frac={infected_frac:.4f}")
#
#     return all_seed_sets, all_nu_values, all_infected_fracs



def load_hypergraph(file_path):
    """从txt文件加载"""
    incidence_matrix, node_id_map, _ = create_incidence_matrix_from_file(file_path)
    if incidence_matrix is None or node_id_map is None:
        raise ValueError(f"Failed to load incidence matrix from {file_path}")
    incidence_matrix = csr_matrix(incidence_matrix)
    node_degrees = np.sum(incidence_matrix, axis=1).A1
    edge_degrees = np.sum(incidence_matrix, axis=0).A1
    non_isolated = node_degrees > 0
    if not non_isolated.all():
        incidence_matrix = incidence_matrix[non_isolated, :]
        new_node_indices = np.cumsum(non_isolated) - 1
        node_id_map = {node: new_node_indices[old_idx] for node, old_idx in node_id_map.items() if non_isolated[old_idx]}
        edge_degrees = np.sum(incidence_matrix, axis=0).A1
        non_empty = edge_degrees > 0
        incidence_matrix = incidence_matrix[:, non_empty]
        node_degrees = np.sum(incidence_matrix, axis=1).A1
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(np.nonzero(adj)), dtype=torch.long)
    print(f"Dataset: {file_path}")
    print(f"Nodes: {incidence_matrix.shape[0]}, Hyperedges: {incidence_matrix.shape[1]}, Isolated nodes: {np.sum(node_degrees == 0)}")
    print(f"Avg node degree: {np.mean(node_degrees):.2f}, Max node degree: {np.max(node_degrees)}")
    print(f"Avg hyperedge size: {np.mean(edge_degrees):.2f}, Max hyperedge size: {np.max(edge_degrees)}")
    return incidence_matrix, edge_index, node_id_map

def load_hypergraph_pickle(file_path):
    """从pickle文件加载"""
    incidence_matrix, node_id_map = process_pickle_file(file_path)
    if incidence_matrix is None or node_id_map is None:
        raise ValueError(f"Failed to load incidence matrix from {file_path}")
    incidence_matrix = csr_matrix(incidence_matrix)
    node_degrees = np.sum(incidence_matrix, axis=1).A1
    edge_degrees = np.sum(incidence_matrix, axis=0).A1
    # 去掉孤立节点
    non_isolated = node_degrees > 0
    if not non_isolated.all():
        incidence_matrix = incidence_matrix[non_isolated, :]
        new_node_indices = np.cumsum(non_isolated) - 1
        node_id_map = {node: new_node_indices[old_idx] for node, old_idx in node_id_map.items() if non_isolated[old_idx]}
        edge_degrees = np.sum(incidence_matrix, axis=0).A1
        non_empty = edge_degrees > 0
        incidence_matrix = incidence_matrix[:, non_empty]
        node_degrees = np.sum(incidence_matrix, axis=1).A1  # 更新
    # 边索引
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(np.nonzero(adj)), dtype=torch.long)
    print(f"Dataset: {file_path}")
    print(f"Nodes: {incidence_matrix.shape[0]}, Hyperedges: {incidence_matrix.shape[1]}, Isolated nodes: {np.sum(node_degrees == 0)}")
    print(f"Avg node degree: {np.mean(node_degrees):.2f}, Max node degree: {np.max(node_degrees)}")
    print(f"Avg hyperedge size: {np.mean(edge_degrees):.2f}, Max hyperedge size: {np.max(edge_degrees)}")
    return incidence_matrix, edge_index, node_id_map


def compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=1.0, max_steps=10000, num_runs=10):
    num_nodes = incidence_matrix.shape[0]
    infected_fractions = []
    for run in range(num_runs):
        initial_infected = np.zeros(num_nodes)
        initial_infected[top_nodes] = 1
        contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu, mu)
        infected_fraction = contagion.simulate(max_steps=max_steps, tolerance=1e-5)
        infected_fractions.append(infected_fraction)
    avg_infected_fraction = np.mean(infected_fractions)
    std_infected_fraction = np.std(infected_fractions)
    print(f"nu={nu:.2f}, lambda_val={lambda_val:.4f}, avg_infected_fraction={avg_infected_fraction:.4f}, std={std_infected_fraction:.4f}")
    return avg_infected_fraction

def cache_baseline_scores(incidence_matrix):
    """计算基线算法的分数"""
    cache_file = "baseline_scores.pkl"
    required_keys = ['DC', 'BC', 'HDC', 'SC']
    num_nodes = incidence_matrix.shape[0]

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                scores = pickle.load(f)
            # 验证完整性和形状
            if isinstance(scores, dict) and all(key in scores for key in required_keys):
                valid = True
                for key, score in scores.items():
                    if not isinstance(score, np.ndarray) or len(score) != num_nodes:
                        print(f"Invalid score for {key}: expected shape ({num_nodes},), got {score.shape if hasattr(score, 'shape') else type(score)}")
                        valid = False
                if valid:
                    return scores
            print(f"Invalid cache content, deleting and recalculating...")
            os.remove(cache_file)
        except Exception as e:
            print(f"Error loading cache: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)

    print("Calculating baseline scores...")
    scores = {
        'DC': compute_dc(incidence_matrix),
        'BC': compute_bc(incidence_matrix),
        'HDC': compute_hdc(incidence_matrix),
        'SC': compute_sc(incidence_matrix)
    }

    # 验证计算结果
    for key, score in scores.items():
        if not isinstance(score, np.ndarray) or len(score) != num_nodes:
            raise ValueError(f"Invalid computed score for {key}: expected array of length {num_nodes}, got {score.shape if hasattr(score, 'shape') else type(score)}")
        print(f"{key} shape: {score.shape}, sample: {score[:5]}")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(scores, f)
        print("Saved baseline scores to cache")
    except Exception as e:
        print(f"Error saving cache: {e}. Proceeding without cache.")

    return scores

def get_max_hyperdegree(incidence_matrix):
    return np.max(np.sum(incidence_matrix, axis=1).A1)

def split_dataset(num_nodes, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio/(1-train_ratio), random_state=42)
    return train_idx, val_idx, test_idx


def generate_dynamic_candidate_sets(incidence_matrix, nu, top_k, lambda_val, num_candidates=50, num_runs=20):
    """
    根据动态加权公式生成多样化的候选种子集，并模拟得到其真实分数。

    Args:
        num_candidates: 要生成的候选集数量
        num_runs: 每个候选集的模拟次数

    Returns:
        candidate_sets: 候选集列表 [num_candidates, top_k]
        candidate_scores: 对应的真实分数列表 [num_candidates]
    """
    num_nodes = incidence_matrix.shape[0]

    # 1. 计算五个特征
    features = {}
    features['HDC'] = compute_hdc(incidence_matrix)
    features['RWHC'] = RWHCCalculator(incidence_matrix).calculate_rwhc()
    features['RWIEC'] = RWIECalculator(incidence_matrix).calculate_rwiec()
    features['Motif'] = compute_motif_coefficient(incidence_matrix)
    features['Overlap'] = compute_overlap_degree(incidence_matrix)

    # 2. 动态权重初步设定
    if nu < 1.3:
        weights = np.array([0.5, 0.2, 0.1, 0.15, 0.05])  # 偏向HDC和Motif
    elif nu < 1.7:
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # 相对均衡
    else:
        weights = np.array([0.2, 0.3, 0.25, 0.15, 0.1])  # 偏向RWHC和RWIEC

    # 3. 计算初始分数并选择Top M节点
    composite_scores = np.zeros(num_nodes)
    feature_keys = ['HDC', 'RWHC', 'RWIEC', 'Motif', 'Overlap']
    for i, key in enumerate(feature_keys):
        # 归一化每个特征
        normalized_feat = (features[key] - np.min(features[key])) / (
                    np.max(features[key]) - np.min(features[key]) + 1e-10)
        composite_scores += weights[i] * normalized_feat

    # 样本池的范围
    M = 3 * top_k
    top_m_indices = np.argsort(composite_scores)[-M:]
    top_k_indices = top_m_indices[-top_k:]  # 纯Top-K集

    # 4. 生成多样化的候选集
    candidate_sets = []

    # 确保包含纯Top-K集
    candidate_sets.append(top_k_indices.copy())

    # 生成其他候选集
    for i in range(num_candidates - 1):
        if np.random.rand() < 0.5:  # 50%的概率进行随机替换
            candidate_set = top_k_indices.copy()
            # 随机替换1-2个节点
            num_replace = np.random.randint(1, 0.3 * top_k) # 可以替换掉最多30%的节点
            for _ in range(num_replace):
                # 随机选择一个要替换的位置
                replace_idx = np.random.randint(0, top_k)
                # 从Top M中但不在当前候选集的节点中随机选择一个替代
                available_nodes = [n for n in top_m_indices if n not in candidate_set]
                if available_nodes:
                    new_node = np.random.choice(available_nodes)
                    candidate_set[replace_idx] = new_node
            candidate_sets.append(candidate_set)
        else:  # 30%的概率完全随机从Top M中选择
            candidate_set = np.random.choice(top_m_indices, size=top_k, replace=False)
            candidate_sets.append(candidate_set)

    # 5. 为每个候选集模拟得到真实分数
    candidate_scores = []
    print(f"为ν={nu:.1f}模拟{len(candidate_sets)}个候选集...")

    for i, candidate_set in enumerate(candidate_sets):
        infected_frac = compute_infected_fraction(
            incidence_matrix, candidate_set, lambda_val, nu,
            mu=1.0, num_runs=num_runs
        )
        candidate_scores.append(infected_frac)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{len(candidate_sets)}个候选集的模拟")

    return candidate_sets, candidate_scores


def prepare_enhanced_training_data(incidence_matrix, lambda_val, nu_values, top_k_ratio=0.04):
    """
    为多个nu值准备训练数据
    """
    num_nodes = incidence_matrix.shape[0]
    top_k = int(num_nodes * top_k_ratio)

    all_candidate_sets = []
    all_nu_values = []
    all_scores = []

    print("开始生成增强版训练数据...")
    for nu in nu_values:
        print(f"处理 ν={nu:.1f}")

        candidate_sets, candidate_scores = generate_dynamic_candidate_sets(
            incidence_matrix, nu, top_k, lambda_val,
            num_candidates=30, num_runs=15
        )

        all_candidate_sets.extend(candidate_sets)
        all_nu_values.extend([nu] * len(candidate_sets))
        all_scores.extend(candidate_scores)

        # 当前nu的最佳表现
        best_idx = np.argmax(candidate_scores)
        print(f"  ν={nu:.1f}最佳候选集分数: {candidate_scores[best_idx]:.4f}")

    return all_candidate_sets, all_nu_values, all_scores