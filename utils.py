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
    """计算motif聚类系数"""
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
    """计算超边重叠度"""
    num_nodes = incidence_matrix.shape[0]
    overlap_degree = np.zeros(num_nodes)

    for i in range(num_nodes):
        # 获取节点i参与的超边
        edges_i = incidence_matrix[i].nonzero()[1]
        if len(edges_i) == 0:
            continue

        total_overlap = 0
        count = 0
        for edge_idx, edge in enumerate(edges_i):
            # 获取该超边的其他节点
            nodes_in_edge = incidence_matrix[:, edge].nonzero()[0]
            # 计算该节点与其他节点的重叠度
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


def compute_coreness(incidence_matrix):
    """基于超图的k-core分解"""
    num_nodes = incidence_matrix.shape[0]
    node_degrees = np.sum(incidence_matrix, axis=1).A1
    coreness = np.zeros(num_nodes, dtype=int)

    # 简单的k-core分解
    remaining = np.ones(num_nodes, dtype=bool)
    k = 1

    while np.any(remaining):
        # 找到当前度大于等于k的节点
        high_degree_nodes = node_degrees >= k
        current_core = high_degree_nodes & remaining

        if not np.any(current_core):
            break

        coreness[current_core] = k
        remaining = remaining & ~current_core
        k += 1

    return coreness


def compute_features(incidence_matrix):
    """计算扩展的特征集"""
    num_nodes = incidence_matrix.shape[0]

    # 原有特征（移除coreness）
    hdc = np.sum(incidence_matrix, axis=1).A1
    rwiec_calc = RWIECalculator(incidence_matrix)
    rwiec = rwiec_calc.calculate_rwiec(gamma=1.0)  # 使用固定gamma
    rwhc_calc = RWHCCalculator(incidence_matrix)
    rwhc = rwhc_calc.calculate_rwhc()

    print("Computing motif coefficients...")
    motif_coeff = compute_motif_coefficient(incidence_matrix)
    print("Computing overlap degrees...")
    overlap_degree = compute_overlap_degree(incidence_matrix)

    # 移除 coreness 计算
    # print("Computing coreness...")
    # coreness = compute_coreness(incidence_matrix)

    feature_list = [hdc, rwiec, rwhc, motif_coeff, overlap_degree]
    feature_names = ['HDC', 'RWIEC', 'RWHC', 'Motif', 'Overlap']

    # 特别处理 RWHC - 放大到合理范围
    rwhc = rwhc * 1000  # 放大1000倍，使其值域在0-22之间

    for i, (feat, name) in enumerate(zip(feature_list, feature_names)):
        if len(feat) != num_nodes:
            raise ValueError(f"{name} feature length {len(feat)} != num_nodes {num_nodes}")

    features = np.vstack(feature_list).T
    print(f"Raw features shape: {features.shape}")

    # 打印处理后的统计信息
    for i, name in enumerate(feature_names):
        print(
            f"{name}: min={features[:, i].min():.4f}, max={features[:, i].max():.4f}, mean={features[:, i].mean():.4f}")

    # 标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print(f"Normalized features shape: {features.shape}")
    return torch.tensor(features, dtype=torch.float)



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



# def split_dataset(num_nodes, train_ratio=0.6, val_ratio=0.1):
#     indices = np.arange(num_nodes)
#     train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
#     val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (1 - train_ratio), random_state=42)
#     return train_idx, val_idx, test_idx

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


def cache_baseline_scores(incidence_matrix, cache_file="baseline_scores.pkl"):
    """缓存机制"""
    # 如果缓存文件存在且有效，直接加载
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_scores = pickle.load(f)
            # 检查缓存的数据是否与当前incidence_matrix匹配
            if len(cached_scores['dc']) == incidence_matrix.shape[0]:
                print("Loading baseline scores from cache")
                return cached_scores['dc'], cached_scores['bc'], cached_scores['hdc'], cached_scores['sc']
        except:
            print("Cache file corrupted, recalculating...")

    # 计算并缓存
    print("Calculating baseline scores...")
    scores_dc = compute_dc(incidence_matrix)
    scores_bc = compute_bc(incidence_matrix)
    scores_hdc = compute_hdc(incidence_matrix)
    scores_sc = compute_sc(incidence_matrix)

    # 保存到缓存
    cached_scores = {
        'dc': scores_dc,
        'bc': scores_bc,
        'hdc': scores_hdc,
        'sc': scores_sc
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_scores, f)

    return scores_dc, scores_bc, scores_hdc, scores_sc

def get_max_hyperdegree(incidence_matrix):
    return np.max(np.sum(incidence_matrix, axis=1).A1)

def split_dataset(num_nodes, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio/(1-train_ratio), random_state=42)
    return train_idx, val_idx, test_idx




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




def prepare_multi_nu_training_data(incidence_matrix, top_k, lambda_val, nu_vals):
    """为多个θ值准备训练目标"""
    num_nodes = incidence_matrix.shape[0]
    y_multi_nu = np.zeros(num_nodes)

    # 对每个θ值计算感染分数并平均
    for nu in nu_vals:
        scores_hdc = compute_hdc(incidence_matrix)
        top_nodes = np.argsort(scores_hdc)[-top_k:]

        infected_frac = compute_infected_fraction(
            incidence_matrix, top_nodes, lambda_val, nu, mu=1.0, num_runs=3
        )
        y_multi_nu[top_nodes] += infected_frac

    y_multi_nu[top_nodes] /= len(nu_vals)
    return y_multi_nu


def create_nu_specific_labels(incidence_matrix, nu_values, lambda_val=0.02, top_k_ratio=0.05):
    """为每个θ值创建特定的监督信号"""
    num_nodes = incidence_matrix.shape[0]
    top_k = int(num_nodes * top_k_ratio)

    # 为每个θ值选择最优的种子节点
    nu_specific_labels = np.zeros((num_nodes, len(nu_values)))

    for i, nu in enumerate(nu_values):
        print(f"Creating labels for nu={nu:.1f}")

        # 测试多种中心性方法，选择最好的作为监督信号
        candidate_scores = [
            compute_hdc(incidence_matrix),
            compute_dc(incidence_matrix),
            compute_bc(incidence_matrix),
            compute_sc(incidence_matrix)
        ]

        best_score = None
        best_performance = 0

        for scores in candidate_scores:
            top_nodes = np.argsort(scores)[-top_k:]

            performance = compute_infected_fraction(
                incidence_matrix, top_nodes, lambda_val, nu,
                mu=1.0, num_runs=3
            )

            if performance > best_performance:
                best_performance = performance
                best_score = scores

        # 基于最佳中心性分数创建监督信号
        nu_specific_labels[:, i] = best_score / (np.max(best_score) + 1e-10)

    return nu_specific_labels


import concurrent.futures
from tqdm import tqdm


def simulate_for_node_wrapper(args):
    """包装函数，用于多进程调用"""
    node_id, incidence_matrix, lambda_val, nu, num_nodes, num_runs = args
    return simulate_for_node(node_id, incidence_matrix, lambda_val, nu, num_nodes, num_runs)


def simulate_for_node(node_id, incidence_matrix, lambda_val, nu, num_nodes, num_runs):
    """实际的模拟函数"""
    initial_infected = np.zeros(num_nodes)
    initial_infected[node_id] = 1
    contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu, mu=1.0)
    infected_frac = contagion.simulate(max_steps=10000)

    return node_id, infected_frac


def create_labels_from_simulation(incidence_matrix, lambda_val, nu, top_k_ratio=0.05, num_runs=5):
    num_nodes = incidence_matrix.shape[0]
    y_true = np.zeros(num_nodes)

    node_degrees = np.sum(incidence_matrix, axis=1).A1
    top_k = int(num_nodes * top_k_ratio)
    target_nodes = np.argsort(node_degrees)[-top_k:]

    print(f"Generating simulation labels for nu={nu:.2f}. Simulating {len(target_nodes)} candidate nodes...")

    for node_id in target_nodes:
        total_infected = 0
        for run in range(num_runs):
            initial_infected = np.zeros(num_nodes)
            initial_infected[node_id] = 1
            contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu, mu=1.0)
            infected_frac = contagion.simulate(max_steps=10000)
            total_infected += infected_frac

        y_true[node_id] = total_infected / num_runs

    return y_true


# def create_multi_nu_simulation_labels(incidence_matrix, lambda_val, nu_values, top_k_ratio=0.1, n_workers=4):
#     """
#     为多个θ值生成真实的模拟标签，构建多任务学习数据集。
#     返回: (num_nodes, len(nu_values)) 的矩阵，每一列对应一个θ值的标签。
#     """
#     num_nodes = incidence_matrix.shape[0]
#     num_nu = len(nu_values)
#     Y_multi_nu = np.zeros((num_nodes, num_nu))
#
#     print(f"Generating simulation labels for {num_nu} ν values...")
#
#     # 并行为每个θ值生成标签
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
#         future_to_nu = {
#             executor.submit(create_labels_from_simulation, incidence_matrix, lambda_val, nu, top_k_ratio): i
#             for i, nu in enumerate(nu_values)
#         }
#
#
#         for future in tqdm(concurrent.futures.as_completed(future_to_nu), total=num_nu, desc="Multi-ν Simulation"):
#             nu_index = future_to_nu[future]
#             try:
#                 y_true = future.result()
#                 Y_multi_nu[:, nu_index] = y_true
#                 print(f"ν={nu_values[nu_index]:.2f} completed. Max influence: {y_true.max():.4f}")
#             except Exception as e:
#                 print(f"Error processing ν={nu_values[nu_index]:.2f}: {e}")
#
#     return Y_multi_nu

def create_multi_nu_simulation_labels(incidence_matrix, lambda_val, nu_values, top_k_ratio=0.1):
    num_nodes = incidence_matrix.shape[0]
    num_nu = len(nu_values)
    Y_multi_nu = np.zeros((num_nodes, num_nu))

    print(f"Generating simulation labels for {num_nu} ν values...")

    for i, nu in enumerate(nu_values):
        y_true = create_labels_from_simulation(
            incidence_matrix, lambda_val, nu, top_k_ratio, num_runs=5
        )
        Y_multi_nu[:, i] = y_true
        print(f"ν={nu:.2f} completed. Max influence: {y_true.max():.4f}")

    return Y_multi_nu



def enhanced_simulate_for_node(node_id, incidence_matrix, lambda_val, nu, num_nodes, num_runs=20):
    """增强的节点模拟函数，增加运行次数和精度"""
    total_infected = 0
    for run in range(num_runs):
        initial_infected = np.zeros(num_nodes)
        initial_infected[node_id] = 1
        contagion = HypergraphContagion(
            incidence_matrix,
            initial_infected,
            lambda_val,
            nu,
            mu=1.0
        )
        infected_frac = contagion.simulate(
            max_steps=20000,  # 增加最大步数
            tolerance=1e-6,  # 更小的容忍度
            stable_steps=300  # 更长的稳定期判断
        )
        total_infected += infected_frac

    return node_id, total_infected / num_runs


def create_smoothed_labels(incidence_matrix, lambda_val, nu, top_k_ratio=0.1, num_runs=20):
    """创建平滑的标签"""
    num_nodes = incidence_matrix.shape[0]
    y_true = np.zeros(num_nodes)

    # 基于节点度的先验分布
    node_degrees = np.sum(incidence_matrix, axis=1).A1
    degree_prior = node_degrees / np.sum(node_degrees)

    top_k = int(num_nodes * top_k_ratio)
    target_nodes = np.argsort(node_degrees)[-top_k:]

    print(f"Generating smoothed labels for nu={nu:.2f}. Simulating {len(target_nodes)} nodes...")

    # 并行计算
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for node_id in target_nodes:
            futures.append(
                executor.submit(
                    enhanced_simulate_for_node,
                    node_id, incidence_matrix, lambda_val, nu, num_nodes, num_runs
                )
            )

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            node_id, avg_infected = future.result()
            y_true[node_id] = avg_infected

    # 标签平滑：使用指数移动平均结合先验分布
    alpha = 0.7  # 平滑系数
    smoothed_labels = alpha * y_true + (1 - alpha) * degree_prior * np.max(y_true)

    return smoothed_labels


def create_high_quality_multi_nu_labels(incidence_matrix, lambda_val, nu_values, top_k_ratio=0.1):
    """创建高质量的多ν标签"""
    num_nodes = incidence_matrix.shape[0]
    num_nu = len(nu_values)
    Y_high_quality = np.zeros((num_nodes, num_nu))

    print(f"Generating high-quality labels for {num_nu} ν values...")

    for i, nu in enumerate(nu_values):
        y_smoothed = create_smoothed_labels(
            incidence_matrix, lambda_val, nu, top_k_ratio, num_runs=25
        )
        Y_high_quality[:, i] = y_smoothed
        print(f"ν={nu:.2f} completed. Max influence: {y_smoothed.max():.4f}")

    return Y_high_quality