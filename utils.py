from scipy import sparse
from sklearn.preprocessing import StandardScaler
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
from randomwalk import HypergraphRandomWalk
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

# def compute_pagerank(incidence_matrix):
#     num_nodes = incidence_matrix.shape[0]
#     node_degrees = np.sum(incidence_matrix, axis=1).A1
#     edge_degrees = np.sum(incidence_matrix, axis=0).A1  # 超边大小作为权重
#     alpha = 0.85
#     max_iter = 100
#     tol = 1e-6
#     H_mat = incidence_matrix.tocsr()
#     weights = edge_degrees / np.max(edge_degrees)
#     edge_degrees = np.sum(incidence_matrix, axis=0).A1
#     node_degrees[node_degrees == 0] = 1
#     edge_degrees[edge_degrees == 0] = 1
#
#     # 构建对角矩阵的逆
#     D_v_inv = sparse.diags(1.0 / node_degrees)  # 节点度矩阵的逆
#     W = sparse.diags(weights)  # 超边权重矩阵
#     D_e_inv = sparse.diags(1.0 / edge_degrees)  # 超边度矩阵的逆
#
#     # 构建转移概率矩阵 P = D_v^{-1} H W D_e^{-1} H^T
#     P = D_v_inv.dot(H_mat).dot(W).dot(D_e_inv).dot(H_mat.T)
#
#     # 转换为CSR格式
#     P = P.tocsr()
#
#     print(f"转移矩阵P的形状: {P.shape}")
#     print(f"P的非零元素数量: {P.nnz}")
#
#     # 初始化PageRank向量
#     x = np.ones(num_nodes) / num_nodes
#     personalization_vector = np.ones(num_nodes) / num_nodes
#
#     print("开始PageRank迭代...")
#
#     # Power Iteration方法
#     for i in range(max_iter):
#         x_new = alpha * P.dot(x) + (1 - alpha) * personalization_vector
#
#         # 检查收敛条件
#         diff = np.linalg.norm(x_new - x, 1)
#
#         if i < 5:  # 打印前几次迭代的差异
#             print(f"迭代 {i + 1}: 差异 = {diff:.10f}")
#
#         if diff < tol:
#             print(f"收敛于第 {i + 1} 次迭代, 最终差异 = {diff:.10f}")
#             x = x_new
#             break
#
#         x = x_new
#     else:
#         print(f"达到最大迭代次数 {max_iter}, 最终差异 = {diff:.10f}")
#
#     return x

def compute_pagerank(incidence_matrix):
    # 创建 HypergraphRandomWalk 实例（从 randomwalk.py）
    rw = HypergraphRandomWalk(incidence_matrix)
    # 计算稳态分布作为 PageRank 分数
    return rw.calculate_stationary_distribution()


# 动态计算rwhc和rwiec

def compute_features(incidence_matrix):
    # if nu_values is None:
    #     nu_values = [1.0]

    features_list = []

    # theta = 1.0 + 0.5 * (nu - 1.0) ** 1.5
    # gamma = 0.8 + 0.3 * (nu - 1.0) ** 1.5

    theta = 1.25
    gamma = 1.0

    # 计算各个特征
    hdc = compute_hdc(incidence_matrix)
    # sc = compute_sc(incidence_matrix)
    rwhc = RWHCCalculator(incidence_matrix).calculate_rwhc(theta=theta)
    rwiec = RWIECalculator(incidence_matrix).calculate_rwiec(gamma=gamma)
    motif = compute_motif_coefficient(incidence_matrix)
    overlap = compute_overlap_degree(incidence_matrix)
    pagerank = compute_pagerank(incidence_matrix)


    # === 新增：特征标准化和诊断 ===
    features_raw = [hdc, rwhc, rwiec, motif, overlap, pagerank]
    feature_names = ['HDC', 'RWHC', 'RWIEC', 'Motif', 'Overlap', 'Pagerank']

    print("=== 特征值诊断 ===")
    for i, (feat, name) in enumerate(zip(features_raw, feature_names)):
        print(f"{name}: min={np.min(feat):.6f}, max={np.max(feat):.6f}, "
              f"mean={np.mean(feat):.6f}, std={np.std(feat):.6f}")
        # 前5个节点的原始分数
        print(f"  Sample scores (first 5 nodes): {feat[:5]}")

    # 鲁棒的标准化方法
    features_normalized = []
    for feat in features_raw:
        if np.std(feat) < 1e-10:  # 常数特征
            normalized = np.ones_like(feat) * 0.5
        else:
            # 先进行对数变换（如果数据有偏）
            if np.max(feat) > 10 * np.median(feat[feat > 0]):
                feat_transformed = np.log1p(feat)
            else:
                feat_transformed = feat
            # 最小最大标准化到[0,1]
            normalized = (feat_transformed - np.min(feat_transformed)) / \
                          (np.max(feat_transformed) - np.min(feat_transformed) + 1e-10)
        features_normalized.append(normalized)

    # 组合特征
    combined_features = np.stack(features_normalized, axis=1)

    # === 新增：显示组合后的效果 ===
    print("=== 组合特征效果 ===")
    composite_scores = np.mean(combined_features, axis=1)
    top_5_indices = np.argsort(composite_scores)[-5:][::-1]  # 从高到低
    print(f"Top 5节点索引: {top_5_indices}")
    print("Top 5节点各特征分数:")
    for idx in top_5_indices:
        feat_values = combined_features[idx]
        print(f"  节点{idx}: {', '.join([f'{name}={val:.3f}' for name, val in zip(feature_names, feat_values)])}")

    features_list.append(combined_features)

    return features_list


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
    # print(f"nu={nu:.2f}, lambda_val={lambda_val:.4f}, avg_infected_fraction={avg_infected_fraction:.4f}, std={std_infected_fraction:.4f}")
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


def generate_dynamic_candidate_sets(incidence_matrix, nu, top_k, lambda_val, num_runs=10):
    """
    生成多样化的候选种子集

    参数:
        num_candidates: 要生成的候选集数量（其实这个也可以根据数据集大小改变）
        num_runs: 每个候选集的模拟次数

    返回:
        candidate_sets: 候选集列表 [num_candidates, top_k]
        candidate_scores: 对应的真实分数列表 [num_candidates]
    """
    num_nodes = incidence_matrix.shape[0]

    # 1. 计算特征
    features = {}
    features['HDC'] = compute_hdc(incidence_matrix)
    features['RWHC'] = RWHCCalculator(incidence_matrix).calculate_rwhc()
    features['RWIEC'] = RWIECalculator(incidence_matrix).calculate_rwiec()
    features['Motif'] = compute_motif_coefficient(incidence_matrix)
    features['Overlap'] = compute_overlap_degree(incidence_matrix)
    features['Pagerank'] = compute_pagerank(incidence_matrix)

    # 2. 权重初步设定
    if nu < 1.3:
        weights = np.array([0.4, 0.2, 0.1, 0.15, 0.05, 0.3])  # 偏向HDC和Motif
    elif nu < 1.7:
        weights = np.array([0.5, 0.2, 0.2, 0.2, 0.1, 0.3])  # 相对均衡
    else:
        weights = np.array([0.5, 0.3, 0.25, 0.15, 0.1, 0.3])  # 偏向RWHC和RWIEC

    # 3. 计算初始分数并选择Top M节点
    composite_scores = np.zeros(num_nodes)
    feature_keys = ['HDC', 'RWHC', 'RWIEC', 'Motif', 'Overlap', 'Pagerank']
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

    # 确保包含纯Top-K集 作为第一个
    candidate_sets.append(top_k_indices.copy())

    num_candidates = 30
    # 生成其他候选集
    for i in range(num_candidates - 1):
        if np.random.rand() < 0.5:  # 50%的概率进行随机替换
            candidate_set = top_k_indices.copy()
            # 随机替换几个节点
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
        else:  # 50%的概率完全随机从Top M中选择
            candidate_set = np.random.choice(top_m_indices, size=top_k, replace=False)
            candidate_sets.append(candidate_set)

    candidate_scores = []
    print(f"为θ={nu:.1f}模拟{len(candidate_sets)}个候选集...")

    for i, candidate_set in enumerate(candidate_sets):
        infected_frac = compute_infected_fraction(
            incidence_matrix, candidate_set, lambda_val, nu,
            mu=1.0, num_runs=num_runs
        )
        candidate_scores.append(infected_frac)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{len(candidate_sets)}个候选集的模拟")

    return candidate_sets, candidate_scores


def prepare_enhanced_training_data(incidence_matrix, lambda_val, nu_values, top_k_ratio):
    """
    为多个θ值准备训练数据
    """
    num_nodes = incidence_matrix.shape[0]
    top_k = int(num_nodes * top_k_ratio)

    all_candidate_sets = []
    all_nu_values = []
    all_scores = []

    print("开始生成训练数据...")
    for nu in nu_values:
        print(f"处理 θ={nu:.1f}")

        candidate_sets, candidate_scores = generate_dynamic_candidate_sets(
            incidence_matrix, nu, top_k, lambda_val,
            num_runs=10
        )

        all_candidate_sets.extend(candidate_sets)
        all_nu_values.extend([nu] * len(candidate_sets))
        all_scores.extend(candidate_scores)

        # 当前θ的最佳表现
        best_idx = np.argmax(candidate_scores)
        print(f"  θ={nu:.1f}最佳候选集分数: {candidate_scores[best_idx]:.4f}")

    return all_candidate_sets, all_nu_values, all_scores


def evaluate_infection_dynamics(incidence_matrix, seed_set, lambda_val, nu, num_runs=5):
    """评估感染动力学特征"""
    all_curves = []
    all_times = []

    for run in range(num_runs):
        initial_infected = np.zeros(incidence_matrix.shape[0])
        initial_infected[seed_set] = 1
        contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu, mu=1.0)

        # 时间序列数据
        avg_infected, curve, steps = contagion.simulate(return_curve=True)
        all_curves.append(curve)
        all_times.append(steps)

    # 计算各种动力学指标
    dynamics_metrics = {
        'final_fraction': np.mean([curve[-1] for curve in all_curves]),
        'time_to_half': np.mean([steps[np.where(np.array(curve) >= 0.5)[0][0]]
                                 for curve in all_curves if np.any(np.array(curve) >= 0.5)]),
        # 'max_growth_rate': np.mean([np.max(np.diff(curve)) for curve in all_curves]),
        'stabilization_time': np.mean([len(steps) for steps in all_times]),
        'area_under_curve': np.mean([np.trapz(curve) for curve in all_curves])
    }

    return dynamics_metrics


def evaluate_seed_set_diversity(incidence_matrix, seed_sets):
    """评估不同方法选择的种子集多样性"""
    from sklearn.metrics import jaccard_score

    diversity_metrics = {}
    method_names = list(seed_sets.keys())

    # 计算两两Jaccard相似度
    similarity_matrix = np.zeros((len(method_names), len(method_names)))
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i != j:
                set1 = set(seed_sets[method1])
                set2 = set(seed_sets[method2])
                similarity = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                similarity_matrix[i, j] = similarity

    diversity_metrics['pairwise_similarity'] = similarity_matrix
    diversity_metrics['average_similarity'] = np.mean(similarity_matrix)

    return diversity_metrics


def assess_network_complexity(incidence_matrix):
    """评估网络复杂度，判断是否值得深度优化"""
    num_nodes = incidence_matrix.shape[0]

    # 计算特征多样性
    features = [
        compute_hdc(incidence_matrix),
        compute_sc(incidence_matrix),
        compute_motif_coefficient(incidence_matrix),
        compute_overlap_degree(incidence_matrix)
    ]

    # 计算特征间的平均相关系数
    corr_matrix = np.corrcoef(features)
    avg_correlation = (np.sum(np.abs(corr_matrix)) - 4) / 12  # 减去对角线，除以组合数

    # 计算节点度的基尼系数
    degrees = compute_hdc(incidence_matrix)
    sorted_degrees = np.sort(degrees)
    n = len(degrees)
    gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_degrees) / (n * np.sum(sorted_degrees))

    complexity_score = (1 - avg_correlation) * gini * num_nodes / 100

    print(f"网络复杂度评估:")
    print(f"  节点数: {num_nodes}")
    print(f"  特征平均相关系数: {avg_correlation:.3f} (越低越好)")
    print(f"  度分布基尼系数: {gini:.3f} (越高越不均匀)")
    print(f"  复杂度分数: {complexity_score:.3f}")

    return complexity_score > 2.0


def analyze_critical_nodes_comparison(incidence_matrix, baseline_scores, enhanced_scores, top_k=10):
    """详细分析关键节点对比，新增超边覆盖分析"""
    print("\n" + "=" * 60)
    print("关键节点对比分析（含超边覆盖）")
    print("=" * 60)

    # 获取各方法的关键节点
    enhanced_top = np.argsort(enhanced_scores)[-top_k:][::-1]
    hdc_top = np.argsort(baseline_scores['HDC'])[-top_k:][::-1]
    dc_top = np.argsort(baseline_scores['DC'])[-top_k:][::-1]
    bc_top = np.argsort(baseline_scores['BC'])[-top_k:][::-1]
    sc_top = np.argsort(baseline_scores['SC'])[-top_k:][::-1]

    # 打印对比
    methods = {
        'Enhanced-NuGNN': enhanced_top,
        'HDC': hdc_top,
        'DC': dc_top,
        'BC': bc_top,
        'SC': sc_top
    }

    print(f"\n各方法Top{top_k}关键节点:")
    for method, nodes in methods.items():
        print(f"{method:15}: {nodes}")

    # === 新增：超边覆盖分析 ===
    print(f"\n{'=' * 60}")
    print("超边覆盖分析")
    print(f"{'=' * 60}")

    # 计算超边大小分布
    edge_sizes = np.sum(incidence_matrix, axis=0).A1  # 每个超边的大小
    total_edges = incidence_matrix.shape[1]  # 总超边数

    print(f"超图基本信息: 总超边数={total_edges}, 超边平均大小={np.mean(edge_sizes):.2f}")
    print(f"{'方法':<15} {'覆盖超边数':<10} {'覆盖率(%)':<10} {'覆盖超边平均大小':<15} {'覆盖超边大小标准差':<15}")
    print(f"{'-' * 70}")

    for method, nodes in methods.items():
        # 计算该节点集覆盖的超边
        covered_edges = set()
        for node in nodes:
            edges_of_node = incidence_matrix[node].nonzero()[1]
            covered_edges.update(edges_of_node)

        coverage_count = len(covered_edges)
        coverage_ratio = (coverage_count / total_edges) * 100

        # 计算覆盖超边的大小统计
        if coverage_count > 0:
            covered_edge_sizes = edge_sizes[list(covered_edges)]
            avg_size = np.mean(covered_edge_sizes)
            std_size = np.std(covered_edge_sizes)
        else:
            avg_size = 0.0
            std_size = 0.0

        print(f"{method:<15} {coverage_count:<10} {coverage_ratio:<10.1f} {avg_size:<15.2f} {std_size:<15.2f}")

    # === 新增：覆盖超边的重叠分析 ===
    print(f"\n{'=' * 60}")
    print("覆盖超边重叠分析")
    print(f"{'=' * 60}")

    # 计算各方法覆盖超边的重叠情况
    covered_edges_dict = {}
    for method, nodes in methods.items():
        covered_edges = set()
        for node in nodes:
            edges_of_node = incidence_matrix[node].nonzero()[1]
            covered_edges.update(edges_of_node)
        covered_edges_dict[method] = covered_edges

    # 计算Enhanced-NuGNN与其他方法的超边重叠度
    enhanced_edges = covered_edges_dict['Enhanced-NuGNN']
    print("Enhanced-NuGNN与其他方法的超边重叠情况:")
    for method, edges in covered_edges_dict.items():
        if method != 'Enhanced-NuGNN':
            overlap = len(enhanced_edges & edges)
            jaccard = overlap / len(enhanced_edges | edges) if len(enhanced_edges | edges) > 0 else 0
            print(f"  vs {method}: 重叠超边数={overlap}, Jaccard相似度={jaccard:.3f}")

    # 原有的重叠度矩阵计算保持不变
    print(f"\n{'=' * 60}")
    print("节点重叠度矩阵 (Top{top_k}):")
    print(f"{'=' * 60}")

    method_names = list(methods.keys())
    overlap_matrix = np.zeros((len(method_names), len(method_names)))

    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            set1 = set(methods[method1])
            set2 = set(methods[method2])
            overlap = len(set1 & set2)
            overlap_matrix[i, j] = overlap

    # 打印重叠矩阵
    print("     " + "".join([f"{name:>15}" for name in method_names]))
    for i, name in enumerate(method_names):
        row = f"{name:15}"
        for j in range(len(method_names)):
            row += f"{overlap_matrix[i, j]:>15}"
        print(row)

    # 分析Enhanced-NuGNN的独特点
    enhanced_set = set(enhanced_top)
    other_sets = [set(methods[name]) for name in method_names if name != 'Enhanced-NuGNN']
    unique_to_enhanced = enhanced_set - set().union(*other_sets)

    if unique_to_enhanced:
        print(f"\nEnhanced-NuGNN独有的关键节点: {sorted(unique_to_enhanced)}")
        # 分析这些独有节点的特性
        node_degrees = compute_hdc(incidence_matrix)
        unique_degrees = node_degrees[list(unique_to_enhanced)]
        print(
            f"这些节点的度分布: 最小={np.min(unique_degrees)}, 最大={np.max(unique_degrees)}, 平均={np.mean(unique_degrees):.2f}")

        # === 新增：分析独有节点的超边覆盖特性 ===
        print("独有节点的超边覆盖特性:")
        for node in sorted(unique_to_enhanced):
            edges_of_node = incidence_matrix[node].nonzero()[1]
            edge_size_info = []
            for edge in edges_of_node:
                edge_size = edge_sizes[edge]
                edge_size_info.append(f"E{edge}({edge_size}节点)")
            print(f"  节点{node}: 覆盖超边 {', '.join(edge_size_info)}")
    else:
        print(f"\nEnhanced-NuGNN没有独有关键节点")

    return methods