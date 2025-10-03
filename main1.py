# θ从1到2

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from UniGGCN import UniGCNRegression
from NDA_HGNN_model import NonlinearDiffusionAwareModel
from nonlinear import HypergraphContagion
from rwhc import RWHCCalculator
from rwiec import RWIECalculator
from utils import load_hypergraph, compute_features, split_dataset, compute_infected_fraction, cache_baseline_scores, \
    load_hypergraph_pickle, prepare_enhanced_training_data, evaluate_seed_set_diversity, evaluate_infection_dynamics, \
    assess_network_complexity, compute_motif_coefficient, compute_overlap_degree, compute_pagerank, \
    analyze_critical_nodes_comparison
from baseline import compute_hdc, compute_dc, compute_bc, compute_sc
from connectivity import hypergraph_natural_connectivity
import matplotlib.pyplot as plt
import pickle
import os
import random
from unigencoder import Unigencoder
import torch.nn.functional as F
from param_tuner import HyperparameterTuner
from connectivity import hypergraph_natural_connectivity
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_infection_loss(model, data, incidence_matrix, nu, lambda_val=0.1, top_k_ratio=0.05):
    model.eval()
    with torch.no_grad():
        node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
        main_scores, _ = model(data, nu, node_degrees)
        scores = main_scores.cpu().numpy().flatten()
        num_nodes = incidence_matrix.shape[0]
        top_k = int(num_nodes * top_k_ratio)
        seed_nodes = np.argsort(scores)[-top_k:]

        # 修复：确保simulate返回数组
        infected_frac = compute_infected_fraction(
            incidence_matrix, seed_nodes, lambda_val, nu, num_runs=3  # 减少模拟次数加速
        )

        # 直接使用返回值，不尝试切片
        return torch.tensor(1 - infected_frac)  # 损失：1 - 感染比例

def compute_connectivity_loss(model, data, incidence_matrix, nu, top_k_ratio=0.05, alpha=0.3):
    model.eval()
    with torch.no_grad():
        node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
        main_scores, _ = model(data, nu, node_degrees)
        scores = main_scores.cpu().numpy().flatten()
        num_nodes = incidence_matrix.shape[0]
        top_k = int(num_nodes * top_k_ratio)
        sorted_nodes = np.argsort(scores)[-top_k:]  # 高分节点
        nodes_to_keep = np.setdiff1d(np.arange(num_nodes), sorted_nodes)
        if len(nodes_to_keep) == 0:
            return torch.tensor(0.0)  # 避免除零
        modified_matrix = incidence_matrix[nodes_to_keep, :]
        edge_sums = np.array(modified_matrix.sum(axis=0)).flatten()
        non_zero_edges = edge_sums > 0
        modified_matrix = modified_matrix[:, non_zero_edges]
        if modified_matrix.shape[1] == 0:
            connectivity_after = -np.inf
        else:
            connectivity_after = hypergraph_natural_connectivity(modified_matrix)
        connectivity_original = hypergraph_natural_connectivity(incidence_matrix)
        drop = (connectivity_original - connectivity_after) / (connectivity_original + 1e-10)
        return -alpha * torch.log(torch.tensor(drop + 1e-10))  # 鼓励更大下降

def evaluate_connectivity_after_removal(incidence_matrix, node_scores, algorithm_name, removal_ratios):
    """
    评估删除不同比例节点后的连通性

    参数:
        incidence_matrix: 超图关联矩阵
        node_scores: 节点重要性分数
        algorithm_name: 算法名称
        removal_ratios: 要删除的比例列表 [0.1, 0.2, ..., 1.0]

    返回:
        每个删除比例对应的连通性值列表
    """
    connectivity_values = []
    num_nodes = incidence_matrix.shape[0]

    # 按重要性排序节点（从最重要到最不重要）
    sorted_nodes = np.argsort(node_scores)[::-1]

    for ratio in removal_ratios:
        num_to_remove = int(num_nodes * ratio)
        if num_to_remove == 0:
            # 不删除任何节点，使用原始矩阵
            modified_matrix = incidence_matrix.copy()
        else:
            # 删除前num_to_remove个最重要的节点
            nodes_to_remove = sorted_nodes[:num_to_remove]
            nodes_to_keep = sorted_nodes[num_to_remove:]

            if len(nodes_to_keep) == 0:
                # 如果删除了所有节点，连通性为负无穷
                connectivity_values.append(-np.inf)
                continue

            # 创建修改后的关联矩阵（只保留剩余的节点）
            if hasattr(incidence_matrix, 'tocsr'):
                modified_matrix = incidence_matrix.tocsr()[nodes_to_keep, :]
            else:
                modified_matrix = incidence_matrix[nodes_to_keep, :]

            # 删除完全为零的列（不再包含任何节点的超边）
            edge_sums = np.array(modified_matrix.sum(axis=0)).flatten()
            non_zero_edges = edge_sums > 0
            modified_matrix = modified_matrix[:, non_zero_edges]

            # 检查是否还有有效的超边
            if modified_matrix.shape[1] == 0:
                connectivity_values.append(-np.inf)
                continue

        # 计算修改后的超图的自然连通性
        try:
            if modified_matrix.shape[0] > 1 and modified_matrix.shape[1] > 0:
                rho = hypergraph_natural_connectivity(modified_matrix)
                connectivity_values.append(rho)
            else:
                # 如果删除后没有足够的节点或超边，连通性为负无穷
                connectivity_values.append(-np.inf)
        except Exception as e:
            print(f"Error computing connectivity for {algorithm_name} at ratio {ratio}: {e}")
            connectivity_values.append(-np.inf)

    return connectivity_values


def compare_connectivity_across_algorithms(incidence_matrix, baseline_scores, enhanced_scores, removal_ratios,
                                           title_suffix=""):
    """
    比较不同算法在删除不同比例节点后的连通性

    参数:
        incidence_matrix: 原始超图关联矩阵
        baseline_scores: 基线算法的分数字典
        enhanced_scores: 增强算法的分数
        removal_ratios: 要删除的比例列表
        title_suffix: 图表标题的后缀

    返回:
        包含所有算法连通性结果的字典
    """
    results = {}

    # 评估增强算法
    print(f"Evaluating NDA-HGNN connectivity...")
    enhanced_connectivity = evaluate_connectivity_after_removal(
        incidence_matrix, enhanced_scores, "NDA-HGNN", removal_ratios
    )
    results['NDA-HGNN'] = enhanced_connectivity

    # 评估基线算法
    for method_name, scores in baseline_scores.items():
        print(f"Evaluating {method_name} connectivity...")
        method_connectivity = evaluate_connectivity_after_removal(
            incidence_matrix, scores, method_name, removal_ratios
        )
        results[method_name] = method_connectivity

    # 绘制结果
    plt.figure(figsize=(10, 6))
    colors = {
        'NDA-HGNN': 'blue',
        'DC': 'red',
        'BC': 'green',
        'HDC': 'orange',
        'SC': 'purple'
    }

    for method, connectivity in results.items():
        plt.plot(removal_ratios, connectivity, label=method, color=colors.get(method, 'black'), marker='o', linewidth=2)

    plt.xlabel('Removal Ratio')
    plt.ylabel('Natural Connectivity (ρ)')
    plt.title(f'Connectivity After Node Removal {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    filename = f'connectivity_comparison-senate-{title_suffix.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return results

def train_enhanced_model(model, data, original_incidence_matrix, train_idx, epochs=300, lr=0.0005, patience=50):
    device = data.x.device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    patience_counter = 0
    # # 假设从 split_dataset 获取 train/val 分割
    # train_idx, val_idx = train_test_split(range(data.x.shape[0]), test_size=0.2, random_state=42)
    # 简单的学习率调度
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    torch.autograd.set_detect_anomaly(True)   # 帮助检测梯度问题
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()

        nu = np.random.choice(nu_vals)
        node_degrees = torch.tensor(compute_hdc(original_incidence_matrix), dtype=torch.float, device=device).view(-1,
                                                                                                                   1)
        main_scores, linear_scores = model(data, nu, node_degrees)

        # 计算损失
        infection_loss = compute_infection_loss(model, data, original_incidence_matrix, nu)
        if epoch % 5 == 0:  # 每5个epoch计算一次连通性损失
            connect_loss = compute_connectivity_loss(model, data, original_incidence_matrix, nu)
        else:
            connect_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # aux_loss = torch.mean((main_scores - linear_scores) ** 2)
        # loss = 0.5 * infection_loss + 0.3 * connect_loss + 0.2 * aux_loss

        # 阶段1 (0-100轮): 主要学习传播特性
        if epoch < 100:
            infection_loss = compute_infection_loss(model, data, original_incidence_matrix, nu)
            loss = infection_loss  # 只关注传播

        # 阶段2 (100-200轮): 逐步引入连通性考虑
        elif epoch < 200:
            infection_loss = compute_infection_loss(model, data, original_incidence_matrix, nu)
            connect_loss = compute_connectivity_loss(model, data, original_incidence_matrix, nu)
            # 渐进式增加连通性权重
            connect_weight = min(0.3 * (epoch - 100) / 100, 0.3)
            loss = (1 - connect_weight) * infection_loss + connect_weight * connect_loss

        # 阶段3 (200轮后): 平衡优化
        else:
            infection_loss = compute_infection_loss(model, data, original_incidence_matrix, nu)
            connect_loss = compute_connectivity_loss(model, data, original_incidence_matrix, nu)
            loss = 0.7 * infection_loss + 0.3 * connect_loss  # 侧重传播，兼顾结构

        # 检查损失有效性
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch}: 无效损失值，跳过")
            continue

        # 确保损失可导
        if not loss.requires_grad:
            loss = loss.clone().requires_grad_(True)

        loss.backward()

        # 梯度检查
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # 如果梯度太大，进行裁剪
        max_grad_norm = 1.0
        if total_norm > max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            print(f"Epoch {epoch}: 梯度裁剪 applied, norm was {total_norm:.4f}")

        optimizer.step()

        # 检查损失是否合理
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch}: 损失异常 {loss.item()}, 跳过更新")
            optimizer.zero_grad()
            continue

        # 验证阶段
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_nu in nu_vals[:2]:  # 验证前2个θ值
                    val_main_scores, _ = model(data, val_nu, node_degrees)
                    # 简单的验证损失：节点分数的方差（鼓励分数有区分度）
                    score_variance = torch.var(val_main_scores)
                    val_loss += -score_variance.item()  # 负号因为我们要最大化方差

                val_loss /= 2

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # 加载最佳模型
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))

    return model

# def train_enhanced_model(model, data, incidence_matrix, train_idx, epochs=300, lr=0.001, patience=50):
#     # 评估网络复杂度
#     complexity = assess_network_complexity(incidence_matrix)
#
#     # 动态调整训练策略
#     if complexity < 2.0:  # 简单网络
#         actual_epochs = min(epochs, 150)
#         actual_lr = lr * 1.5
#     else:  # 复杂网络
#         actual_epochs = epochs
#         actual_lr = lr
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     # 准备节点度特征
#     node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=device).view(-1, 1)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
#
#     best_loss = float('inf')
#     patience_counter = 0
#
#     for epoch in range(actual_epochs):
#         model.train()
#         total_loss = 0
#         total_main_loss = 0
#         total_aux_loss = 0
#
#         # 随机打乱训练样本
#         indices = torch.randperm(len(data.seed_sets))
#
#         for idx in indices:
#             seed_set = data.seed_sets[idx]
#             nu_val = data.seed_set_nu[idx]
#             true_score = data.seed_set_labels[idx]
#
#             optimizer.zero_grad()
#
#             # 前向传播
#             main_scores, linear_scores = model(data, nu_val, node_degrees)
#
#             # 计算主损失：种子集总分 vs 真实分数
#             seed_set_scores = main_scores[seed_set]
#             predicted_score = torch.sum(seed_set_scores)
#             loss_main = F.mse_loss(predicted_score, true_score)
#
#             # 计算辅助损失：鼓励主分数与线性分数分布相似
#             loss_aux = F.mse_loss(main_scores, linear_scores.detach())
#
#             # 组合损失
#             alpha = 0.1  # 辅助损失权重
#             loss = loss_main + alpha * loss_aux
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_main_loss += loss_main.item()
#             total_aux_loss += loss_aux.item()
#
#         # 验证和早停
#         avg_loss = total_loss / len(indices)
#         avg_main_loss = total_main_loss / len(indices)
#         avg_aux_loss = total_aux_loss / len(indices)
#
#         if epoch % 20 == 0:
#             print(
#                 f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} (Main: {avg_main_loss:.4f}, Aux: {avg_aux_loss:.4f}) | LR: {optimizer.param_groups[0]["lr"]:.6f}')
#
#         # 早停
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), 'best_enhanced_model.pth')
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break
#
#         scheduler.step(avg_loss)
#
#     # 加载最佳模型
#     model.load_state_dict(torch.load('best_enhanced_model.pth', map_location=device))
#     return model


def analyze_auxiliary_weights(model, data, nu_values, incidence_matrix):
    """分析辅助网络学到的特征权重"""
    device = next(model.parameters()).device
    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=device).view(-1, 1)

    print("辅助网络学到的特征权重随nu的变化:")
    feature_names = ['HDC', 'RWHC', 'RWIEC', 'Motif', 'Overlap', 'Pagerank']

    for nu in nu_values:
        with torch.no_grad():
            _, _, aux_weights = model(data, nu, node_degrees, return_auxiliary=True)

        weights = aux_weights.squeeze().cpu().numpy()
        print(f"ν = {nu:.1f}: {', '.join([f'{name}:{w:.3f}' for name, w in zip(feature_names, weights)])}")


def train_with_seed_sets(model, data, epochs=200, lr=0.001, patience=30, focus_nu_range=(1.3, 1.8)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    mse_loss = nn.MSELoss()
    ranking_loss = nn.MarginRankingLoss(margin=0.1)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(data.seed_sets))
        for idx in indices:
            seed_set = data.seed_sets[idx]
            nu_val = data.seed_set_nu[idx].item()
            true_label = data.seed_set_labels[idx]
            if focus_nu_range and not (focus_nu_range[0] <= nu_val <= focus_nu_range[1]):
                if np.random.rand() < 0.7:
                    continue
            optimizer.zero_grad()
            node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=device).view(-1, 1)
            node_scores = model(data, nu_val, node_degrees)
            seed_set_scores = node_scores[seed_set]
            predicted_score = torch.sum(seed_set_scores)
            loss_mse = mse_loss(predicted_score, true_label)
            if idx > 0:
                other_idx = idx - 1
                other_score = torch.sum(node_scores[data.seed_sets[other_idx]])
                other_label = data.seed_set_labels[other_idx]
                loss_rank = ranking_loss(predicted_score, other_score,
                                         torch.tensor(1.0 if true_label > other_label else -1.0))
                weight_rank = 0.4 + 0.2 * (nu_val - 1.0)
                loss = (1 - weight_rank) * loss_mse + weight_rank * loss_rank
            else:
                loss = loss_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(indices):.4f}")
        val_loss = 0
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    return model


def focused_nu_training(model, data, nu_values, Y_real, focus_nu=1.4, epochs=200, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    Y_real_tensor = torch.tensor(Y_real, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    criterion = nn.MSELoss()

    train_idx, val_idx, test_idx = split_dataset(data.num_nodes)

    # 找到focus_nu对应的索引
    focus_idx = np.where(np.isclose(nu_values, focus_nu))[0][0]

    best_val_loss = float('inf')
    best_overall_loss = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        model.train()

        # 动态调整训练策略：前期全面训练，后期专注薄弱点
        if epoch < epochs // 2:
            # 前期：全面训练所有ν值
            nu_idx = np.random.randint(len(nu_values))
        else:
            # 后期：50%概率训练focus_nu，50%概率训练其他θ
            if np.random.rand() < 0.5:
                nu_idx = focus_idx
            else:
                other_indices = [i for i in range(len(nu_values)) if i != focus_idx]
                nu_idx = np.random.choice(other_indices)

        nu = nu_values[nu_idx]

        optimizer.zero_grad()
        out = model(data, nu)
        y_true = Y_real_tensor[:, nu_idx].view(-1, 1)

        loss = criterion(out[train_idx], y_true[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 验证：同时检查focus_nu和整体性能
        model.eval()
        with torch.no_grad():
            # 检查focus_nu的性能
            focus_out = model(data, focus_nu)
            focus_true = Y_real_tensor[:, focus_idx].view(-1, 1)
            focus_val_loss = criterion(focus_out[val_idx], focus_true[val_idx])

            # 检查整体性能
            overall_val_loss = 0
            for i, nu_val in enumerate(nu_values):
                val_out = model(data, nu_val)
                val_true = Y_real_tensor[:, i].view(-1, 1)
                overall_val_loss += criterion(val_out[val_idx], val_true[val_idx]).item()
            overall_val_loss /= len(nu_values)

        scheduler.step(overall_val_loss)

        if epoch % 20 == 0:
            print(f'Epoch {epoch}: ν={nu:.1f}, Train Loss: {loss.item():.4f}, '
                  f'Focus ν Loss: {focus_val_loss.item():.4f}, Overall Val Loss: {overall_val_loss:.4f}')

        # 早停
        if overall_val_loss < best_overall_loss:
            best_overall_loss = overall_val_loss
            best_val_loss = focus_val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return model


def train_model(model, data, train_idx, val_idx, epochs=500, lr=0.001, patience=50):
    """监督学习训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.Pv = data.Pv.to(device)
    data.PvT = data.PvT.to(device)
    data.y = data.y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()

    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[val_idx], data.y[val_idx])
        scheduler.step(val_loss)
        model.train()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 50 == 0:
            print(
                f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载最佳模型
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return model, train_losses, val_losses


def evaluate_enhanced_model(model, incidence_matrix, data, nu_vals, lambda_val, top_k_ratio, num_runs=20):
    results = {}
    all_seed_sets = {}

    for nu in nu_vals:
        top_k = int(incidence_matrix.shape[0] * top_k_ratio)
        results[nu] = {}

        # === 新增：打印当前ν值和top_k ===
        print(f"\n=== ν={nu:.1f}, 选择top_{top_k}个节点 ===")

        nu_seed_sets = {}

        model.eval()
        with torch.no_grad():
            node_degrees = torch.tensor(compute_hdc(incidence_matrix),
                                        dtype=torch.float, device=data.x.device).view(-1, 1)
            main_scores, linear_scores = model(data, nu, node_degrees)
            enhanced_nodes = np.argsort(main_scores.cpu().numpy().flatten())[-top_k:]
            nu_seed_sets['NDA-HGNN'] = enhanced_nodes

            # === 新增：打印NDA-HGNN的关键节点和分数 ===
            enhanced_scores = main_scores.cpu().numpy().flatten()
            top_enhanced_indices = np.argsort(enhanced_scores)[-top_k:][::-1]  # 从高到低
            print(f"NDA-HGNN Top{top_k}节点索引: {top_enhanced_indices}")
            print(f"NDA-HGNN Top{top_k}节点分数: {enhanced_scores[top_enhanced_indices]}")

            # === 修复：评估NDA-HGNN的性能并添加到results中 ===
            enhanced_dynamics = evaluate_infection_dynamics(
                incidence_matrix, enhanced_nodes, lambda_val, nu, num_runs=num_runs
            )
            results[nu]['NDA-HGNN'] = {
                'final_fraction': enhanced_dynamics['final_fraction'],
                'dynamics': enhanced_dynamics
            }

            # 评估基线方法
        try:
            baseline_scores = cache_baseline_scores(incidence_matrix)

            for method in ['HDC', 'DC', 'BC', 'SC']:
                scores = baseline_scores[method]
                top_nodes = np.argsort(scores)[-top_k:][::-1]  # 从高到低排序
                nu_seed_sets[method] = top_nodes

                print(f"\n{method}方法:")
                print(f"Top{top_k}节点索引: {top_nodes}")
                print(f"Top{top_k}节点分数: {scores[top_nodes]}")

                # 可选：打印节点度信息用于对比
                node_degrees = compute_hdc(incidence_matrix)
                print(f"Top{top_k}节点度: {node_degrees[top_nodes]}")

                method_dynamics = evaluate_infection_dynamics(
                    incidence_matrix, top_nodes, lambda_val, nu, num_runs=num_runs
                )
                results[nu][method] = {
                    'final_fraction': method_dynamics['final_fraction'],
                    'dynamics': method_dynamics
                }

        except KeyError as e:
            print(f"KeyError in baseline scores: {e}")
        except Exception as e:
            print(f"Error evaluating baseline method: {e}")

            # 评估种子集多样性
        try:
            diversity_metrics = evaluate_seed_set_diversity(incidence_matrix, nu_seed_sets)
            results[nu]['diversity'] = diversity_metrics
        except Exception as e:
            print(f"Error evaluating diversity: {e}")
            results[nu]['diversity'] = {}

        all_seed_sets[nu] = nu_seed_sets

        # === 新增：打印方法间重叠度对比 ===
        print(f"\n--- 方法间重叠度对比 (ν={nu:.1f}) ---")
        enhanced_set = set(nu_seed_sets['NDA-HGNN'])
        for method, nodes in nu_seed_sets.items():
            if method != 'NDA-HGNN':
                method_set = set(nodes)
                overlap = len(enhanced_set & method_set)
                jaccard = overlap / len(enhanced_set | method_set) if len(enhanced_set | method_set) > 0 else 0
                print(f"NDA-HGNN vs {method}: 重叠{overlap}个节点, Jaccard相似度={jaccard:.3f}")

        # === 修复：确保NDA-HGNN的结果存在再打印 ===
        if 'NDA-HGNN' in results[nu]:
            print(f"ν={nu:.1f}: NDA-HGNN: {results[nu]['NDA-HGNN']['final_fraction']:.4f}")
        else:
            print(f"ν={nu:.1f}: NDA-HGNN结果缺失")

    return results, all_seed_sets


def plot_results(nu_vals, results):
    colors = {
        'NDA-HGNN': 'blue',
        'DC': 'red',
        'BC': 'green',
        'HDC': 'orange',
        'SC': 'purple'
    }
    plt.figure(figsize=(10, 6))
    if len(nu_vals) > 0 and results:  # 检查非空：len(nu_vals) 兼容 NumPy 数组
        algorithms = list(results[nu_vals[0]].keys())  # 从第一个 nu 获取算法名
        for alg in algorithms:
            fractions = [results[nu].get(alg, 0.0) for nu in nu_vals]
            plt.plot(nu_vals, fractions, label=alg, color=colors.get(alg, 'black'), marker='o')
    else:
        print("Warning: No results to plot")
    plt.xlabel('ν')
    plt.ylabel('Infected Fraction')
    plt.title('Senate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('senate2-dy-black .png')
    plt.show()


def plot_enhanced_results(nu_vals, results, all_seed_sets):
    """多维度可视化结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 最终感染比例
    colors = {'NDA-HGNN': 'blue', 'DC': 'red', 'BC': 'green', 'HDC': 'orange', 'SC': 'purple'}
    for method in ['NDA-HGNN', 'DC', 'BC', 'HDC', 'SC']:
        fractions = [results[nu][method]['final_fraction'] for nu in nu_vals]
        ax1.plot(nu_vals, fractions, label=method, color=colors.get(method, 'black'), marker='o', linewidth=2)
    ax1.set_xlabel('θ')
    ax1.set_ylabel('Final Infection Fraction')
    ax1.set_title('Senate')
    ax1.legend()
    ax1.grid(True)

    # 2. 达到50%感染的时间
    for method in ['NDA-HGNN', 'DC', 'BC', 'HDC', 'SC']:
        times = [results[nu][method]['dynamics']['time_to_half'] for nu in nu_vals]
        ax2.plot(nu_vals, times, label=method, color=colors.get(method, 'black'), marker='s', linewidth=2)
    ax2.set_xlabel('θ')
    ax2.set_ylabel('Time to 50% Infection')
    ax2.set_title('Senate')
    ax2.legend()
    ax2.grid(True)

    # # 3. 种子集多样性（平均相似度）
    # avg_similarities = {method: [] for method in ['DC', 'BC', 'HDC', 'SC']}
    # for nu in nu_vals:
    #     similarity_to_enhanced = []
    #     for method in ['DC', 'BC', 'HDC', 'SC']:
    #         enhanced_set = set(all_seed_sets[nu]['NDA-HGNN'])
    #         method_set = set(all_seed_sets[nu][method])
    #         similarity = len(enhanced_set & method_set) / len(enhanced_set | method_set)
    #         avg_similarities[method].append(similarity)
    #
    # for method, similarities in avg_similarities.items():
    #     ax3.plot(nu_vals, similarities, label=method, color=colors.get(method, 'black'), marker='^', linewidth=2)
    # ax3.set_xlabel('ν')
    # ax3.set_ylabel('Similarity to NDA-HGNN')
    # ax3.set_title('Senate')
    # ax3.legend()
    # ax3.grid(True)

    # # 4. 最大增长率
    # for method in ['NDA-HGNN', 'DC', 'BC', 'HDC', 'SC']:
    #     growth_rates = [results[nu][method]['dynamics']['max_growth_rate'] for nu in nu_vals]
    #     ax4.plot(nu_vals, growth_rates, label=method, color=colors.get(method, 'black'), marker='d', linewidth=2)
    # ax4.set_xlabel('ν')
    # ax4.set_ylabel('Maximum Growth Rate')
    # ax4.set_title('Senate')
    # ax4.legend()
    # ax4.grid(True)

    plt.tight_layout()
    plt.savefig('senate2-.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 清理缓存文件
    for cache_file in ["dr_ugcn_scores2.pkl", "baseline_scores.pkl"]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"Deleted cache file: {cache_file}")
            except Exception as e:
                print(f"Error removing {cache_file}: {e}")

    file_path = "hyperedges-senate-committees.txt"
    incidence_matrix, edge_index, node_id_map = load_hypergraph(file_path)
    print(f"Original incidence matrix shape: {incidence_matrix.shape}")
    num_nodes = incidence_matrix.shape[0]

    # 保存原始关联矩阵的副本，用于后续实验
    original_incidence_matrix = incidence_matrix.copy()

    nu_vals = np.arange(1.0, 2.1, 0.1)
    # 根据网络规模动态调整λ值
    if num_nodes < 100:
        lambda_vals = 0.02
    elif num_nodes < 500:
        lambda_vals = 0.06
    elif num_nodes < 1000:
        lambda_vals = 0.8
    elif num_nodes < 2000:
        lambda_vals = 0.2
    else:
        lambda_vals = 0.4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 初始阶段：基于五个特征值的初版top-k连通性比较 ===
    print("=== 初始阶段：基于五个特征值的初版top-k连通性比较 ===")

    # 使用原始矩阵的副本来进行连通性比较
    connectivity_matrix = original_incidence_matrix.copy()

    # 计算基线算法分数
    baseline_scores = cache_baseline_scores(connectivity_matrix)

    # 计算初版增强算法分数（基于五个特征的组合）
    scores_hdc = compute_hdc(connectivity_matrix)
    scores_rwhc = RWHCCalculator(connectivity_matrix).calculate_rwhc()
    scores_rwiec = RWIECalculator(connectivity_matrix).calculate_rwiec()
    scores_motif = compute_motif_coefficient(connectivity_matrix)
    scores_overlap = compute_overlap_degree(connectivity_matrix)
    scores_pagerank = compute_pagerank(incidence_matrix)


    # 归一化特征
    def normalize_scores(scores):
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)


    normalized_hdc = normalize_scores(scores_hdc)
    normalized_rwhc = normalize_scores(scores_rwhc)
    normalized_rwiec = normalize_scores(scores_rwiec)
    normalized_motif = normalize_scores(scores_motif)
    normalized_overlap = normalize_scores(scores_overlap)
    normalized_pagerank = normalize_scores(scores_pagerank)

    # 初始组合分数（平均权重）
    initial_enhanced_scores = (normalized_hdc + normalized_rwhc + normalized_rwiec + normalized_motif + normalized_overlap + normalized_pagerank) / 6

    # 定义要删除的比例
    removal_ratios = np.arange(0.1, 1.1, 0.1)

    # 初始连通性比较
    initial_connectivity_results = compare_connectivity_across_algorithms(
        connectivity_matrix, baseline_scores, initial_enhanced_scores, removal_ratios, "Initial"
    )
    # === 训练前的关键节点对比 ===
    print("\n=== 训练前关键节点对比 ===")
    initial_methods = analyze_critical_nodes_comparison(
        original_incidence_matrix, baseline_scores, initial_enhanced_scores, top_k=15
    )



    # === 后续实验步骤使用原始矩阵 ===
    print("=== 生成训练数据 ===")
    # 使用原始矩阵进行后续实验
    features = compute_features(original_incidence_matrix)
    if isinstance(features, list):
        features = features[0]
    print(f"Features shape: {features.shape}")
    features = torch.tensor(features, dtype=torch.float, device=device)

    data = Data(
        x=features,
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    unig_encoder = Unigencoder(
        in_channels=features.shape[1],
        hidden_channels=512,
        out_channels=256,
        num_layers=2,
        dropout=0.4
    )
    print("Expanding data with UniGEncoder...")
    data = unig_encoder.d_expansion(data)

    # 生成训练数据使用原始矩阵
    all_seed_sets, all_nu_values, all_infected_fracs = prepare_enhanced_training_data(
        original_incidence_matrix, lambda_vals, nu_vals, top_k_ratio=0.05
    )

    # 转换为tensor并移动到设备
    data.seed_sets = all_seed_sets
    data.seed_set_nu = torch.tensor(all_nu_values, dtype=torch.float, device=device)
    data.seed_set_labels = torch.tensor(all_infected_fracs, dtype=torch.float, device=device)
    print(f"生成了 {len(all_seed_sets)} 个高质量训练样本")

    # complexity_adequate = assess_network_complexity(original_incidence_matrix)
    # if not complexity_adequate:
    #     print("网络复杂度较低，建议使用标准模型")
    #     # 使用简化版模型
    #     model = NonlinearDiffusionAwareModel(
    #         in_channels=features.shape[1],
    #         hidden_channels=128,
    #         out_channels=1,
    #         num_raw_features=5,
    #         num_heads=3
    #     )
    # else:
    #     print("网络复杂度足够，使用增强模型")
    #     model = NonlinearDiffusionAwareModel(
    #         in_channels=features.shape[1],
    #         hidden_channels=256,
    #         out_channels=1,
    #         num_raw_features=5,
    #         num_heads=4
    #     )

    tuner = HyperparameterTuner(incidence_matrix, data, nu_vals)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    # 临时关闭optuna的进度输出
    import optuna.logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("开始超参数调优...")
    best_params = tuner.tune(n_trials=50)
    # 调优完成后恢复
    optuna.logging.set_verbosity(optuna.logging.INFO)
    if not best_params or any(key not in best_params for key in ['hidden_channels', 'num_heads']):
        print("智能调参失败，使用默认参数")
        best_params = {'hidden_channels': 128, 'num_layers': 2, 'num_heads': 4,
                       'dropout': 0.3, 'lr': 0.001, 'top_k_ratio': 0.05}
    model = NonlinearDiffusionAwareModel(
        in_channels=data.x.shape[1],
        hidden_channels=best_params['hidden_channels'],
        out_channels=1,
        num_raw_features=6,
        num_heads=best_params['num_heads']
    )

    print("=== 训练模型 ===")
    trained_model = train_enhanced_model(
        model, data, original_incidence_matrix, train_idx=range(len(all_seed_sets)),
        epochs=300, lr=0.0005, patience=50
    )

    print("=== 评估模型性能 ===")
    results, all_seed_sets = evaluate_enhanced_model(trained_model, original_incidence_matrix, data, nu_vals,
                                                     lambda_vals, top_k_ratio=0.05)

    # === 训练后阶段：基于训练后模型的关键节点排序连通性比较 ===
    print("=== 训练后阶段：基于训练后模型的关键节点排序连通性比较 ===")

    # 使用原始矩阵的副本进行训练后连通性比较
    trained_connectivity_matrix = original_incidence_matrix.copy()

    # 获取训练后模型的节点分数（使用nu=1.5作为代表值）
    trained_model.eval()
    with torch.no_grad():
        node_degrees = torch.tensor(compute_hdc(trained_connectivity_matrix), dtype=torch.float, device=device).view(-1,
                                                                                                                     1)
        main_scores, _ = trained_model(data, 1.5, node_degrees)
        trained_enhanced_scores = main_scores.cpu().numpy().flatten()

    # 重新计算基线分数（使用训练后连通性矩阵）
    trained_baseline_scores = {}
    trained_baseline_scores['DC'] = compute_dc(trained_connectivity_matrix)
    trained_baseline_scores['BC'] = compute_bc(trained_connectivity_matrix)
    trained_baseline_scores['HDC'] = compute_hdc(trained_connectivity_matrix)
    trained_baseline_scores['SC'] = compute_sc(trained_connectivity_matrix)

    # 训练后连通性比较
    trained_connectivity_results = compare_connectivity_across_algorithms(
        trained_connectivity_matrix, trained_baseline_scores, trained_enhanced_scores, removal_ratios, "Trained"
    )
    # === 新增：训练后的关键节点对比 ===
    print("\n=== 训练后关键节点对比 ===")
    trained_methods = analyze_critical_nodes_comparison(
        trained_connectivity_matrix, trained_baseline_scores, trained_enhanced_scores, top_k=15
    )

    # 分析训练前后的变化
    print("\n=== 训练前后关键节点变化分析 ===")
    enhanced_before = set(initial_methods['NDA-HGNN'])
    enhanced_after = set(trained_methods['NDA-HGNN'])
    changed_nodes = enhanced_before.symmetric_difference(enhanced_after)
    print(f"训练前后NDA-HGNN关键节点变化数量: {len(changed_nodes)}")
    print(f"新增节点: {enhanced_after - enhanced_before}")
    print(f"减少节点: {enhanced_before - enhanced_after}")

    print("=== 生成综合评估图表 ===")
    plot_enhanced_results(nu_vals, results, all_seed_sets)

    # 打印详细结果
    print("\n=== 详细性能分析 ===")
    for nu in nu_vals:
        print(f"\nν = {nu:.1f}:")
        for method in ['NDA-HGNN', 'DC', 'BC', 'HDC', 'SC']:
            data = results[nu][method]
            print(f"  {method}: {data['final_fraction']:.4f} "
                  f"(50%时间: {data['dynamics']['time_to_half']:.0f})")
