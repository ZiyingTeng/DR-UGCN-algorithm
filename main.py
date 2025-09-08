# θ从1到2

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from UniGGCN import UniGCNRegression
from new_aware_model import NuAwareUniGCN
from rwhc import RWHCCalculator
from rwiec import RWIECalculator
from utils import load_hypergraph, compute_features, split_dataset, compute_infected_fraction, cache_baseline_scores, \
    load_hypergraph_pickle, prepare_multi_nu_training_data, create_nu_specific_labels, \
    create_multi_nu_simulation_labels, create_high_quality_multi_nu_labels
from baseline import compute_hdc, compute_dc, compute_bc, compute_sc
import matplotlib.pyplot as plt
import pickle
import os
import random
from unigencoder import Unigencoder

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def focused_nu_training(model, data, nu_values, Y_real, focus_nu=1.4, epochs=200, lr=0.001):
    """专注于特定ν值的训练，同时保持整体性能"""
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
            # 后期：50%概率训练focus_nu，50%概率训练其他ν
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

        # 早停策略：基于整体性能
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


def prepare_training_data(incidence_matrix, features, edge_index, top_k, lambda_val, nu_vals):
    """使用多种中心性组合生成训练数据"""
    num_nodes = incidence_matrix.shape[0]
    scores_hdc = compute_hdc(incidence_matrix)
    rwhc_calc = RWHCCalculator(incidence_matrix)
    scores_rwhc = rwhc_calc.calculate_rwhc()
    rwiec_calc = RWIECalculator(incidence_matrix)
    scores_rwiec = rwiec_calc.calculate_rwiec(gamma=1.0)

    # 动态加权组合
    composite_scores = []
    for nu in nu_vals:
        if nu < 1.3:
            # 低非线性：偏向HDC和RWHC
            weights = [0.5, 0.3, 0.2]
        elif nu < 1.7:
            # 中等非线性
            weights = [0.4, 0.3, 0.3]
        else:
            # 高非线性：偏向RWIEC
            weights = [0.3, 0.3, 0.4]

        composite_score = (weights[0] * scores_hdc +
                           weights[1] * scores_rwhc +
                           weights[2] * scores_rwiec)
        composite_scores.append(composite_score)

    # 平均
    avg_scores = np.mean(composite_scores, axis=0)
    top_nodes = np.argsort(avg_scores)[-top_k:]

    y = np.zeros(num_nodes)
    for nu in nu_vals:
        infected_frac = compute_infected_fraction(
            incidence_matrix, top_nodes, lambda_val, nu,
            mu=1.0, num_runs=5
        )
        y[top_nodes] += infected_frac

    y[top_nodes] /= len(nu_vals)
    return torch.tensor(y, dtype=torch.float).reshape(-1, 1)





# def evaluate_algorithms(incidence_matrix, features, edge_index, top_k, lambda_vals, nu_vals):
#     set_seed(42)
#     num_nodes = incidence_matrix.shape[0]
#     print(f"Number of nodes: {num_nodes}, Top k: {top_k}")
#
#     # 分析特征权重分布
#     print("\n=== Feature Weight Analysis ===")
#     for nu in [1.0, 1.3, 1.6, 2.0]:
#         weights = get_feature_weights(nu)
#         print(f"nu={nu:.1f}: {weights}")
#
#     lambda_val = lambda_vals[0]
#     y = prepare_training_data(incidence_matrix, features, edge_index, top_k, lambda_val, nu_vals)
#
#     data = Data(
#         x=features,
#         edge_index=edge_index,
#         y=y,
#         num_nodes=features.shape[0]  # 确保有 num_nodes 属性
#     )
#
#     model = UniGCNRegression(
#         in_channels=features.shape[1],
#         hidden_channels=128,
#         out_channels=1,
#         num_layers=3,
#         dropout=0.3
#     )
#
#     data = model.unig_encoder.d_expansion(data)
#
#     train_idx, val_idx, test_idx = split_dataset(num_nodes)
#
#     print("Training DR-UGCN model...")
#     model, train_losses, val_losses = train_model(model, data, train_idx, val_idx, epochs=200, lr=0.0001, patience=20)
#     model.eval()
#     cache_file = "dr_ugcn_scores.pkl"
#
#     with torch.no_grad():
#         scores_dr_ugcn = model(data).squeeze().cpu().numpy()
#         scores_dr_ugcn = np.clip(scores_dr_ugcn, 0, 1)
#
#     with open(cache_file, 'wb') as f:
#         pickle.dump(scores_dr_ugcn, f)
#
#     scores_dc, scores_bc, scores_hdc, scores_sc = cache_baseline_scores(incidence_matrix)
#
#     results = {alg: [] for alg in ['DR-UGCN', 'DC', 'BC', 'HDC', 'SC']}
#
#     for nu in nu_vals:
#         print(f"\nEvaluating at nu={nu:.1f}")
#         for alg, scores in [
#             ('DR-UGCN', scores_dr_ugcn),
#             ('DC', scores_dc),
#             ('BC', scores_bc),
#             ('HDC', scores_hdc),
#             ('SC', scores_sc)
#         ]:
#             current_top_k = min(top_k, len(scores))
#             top_nodes = np.argsort(scores)[-current_top_k:]
#
#             infected_frac = compute_infected_fraction(
#                 incidence_matrix, top_nodes, lambda_vals[0], nu,
#                 mu=1.0, num_runs=10
#             )
#             results[alg].append(infected_frac)
#             print(f"  {alg}: infected_frac={infected_frac:.4f}")
#
#     return results, train_losses, val_losses


def evaluate_enhanced_model(model, incidence_matrix, data, nu_values, lambda_val, top_k):
    """评估改进的Nu-Aware模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    scores_dc, scores_bc, scores_hdc, scores_sc = cache_baseline_scores(incidence_matrix)

    results = {alg: [] for alg in ['Enhanced-NuGNN', 'DC', 'BC', 'HDC', 'SC']}

    model.eval()
    with torch.no_grad():
        for nu in nu_values:
            print(f"Evaluating at nu={nu:.1f}")

            scores_enhanced = model(data, nu).squeeze().cpu().numpy()

            algorithm_scores = [
                ('Enhanced-NuGNN', scores_enhanced),
                ('DC', scores_dc),
                ('BC', scores_bc),
                ('HDC', scores_hdc),
                ('SC', scores_sc)
            ]

            for alg, scores in algorithm_scores:
                current_top_k = min(top_k, len(scores))
                top_nodes = np.argsort(scores)[-current_top_k:]

                infected_frac = compute_infected_fraction(
                    incidence_matrix, top_nodes, lambda_val, nu,
                    mu=1.0, num_runs=5
                )
                results[alg].append(infected_frac)
                print(f"  {alg}: infected_frac={infected_frac:.4f}")

    return results

# def evaluate_nu_aware_algorithms(incidence_matrix, features, edge_index, top_k, lambda_vals, nu_vals):
#     num_nodes = incidence_matrix.shape[0]
#
#     # 初始化模型
#     model = EnhancedNuAwareHyperGNN(
#         in_channels=features.shape[1],
#         hidden_channels=128,
#         out_channels=1,
#         num_nu_bins=len(nu_vals)
#     )
#
#     # 准备训练数据（使用多个θ值的混合目标）
#     lambda_val = lambda_vals[0]
#     y = prepare_multi_nu_training_data(incidence_matrix, top_k, lambda_val, nu_vals)
#     y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
#
#     data = Data(x=features, edge_index=edge_index, y=y, num_nodes=num_nodes)
#
#     train_idx, val_idx, test_idx = split_dataset(num_nodes)
#
#     print("Training Nu-aware model...")
#     model, train_losses, val_losses = stable_nu_aware_training(
#         model, data, train_idx, val_idx, nu_vals, epochs=150, lr=0.001
#     )
#
#     model.eval()
#     results = {alg: [] for alg in ['DR-UGCN', 'DC', 'BC', 'HDC', 'SC']}
#
#     with torch.no_grad():
#         for nu in nu_vals:
#             scores_dr_ugcn = model(data, nu).squeeze().cpu().numpy()
#
#             for alg, scores in [
#                 ('DR-UGCN', scores_dr_ugcn),
#                 ('DC', compute_dc(incidence_matrix)),
#                 ('BC', compute_bc(incidence_matrix)),
#                 ('HDC', compute_hdc(incidence_matrix)),
#                 ('SC', compute_sc(incidence_matrix))
#             ]:
#                 top_nodes = np.argsort(scores)[-top_k:]
#                 infected_frac = compute_infected_fraction(
#                     incidence_matrix, top_nodes, lambda_val, nu, mu=1.0, num_runs=5
#                 )
#                 results[alg].append(infected_frac)
#                 print(f"  {alg}: infected_frac={infected_frac:.4f}")
#
#     return results, train_losses, val_losses

def plot_results(nu_vals, results, train_losses=None, val_losses=None):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    colors = {'Enhanced-NuGNN': 'blue', 'DC': 'green', 'BC': 'red', 'HDC': 'purple', 'SC': 'orange'}

    for alg, fractions in results.items():
        plt.plot(nu_vals, fractions, label=alg, color=colors[alg], marker='o')

    plt.xlabel('Nonlinearity Degree (θ)')
    plt.ylabel('Infected Fraction')
    plt.title('Senate')
    plt.xticks(nu_vals)
    plt.legend()
    plt.grid(True)

    if train_losses and val_losses:
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('senate2.82.png')
    plt.show()


if __name__ == "__main__":
    # 清理旧缓存
    for cache_file in ["dr_ugcn_scores2.pkl", "baseline_scores.pkl"]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"Deleted cache file: {cache_file}")
            except PermissionError:
                print(f"Warning: Cannot remove {cache_file}, file in use")

    file_path = "hyperedges-senate-committees.txt"
    incidence_matrix, edge_index, node_id_map = load_hypergraph(file_path)

    print(f"Original incidence matrix shape: {incidence_matrix.shape}")

    num_nodes = incidence_matrix.shape[0]
    top_k = int(0.08 * num_nodes)
    nu_vals = np.arange(1.0, 2.1, 0.1)
    lambda_vals = [0.02]

    features = compute_features(incidence_matrix)
    print(f"Features shape: {features.shape}")

    # 创建基础数据对象
    data = Data(
        x=features,
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    # 使用 UniGEncoder 进行数据扩展（生成 Pv 和 PvT）
    unig_encoder = Unigencoder(
        in_channels=features.shape[1],
        hidden_channels=128,
        out_channels=128,
        num_layers=1,
        dropout=0.3
    )

    print("Expanding data with UniGEncoder...")
    data = unig_encoder.d_expansion(data)  # 这会添加 Pv 和 PvT 属性

    # 1. 生成高质量标签
    print("=== Generating High-Quality Simulation Labels ===")
    Y_high_quality = create_high_quality_multi_nu_labels(
        incidence_matrix, lambda_vals[0], nu_vals, top_k_ratio=0.1
    )

    # 2. 添加标签到数据
    y_avg = np.mean(Y_high_quality, axis=1)
    data.y = torch.tensor(y_avg, dtype=torch.float).view(-1, 1)
    data.Y_real = torch.tensor(Y_high_quality, dtype=torch.float)

    # 3. 初始化新模型
    model = NuAwareUniGCN(
        in_channels=features.shape[1],
        hidden_channels=128,
        out_channels=1
    )

    print(f"Model initialized with in_channels={features.shape[1]}")

    # 4. 简化训练
    print("=== Simplified Nu-Aware Training ===")
    trained_model = focused_nu_training(
        model, data, nu_vals, Y_high_quality,
        focus_nu=1.4, epochs=200, lr=0.001
    )

    # 5. 评估
    results = evaluate_enhanced_model(trained_model, incidence_matrix, data, nu_vals, lambda_vals[0], top_k)
    plot_results(nu_vals, results)

    # print(f"Incidence matrix shape: {incidence_matrix.shape}")
    # print(f"Features shape: {features.shape}")
    # print(f"Edge index shape: {edge_index.shape}")
    #
    # # results, train_losses, val_losses = evaluate_algorithms(
    # #     incidence_matrix, features, edge_index, top_k, lambda_vals, nu_vals
    # # )
    #
    # print(f"Creating nu-specific training labels...")
    #
    # nu_specific_labels = create_nu_specific_labels(
    #     incidence_matrix, nu_vals, lambda_vals[0], 0.05
    # )
    #
    # y_multi_nu = np.mean(nu_specific_labels, axis=1)
    # y_tensor = torch.tensor(y_multi_nu, dtype=torch.float).reshape(-1, 1)
    #
    # data = Data(
    #     x=features,
    #     edge_index=edge_index,
    #     y=y_tensor,
    #     num_nodes=num_nodes,
    #     nu_specific_labels=torch.tensor(nu_specific_labels, dtype=torch.float)
    # )
    #
    # model = EnhancedNuAwareHyperGNN(
    #     in_channels=features.shape[1],
    #     hidden_channels=128,  # 这里需要明确指定hidden_channels大小
    #     out_channels=1
    # )
    #
    # train_idx, val_idx, test_idx = split_dataset(num_nodes)
    #
    # print("Starting stable nu-aware training...")
    # trained_model, train_losses, val_losses = stable_nu_aware_training(
    #     model, data, train_idx, val_idx, nu_vals, epochs=200, lr=0.0005
    # )
    #
    # results = evaluate_enhanced_model(
    #     trained_model, incidence_matrix, data, nu_vals, lambda_vals[0], top_k
    # )
    #
    # plot_results(nu_vals, results, train_losses, val_losses)

