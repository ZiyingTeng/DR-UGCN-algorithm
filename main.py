import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from UniGGCN import UniGCNRegression
from new_aware_model import NuAwareUniGCN, NuAwareUniGCNWithFiLM, EnhancedNuAwareModel
from rwhc import RWHCCalculator
from rwiec import RWIECalculator
from utils import load_hypergraph, compute_features, split_dataset, compute_infected_fraction, cache_baseline_scores, \
    load_hypergraph_pickle,prepare_enhanced_training_data
from baseline import compute_hdc, compute_dc, compute_bc, compute_sc
import matplotlib.pyplot as plt
import pickle
import os
import random
from unigencoder import Unigencoder
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_enhanced_model(model, data, incidence_matrix, train_idx, epochs=300, lr=0.001, patience=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=device).view(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_aux_loss = 0

        # 随机打乱训练样本
        indices = torch.randperm(len(data.seed_sets))

        for idx in indices:
            seed_set = data.seed_sets[idx]
            nu_val = data.seed_set_nu[idx]
            true_score = data.seed_set_labels[idx]

            optimizer.zero_grad()

            # 前向传播
            main_scores, linear_scores = model(data, nu_val, node_degrees)

            # 主损失
            seed_set_scores = main_scores[seed_set]
            predicted_score = torch.sum(seed_set_scores)
            loss_main = F.mse_loss(predicted_score, true_score)

            # 辅助损失
            loss_aux = F.mse_loss(main_scores, linear_scores.detach())

            alpha = 0.1  # 辅助损失的权重
            loss = loss_main + alpha * loss_aux # 这里有一个问题 我不确定将主损失和辅助损失加权结合这一操作是否符合逻辑 是否能够让模型真正学习到其表达的内容

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_main_loss += loss_main.item()
            total_aux_loss += loss_aux.item()

        avg_loss = total_loss / len(indices)
        avg_main_loss = total_main_loss / len(indices)
        avg_aux_loss = total_aux_loss / len(indices)

        if epoch % 20 == 0:
            print(
                f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} (Main: {avg_main_loss:.4f}, Aux: {avg_aux_loss:.4f}) | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_enhanced_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(avg_loss)

    # 加载最佳模型
    model.load_state_dict(torch.load('best_enhanced_model.pth', map_location=device))
    return model


def analyze_auxiliary_weights(model, data, nu_values, incidence_matrix):
    """分析辅助网络学到的特征权重"""
    device = next(model.parameters()).device
    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=device).view(-1, 1)

    print("辅助网络学到的特征权重随θ的变化:")
    feature_names = ['HDC', 'RWHC', 'RWIEC', 'Motif', 'Overlap']

    for nu in nu_values:
        with torch.no_grad():
            _, _, aux_weights = model(data, nu, node_degrees, return_auxiliary=True)

        weights = aux_weights.squeeze().cpu().numpy()
        print(f"ν = {nu:.1f}: {', '.join([f'{name}:{w:.3f}' for name, w in zip(feature_names, weights)])}")


def train_with_seed_sets(model, data, epochs=200, lr=0.001, patience=30, focus_nu_range=(1.3, 1.8)):
    """着眼于节点集的训练"""
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
                loss_rank = ranking_loss(predicted_score, other_score, torch.tensor(1.0 if true_label > other_label else -1.0))
                weight_rank = 0.4 + 0.2 * (nu_val - 1.0)
                loss = (1 - weight_rank) * loss_mse + weight_rank * loss_rank
            else:
                loss = loss_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(indices):.4f}")
        val_loss = 0  # Add validation logic if needed
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
    """专注于特定θ值的训练，同时保持整体性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    Y_real_tensor = torch.tensor(Y_real, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    criterion = nn.MSELoss()

    train_idx, val_idx, test_idx = split_dataset(data.num_nodes)

    focus_idx = np.where(np.isclose(nu_values, focus_nu))[0][0]

    best_val_loss = float('inf')
    best_overall_loss = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        model.train()

        if epoch < epochs // 2:
            # 前期全面训练所有θ值
            nu_idx = np.random.randint(len(nu_values))
        else:
            # 后期50%概率训练focus_nu，50%概率训练其他ν
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


def evaluate_enhanced_model(model, incidence_matrix, data, nu_vals, lambda_val, top_k_base=0.05, num_runs=100):
    results = {}
    for nu in nu_vals:
        top_k_ratio = max(0.02, top_k_base - 0.03 * (nu - 1.0))
        top_k = int(incidence_matrix.shape[0] * top_k_ratio)
        model.eval()
        with torch.no_grad():
            node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1,
                                                                                                                     1)

            main_scores, linear_scores = model(data, nu, node_degrees)
            node_scores = main_scores.cpu().numpy().flatten()

        top_nodes = np.argsort(node_scores)[-top_k:]
        infected_frac = compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=1.0,
                                                  num_runs=num_runs)
        results[nu] = {'Enhanced-NuGNN': infected_frac}

        try:
            baseline_scores = cache_baseline_scores(incidence_matrix)
            for method in ['DC', 'BC', 'HDC', 'SC']:
                scores = baseline_scores[method]
                top_nodes = np.argsort(scores)[-top_k:]
                infected_frac = compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=1.0,
                                                          num_runs=num_runs)
                results[nu][method] = infected_frac
        except KeyError as e:
            print(f"KeyError in baseline scores: {e}. Forcing recalculation...")
            os.remove("baseline_scores.pkl")
            baseline_scores = cache_baseline_scores(incidence_matrix)
            for method in ['DC', 'BC', 'HDC', 'SC']:
                scores = baseline_scores[method]
                top_nodes = np.argsort(scores)[-top_k:]
                infected_frac = compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=1.0,
                                                          num_runs=num_runs)
                results[nu][method] = infected_frac

        print(
            f"nu={nu:.1f}: Enhanced-NuGNN: {results[nu]['Enhanced-NuGNN']:.4f}, DC: {results[nu].get('DC', 0.0):.4f}, BC: {results[nu].get('BC', 0.0):.4f}, HDC: {results[nu].get('HDC', 0.0):.4f}, SC: {results[nu].get('SC', 0.0):.4f}")
    return results

def plot_results(nu_vals, results):
    colors = {
        'Enhanced-NuGNN': 'blue',
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


if __name__ == "__main__":
    set_seed(42)

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

    nu_vals = np.arange(1.0, 2.1, 0.1)
    lambda_vals = [0.04]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = compute_features(incidence_matrix, nu_values=nu_vals)
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
        hidden_channels=256,
        out_channels=256,
        num_layers=1,
        dropout=0.3
    )
    print("Expanding data with UniGEncoder...")
    data = unig_encoder.d_expansion(data)

    print("=== 生成训练数据 ===")
    all_seed_sets, all_nu_values, all_infected_fracs = prepare_enhanced_training_data(
        incidence_matrix, lambda_vals[0], nu_vals, top_k_ratio=0.04
    )

    # 转换为tensor并移动到设备
    data.seed_sets = all_seed_sets
    data.seed_set_nu = torch.tensor(all_nu_values, dtype=torch.float, device=device)
    data.seed_set_labels = torch.tensor(all_infected_fracs, dtype=torch.float, device=device)
    print(f"生成了 {len(all_seed_sets)} 个高质量训练样本")

    model = EnhancedNuAwareModel(
        in_channels=features.shape[1],
        hidden_channels=256,
        out_channels=1,
        num_raw_features=5  # HDC, RWHC, RWIEC, Motif, Overlap
    )

    print("=== 训练模型 ===")
    trained_model = train_enhanced_model(
        model, data, incidence_matrix, train_idx=range(len(all_seed_sets)),
        epochs=300, lr=0.001, patience=50
    )
    
    print("=== 评估模型性能 ===")
    results = evaluate_enhanced_model(trained_model, incidence_matrix, data, nu_vals, lambda_vals[0])

    plot_results(nu_vals, results)

    print("\n=== 分析辅助权重 ===")
    analyze_auxiliary_weights(trained_model, data, nu_vals, incidence_matrix)



