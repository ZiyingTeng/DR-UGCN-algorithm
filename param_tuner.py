import optuna
import numpy as np
import torch
from baseline import compute_hdc, compute_bc, compute_sc
from connectivity import hypergraph_natural_connectivity
from NDA_HGNN_model import *
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from nonlinear import HypergraphContagion
from rwhc import RWHCCalculator
from rwiec import RWIECalculator
from utils import *


def compute_infection_loss_2(model, data, incidence_matrix, nu, lambda_val=0.1, top_k_ratio=0.05):
    model.eval()
    with torch.no_grad():
        node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
        main_scores, _ = model(data, nu, node_degrees)
        scores = main_scores.cpu().numpy().flatten()
        num_nodes = incidence_matrix.shape[0]
        top_k = int(num_nodes * top_k_ratio)
        seed_nodes = np.argsort(scores)[-top_k:]
        initial_infected = np.zeros(num_nodes)
        initial_infected[seed_nodes] = 1
        contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu)
        simulated_fraction, _, _ = contagion.simulate(return_curve=True)
        target_fraction = np.mean(simulated_fraction[-100:]) if len(simulated_fraction) > 100 else np.mean(simulated_fraction)
        return torch.tensor(1 - target_fraction)

def compute_infection_loss(model, data, incidence_matrix, nu, lambda_val=0.1, top_k_ratio=0.05):
    """计算感染损失"""
    model.train()  # 保留梯度
    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
    main_scores, _ = model(data, nu, node_degrees)
    scores = main_scores.cpu().detach().numpy().flatten()
    num_nodes = incidence_matrix.shape[0]
    top_k = int(num_nodes * top_k_ratio)
    seed_nodes = np.argsort(scores)[-top_k:]

    # 感染模拟
    with torch.no_grad():
        infected_frac = compute_infected_fraction(
            incidence_matrix, seed_nodes, lambda_val, nu, num_runs=3
        )

    # 创建可求导的损失张量
    infection_loss = 1 - torch.tensor(infected_frac, device=data.x.device, requires_grad=True)
    return infection_loss


def compute_proxy_infection_loss(model, data, incidence_matrix, nu):
    """感染率代理损失"""
    model.train()

    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
    main_scores, _ = model(data, nu, node_degrees)

    hdc = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=main_scores.device)
    rwhc = torch.tensor(RWHCCalculator(incidence_matrix).calculate_rwhc(), dtype=torch.float, device=main_scores.device)
    pagerank = torch.tensor(compute_pagerank(incidence_matrix), dtype=torch.float, device=main_scores.device)

    if nu < 1.3:
        # 低θ：传播主要依赖直接连接，侧重度中心性
        base_weights = torch.tensor([0.5, 0.3, 0.2], device=main_scores.device)  # HDC, RWHC, PageRank
        interaction_weight = 0.1
    elif nu < 1.7:
        # 中θ：平衡各种机制
        base_weights = torch.tensor([0.4, 0.4, 0.2], device=main_scores.device)
        interaction_weight = 0.15
    else:
        # 高θ：集体感染效应强，侧重随机游走
        base_weights = torch.tensor([0.3, 0.5, 0.2], device=main_scores.device)
        interaction_weight = 0.2

    hdc_rwhc_interaction = hdc * rwhc  # 度中心性与随机游走的协同 特征交互

    proxy_target = (
            base_weights[0] * hdc +
            base_weights[1] * rwhc +
            base_weights[2] * pagerank +
            interaction_weight * hdc_rwhc_interaction
    )

    node_degrees_np = np.sum(incidence_matrix, axis=1).A1
    adj = incidence_matrix.dot(incidence_matrix.T) - sparse.diags(node_degrees_np)

    adj_dense = torch.tensor(adj.toarray(), dtype=torch.float, device=main_scores.device)

    # 计算邻居的平均影响力
    neighbor_influence = torch.matmul(adj_dense, proxy_target.unsqueeze(1)).squeeze()

    # 邻居度归一化
    neighbor_degrees = torch.sum(adj_dense, dim=1)
    neighbor_influence = neighbor_influence / (neighbor_degrees + 1e-10)

    # 自身特征 + 邻居影响
    proxy_target = 0.7 * proxy_target + 0.3 * neighbor_influence

    proxy_target = (proxy_target - proxy_target.min()) / (proxy_target.max() - proxy_target.min() + 1e-10)  # 防止梯度爆炸

    loss = F.mse_loss(main_scores.squeeze(), proxy_target)
    return loss


def compute_connectivity_loss_2(model, data, incidence_matrix, nu, top_k_ratio=0.05, alpha=0.3):
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
        return -alpha * torch.log(torch.tensor(drop + 1e-10))

def compute_connectivity_loss(model, data, incidence_matrix, nu, top_k_ratio=0.05, alpha=0.3):
    """计算连通性损失，保留梯度信息"""
    model.train()

    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
    main_scores, _ = model(data, nu, node_degrees)
    scores = main_scores.cpu().detach().numpy().flatten()
    num_nodes = incidence_matrix.shape[0]
    top_k = int(num_nodes * top_k_ratio)
    sorted_nodes = np.argsort(scores)[-top_k:]

    with torch.no_grad():
        nodes_to_keep = np.setdiff1d(np.arange(num_nodes), sorted_nodes)
        if len(nodes_to_keep) == 0:
            return torch.tensor(0.0, device=data.x.device, requires_grad=True)

        modified_matrix = incidence_matrix[nodes_to_keep, :]
        edge_sums = np.array(modified_matrix.sum(axis=0)).flatten()
        non_zero_edges = edge_sums > 0
        modified_matrix = modified_matrix[:, non_zero_edges]

        if modified_matrix.shape[1] == 0:
            connectivity_after = 0.0
        else:
            connectivity_after = hypergraph_natural_connectivity(modified_matrix)

            connectivity_original = hypergraph_natural_connectivity(incidence_matrix)
            if len(nodes_to_keep) == 0:
                return torch.tensor(0.0)

            modified_matrix = incidence_matrix[nodes_to_keep, :]
            edge_sums = np.array(modified_matrix.sum(axis=0)).flatten()
            non_zero_edges = edge_sums > 0
            modified_matrix = modified_matrix[:, non_zero_edges]

            if modified_matrix.shape[0] < 2 or modified_matrix.shape[1] == 0:
                return torch.tensor(0.0)  # 无法计算连通性

            try:
                connectivity_after = hypergraph_natural_connectivity(modified_matrix)
            except Exception as e:
                print(f"连通性计算错误: {e}")
                return torch.tensor(0.0)

            # 确保下降值合理
            drop = (connectivity_original - connectivity_after) / (connectivity_original + 1e-10)
            drop = np.clip(drop, 0.0, 1.0)

            return -alpha * torch.log(torch.tensor(drop + 1e-10))


def compute_proxy_connectivity_loss(model, data, incidence_matrix, nu):
    model.train()

    node_degrees = torch.tensor(compute_hdc(incidence_matrix), dtype=torch.float, device=data.x.device).view(-1, 1)
    main_scores, _ = model(data, nu, node_degrees)

    # 选择对网络结构影响大的节点
    rwiec = torch.tensor(RWIECalculator(incidence_matrix).calculate_rwiec(), dtype=torch.float,
                         device=main_scores.device)
    overlap = torch.tensor(compute_overlap_degree(incidence_matrix), dtype=torch.float, device=main_scores.device)
    motif = torch.tensor(compute_motif_coefficient(incidence_matrix), dtype=torch.float, device=main_scores.device)

    # 连通性重要性组合：
    # - RWIEC: 信息熵高表示结构关键位置
    # - Overlap: 重叠度高表示桥梁作用强
    # - Motif: motif参与度高表示局部结构重要
    connectivity_importance = 0.4 * rwiec + 0.4 * overlap + 0.2 * motif

    # 归一化
    connectivity_importance = (connectivity_importance - connectivity_importance.min()) / (
            connectivity_importance.max() - connectivity_importance.min() + 1e-10)

    loss = F.mse_loss(main_scores.squeeze(), connectivity_importance)
    return loss

class HyperparameterTuner:
    def __init__(self, incidence_matrix, data, nu_values):
        self.incidence_matrix = incidence_matrix
        self.data = data
        self.nu_values = nu_values
        self.best_score = -float('inf')
        self.best_params = None

    def objective(self, trial):
        # 关键参数搜索空间
        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 2, 5),
            'num_heads': trial.suggest_int('num_heads', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            # 'alpha_aux': trial.suggest_float('alpha_aux', 0.05, 0.3),  # 辅助损失权重
            'top_k_ratio': trial.suggest_float('top_k_ratio', 0.02, 0.07),
        }

        # 根据网络大小调整参数范围
        num_nodes = self.incidence_matrix.shape[0]
        if num_nodes < 100:
            params['hidden_channels'] = trial.suggest_categorical('hidden_channels_small', [32, 64, 128])
        elif num_nodes > 1000:
            params['hidden_channels'] = trial.suggest_categorical('hidden_channels_large', [256, 512])

        try:
            model = NonlinearDiffusionAwareModel(
                in_channels=self.data.x.shape[1],
                hidden_channels=params['hidden_channels'],
                out_channels=1,
                num_raw_features=6,
                num_heads=params['num_heads']
            )

            # 快速验证
            score = self.quick_validate(model, params)
            return score

        except Exception as e:
            print(f"Trial failed: {e}")
            return -float('inf')

        except Exception as e:
            print(f"Trial failed: {e}")
            return -float('inf')

    def quick_validate(self, model, params):
        """快速验证方法"""
        device = next(model.parameters()).device
        model.to(device)
        self.data = self.data.to(device)

        optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)

        try:
            for epoch in range(20):
                model.train()
                optimizer.zero_grad()

                nu = np.random.choice(self.nu_values)

                # 前向传播
                scores, _ = model(self.data, nu, node_degrees=None)

                infection_loss = compute_infection_loss(model, self.data, self.incidence_matrix, nu,
                                                        top_k_ratio=params['top_k_ratio'])
                connect_loss = compute_connectivity_loss(model, self.data, self.incidence_matrix, nu,
                                                         top_k_ratio=params['top_k_ratio'])

                # 组合损失
                loss = 0.6 * infection_loss + 0.4 * connect_loss

                # 检查损失是否可导
                if not loss.requires_grad:
                    loss = loss.clone().requires_grad_(True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 快速评估
            model.eval()
            total_score = 0
            eval_nu_values = self.nu_values[:3]

            for nu in eval_nu_values:
                with torch.no_grad():
                    scores, _ = model(self.data, nu, node_degrees=None)
                    scores_np = scores.cpu().numpy().flatten()

                    top_k = int(len(scores_np) * params['top_k_ratio'])
                    top_scores = np.sort(scores_np)[-top_k:]
                    avg_top_score = np.mean(top_scores)

                    total_score += avg_top_score

            return total_score / len(eval_nu_values)

        except Exception as e:
            print(f"Validation failed: {e}")
            return -float('inf')

    def tune(self, n_trials=25):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        print("=== 最佳参数组合 ===")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        return study.best_params

