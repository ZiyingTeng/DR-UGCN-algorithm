import numpy as np
from scipy.sparse import issparse
from randomwalk import HypergraphRandomWalk


def calculate_rwhc_with_decay(self, distance_decay=1.25, max_iter=1000, tol=1e-6, lambda_val=5e-3, alpha=0.7):
    """计算RWHC"""
    heat = self.initialize_heat(lambda_val, distance_decay)
    laplacian = self.build_laplacian_matrix()
    initial_heat = heat.copy()

    for _ in range(max_iter):
        heat_new = alpha * (heat - laplacian @ heat) + (1 - alpha) * initial_heat
        if np.linalg.norm(heat_new - heat) < tol:
            break
        heat = heat_new

    return heat

class RWHCCalculator:
    def __init__(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix
        self.n_nodes, self.n_edges = incidence_matrix.shape
        if self.n_edges == 0:
            raise ValueError("No hyperedges in incidence matrix")
        self.node_degrees = np.sum(incidence_matrix, axis=1)
        if issparse(incidence_matrix):
            self.node_degrees = self.node_degrees.A1 if hasattr(self.node_degrees, 'A1') else self.node_degrees.ravel()
        self.edge_degrees = np.sum(incidence_matrix, axis=0)
        if issparse(incidence_matrix):
            self.edge_degrees = self.edge_degrees.A1 if hasattr(self.edge_degrees, 'A1') else self.edge_degrees.ravel()
        self.rw = HypergraphRandomWalk(incidence_matrix)
        self.transition_matrix = self.rw.transition_matrix

    def calculate_edge_mass(self):
        """计算超边质量"""
        avg_node_degree = np.mean(self.node_degrees)
        edge_mass = np.sqrt(self.edge_degrees ** 2 + avg_node_degree ** 2)
        return edge_mass

    def calculate_edge_distance(self):
        """计算Jaccard距离"""
        edge_distance = np.ones((self.n_edges, self.n_edges)) * np.inf
        for i in range(self.n_edges):
            for j in range(self.n_edges):
                if i == j:
                    edge_distance[i, j] = 0
                    continue
                if issparse(self.incidence_matrix):
                    nodes_i = self.incidence_matrix[:, i].nonzero()[0]
                    nodes_j = self.incidence_matrix[:, j].nonzero()[0]
                else:
                    nodes_i = np.where(self.incidence_matrix[:, i])[0]
                    nodes_j = np.where(self.incidence_matrix[:, j])[0]
                intersection = len(set(nodes_i) & set(nodes_j))
                union = len(set(nodes_i) | set(nodes_j))
                if union > 0:
                    edge_distance[i, j] = 1 - intersection / union
        return edge_distance

    def calculate_edge_force(self, edge_mass, edge_distance, lambda_val=5e-3, theta=1.25):
        """基于重力模型的超边作用力"""
        edge_force = np.zeros((self.n_edges, self.n_edges))
        for i in range(self.n_edges):
            for j in range(self.n_edges):
                if edge_distance[i, j] > 0 and edge_distance[i, j] < np.inf:
                    edge_force[i, j] = lambda_val * (edge_mass[i] * edge_mass[j]) / (edge_distance[i, j] ** theta + 1e-10)
        return edge_force

    def build_laplacian_matrix(self):
        pi = self.rw.calculate_stationary_distribution(alpha=0.85)
        phi = np.diag(pi)
        laplacian = phi - 0.5 * (phi @ self.transition_matrix + self.transition_matrix.T @ phi)
        return laplacian

    def initialize_heat(self, lambda_val=5e-3, distance_decay=1.25):
        """初始化热力值"""
        heat = np.zeros(self.n_nodes)
        edge_mass = self.calculate_edge_mass()
        edge_distance = self.calculate_edge_distance()
        edge_force = self.calculate_edge_force(edge_mass, edge_distance, lambda_val, distance_decay)
        edge_importance = np.sum(edge_force, axis=1)

        for e in range(self.n_edges):
            if issparse(self.incidence_matrix):
                nodes_e = self.incidence_matrix[:, e].nonzero()[0]
            else:
                nodes_e = np.where(self.incidence_matrix[:, e])[0]
            if len(nodes_e) > 0:
                weights = self.node_degrees[nodes_e] / (np.sum(self.node_degrees[nodes_e]) + 1e-10)
                for u_idx, u in enumerate(nodes_e):
                    heat[u] += edge_importance[e] * weights[u_idx]

        return heat

    def calculate_rwhc(self, max_iter=1000, tol=1e-6, lambda_val=5e-3, theta=1.25, alpha=0.7):
        """计算RWHC，保持数值幅度"""
        heat = self.initialize_heat(lambda_val, theta)
        laplacian = self.build_laplacian_matrix()
        initial_heat = heat.copy()

        for _ in range(max_iter):
            heat_new = alpha * (heat - laplacian @ heat) + (1 - alpha) * initial_heat

            # 移除迭代中的归一化
            # if np.sum(heat_new) > 0:
            #     heat_new /= np.sum(heat_new)

            if np.linalg.norm(heat_new - heat) < tol:
                break
            heat = heat_new

        return heat