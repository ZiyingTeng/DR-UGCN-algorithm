import numpy as np
from scipy.sparse import issparse

class HypergraphRandomWalk:
    def __init__(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix
        self.n_nodes, self.n_edges = incidence_matrix.shape
        self.node_degrees = np.sum(incidence_matrix, axis=1)
        if issparse(incidence_matrix):
            self.node_degrees = self.node_degrees.A1 if hasattr(self.node_degrees, 'A1') else self.node_degrees
        self.edge_degrees = np.sum(incidence_matrix, axis=0)
        if issparse(incidence_matrix):
            self.edge_degrees = self.edge_degrees.A1 if hasattr(self.edge_degrees, 'A1') else self.edge_degrees
        if np.all(self.edge_degrees == 0):
            raise ValueError("Hypergraph has no edges, cannot build transition matrix")
        self.transition_matrix = self.build_transition_matrix()

    def build_transition_matrix(self):
        """基于随机游走的转移矩阵"""
        transition_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for v in range(self.n_nodes):
            if issparse(self.incidence_matrix):
                edges_v = self.incidence_matrix[v].nonzero()[1]
            else:
                edges_v = np.where(self.incidence_matrix[v])[0]
            if len(edges_v) == 0:
                continue  # 孤立节点
            # p(e|v)
            edge_probs = self.edge_degrees[edges_v] / (np.sum(self.edge_degrees[edges_v]) + 1e-10)
            for e_idx, e in enumerate(edges_v):
                if issparse(self.incidence_matrix):
                    nodes_e = self.incidence_matrix[:, e].nonzero()[0]
                else:
                    nodes_e = np.where(self.incidence_matrix[:, e])[0]
                if len(nodes_e) == 0:
                    continue  # 空
                # p(u|e)
                node_probs = self.node_degrees[nodes_e] / (np.sum(self.node_degrees[nodes_e]) + 1e-10)
                for u_idx, u in enumerate(nodes_e):
                    # 转移概率 p(v→u)
                    transition_matrix[v, u] += edge_probs[e_idx] * node_probs[u_idx]
        return transition_matrix

    def calculate_stationary_distribution(self, max_iter=1000, tol=1e-6, alpha=0.85):
        """静态分布，使用PageRank平滑"""
        pi = np.ones(self.n_nodes) / self.n_nodes
        for _ in range(max_iter):
            pi_new = alpha * (pi @ self.transition_matrix) + (1 - alpha) / self.n_nodes
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new
        return pi