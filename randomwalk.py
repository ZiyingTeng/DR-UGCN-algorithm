import numpy as np
from scipy.sparse import issparse


class HypergraphRandomWalk:
    """Hypergraph random walk model implementation"""

    def __init__(self, incidence_matrix):
        """
        Initialize hypergraph random walk model

        Parameters:
        incidence_matrix: Hypergraph incidence matrix, shape [num_nodes, num_edges], elements 0 or 1
        """
        self.incidence_matrix = incidence_matrix
        self.n_nodes, self.n_edges = incidence_matrix.shape
        self.node_degrees = np.sum(incidence_matrix, axis=1)  # Node hyperdegrees
        if issparse(incidence_matrix):
            self.node_degrees = self.node_degrees.A1 if hasattr(self.node_degrees, 'A1') else self.node_degrees
        self.edge_degrees = np.sum(incidence_matrix, axis=0)  # Hyperedge degrees
        if issparse(incidence_matrix):
            self.edge_degrees = self.edge_degrees.A1 if hasattr(self.edge_degrees, 'A1') else self.edge_degrees
        self.transition_matrix = self.build_transition_matrix()

    def build_transition_matrix(self):
        """
        Build the transition matrix for hypergraph random walk

        Transition process: node → hyperedge → node
        p(e|v) = |e| / sum(|e'|)
        p(u|e) = d_H(u) / sum(d_H(w))
        p_vu = sum(p(e|v) * p(u|e))
        """
        transition_matrix = np.zeros((self.n_nodes, self.n_nodes))

        for v in range(self.n_nodes):
            # Get hyperedges node v participates in
            if issparse(self.incidence_matrix):
                edges_v = self.incidence_matrix[v].nonzero()[1]  # Get column indices of non-zero elements
            else:
                edges_v = np.where(self.incidence_matrix[v])[0]

            if len(edges_v) == 0:
                continue  # Isolated node, transition probabilities are 0

            # Calculate probability of selecting each hyperedge p(e|v)
            edge_probs = self.edge_degrees[edges_v] / np.sum(self.edge_degrees[edges_v])

            for e_idx, e in enumerate(edges_v):
                # Get nodes in hyperedge e
                if issparse(self.incidence_matrix):
                    nodes_e = self.incidence_matrix[:, e].nonzero()[0]  # Get row indices of non-zero elements
                else:
                    nodes_e = np.where(self.incidence_matrix[:, e])[0]

                if len(nodes_e) == 0:
                    continue  # Empty hyperedge, skip

                # Calculate probability of selecting each node p(u|e)
                node_probs = self.node_degrees[nodes_e] / np.sum(self.node_degrees[nodes_e])

                for u_idx, u in enumerate(nodes_e):
                    # Accumulate transition probability p(v→u)
                    transition_matrix[v, u] += edge_probs[e_idx] * node_probs[u_idx]

        return transition_matrix

    def calculate_stationary_distribution(self, max_iter=1000, tol=1e-6):
        """
        Calculate the stationary distribution π
        π = πP and sum(π) = 1
        """
        # Initial uniform distribution
        pi = np.ones(self.n_nodes) / self.n_nodes

        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new

        return pi