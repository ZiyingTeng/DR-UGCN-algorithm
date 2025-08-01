import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse.linalg import eigsh

from connectivity import hypergraph_natural_connectivity

def compute_dc(incidence_matrix):
    """Compute Degree Centrality (DC) on projected graph."""
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    G = nx.from_scipy_sparse_array(adj)
    dc = nx.degree_centrality(G)
    return np.array([dc[i] for i in range(len(dc))])

def compute_bc(incidence_matrix):
    """Compute Betweenness Centrality (BC) on projected graph."""
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    G = nx.from_scipy_sparse_array(adj)
    bc = nx.betweenness_centrality(G)
    return np.array([bc[i] for i in range(len(bc))])

def compute_hdc(incidence_matrix):
    """Compute Hyperdegree Centrality (HDC)."""
    return np.sum(incidence_matrix, axis=1).A1

def compute_sc(incidence_matrix):
    """Compute Spectral Centrality (SC) based on natural connectivity."""
    rho = hypergraph_natural_connectivity(incidence_matrix)
    # Approximate SC as normalized eigenvector of largest eigenvalue
    H = csr_matrix(incidence_matrix)
    A = H.dot(H.T)
    A.setdiag(0)
    _, eigenvectors = eigsh(A, k=1, which='LM')
    sc = np.abs(eigenvectors[:, 0])
    sc = sc / np.sum(sc)
    return sc