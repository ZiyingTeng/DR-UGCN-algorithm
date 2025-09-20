import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from connectivity import hypergraph_natural_connectivity

def compute_dc(incidence_matrix):
    """度中心性"""
    num_nodes = incidence_matrix.shape[0]
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    G = nx.from_scipy_sparse_array(adj)
    dc = nx.degree_centrality(G)
    dc_array = np.zeros(num_nodes)
    for i in range(num_nodes):
        dc_array[i] = dc.get(i, 0)
    return dc_array

def compute_bc(incidence_matrix):
    """介数中心性"""
    num_nodes = incidence_matrix.shape[0]
    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    G = nx.from_scipy_sparse_array(adj)
    bc = nx.betweenness_centrality(G)
    bc_array = np.zeros(num_nodes)
    for i in range(num_nodes):
        bc_array[i] = bc.get(i, 0)
    return bc_array

def compute_hdc(incidence_matrix):
    """超度中心性"""
    return np.sum(incidence_matrix, axis=1).A1

def compute_sc(incidence_matrix):
    """子图中心性"""
    num_nodes = incidence_matrix.shape[0]
    rho = hypergraph_natural_connectivity(incidence_matrix)
    H = csr_matrix(incidence_matrix)
    A = H.dot(H.T)
    A.setdiag(0)
    _, eigenvectors = eigsh(A, k=1, which='LM')
    sc = np.abs(eigenvectors[:, 0])
    if len(sc) != num_nodes:
        raise ValueError(f"SC length {len(sc)} != num_nodes {num_nodes}")
    sc = sc / (np.sum(sc) + 1e-10)

    return sc
