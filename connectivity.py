import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

def hypergraph_natural_connectivity(incidence_matrix):
    if not isinstance(incidence_matrix, csr_matrix):
        H = csr_matrix(incidence_matrix, dtype=np.float64)
    else:
        H = incidence_matrix
    N = H.shape[0]
    if N < 2:
        raise ValueError("Hypergraph too small for eigenvalue computation")
    node_hyperdegrees = H.sum(axis=1).A1
    D = diags(node_hyperdegrees, format='csr')
    A = H.dot(H.T) - D
    A.setdiag(0)
    eigenvalues, _ = eigsh(A, k=1, which='LM')
    max_lambda = eigenvalues[0]
    eigenvalues, _ = eigsh(A, k=min(N-1, A.shape[0]-1), which='LM')
    shifted_exp_sum = np.sum(np.exp(np.clip(eigenvalues - max_lambda, -100, 100)))
    rho = np.log(shifted_exp_sum / N) + max_lambda
    return rho