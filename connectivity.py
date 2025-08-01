import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def hypergraph_natural_connectivity(incidence_matrix):
    # Ensure incidence_matrix is sparse
    if not isinstance(incidence_matrix, csr_matrix):
        H = csr_matrix(incidence_matrix, dtype=np.float64)
    else:
        H = incidence_matrix

    N = H.shape[0]

    node_hyperdegrees = H.sum(axis=1).A1
    D = diags(node_hyperdegrees, format='csr')

    A = H.dot(H.T) - D
    A.setdiag(0)

    # Use sparse eigenvalue solver for the largest eigenvalue
    eigenvalues, _ = eigsh(A, k=1, which='LM')  # Largest magnitude
    max_lambda = eigenvalues[0]

    # For all eigenvalues, use eigsh with k=N-1 to get all non-trivial eigenvalues
    eigenvalues, _ = eigsh(A, k=min(N-1, A.shape[0]-1), which='LM')
    shifted_exp_sum = np.sum(np.exp(eigenvalues - max_lambda))
    rho = np.log(shifted_exp_sum / N) + max_lambda

    return rho