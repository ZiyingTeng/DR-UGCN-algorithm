import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dealtxt import create_incidence_matrix_from_file
from connectivity import hypergraph_natural_connectivity
from rwiec import RWIECalculator
from rwhc import RWHCCalculator
from nonlinear import HypergraphContagion


def load_hypergraph(file_path):
    """Load hypergraph from txt file and return incidence matrix and edge index."""
    incidence_matrix, node_id_map, hyperedge_names = create_incidence_matrix_from_file(file_path)
    incidence_matrix = csr_matrix(incidence_matrix)
    print(f"Number of nodes: {incidence_matrix.shape[0]}, Number of hyperedges: {incidence_matrix.shape[1]}")

    node_degrees = np.sum(incidence_matrix, axis=1).A1
    print(f"Average node hyperdegree: {np.mean(node_degrees):.2f}")
    print(f"Number of isolated nodes: {np.sum(node_degrees == 0)}")

    edge_sizes = np.sum(incidence_matrix, axis=0).A1
    print(f"Average hyperedge size: {np.mean(edge_sizes):.2f}")
    print(f"Maximum hyperedge size: {np.max(edge_sizes)}")

    adj = incidence_matrix.dot(incidence_matrix.T)
    adj.setdiag(0)
    edge_index = torch.tensor(np.vstack(np.nonzero(adj)), dtype=torch.long)

    return incidence_matrix, edge_index, node_id_map


def compute_features(incidence_matrix):
    """Compute HDC, RWIEC, RWHC, and clustering coefficient features and normalize them."""
    hdc = np.sum(incidence_matrix, axis=1).A1
    rwiec_calc = RWIECalculator(incidence_matrix)
    rwiec = rwiec_calc.calculate_rwiec()
    rwhc_calc = RWHCCalculator(incidence_matrix)
    rwhc = rwhc_calc.calculate_rwhc()

    # Compute clustering coefficient
    adj = incidence_matrix.dot(incidence_matrix.T).toarray()
    np.fill_diagonal(adj, 0)
    clustering = np.zeros(incidence_matrix.shape[0])
    for i in range(incidence_matrix.shape[0]):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            continue
        sub_adj = adj[np.ix_(neighbors, neighbors)]
        edges = np.sum(sub_adj) / 2
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        clustering[i] = edges / possible_edges if possible_edges > 0 else 0

    features = np.vstack([hdc, rwiec, rwhc, clustering]).T
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.tensor(features, dtype=torch.float)


def split_dataset(num_nodes, train_ratio=0.6, val_ratio=0.1):
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (1 - train_ratio), random_state=42)
    return train_idx, val_idx, test_idx


def compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu, mu=0.015, max_steps=1000, num_runs=30):
    """Run nonlinear contagion model and return average infected fraction."""
    num_nodes = incidence_matrix.shape[0]
    infected_fractions = []
    for run in range(num_runs):
        initial_infected = np.zeros(num_nodes)
        initial_infected[top_nodes] = 1
        contagion = HypergraphContagion(incidence_matrix, initial_infected, lambda_val, nu, mu)
        infected_fraction = contagion.simulate(max_steps=max_steps, verbose=False)
        infected_fractions.append(infected_fraction)
    mean_infected = np.mean(infected_fractions)
    return mean_infected


def get_max_hyperdegree(incidence_matrix):
    node_degrees = np.sum(incidence_matrix, axis=1).A1
    max_hyperdegree = np.max(node_degrees)
    max_node_idx = np.argmax(node_degrees)
    print(f"Maximum node hyperdegree: {max_hyperdegree}, Node with max hyperdegree: {max_node_idx}")
    return max_hyperdegree