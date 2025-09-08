import numpy as np

def create_incidence_matrix_from_file(file_path, node_id_map=None):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    hyperedges = [line.strip().split(',') for line in lines]
    if not hyperedges:
        raise ValueError("Input file is empty")
    if node_id_map is None:
        all_nodes = set()
        for he in hyperedges:
            all_nodes.update(he)
        all_nodes = sorted(all_nodes, key=int)
        node_id_map = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(node_id_map)
    num_hyperedges = len(hyperedges)
    incidence_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)
    hyperedge_names = []
    for j, he_nodes in enumerate(hyperedges):
        hyperedge_names.append(f"HE_{j}")
        try:
            nodes = [int(node) for node in he_nodes]
        except ValueError:
            raise ValueError(f"Invalid node ID in hyperedge {j}: {he_nodes}")
        for node_name in he_nodes:
            if node_name in node_id_map:
                node_id = node_id_map[node_name]
                incidence_matrix[node_id, j] = 1
    node_degrees = np.sum(incidence_matrix, axis=1)
    non_isolated = node_degrees > 0
    incidence_matrix = incidence_matrix[non_isolated, :]
    filtered_node_id_map = {
        node: idx for idx, (node, old_idx) in enumerate(node_id_map.items()) if non_isolated[old_idx]
    }
    print(f"Original nodes: {num_nodes}, After filtering isolated nodes: {incidence_matrix.shape[0]}")
    return incidence_matrix, filtered_node_id_map, hyperedge_names