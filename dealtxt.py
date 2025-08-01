import numpy as np


def create_incidence_matrix_from_file(file_path, node_id_map=None):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    hyperedges = [line.strip().split(',') for line in lines]

    if node_id_map is None:
        all_nodes = set()
        for he in hyperedges:
            all_nodes.update(he)
        node_id_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}

    num_nodes = len(node_id_map)
    num_hyperedges = len(hyperedges)

    incidence_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)

    hyperedge_names = []
    for j, he_nodes in enumerate(hyperedges):
        hyperedge_names.append(f"HE_{j}")  # 为当前超边创建名称

        for node_name in he_nodes:
            if node_name in node_id_map:
                node_id = node_id_map[node_name]
                incidence_matrix[node_id, j] = 1

    return incidence_matrix, node_id_map, hyperedge_names


# 使用示例
if __name__ == "__main__":

    H, node_map, hyperedges = create_incidence_matrix_from_file("../code原码/code/hyperedges-house-committees.txt")

    print("\n关联矩阵:")
    print(H)