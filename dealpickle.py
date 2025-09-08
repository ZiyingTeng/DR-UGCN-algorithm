import numpy as np
import pickle
from collections import Counter

def generate_incidence_matrix(matrices, n_nodes, n_edges):
    """根据超边列表生成关联矩阵"""
    if not matrices:
        print("Error: No hyperedges provided to generate incidence matrix")
        return None

    try:
        incidence_matrix = np.zeros((n_nodes, n_edges), dtype=int)
        for edge_idx, edge in enumerate(matrices):
            for node_idx in edge:
                incidence_matrix[node_idx, edge_idx] = 1
        return incidence_matrix
    except Exception as e:
        print(f"Error in generate_incidence_matrix: {e}")
        return None

def process_pickle_file(file_path):
    """读取pickle文件并生成关联矩阵和节点映射"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            print(f"Error: Pickle file {file_path} does not contain a dictionary, got {type(data)}")
            return None, None

        # 收集所有唯一的节点ID（支持字符串和整数）
        node_set = set()
        for key, values in data.items():
            if not isinstance(values, (list, set, tuple)):
                print(f"Error: Invalid values type {type(values)} for key={key}")
                continue
            node_set.add(key)
            node_set.update(values)

        if not node_set:
            print(f"Error: No valid nodes found in {file_path}")
            return None, None

        # 节点ID到整数索引的映射
        node_id_map = {node: idx for idx, node in enumerate(sorted(node_set, key=str))}
        n_nodes = len(node_id_map)

        # 将超边转换为整数索引
        matrices = []
        for key, values in data.items():
            if not isinstance(values, (list, set, tuple)):
                continue
            # 将键和值映射到整数索引
            combined_set = {node_id_map[key]}.union(node_id_map[v] for v in values if v in node_id_map)
            int_edge = sorted(combined_set)  # Sort integers only
            if int_edge:
                matrices.append(int_edge)

        if not matrices:
            print(f"Error: No valid hyperedges generated from {file_path}")
            return None, None

        # 统计信息
        all_numbers = [num for matrix in matrices for num in matrix]
        counter = Counter(all_numbers)
        max_length = max(len(matrix) for matrix in matrices)
        print(f"Pickle data stats: Nodes={len(counter)}, Hyperedges={len(matrices)}, Max hyperedge size={max_length}")

        incidence_matrix = generate_incidence_matrix(matrices, n_nodes, len(matrices))
        if incidence_matrix is None:
            print("Error: Failed to generate incidence matrix")
            return None, None

        return incidence_matrix, node_id_map

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None
    except Exception as e:
        print(f"Error processing pickle file {file_path}: {e}")
        return None, None