import numpy as np
import pickle
from collections import Counter


def generate_incidence_matrix(matrices):
    """根据超边列表生成关联矩阵"""
    if not matrices:
        return None

    # 确定最大节点索引
    max_node = max(max(edge) for edge in matrices)
    n_nodes = max_node + 1
    n_edges = len(matrices)

    # 初始化关联矩阵
    incidence_matrix = np.zeros((n_nodes, n_edges), dtype=int)

    # 填充关联矩阵
    for edge_idx, edge in enumerate(matrices):
        for node_idx in edge:
            incidence_matrix[node_idx, edge_idx] = 1

    return incidence_matrix


def process_pickle_file(file_path):
    """读取pickle文件并生成关联矩阵"""
    try:
        # 打开pickle文件并加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 处理数据生成矩阵列表
        matrices = []
        for key, values in data.items():
            # 将键合并到数值集合中
            combined_set = {key}.union(values)
            # 转换为排序列表
            matrix = sorted(combined_set)
            matrices.append(matrix)

        # 统计信息
        all_numbers = [num for matrix in matrices for num in matrix]
        counter = Counter(all_numbers)
        max_length = max(len(matrix) for matrix in matrices)

        # 生成并返回关联矩阵
        incidence_matrix = generate_incidence_matrix(matrices)
        return incidence_matrix

    except Exception as e:
        print(f"处理pickle文件时出错: {e}")
        return None

