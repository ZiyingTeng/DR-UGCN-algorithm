import numpy as np
from randomwalk import HypergraphRandomWalk
from dealpickle import process_pickle_file

class RWIECalculator:
    def __init__(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix
        self.n_nodes = incidence_matrix.shape[0]
        self.rw = HypergraphRandomWalk(incidence_matrix)
        self.transition_matrix = self.rw.transition_matrix
        self.stationary_dist = self.rw.calculate_stationary_distribution(alpha=0.85)
        if not np.isclose(np.sum(self.stationary_dist), 1.0, atol=1e-5):
            raise ValueError("Stationary distribution not normalized")

    def calculate_dynamic_entropy(self, gamma=1.0):
        """计算动态转移熵"""
        dynamic_entropy = np.zeros(self.n_nodes)
        for v in range(self.n_nodes):
            non_zero_idx = np.where(self.transition_matrix[v] > 0)[0]
            non_zero_probs = self.transition_matrix[v, non_zero_idx]
            if len(non_zero_probs) > 0:
                weighted_probs = non_zero_probs ** gamma
                if np.sum(weighted_probs) == 0:
                    dynamic_entropy[v] = 0
                    continue
                weighted_probs /= np.sum(weighted_probs) + 1e-10
                entropy = -np.sum(weighted_probs * np.log(weighted_probs + 1e-10))
                dynamic_entropy[v] = entropy
        return dynamic_entropy

    def calculate_stationary_entropy(self):
        """计算平稳分布熵"""
        stationary_entropy = -self.stationary_dist * np.log(self.stationary_dist + 1e-10)
        return stationary_entropy

    def calculate_rwiec(self, gamma=1.0):
        """计算RWIEC：I_R(v) = H_dyn(v) + H_stat(v)"""
        dyn_entropy = self.calculate_dynamic_entropy(gamma)
        stat_entropy = self.calculate_stationary_entropy()
        if len(dyn_entropy) != len(stat_entropy):
            raise ValueError("Dynamic and stationary entropy length mismatch")
        rwiec = dyn_entropy + stat_entropy
        return rwiec


# import numpy as np
# from randomwalk import HypergraphRandomWalk
# from dealpickle import process_pickle_file
#
# class RWIECalculator:
#
#     def __init__(self, incidence_matrix):
#         """
#         初始化RWIEC计算器
#
#         参数:
#         incidence_matrix: 超图关联矩阵，形状为[节点数, 超边数]
#         """
#         self.incidence_matrix = incidence_matrix
#         self.n_nodes = incidence_matrix.shape[0]
#         self.rw = HypergraphRandomWalk(incidence_matrix)
#         self.transition_matrix = self.rw.transition_matrix
#         self.stationary_dist = self.rw.calculate_stationary_distribution()
#
#     def calculate_dynamic_entropy(self):
#         """
#         计算动态转移熵：描述节点作为信息源的活跃性和邻居分布不确定性
#         H_dyn(v) = -1/N * sum(p_vu * log(p_vu)) ，仅对p_vu>0的项求和
#         """
#         dynamic_entropy = np.zeros(self.n_nodes)
#
#         for v in range(self.n_nodes):
#             # 提取非零转移概率
#             non_zero_idx = np.where(self.transition_matrix[v] > 0)[0]
#             non_zero_probs = self.transition_matrix[v, non_zero_idx]
#
#             if len(non_zero_probs) > 0:
#                 # 对概率取对数并加权求和
#                 entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
#                 dynamic_entropy[v] = entropy / self.n_nodes  # 归一化
#
#         return dynamic_entropy
#
#     def calculate_stationary_entropy(self):
#         """
#         计算平稳分布熵：H_stat(v) = -pi_v * log(pi_v)
#         """
#         stationary_entropy = -self.stationary_dist * np.log(self.stationary_dist + 1e-10)
#         return stationary_entropy
#
#     def calculate_rwiec(self):
#         """
#         计算RWIEC：I_R(v) = H_dyn(v) + H_stat(v)
#         """
#         dyn_entropy = self.calculate_dynamic_entropy()
#         stat_entropy = self.calculate_stationary_entropy()
#         rwiec = dyn_entropy + stat_entropy
#         return rwiec


