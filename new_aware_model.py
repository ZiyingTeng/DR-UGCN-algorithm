import torch
import torch.nn as nn
import torch.nn.functional as F
from UniGGCN import UniGCNRegression


class NuAwareUniGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # 可学习的特征权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels),  # 输出与特征数相同的权重
            nn.Softmax(dim=-1)
        )

        # UniGCN骨干网络 - 确保输入输出维度匹配
        self.backbone = UniGCNRegression(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # 输出隐藏维度，不是1
            num_layers=3,
            dropout=0.3
        )

        # 基于θ的注意力机制
        self.nu_attention = nn.Sequential(
            nn.Linear(hidden_channels + 1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, data, nu):
        # 生成特征权重
        nu_tensor = torch.tensor([nu], device=data.x.device, dtype=torch.float).view(1, 1)
        feature_weights = self.weight_generator(nu_tensor)  # [1, in_channels]

        # 应用特征权重
        weighted_x = data.x * feature_weights

        # 使用加权后的特征进行前向传播
        data_copy = data.clone()
        data_copy.x = weighted_x

        # 获取UniGCN的输出
        node_embeddings = self.backbone(data_copy)  # [num_nodes, hidden_channels]

        # 基于ν的注意力机制
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        attention_input = torch.cat([node_embeddings, nu_expanded], dim=1)
        attention_weights = self.nu_attention(attention_input)

        # 应用注意力权重
        attended_embeddings = node_embeddings * attention_weights

        # 最终输出
        output = self.output_layer(attended_embeddings)

        return output

# FiLM机制

class NuAwareUniGCNWithFiLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # FiLM 层
        self.film = FiLMLayer(in_channels)

        # UniGCN骨干网络
        self.backbone = UniGCNRegression(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=3,
            dropout=0.3
        )

        # 基于θ和节点度的注意力机制
        self.nu_attention = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels // 2),  # +2 for nu and node_degrees
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, data, nu, node_degrees=None):
        # 应用FiLM变换
        filmed_x = self.film(data.x, nu)

        # 使用变换后的特征进行前向传播
        data_copy = data.clone()
        data_copy.x = filmed_x

        # 获取UniGCN的输出
        node_embeddings = self.backbone(data_copy)

        # 基于θ和节点度的注意力机制
        nu_tensor = torch.tensor([nu], device=node_embeddings.device, dtype=torch.float).view(1, 1)
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        if node_degrees is None:
            node_degrees = torch.zeros(node_embeddings.size(0), 1, device=node_embeddings.device)  # 默认值
        attention_input = torch.cat([node_embeddings, nu_expanded, node_degrees], dim=1)
        attention_weights = self.nu_attention(attention_input)

        # 应用注意力权重
        attended_embeddings = node_embeddings * attention_weights

        # 最终输出
        output = self.output_layer(attended_embeddings)

        return output


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""

    def __init__(self, feature_dim, condition_dim=32):
        super().__init__()
        self.condition_encoder = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, feature_dim * 2)
        )

    def forward(self, x, condition):
        if not torch.is_tensor(condition):
            condition = torch.tensor([condition], device=x.device, dtype=torch.float)
        if condition.dim() == 0:
            condition = condition.view(1, 1)
        elif condition.dim() == 1:
            condition = condition.unsqueeze(1)

        gamma_beta = self.condition_encoder(condition)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + 0.5 * gamma) * x + 0.3 * beta


class EnhancedNuAwareModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_raw_features=6, num_heads=4):
        super().__init__()
        self.num_raw_features = num_raw_features
        self.num_heads = num_heads

        # === 改进的特征处理 ===
        # 使用无参数化的BatchNorm（只标准化，不学习缩放参数）
        self.feature_norm = nn.BatchNorm1d(in_channels, affine=False)

        # 特征重要性保护参数（可学习，但有约束）
        self.feature_importance = nn.Parameter(torch.ones(num_raw_features))
        # 初始化保护参数：HDC权重较高
        with torch.no_grad():
            self.feature_importance.data = torch.tensor([1.5, 1.0, 1.0, 0.8, 0.8, 1.2])  # HDC, RWHC, RWIEC, Motif, Overlap, Pagerank

        # 主通路 - 非线性黑箱
        self.film = FiLMLayer(in_channels)
        self.backbone = UniGCNRegression(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=0.3
        )

        # 简化注意力机制（减少参数量）
        self.multi_head_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels + 2, hidden_channels // 4),  # 减少维度
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_channels // 4, 1),
                nn.Sigmoid()
            ) for _ in range(min(num_heads, 4))  # 限制注意力头数量
        ])

        self.attention_combiner = nn.Linear(len(self.multi_head_attention), 1)

        # 简化输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 减少dropout
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Tanh()  # 使用Tanh替代Sigmoid，避免梯度消失
        )

        # # 辅助通路 - 增加约束
        # self.auxiliary_net = nn.Sequential(
        #     nn.Linear(1, 16),  # 减少隐藏单元
        #     nn.ReLU(),
        #     nn.Linear(16, num_raw_features),
        #     nn.Softmax(dim=-1)
        # )

        # 简化记忆机制
        self.memory_bank = nn.Parameter(torch.randn(10, hidden_channels) * 0.1)  # 减少记忆模式

        # self._initialize_with_prior_knowledge()

    # def _initialize_with_prior_knowledge(self):
    #     """基于先验知识初始化"""
    #     # 更温和的初始化
    #     for layer in self.auxiliary_net:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_uniform_(layer.weight, gain=0.5)  # 减小初始化规模
    #             if layer.bias is not None:
    #                 nn.init.constant_(layer.bias, 0.1)
    #
    #     # 辅助网络偏置：鼓励HDC在初期占主导
    #     with torch.no_grad():
    #         self.auxiliary_net[2].bias.data = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1, 0.1])  # 更强调HDC

    def forward(self, data, nu, node_degrees=None, return_auxiliary=False):
        # 准备θ张量
        if not isinstance(nu, torch.Tensor):
            nu_tensor = torch.tensor([nu], device=data.x.device, dtype=torch.float)
        else:
            nu_tensor = nu

        if nu_tensor.dim() == 0:
            nu_tensor = nu_tensor.view(1)
        if nu_tensor.dim() == 1:
            nu_tensor = nu_tensor.unsqueeze(1)

        # === 改进的特征保护 ===
        raw_features = data.x[:, :self.num_raw_features]

        # 应用特征重要性保护（带约束）
        protected_features = raw_features * torch.clamp(self.feature_importance, 0.5, 2.0)

        # 组合特征
        if data.x.shape[1] > self.num_raw_features:
            other_features = data.x[:, self.num_raw_features:]
            processed_features = torch.cat([protected_features, other_features], dim=1)
        else:
            processed_features = protected_features

        # 温和的标准化
        processed_features = self.feature_norm(processed_features)

        # # 辅助通路：生成特征权重（带温度调节）
        # aux_weights = self.auxiliary_net(nu_tensor)
        #
        # # 计算线性分数（作为参考基准）
        # aux_weights_expanded = aux_weights.expand(raw_features.size(0), -1)
        # linear_scores = torch.sum(aux_weights_expanded * raw_features, dim=1, keepdim=True)

        # 主通路：非线性变换
        filmed_x = self.film(processed_features, nu_tensor)
        data_copy = data.clone()
        data_copy.x = filmed_x

        # 使用更稳定的骨干网络
        node_embeddings = self.backbone(data_copy)

        # 处理节点度
        if node_degrees is None:
            node_degrees = torch.zeros(node_embeddings.size(0), 1, device=node_embeddings.device)

        # 简化注意力机制
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        attention_input = torch.cat([node_embeddings, nu_expanded, node_degrees], dim=1)

        head_outputs = []
        for head in self.multi_head_attention:
            head_weights = head(attention_input)
            head_outputs.append(head_weights)

        # 组合多头结果
        attention_weights = torch.cat(head_outputs, dim=1)
        combined_weights = self.attention_combiner(attention_weights)

        # 轻量记忆增强
        memory_enhanced = self._apply_memory_enhancement(node_embeddings)

        # 应用注意力权重
        attended_embeddings = memory_enhanced * torch.sigmoid(combined_weights)  # 使用sigmoid约束权重范围

        # 最终输出（带输出约束）
        main_scores = self.output_layer(attended_embeddings)

        # 确保输出在合理范围内
        main_scores = torch.tanh(main_scores)  # 约束到[-1, 1]

        linear_scores_placeholder = torch.zeros_like(main_scores)

        # if return_auxiliary:
        #     return main_scores, linear_scores, aux_weights, attention_weights
        if return_auxiliary:
            # 返回占位符值
            return main_scores, linear_scores_placeholder, torch.zeros(self.num_raw_features), attention_weights
        return main_scores, linear_scores_placeholder

    def _apply_memory_enhancement(self, embeddings):
        """轻量级记忆增强"""
        if self.memory_bank.size(0) == 0:
            return embeddings

        # 计算相似度（带温度调节）
        similarity = torch.matmul(embeddings, self.memory_bank.t()) / torch.sqrt(
            torch.tensor(embeddings.size(-1), dtype=torch.float)
        )
        attention_weights = F.softmax(similarity, dim=1)

        # 加权记忆反馈
        memory_output = torch.matmul(attention_weights, self.memory_bank)

        # 温和的残差连接
        return embeddings + 0.1 * memory_output  # 减小记忆影响
