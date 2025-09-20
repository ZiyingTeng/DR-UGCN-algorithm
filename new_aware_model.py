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
            nn.Linear(32, in_channels),
            nn.Softmax(dim=-1)
        )

        # UniGCN backbone
        self.backbone = UniGCNRegression(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
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

        # 基于θ的注意力机制
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        attention_input = torch.cat([node_embeddings, nu_expanded], dim=1)
        attention_weights = self.nu_attention(attention_input)

        # 应用注意力权重
        attended_embeddings = node_embeddings * attention_weights

        output = self.output_layer(attended_embeddings)

        return output

# FiLM机制

class NuAwareUniGCNWithFiLM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # FiLM 层
        self.film = FiLMLayer(in_channels)

        # UniGCN backbone
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

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid()
        )

    def forward(self, data, nu, node_degrees=None):
        filmed_x = self.film(data.x, nu)

        # 使用变换后的特征进行前向传播
        data_copy = data.clone()
        data_copy.x = filmed_x

        # 获取UniGCN的输出
        node_embeddings = self.backbone(data_copy)

        # 基于ν和节点度的注意力机制
        nu_tensor = torch.tensor([nu], device=node_embeddings.device, dtype=torch.float).view(1, 1)
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        if node_degrees is None:
            node_degrees = torch.zeros(node_embeddings.size(0), 1, device=node_embeddings.device)  # 默认值
        attention_input = torch.cat([node_embeddings, nu_expanded, node_degrees], dim=1)
        attention_weights = self.nu_attention(attention_input)

        # 应用注意力权重
        attended_embeddings = node_embeddings * attention_weights

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
    def __init__(self, in_channels, hidden_channels, out_channels, num_raw_features=5):
        super().__init__()
        self.num_raw_features = num_raw_features

        # 主通路 - 非线性黑箱
        self.film = FiLMLayer(in_channels)
        self.backbone = UniGCNRegression(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=3,
            dropout=0.3
        )

        # 基于θ和节点度的注意力机制
        self.nu_attention = nn.Sequential(
            nn.Linear(hidden_channels + 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )

        # 辅助通路 - 可解释权重生成
        self.auxiliary_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_raw_features),
            nn.Softmax(dim=-1)
        )

        self._initialize_auxiliary_net()

    def _initialize_auxiliary_net(self):
        """用合理的初始值初始化辅助网络"""
        for layer in self.auxiliary_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, data, nu, node_degrees=None, return_auxiliary=False):
        if not isinstance(nu, torch.Tensor):
            nu_tensor = torch.tensor([nu], device=data.x.device, dtype=torch.float)
        else:
            nu_tensor = nu

        if nu_tensor.dim() == 0:
            nu_tensor = nu_tensor.view(1)
        if nu_tensor.dim() == 1:
            nu_tensor = nu_tensor.unsqueeze(1)

        # 辅助通路：生成特征权重
        aux_weights = self.auxiliary_net(nu_tensor)  # [1, num_raw_features]

        # 计算线性分数
        raw_features = data.x[:, :self.num_raw_features]
        aux_weights_expanded = aux_weights.expand(raw_features.size(0), -1)
        linear_scores = torch.sum(aux_weights_expanded * raw_features, dim=1, keepdim=True)

        # 主通路：非线性变换
        filmed_x = self.film(data.x, nu_tensor)
        data_copy = data.clone()
        data_copy.x = filmed_x

        node_embeddings = self.backbone(data_copy)

        if node_degrees is None:
            node_degrees = torch.zeros(node_embeddings.size(0), 1, device=node_embeddings.device)

        # 注意力机制
        nu_expanded = nu_tensor.expand(node_embeddings.size(0), -1)
        attention_input = torch.cat([node_embeddings, nu_expanded, node_degrees], dim=1)
        attention_weights = self.nu_attention(attention_input)
        attended_embeddings = node_embeddings * attention_weights

        main_scores = self.output_layer(attended_embeddings)

        if return_auxiliary:
            return main_scores, linear_scores, aux_weights

        return main_scores, linear_scores

