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

        # 基于ν的注意力机制
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