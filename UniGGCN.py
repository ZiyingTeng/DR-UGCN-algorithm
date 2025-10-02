from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from unigencoder import Unigencoder


class UniGCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=2, dropout=0.3):  # 默认hidden_channels=64, num_layers=2
        super().__init__()
        self.dropout = dropout
        self.unig_encoder = Unigencoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=1,
            dropout=dropout,
            Normalization='bn',
            InputNorm=True
        )
        first_layer_input = in_channels + hidden_channels
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gcn_layers.append(GCNConv(first_layer_input, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        for i in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # 回归器
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        self.unig_encoder.reset_parameters()
        for layer in self.gcn_layers:
            layer.reset_parameters()
        for layer in self.regressor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, data):
        x_orig = data.x
        x_unig = self.unig_encoder(data)

        # 拼接原始特征和UniG特征
        x = torch.cat([x_orig, x_unig], dim=-1)

        previous_outputs = []  # 保存各层输出用于跳跃连接

        for i, (conv, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            residual = x if i > 0 else 0  # 第一层无残差

            x = conv(x, data.edge_index)
            x = norm(x)

            if i > 0 and previous_outputs:
                # 从previous_outputs中选择一个合适的层进行跳跃连接
                if len(previous_outputs) > 0:
                    skip_source = previous_outputs[-1]  # 使用上一层的输出
                    if skip_source.shape[-1] != x.shape[-1]:
                        # 如果维度不匹配，使用线性投影
                        skip_projection = nn.Linear(skip_source.shape[-1], x.shape[-1]).to(x.device)
                        skip_source = skip_projection(skip_source)
                    x = x + skip_source  # 残差连接

            previous_outputs.append(x.clone())

            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.regressor(x)