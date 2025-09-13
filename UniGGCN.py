
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from unigencoder import Unigencoder


class UniGCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.unig_encoder = Unigencoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # 输出隐藏维度
            num_layers=1,
            dropout=dropout,
            Normalization='bn',
            InputNorm=True
        )

        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层
        self.gcn_layers.append(GCNConv(6, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        # 中间层
        for i in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # 最后一层
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
        x = data.x  # 直接使用原始特征
        for i, (conv, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            x = conv(x, data.edge_index)
            x = norm(x)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.regressor(x)
        # x_orig = data.x
        # # UniG编码
        # x_unig = self.unig_encoder(data)
        # # 拼接原始特征和UniG特征
        # x = torch.cat([x_orig, x_unig], dim=-1)
        #
        # # GCN层
        # for i, (conv, norm) in enumerate(zip(self.gcn_layers, self.norms)):
        #     x = conv(x, data.edge_index)
        #     x = norm(x)
        #     if i < len(self.gcn_layers) - 1:
        #         x = F.relu(x)
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        #
        # return self.regressor(x)