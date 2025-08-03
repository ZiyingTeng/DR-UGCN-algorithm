import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from unigencoder import PlainUnigencoder


class UniGCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.3, batch_norm=True):
        super(UniGCNRegression, self).__init__()
        self.unig_encoder = PlainUnigencoder(in_channels, hidden_channels, out_channels)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.batch_norm = batch_norm

    def forward(self, data):
        x, edge_index = data.x, data.PvT
        for i, conv in enumerate(self.convs[:-1]):
            residual = x
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:
                x = x + residual  # Residual connection
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)