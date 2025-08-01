import torch
import torch.nn as nn
import torch.nn.functional as F
from unigencoder import PlainUnigencoder

class UniGCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.unig_encoder = PlainUnigencoder(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, Pv, PvT, edge_index = data.x, data.Pv, data.PvT, data.edge_index
        x = self.unig_encoder(x, Pv, PvT, edge_index)
        x = self.linear(x)
        x = torch.sigmoid(x)  # Constrain output to [0, 1]
        return x