import torch
from torch_geometric.utils import to_dense_adj


class PlainUnigencoder:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def d_expansion(self, data):
        edge_index = data.edge_index
        adj = to_dense_adj(edge_index)[0]
        N = adj.shape[0]
        # Create sparse COO tensor
        indices = edge_index
        values = torch.ones(edge_index.shape[1]).to(edge_index.device)
        data.Pv = torch.sparse_coo_tensor(indices, values, (N, N))
        data.PvT = data.Pv.t()
        return data