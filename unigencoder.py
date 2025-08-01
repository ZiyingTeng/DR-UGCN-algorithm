import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sklearn.preprocessing as pp
import torch_sparse
import csv


class PlainUnigencoder(nn.Module):
    @staticmethod
    def d_expansion(data):
        V, E = data.edge_index
        edge_dict = {}
        for i in range(data.edge_index.shape[1]):
            e_idx = E[i].item()
            if e_idx not in edge_dict:
                edge_dict[e_idx] = []
            edge_dict[e_idx].append(V[i].item())

        print(f"Total hyperedges: {len(edge_dict)}, Hyperedge indices: {sorted(edge_dict.keys())}")
        pv_rows = []
        pv_cols = []
        for i, e_idx in enumerate(sorted(edge_dict.keys())):
            neighbor_indices = np.array(edge_dict[e_idx], dtype=np.int32)
            for p in neighbor_indices:
                pv_rows.append(i)
                pv_cols.append(p)
        pv_rows = torch.tensor(pv_rows)
        pv_cols = torch.tensor(pv_cols)
        pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
        pv_values = torch.ones_like(pv_rows, dtype=torch.float32)
        N_vertex = V.max() + 1
        N_hyperedge = len(edge_dict)
        Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[N_hyperedge, N_vertex]).coalesce()
        PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, N_hyperedge]).coalesce()

        data.Pv = Pv
        data.PvT = PvT
        return data

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels
        self.conv_out = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, Pv, PvT, edge_index):
        for conv in self.convs:
            x = torch_sparse.spmm(PvT.indices(), PvT.values(), PvT.size(0), PvT.size(1), x)
            x = torch_sparse.spmm(Pv.indices(), Pv.values(), Pv.size(0), Pv.size(1), x)
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch_sparse.spmm(PvT.indices(), PvT.values(), PvT.size(0), PvT.size(1), x)
        x = torch_sparse.spmm(Pv.indices(), Pv.values(), Pv.size(0), Pv.size(1), x)
        x = self.conv_out(x, edge_index)
        return x