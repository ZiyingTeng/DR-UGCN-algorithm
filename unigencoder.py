import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing as pp
import torch_sparse
from torch_geometric import edge_index
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class Unigencoder(nn.Module):
    @staticmethod
    def d_expansion(data):
        # data = Data(num_nodes=data.num_nodes, edge_index=data.edge_index)
        V, E = data.edge_index
        if data.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        edge_dict = {}
        for i in range(data.edge_index.shape[1]):
            if E[i].item() not in edge_dict:
                edge_dict[E[i].item()] = []
            edge_dict[E[i].item()].append(V[i].item())

        def cosine_similarity_dense_small(x):
            norm = F.normalize(x, p=2., dim=-1)
            sim = norm.mm(norm.t())
            return sim

        N_vertex = V.max() + 1
        sim = cosine_similarity_dense_small(data.x)
        threshold = torch.median(sim.flatten()).item() if sim.numel() > 0 else 0.5
        pv_rows = []
        pv_cols = []
        for i in range(len(edge_dict)):
            neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            res_idx = torch.tensor(neighbor_indices)
            if len(neighbor_indices) > 1:
                idx = torch.tensor(neighbor_indices)
                new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
                new_sim[torch.eye(len(idx)).bool()] = 0
                del_idx = (new_sim < threshold).all(dim=1)
                keep_idx = ~del_idx
                res_idx = torch.masked_select(idx, keep_idx)
            for p in res_idx:
                pv_rows.append(i)
                pv_cols.append(p)
        pv_rows = torch.tensor(pv_rows)
        pv_cols = torch.tensor(pv_cols)
        pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
        pv_values = torch.ones_like(pv_rows, dtype=torch.float32)
        Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[len(edge_dict), N_vertex])
        PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, len(edge_dict)])
        data.Pv = Pv
        data.PvT = PvT
        return data

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.5, Normalization='bn', InputNorm=False):
        super(Unigencoder, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.gat = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.cls = nn.Linear(out_channels, out_channels)
        self.dropout = dropout
        self.norm = nn.BatchNorm1d(hidden_channels) if Normalization == 'bn' else nn.LayerNorm(hidden_channels)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.gat.reset_parameters()
        self.cls.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, data):
        x = data.x
        Pv, PvT = data.Pv, data.PvT
        x = torch.spmm(Pv, x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = self.norm(x)
        x = self.gat(x, data.edge_index)
        x = torch.spmm(PvT, x)
        return x
