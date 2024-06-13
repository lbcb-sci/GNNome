from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers


class SymGatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, normalization, dropout=None):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.SymGatedGCN(hidden_features, hidden_features, normalization, dropout) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e


class GatedGCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, normalization, dropout=None):
        super().__init__()
        self.convs = nn.ModuleList([
            layers.GatedGCN(hidden_features, hidden_features, normalization, dropout) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)):
            h, e = self.convs[i](graph, h, e)
        return h, e


class GCN_processor(nn.Module):
    def __init__(self, num_layers, hidden_features):
        super().__init__()
        self.convs = nn.ModuleList([
            GraphConv(hidden_features, hidden_features, weight=True, bias=True) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            h = F.relu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e
    

class GAT_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout=0.0, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        print(f'Using dropout:', dropout)
        self.convs = nn.ModuleList([
            GATConv(hidden_features, hidden_features, num_heads=self.num_heads, feat_drop=dropout, attn_drop=0) for _ in range(num_layers)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(self.num_heads * hidden_features, hidden_features) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            heads = self.convs[i](graph, h)
            h = torch.cat(tuple(heads[:,j,:] for j in range(self.num_heads)), dim=1)
            h = self.linears[i](h)
            h = F.relu(h)
        heads = self.convs[-1](graph, h)
        h = torch.cat(tuple(heads[:,j,:] for j in range(self.num_heads)), dim=1)
        h = self.linears[-1](h)
        return h, e
    
    
class SAGE_processor(nn.Module):
    def __init__(self, num_layers, hidden_features, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(hidden_features, hidden_features, 'mean', feat_drop=dropout) for _ in range(num_layers)
        ])

    def forward(self, graph, h, e):
        for i in range(len(self.convs)-1):
            h = F.relu(self.convs[i](graph, h))
        h = self.convs[-1](graph, h)
        return h, e