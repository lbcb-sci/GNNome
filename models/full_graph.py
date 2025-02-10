import torch
import torch.nn as nn

import dgl

import layers


class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_ne_features, num_layers, hidden_edge_scores, normalization, dropout=None):
        super().__init__()
        self.node_encoder = layers.NodeEncoder(node_features, hidden_ne_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_ne_features, hidden_features)
        self.gnn = layers.SymGatedGCN_processor(num_layers, hidden_features, normalization, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_ne_features, num_layers, hidden_edge_scores, normalization, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(node_features, hidden_ne_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_ne_features, hidden_features)
        self.gnn = layers.GatedGCN_processor(num_layers, hidden_features, normalization, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            x, e = self.gnn(graph, x, e)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            e = torch.cat((e, e), dim=0)
            x, e = self.gnn(g, x, e)
            e = e[:graph.num_edges()]
        scores = self.predictor(graph, x, e)
        return scores


class GCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_ne_features, num_layers, hidden_edge_scores, normalization, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(node_features, hidden_ne_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_ne_features, hidden_features)
        self.gnn = layers.GCN_processor(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores
    
    
class GATModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_ne_features, num_layers, hidden_edge_scores, normalization, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(node_features, hidden_ne_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_ne_features, hidden_features)
        self.gnn = layers.GAT_processor(num_layers, hidden_features, dropout=dropout, num_heads=3)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):       
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class SAGEModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_ne_features, num_layers, hidden_edge_scores, normalization, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        self.node_encoder = layers.NodeEncoder(node_features, hidden_ne_features, hidden_features)
        self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_ne_features, hidden_features)
        self.gnn = layers.SAGE_processor(num_layers, hidden_features, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.node_encoder(x)
        e = self.edge_encoder(e)
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores