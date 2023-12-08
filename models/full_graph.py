import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import layers


class SymGatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None):
        super().__init__()
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.SymGatedGCN_processor(num_layers, hidden_features, batch_norm, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        # x = self.linear_pe(pe) 
        x = self.linear1_node(pe)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class GatedGCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GatedGCN_processor(num_layers, hidden_features, batch_norm, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        # x = self.linear_pe(pe) 
        x = self.linear1_node(pe)
        x = torch.relu(x)
        x = self.linear2_node(x)
        
        e = torch.cat((e, e), dim=0)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)

        if self.directed:
            x, e = self.gnn(graph, x, e)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            x, e = self.gnn(g, x, e)
            e = e[:graph.num_edges()]
        scores = self.predictor(graph, x, e)
        return scores


class GCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GCN_processor(num_layers, hidden_features)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear1_node(pe)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores
    
    
class GATModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GAT_processor(num_layers, hidden_features, dropout=dropout, num_heads=3)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):       
        x = self.linear1_node(pe)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores


class SAGEModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, hidden_edge_features, num_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=None, directed=True):
        super().__init__()
        self.directed = directed
        hidden_node_features = hidden_edge_features
        self.linear1_node = nn.Linear(node_features, hidden_node_features)
        self.linear2_node = nn.Linear(hidden_node_features, hidden_features)
        self.linear1_edge = nn.Linear(edge_features, hidden_edge_features) 
        self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.SAGE_processor(num_layers, hidden_features, dropout=dropout)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e, pe):
        x = self.linear1_node(pe)
        x = torch.relu(x)
        x = self.linear2_node(x)

        e = self.linear1_edge(e)
        e = torch.relu(e)
        e = self.linear2_edge(e)
        
        if self.directed:
            g = dgl.add_self_loop(graph)
        else:
            g = dgl.add_reverse_edges(graph, copy_edata=True)
            g = dgl.add_self_loop(g)
        x, e = self.gnn(g, x, e)
        scores = self.predictor(graph, x, e)
        return scores