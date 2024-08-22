import torch
import torch.nn as nn


class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores):
        super().__init__()
        self.W1 = nn.Linear(3 * in_features, hidden_edge_scores) 
        self.W2 = nn.Linear(hidden_edge_scores, 32)
        self.W3 = nn.Linear(32, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        h = self.W1(data)
        h = torch.relu(h)
        score = self.W3(torch.relu(self.W2(h)))
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            if len(set(graph.ntypes)) == 1 and len(set(graph.etypes)) == 1:
                graph.edata['e'] = e
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']
            else:
                graph.edges['real'].data['e'] = e
                graph.apply_edges(self.apply_edges, etype='real')
                return graph.edges['real'].data['score']

