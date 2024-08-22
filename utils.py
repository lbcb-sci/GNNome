import os
import pickle
import random
import subprocess

import torch
import numpy as np
import dgl

from Bio import Seq, SeqIO
from scipy import sparse as sp 
from sklearn.metrics import precision_recall_curve, average_precision_score

from hyperparameters import get_hyperparameters


def set_seed(seed=42):
    """Set random seed to enable reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        A number used to set the random seed

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def extract_contigs(path, idx):
    gfa_path = os.path.join(path, f'{idx}_asm.bp.p_ctg.gfa')
    asm_path = os.path.join(path, f'{idx}_assembly.fasta')
    contigs = []
    with open(gfa_path) as f:
        n = 0
        for line in f.readlines():
            line = line.strip()
            if line[0] != 'S':
                continue
            seq=Seq.Seq(line.split()[2])
            ctg = SeqIO.SeqRecord(seq, description=f'contig_{n}', id=f'contig_{n}')
            contigs.append(ctg)
            n += 1
        SeqIO.write(contigs, asm_path, 'fasta')
    # subprocess.run(f'rm {path}/output.csv', shell=True)


def preprocess_graph(g):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    if len(set(g.ntypes)) == 1 and len(set(g.etypes)) == 1:
        ol_len = g.edata['overlap_length'].float()
        ol_sim = g.edata['overlap_similarity']
        ol_len = (ol_len - ol_len.mean()) / ol_len.std()
        if get_hyperparameters()['use_similarities']:
            g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
        else:
            g.edata['e'] = ol_len.unsqueeze(-1)
    else:
        ol_len = g.edges['real'].data['overlap_length'].float()
        ol_sim = g.edges['real'].data['overlap_similarity']
        ol_len = (ol_len - ol_len.mean()) / ol_len.std()
        if get_hyperparameters()['use_similarities']:
            g.edges['real'].data['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
        else:
            g.edges['real'].data['e'] = ol_len.unsqueeze(-1)
    return g


def add_positional_encoding(g):
    """
        Initializing positional encoding with k-RW-PE
    """
    if len(set(g.ntypes)) == 1 and len(set(g.etypes)) == 1:
        g.ndata['in_deg'] = g.in_degrees().float()
        g.ndata['out_deg'] = g.out_degrees().float()
    else:
        g.ndata['in_deg'] = g.in_degrees(etype='real').float()
        g.ndata['out_deg'] = g.out_degrees(etype='real').float()
    pe_dim = get_hyperparameters()['nb_pos_enc']
    pe_type = get_hyperparameters()['type_pos_enc']
    
    if pe_dim == 0:
        return g

    if pe_type == 'RW':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A @ Dinv  
        M = RW
        # Iterate
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(pe_dim-1):
            M_power = M_power @ M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    if pe_type == 'PR':
        # k-step PageRank features
        A = g.adjacency_matrix(scipy_fmt="csr")
        D = A.sum(axis=1) # out degree
        Dinv = 1./ (D+1e-9); Dinv[D<1e-9] = 0 # take care of nodes without outgoing edges
        Dinv = sp.diags(np.squeeze(np.asarray(Dinv)), dtype=float) # D^-1 
        P = (Dinv @ A).T 
        n = A.shape[0]
        One = np.ones([n])
        x = One/ n
        PE = [] 
        alpha = 0.95 
        for _ in range(pe_dim): 
            x = alpha* P.dot(x) + (1.0-alpha)/n* One 
            PE.append(torch.from_numpy(x).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    return g


def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'


def get_walks(idx, data_path):
    walk_path = os.path.join(data_path, f'solutions/{idx}_gt.pkl')
    walks = pickle.load(open(walk_path, 'rb'))
    return walks


def get_correct_ne(idx, data_path):
    nodes_path = os.path.join(data_path, f'solutions/{idx}_nodes.pkl')
    edges_path = os.path.join(data_path, f'solutions/{idx}_edges.pkl')
    nodes_gt = pickle.load(open(nodes_path, 'rb'))
    edges_gt = pickle.load(open(edges_path, 'rb'))
    return nodes_gt, edges_gt


def get_info(idx, data_path, type):
    info_path = os.path.join(data_path, 'info', f'{idx}_{type}.pkl')
    info = pickle.load(open(info_path, 'rb'))
    return info


def unpack_data(data, info_all, use_reads):
    idx, graph = data
    idx = idx if isinstance(idx, int) else idx.item()
    pred = info_all['preds'][idx]
    succ = info_all['succs'][idx]
    if use_reads:
        reads = info_all['reads'][idx]
    else:
        reads = None
    edges = info_all['edges'][idx]
    return idx, graph, pred, succ, reads, edges


def load_graph_data(num_graphs, data_path, use_reads):
    info_all = {
        'preds': [],
        'succs': [],
        'reads': [],
        'edges': [],
    }
    for idx in range(num_graphs):
        info_all['preds'].append(get_info(idx, data_path, 'pred'))
        info_all['succs'].append(get_info(idx, data_path, 'succ'))
        if use_reads:
            info_all['reads'].append(get_info(idx, data_path, 'reads'))
        info_all['edges'].append(get_info(idx, data_path, 'edges'))
    return info_all


def print_graph_info(idx, graph):
    """Print the basic information for the graph with index idx."""
    print('\n---- GRAPH INFO ----')
    print('Graph index:', idx)
    print('Number of nodes:', graph.num_nodes())
    print('Number of edges:', len(graph.edges()[0]))


def print_prediction(walk, current, neighbors, actions, choice, best_neighbor):
    """Print summary of the prediction for the current position."""
    print('\n-----predicting-----')
    print('previous:\t', None if len(walk) < 2 else walk[-2])
    print('current:\t', current)
    print('neighbors:\t', neighbors[current])
    print('actions:\t', actions.tolist())
    print('choice:\t\t', choice)
    print('ground truth:\t', best_neighbor)


def calculate_tfpn(edge_predictions, edge_labels):
    edge_predictions = torch.round(torch.sigmoid(edge_predictions))
    TP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==1)).item()
    TN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==0)).item()
    FP = torch.sum(torch.logical_and(edge_predictions==1, edge_labels==0)).item()
    FN = torch.sum(torch.logical_and(edge_predictions==0, edge_labels==1)).item()
    return TP, TN, FP, FN


def calculate_metrics(TP, TN, FP, FN):
    try: 
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def calculate_metrics_inverse(TP, TN, FP, FN): 
    TP, TN = TN, TP
    FP, FN = FN, FP
    try: 
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = TP / (TP + 0.5 * (FP + FN) )
    except ZeroDivisionError:
        f1 = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, precision, recall, f1


def get_precision_recall_curve(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    return precision, recall, thresholds


def get_precision_recall_curve_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, preds, pos_label=0)
    return precision, recall, thresholds


# Actually computes average_precision_score instead of AUC-PC
def get_aps(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds)
    return auc_pc


# Actually computes average_precision_score instead of AUC-PC
def get_aps_inverse(preds, labels):
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    preds = 1 - preds
    labels = labels.cpu().numpy()
    auc_pc = average_precision_score(labels, preds, pos_label=0)
    return auc_pc
