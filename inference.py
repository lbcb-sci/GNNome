import argparse
import os
import sys
import pickle
import random
import math
from tqdm import tqdm 
import collections
import time
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager

import torch
import torch.nn.functional as F
import dgl

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import evaluate
import utils

DEBUG = False
RANDOM = False

def connectComplement_new(g):
    num_added_edges = int(g.num_nodes()/2)
    d = {'overlap_similarity': torch.ones(num_added_edges, dtype=torch.float32),
             'e': torch.tensor(num_added_edges*[[0,0]], dtype=torch.float32),
             'overlap_length': torch.tensor(num_added_edges*[-1], dtype=torch.int64),
             'prefix_length': torch.tensor(num_added_edges*[-1], dtype=torch.int64)}
    src = torch.arange(0, g.num_nodes(), 2).to(torch.int32)
    dst = torch.arange(1, g.num_nodes(), 2).to(torch.int32)
    g.add_edges(src, dst, d)
    g.add_edges(dst, src, d)

def connectComplement_hetero(input):
    dict = {('read', 'real', 'read'):((input.edges()[0].to(torch.int64), input.edges()[1].to(torch.int64))),
            ('read', 'virtual', 'read'): (torch.cat((torch.tensor(range(0, input.num_nodes(), 2)),
                                                    torch.tensor(range(1, input.num_nodes(), 2)))).to(torch.int64),
                                          torch.cat((torch.tensor(range(1, input.num_nodes(), 2)),
                                                    torch.tensor(range(0, input.num_nodes(), 2)))).to(torch.int64))}
    out = dgl.heterograph(dict)
    out.ndata['read_length'] = input.ndata['read_length']
    out.ndata['x'] = input.ndata['x']
    out.ndata['in_deg'] = input.ndata['in_deg']
    out.ndata['out_deg'] = input.ndata['out_deg']
    out.edata['overlap_similarity'] = {
        'real': input.edata['overlap_similarity'],
        'virtual': torch.ones(out.num_edges('virtual'), dtype=torch.float32)
        }
    out.edata['overlap_length'] = {
        'real': input.edata['overlap_length'],
        'virtual': torch.neg(torch.ones(out.num_edges('virtual'), dtype=torch.int64))
        }
    out.edata['prefix_length'] = {
        'real': input.edata['prefix_length'],
        'virtual': torch.neg(torch.ones(out.num_edges('virtual'), dtype=torch.int64))
        }
    out.edata['e'] = {
        'real': input.edata['e'],
        'virtual': torch.zeros(out.num_edges('virtual'), dtype=torch.float32)
        }
    d = {'overlap_similarity': torch.tensor([1], dtype=torch.float32),
        'overlap_length': torch.tensor([-1], dtype=torch.int64),
        'prefix_length': torch.tensor([-1], dtype=torch.int64),
        'e': torch.tensor([0], dtype=torch.float32)}
    #out = dgl.add_edges(out, torch.arange(1, input.num_nodes(), 2).to(torch.int32), torch.arange(0, input.num_nodes(), 2).to(torch.int32), d, ('read', 'virtual', 'read'))
    return out

def get_contig_length(walk, graph):
    total_length = 0
    idx_src = walk[:-1]
    idx_dst = walk[1:]
    if len(set(graph.ntypes)) == 1 and len(set(graph.etypes)) == 1:
        prefix = graph.edges[idx_src, idx_dst].data['prefix_length']
    else:
        prefix = graph.edges['real'].data['prefix_length'][graph.edge_ids(idx_src, idx_dst, etype='real')]
    total_length = prefix.sum().item()
    total_length += graph.ndata['read_length'][walk[-1]]
    return total_length


def get_subgraph(g, visited, device):
    """Remove the visited nodes from the graph."""
    remove_node_idx = torch.LongTensor([item for item in visited])
    list_node_idx = torch.arange(g.num_nodes())
    keep_node_idx = torch.ones(g.num_nodes())
    keep_node_idx[remove_node_idx] = 0
    keep_node_idx = list_node_idx[keep_node_idx==1].to(torch.int64).to(device)
    g = g.long()
    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device)
    map_subg_to_g = sub_g.ndata[dgl.NID]
    return sub_g, map_subg_to_g


def sample_edges(prob_edges, nb_paths):
    """Sample edges with Bernoulli sampling."""
    if prob_edges.shape[0] > 2**24:
        prob_edges = prob_edges[:2**24]  # torch.distributions.categorical.Categorical does not support tensors longer than 2**24
        
    if RANDOM:
        idx_edges = torch.randint(0, prob_edges.shape[0], (nb_paths,))
        return idx_edges
    
    prob_edges = prob_edges.masked_fill(prob_edges<1e-9, 1e-9)
    prob_edges = prob_edges/ prob_edges.sum()
    prob_edges_nb_paths = prob_edges.repeat(nb_paths, 1)
    idx_edges = torch.distributions.categorical.Categorical(prob_edges_nb_paths).sample()
    return idx_edges


def greedy_forwards(start, logProbs, neighbors, predecessors, edges, visited_old):
    """Greedy walk forwards."""
    current = start
    walk = []
    visited = set()
    sumLogProb = torch.tensor([0.0])
    iteration = 0
    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        neighs_current = neighbors[current]
        if len(neighs_current) == 0:
            break 
        if len(neighs_current) == 1:
            neighbor = neighs_current[0]
            if neighbor in visited_old or neighbor in visited:
                break
            else:
                sumLogProb += logProbs[edges[current, neighbor]]
                current = neighbor
                continue
        masked_neighbors = [n for n in neighs_current if not (n in visited_old or n in visited)]
        neighbor_edges = [edges[current, n] for n in masked_neighbors]
        if not neighbor_edges:
            break
        neighbor_p = logProbs[neighbor_edges]

        if RANDOM:
            index = torch.randint(0, neighbor_p.shape[0], (1,))
            logProb = neighbor_p[index]
        else:
            logProb, index = torch.topk(neighbor_p, k=1, dim=0)

        # logProb, index = torch.topk(neighbor_p, k=1, dim=0)
        sumLogProb += logProb
        iteration += 1
        current = masked_neighbors[index]
    return walk, visited, sumLogProb


def greedy_backwards_rc(start, logProbs, predecessors, neighbors, edges, visited_old):
    """Greedy walk backwards."""
    current = start ^ 1
    walk = []
    visited = set()
    sumLogProb = torch.tensor([0.0])
    iteration = 0
    while True:
        walk.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        neighs_current = neighbors[current]
        if len(neighs_current) == 0:
            break 
        if len(neighs_current) == 1:
            neighbor = neighs_current[0]
            if neighbor in visited_old or neighbor in visited:
                break
            else:
                sumLogProb += logProbs[edges[current, neighbor]]
                current = neighbor
                continue
        masked_neighbors = [n for n in neighs_current if not (n in visited_old or n in visited)]
        neighbor_edges = [edges[current, n] for n in masked_neighbors]
        if not neighbor_edges:
            break
        neighbor_p = logProbs[neighbor_edges]

        if RANDOM:
            index = torch.randint(0, neighbor_p.shape[0], (1,))
            logProb = neighbor_p[index]
        else:
            logProb, index = torch.topk(neighbor_p, k=1, dim=0)

        # logProb, index = torch.topk(neighbor_p, k=1, dim=0)
        sumLogProb += logProb
        iteration += 1
        current = masked_neighbors[index]
    walk = list(reversed([w ^ 1 for w in walk]))
    return walk, visited, sumLogProb
    

def run_greedy_both_ways(src, dst, logProbs, succs, preds, edges, visited):
    walk_f, visited_f, sumLogProb_f = greedy_forwards(dst, logProbs, succs, preds, edges, visited)
    walk_b, visited_b, sumLogProb_b = greedy_backwards_rc(src, logProbs, preds, succs, edges, visited | visited_f)
    return walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b


def get_contigs_greedy(g, succs, preds, edges, nb_paths=50, len_threshold=20, use_labels=False, checkpoint_dir=None, load_checkpoint=False, device='cpu', threads=32):
    """Iteratively search for contigs in a graph until the threshold is met."""
    g = g.to('cpu')
    all_contigs = []
    all_walks_len = []
    all_contigs_len = []
    visited = set()
    idx_contig = -1

    B = 1

    if len(set(g.ntypes)) == 1 and len(set(g.etypes)) == 1:
        if use_labels:
            scores = g.edata['y'].to('cpu')
            scores = scores.masked_fill(scores<1e-9, 1e-9)
            logProbs = torch.log(scores)
        else:
            scores = g.edata['score'].to('cpu')
            logProbs = torch.log(torch.sigmoid(g.edata['score'].to('cpu')))
    else:
        if use_labels:
            scores = g.edges['real'].data['y'].to('cpu')
            scores = scores.masked_fill(scores<1e-9, 1e-9)
            logProbs = torch.log(scores)
        else:
            scores = g.edges['real'].data['score'].to('cpu')
            logProbs = torch.log(torch.sigmoid(g.edges['real'].data['score'].to('cpu')))

    print(f'Starting to decode with greedy...')
    print(f'num_candidates: {nb_paths}, len_threshold: {len_threshold}\n')

    ckpt_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')
    if load_checkpoint and os.path.isfile(ckpt_file):
        print(f'Loading checkpoint from: {checkpoint_dir}\n')
        checkpoint = pickle.load(open(f'{checkpoint_dir}/checkpoint.pkl', 'rb'))
        all_contigs = checkpoint['walks']
        visited = checkpoint['visited']
        idx_contig = len(all_contigs) - 1
        all_walks_len = checkpoint['all_walks_len']
        all_contigs_len = checkpoint['all_contigs_len']

    while True:
        idx_contig += 1       
        time_start_sample_edges = datetime.now()
        sub_g, map_subg_to_g = get_subgraph(g, visited, 'cpu')
        if sub_g.num_edges() == 0:
            break
        
        if len(set(sub_g.ntypes)) == 1 and len(set(sub_g.etypes)) == 1:
            if use_labels:  # Debugging
                prob_edges = sub_g.edata['y']
            else:
                prob_edges = torch.sigmoid(sub_g.edata['score']).squeeze()
        else:
            if use_labels:  # Debugging
                prob_edges = sub_g.edges['real'].data['y']
            else:
                prob_edges = torch.sigmoid(sub_g.edges['real'].data['score']).squeeze()

        idx_edges = sample_edges(prob_edges, nb_paths)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_sample_edges)
        print(f'Elapsed time (sample edges): {elapsed}')

        all_walks = []
        all_visited_iter = []

        all_contig_lens = []
        all_sumLogProbs = []
        all_meanLogProbs = []
        all_meanLogProbs_scaled = []

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {g.num_nodes() - len(visited)}, nb_original_nodes: {g.num_nodes()}')

        # Get nb_paths paths for a single iteration, then take the longest one
        time_start_get_candidates = datetime.now()

        with ThreadPoolExecutor(1) as executor:
            if DEBUG:
                print(f'Starting with greedy for one candidate', flush=True)
                all_cand_time = datetime.now()
            results = {}
            start_times = {}
            if len(set(sub_g.ntypes)) == 1 and len(set(sub_g.etypes)) == 1:
                for e, idx in enumerate(idx_edges):
                    src_init_edges = map_subg_to_g[sub_g.edges()[0][idx]].item()
                    dst_init_edges = map_subg_to_g[sub_g.edges()[1][idx]].item()
                    start_times[e] = datetime.now()
                    if DEBUG:
                        print(f'About to submit job - decoding from edge {e}: {src_init_edges, dst_init_edges}', flush=True)
                    future = executor.submit(run_greedy_both_ways, src_init_edges, dst_init_edges, logProbs, succs, preds, edges, visited)
                    results[(src_init_edges, dst_init_edges)] = (future, e)
            else:
                for e, idx in enumerate(idx_edges):
                    src_init_edges = map_subg_to_g[sub_g.edges(etype='real')[0][idx]].item()
                    dst_init_edges = map_subg_to_g[sub_g.edges(etype='real')[1][idx]].item()
                    start_times[e] = datetime.now()
                    if DEBUG:
                        print(f'About to submit job - decoding from edge {e}: {src_init_edges, dst_init_edges}', flush=True)
                    future = executor.submit(run_greedy_both_ways, src_init_edges, dst_init_edges, logProbs, succs, preds, edges, visited)
                    results[(src_init_edges, dst_init_edges)] = (future, e)
            if DEBUG:
                process = psutil.Process(os.getpid())
                children = process.children(recursive=True)
                print(f'Processes ran: {e+1}\n' \
                      f'Time needed: {utils.timedelta_to_str(datetime.now() - all_cand_time)}\n' \
                      f'Current process ID: {os.getpid()}\n' \
                      f'Total memory used (MB): {process.memory_info().rss / 1024 ** 2}', flush=True)
                if len(children) == 0:
                    print(f'Process has no children!')
                for child in children:
                    print(f'Child pid is {child.pid}', flush=True)
                print()

            indx = 0
            for k, (f, e) in results.items():  # key, future -> Why did I not name this properly?
                walk_f, walk_b, visited_f, visited_b, sumLogProb_f, sumLogProb_b = f.result()
                if DEBUG:
                    print(f'Finished with candidate {e}: {k}\t' \
                        f'Time needed: {utils.timedelta_to_str(datetime.now() - start_times[e])}')         
                walk_it = walk_b + walk_f
                visited_iter = visited_f | visited_b
                sumLogProb_it = sumLogProb_f.item() + sumLogProb_b.item()
                len_walk_it = len(walk_it)
                len_contig_it = get_contig_length(walk_it, g).item()
                if k[0] == k[1]:
                    len_walk_it = 1
                
                if len_walk_it > 2:
                    meanLogProb_it = sumLogProb_it / (len_walk_it - 2)  # len(walk_f) - 1 + len(walk_b) - 1  <-> starting edge is neglected
                    try:
                        meanLogProb_scaled_it = meanLogProb_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        meanLogProb_scaled_it = 0
                elif len_walk_it == 2:
                    meanLogProb_it = 0.0
                    try:
                        meanLogProb_scaled_it = meanLogProb_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        meanLogProb_scaled_it = 0
                else:  # len_walk_it == 1 <-> SELF-LOOP!
                    len_contig_it = 0
                    sumLogProb_it = 0.0
                    meanLogProb_it = 0.0
                    meanLogprob_scaled_it = 0.0
                    print(f'SELF-LOOP!')
                print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12} ' \
                      f'sumLogProb={sumLogProb_it:<12.3f} meanLogProb={meanLogProb_it:<12.4} meanLogProb_scaled={meanLogProb_scaled_it:<12.4}')

                indx += 1
                all_walks.append(walk_it)
                all_visited_iter.append(visited_iter)
                all_contig_lens.append(len_contig_it)
                all_sumLogProbs.append(sumLogProb_it)
                all_meanLogProbs.append(meanLogProb_it)
                all_meanLogProbs_scaled.append(meanLogProb_scaled_it)

        best = max(all_contig_lens)
        idxx = all_contig_lens.index(best)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_candidates)
        print(f'Elapsed time (get_candidates): {elapsed}')

        best_walk = all_walks[idxx]
        best_visited = all_visited_iter[idxx]

        # Add all jumped-over nodes
        time_start_get_visited = datetime.now()
        trans = set()
        for ss, dd in zip(best_walk[:-1], best_walk[1:]):
            t1 = set(succs[ss]) & set(preds[dd])
            t2 = {t^1 for t in t1}
            trans = trans | t1 | t2
        best_visited = best_visited | trans

        best_contig_len = all_contig_lens[idxx]
        best_sumLogProb = all_sumLogProbs[idxx]
        best_meanLogProb = all_meanLogProbs[idxx]
        best_meanLogProb_scaled = all_meanLogProbs_scaled[idxx]

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_visited)
        print(f'Elapsed time (get visited): {elapsed}')

        print(f'\nChosen walk with index: {idxx}')
        print(f'len_walk={len(best_walk):<8} len_contig={best_contig_len:<12} ' \
              f'sumLogProb={best_sumLogProb:<12.3f} meanLogProb={best_meanLogProb:<12.4} meanLogProb_scaled={best_meanLogProb_scaled:<12.4}\n')
        
        if best_contig_len < 70000:
            break

        all_contigs.append(best_walk)
        visited |= best_visited
        all_walks_len.append(len(best_walk))
        all_contigs_len.append(best_contig_len)
        print(f'All walks len: {all_walks_len}')
        print(f'All contigs len: {all_contigs_len}\n')

        if len(all_contigs) % 10 == 0:
            checkpoint = {
                'walks': all_contigs,
                'visited': visited,
                'all_walks_len': all_walks_len,
                'all_contigs_len': all_contigs_len
            }
            if not DEBUG:
                try:
                    pickle.dump(checkpoint, open(f'{checkpoint_dir}/checkpoint_tmp.pkl', 'wb'))
                    os.rename(f'{checkpoint_dir}/checkpoint_tmp.pkl', f'{checkpoint_dir}/checkpoint.pkl')
                except OSError:
                    print(f'Checkpoint was not saved. Last available checkopint: {checkpoint_dir}/checkpoint.pkl')
                    raise

    return all_contigs


def inference(data_path, model_path, assembler, savedir, graph_type, device='cpu', dropout=None):
    """Using a pretrained model, get walks and contigs on new data."""
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    nb_pos_enc = hyperparameters['nb_pos_enc']

    batch_norm = hyperparameters['batch_norm']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']

    strategy = hyperparameters['strategy']
    B = hyperparameters['B']
    nb_paths = hyperparameters['num_decoding_paths']
    len_threshold = hyperparameters['len_threshold']
    use_labels = hyperparameters['decode_with_labels']
    load_checkpoint = hyperparameters['load_checkpoint']
    threads = hyperparameters['num_threads']

    # random_search = hyperparameters['random_search']

    # assembly_path = hyperparameters['asms_path']

    device = 'cpu'  # Hardcode, because we cannot do inference on a GPU - usually not enough memory to load the whole graph
    utils.set_seed(seed)
    time_start = datetime.now()

    ds = AssemblyGraphDataset(data_path, assembler)

    inference_dir = os.path.join(savedir, 'decode')
    if not os.path.isdir(inference_dir):
        os.makedirs(inference_dir)

    checkpoint_dir = os.path.join(savedir, 'checkpoint')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    walks_per_graph = []
    contigs_per_graph = []

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'\nelapsed time (loading network and data): {elapsed}\n')
    
    for idx, g in ds:
        if graph_type == 'homo':
            # Get scores
            connectComplement_new(g)
            print(f'==== Processing graph {idx} ====')
            with torch.no_grad():
                time_start_get_scores = datetime.now()
                g = g.to(device)
                x = g.ndata['x'].to(device)
                e = g.edata['e'].to(device)
                pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
                pe_in = (pe_in - pe_in.mean()) / pe_in.std()
                pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
                pe_out = (pe_out - pe_out.mean()) / pe_out.std()
                pe = torch.cat((pe_in, pe_out), dim=1)  # No PageRank
                
                if use_labels:  # Debugging
                    print('Decoding with labels...')
                    g.edata['score'] = g.edata['y'].clone()
                else:
                    print('Decoding with model scores...')
                    predicts_path = os.path.join(inference_dir, f'{idx}_predicts.pt')
                    if os.path.isfile(predicts_path):
                        print(f'Loading the scores from:\n{predicts_path}\n')
                        g.edata['score'] = torch.load(predicts_path)
                    elif RANDOM:
                        g.edata['score'] = torch.ones_like(g.edata['prefix_length']) * 10
                    else:
                        print(f'Loading model parameters from: {model_path}')
                        model = models.SymGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=dropout)
                        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                        model.eval()
                        model.to(device)
                        print(f'Computing the scores with the model...\n')
                        edge_predictions = model(g, x, e, pe)
                        g.edata['score'] = edge_predictions.squeeze()
                        torch.save(g.edata['score'], os.path.join(inference_dir, f'{idx}_predicts.pt'))
                        g = g.edge_subgraph(g.filter_edges(lambda edges: edges.data['prefix_length'] != -1).to(torch.int32), relabel_nodes=False)
        if graph_type == 'hetero':
            # Get scores
            g = connectComplement_hetero(g)
            print(f'==== Processing graph {idx} ====')
            with torch.no_grad():
                time_start_get_scores = datetime.now()
                g = g.to(device)
                x = g.ndata['x'].to(device)
                e = g.edges['real'].data['e'].to(device)
                pe_in = g.ndata['in_deg'].unsqueeze(1).to(device)
                pe_in = (pe_in - pe_in.mean()) / pe_in.std()
                pe_out = g.ndata['out_deg'].unsqueeze(1).to(device)
                pe_out = (pe_out - pe_out.mean()) / pe_out.std()
                pe = torch.cat((pe_in, pe_out), dim=1)  # No PageRank
                
                if use_labels:  # Debugging
                    print('Decoding with labels...')
                    g.edges['real'].data['score'] = g.edges['real'].data['y'].clone()
                else:
                    print('Decoding with model scores...')
                    predicts_path = os.path.join(inference_dir, f'{idx}_predicts.pt')
                    if os.path.isfile(predicts_path):
                        print(f'Loading the scores from:\n{predicts_path}\n')
                        g.edges['real'].data['score'] = torch.load(predicts_path)
                    elif RANDOM:
                        g.edges['real'].data['score'] = torch.ones_like(g.edges['real'].data['prefix_length']) * 10
                    else:
                        print(f'Loading model parameters from: {model_path}')
                        model = models.SymGatedGCNModel_hetero(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=dropout)
                        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                        model.eval()
                        model.to(device)
                        print(f'Computing the scores with the model...\n')
                        edge_predictions = model(g, x, e, pe)  # forward-pass
                        g.edges['real'].data['score'] = edge_predictions.squeeze()
                        torch.save(g.edges['real'].data['score'], os.path.join(inference_dir, f'{idx}_predicts.pt'))
                        g = g.edge_subgraph({('read', 'real', 'read'): g.edges(form='eid', etype='real')}, relabel_nodes=False)

            elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_scores)
            print(f'elapsed time (get_scores): {elapsed}')

        # Load info data
        print(f'Loading successors...')
        with open(f'{data_path}/{assembler}/info/{idx}_succ.pkl', 'rb') as f_succs:
            succs = pickle.load(f_succs)
        print(f'Loading predecessors...')
        with open(f'{data_path}/{assembler}/info/{idx}_pred.pkl', 'rb') as f_preds:
            preds = pickle.load(f_preds)
        print(f'Loading edges...')

        with open(f'{data_path}/{assembler}/info/{idx}_edges.pkl', 'rb') as f_edges:
            edges = pickle.load(f_edges)
        print(f'Done loading the auxiliary graph data!')

        # Get walks
        time_start_get_walks = datetime.now()
        
        # Some prefixes can be <0 and that messes up the assemblies
        if graph_type == 'homo':
            g.edata['prefix_length'] = g.edata['prefix_length'].masked_fill(g.edata['prefix_length']<0, 0)
        if graph_type == 'hetero':
            g.edges['real'].data['prefix_length'] = g.edges['real'].data['prefix_length'].masked_fill(g.edges['real'].data['prefix_length']<0, 0)
        if strategy == 'greedy':
            walks = get_contigs_greedy(g, succs, preds, edges, nb_paths, len_threshold, use_labels, checkpoint_dir, load_checkpoint, device='cpu', threads=threads)
        else:
            print('Invalid decoding strategy')
            raise Exception
 
        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_walks)
        print(f'elapsed time (get_walks): {elapsed}')
        inference_path = os.path.join(inference_dir, f'{idx}_walks.pkl')
        pickle.dump(walks, open(f'{inference_path}', 'wb'))
        
        print(f'Loading reads...')
        with open(f'{data_path}/{assembler}/info/{idx}_reads.pkl', 'rb') as f_reads:
            reads = pickle.load(f_reads)
        print(f'Done!')
        
        time_start_get_contigs = datetime.now()
        contigs = evaluate.walk_to_sequence(walks, g, reads, edges)
        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_contigs)
        print(f'elapsed time (get_contigs): {elapsed}')

        assembly_dir = os.path.join(savedir, f'assembly')
        if not os.path.isdir(assembly_dir):
            os.makedirs(assembly_dir)
        evaluate.save_assembly(contigs, assembly_dir, idx)
        walks_per_graph.append(walks)
        contigs_per_graph.append(contigs)

    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'elapsed time (total): {elapsed}')
    
    if DEBUG:
        exit(0)

    print(f'Found contigs for {data_path}!')
    print(f'Model used: {model_path}')
    print(f'Assembly saved in: {savedir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the dataset')
    parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('--out', type=str, help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Path to the model')
    parser.add_argument('--gtype', type=str, default=None, help='DGL graph type')
    args = parser.parse_args()

    data = args.data
    asm = args.asm
    out = args.out
    model = args.model
    gtype = args.gtype
    if not model:
        model = 'weights/weights.pt'

    inference(data_path=data, assembler=asm, model_path=model, savedir=out, graph_type=gtype)
