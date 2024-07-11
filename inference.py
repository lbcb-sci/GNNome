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

import heapq
import torch
import torch.nn.functional as F
import dgl
import warnings

from decoding_algorithms import greedy_search, depth_d_search, top_k_search, semi_random_search, \
    random_with_weights_search, random_search, beam_search
from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import evaluate
import utils

DEBUG = False

hyperparameters = get_hyperparameters()


def get_contig_length(walk, graph):
    total_length = 0
    idx_src = walk[:-1]
    idx_dst = walk[1:]
    prefix = graph.edges[idx_src, idx_dst].data['prefix_length']
    total_length = prefix.sum().item()
    total_length += graph.ndata['read_length'][walk[-1]]
    return total_length


def get_subgraph(graph, visited, device):
    """Remove the visited nodes from the graph."""
    remove_node_idx = torch.LongTensor([item for item in visited])
    list_node_idx = torch.arange(graph.num_nodes())
    keep_node_idx = torch.ones(graph.num_nodes())
    keep_node_idx[remove_node_idx] = 0
    keep_node_idx = list_node_idx[keep_node_idx==1].int().to(device)

    sub_g = dgl.node_subgraph(graph, keep_node_idx, store_ids=True)
    sub_g.ndata['idx_nodes'] = torch.arange(sub_g.num_nodes()).to(device)
    map_subg_to_g = sub_g.ndata[dgl.NID]
    return sub_g, map_subg_to_g


def sample_edges(prob_edges, nb_paths):
    """Sample edges with Bernoulli sampling."""
    if prob_edges.shape[0] > 2**24:
        prob_edges = prob_edges[:2**24]  # torch.distributions.categorical.Categorical does not support tensors longer than 2**24
        
    random_search = False
    if random_search:
        idx_edges = torch.randint(0, prob_edges.shape[0], (nb_paths,))
        return idx_edges
    
    prob_edges = prob_edges.masked_fill(prob_edges<1e-9, 1e-9)
    prob_edges = prob_edges/ prob_edges.sum()
    prob_edges_nb_paths = prob_edges.repeat(nb_paths, 1)
    idx_edges = torch.distributions.categorical.Categorical(prob_edges_nb_paths).sample()
    return idx_edges


def greedy_forwards(start, heuristic_values, neighbors, predecessors, edges, visited_old, graph, parameters):
    """Greedy walk forwards."""
    strategy = parameters['strategy']
    if strategy == 'greedy':
        return greedy_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    if strategy == 'depth_d':
        return depth_d_search(start, heuristic_values, neighbors, edges, visited_old, graph, parameters)
    if strategy == 'top_k':
        return top_k_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    if strategy == 'semi_random':
        return semi_random_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    if strategy == 'weighted_random':
        return random_with_weights_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    if strategy == 'random_search':
        return random_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    if strategy == 'beam':
        return beam_search(start, heuristic_values, neighbors, edges, visited_old, parameters)
    raise ValueError('Unknown strategy. Aborting process...')


def greedy_backwards_rc(start, heuristic_values, predecessors, neighbors, edges, visited_old, graph, parameters):
    """Greedy walk backwards."""
    walk, visited, sum_heuristic_value = greedy_forwards(start ^ 1, heuristic_values, neighbors, predecessors, edges, visited_old, graph, parameters)
    walk = list(reversed([edge_id ^ 1 for edge_id in walk]))
    return walk, visited, sum_heuristic_value
    

def run_greedy_both_ways(src, dst, heuristic_values, succs, preds, edges, visited, graph, parameters):
    walk_f, visited_f, sum_heuristic_value_f = greedy_forwards(dst, heuristic_values, succs, preds, edges, visited, graph, parameters)
    walk_b, visited_b, sum_heuristic_value_b = greedy_backwards_rc(src, heuristic_values, preds, succs, edges, visited | visited_f, graph, parameters)
    return walk_f, walk_b, visited_f, visited_b, sum_heuristic_value_f, sum_heuristic_value_b


def get_contigs_greedy(graph, succs, preds, edges, parameters, nb_paths=50, len_threshold=20, use_labels=False, checkpoint_dir=None, load_checkpoint=False, device='cpu', threads=32):
    """Iteratively search for contigs in a graph until the threshold is met."""
    graph = graph.to('cpu')
    all_contigs = []
    all_walks_len = []
    all_contigs_len = []
    visited = set()
    idx_contig = -1

    B = 1

    if use_labels:
        scores = graph.edata['y'].to('cpu')
        scores = scores.masked_fill(scores<1e-9, 1e-9)
        probs = scores
    else:
        scores = graph.edata['score'].to('cpu')
        probs = torch.sigmoid(graph.edata['score'].to('cpu'))
    p = lambda src, dst: probs[edges[src, dst]]
    l = lambda src, dst: graph.ndata['read_length'][dst]
    heuristic_function = hyperparameters['heuristic_function']
    g = lambda src, dst: heuristic_function(p(src, dst), l(src, dst))
    heuristic_values = torch.tensor([g(src, dst) for (src, dst) in edges])

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
        sub_g, map_subg_to_g = get_subgraph(graph, visited, 'cpu')
        if sub_g.num_edges() == 0:
            break
        
        if use_labels:  # Debugging
            prob_edges = sub_g.edata['y']
        else:
            prob_edges = torch.sigmoid(sub_g.edata['score']).squeeze()

        idx_edges = sample_edges(prob_edges, nb_paths)

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_sample_edges)
        print(f'Elapsed time (sample edges): {elapsed}')

        all_walks = []
        all_visited_iter = []

        all_contig_lens = []
        all_sum_heuristic_value = []
        all_mean_heuristic_value = []
        all_mean_heuristic_value_scaled = []

        print(f'\nidx_contig: {idx_contig}, nb_processed_nodes: {len(visited)}, ' \
              f'nb_remaining_nodes: {graph.num_nodes() - len(visited)}, nb_original_nodes: {graph.num_nodes()}')

        # Get nb_paths paths for a single iteration, then take the longest one
        time_start_get_candidates = datetime.now()

        with ThreadPoolExecutor(1) as executor:
            if DEBUG:
                print(f'Starting with greedy for one candidate', flush=True)
                all_cand_time = datetime.now()
            results = {}
            start_times = {}
            for e, idx in enumerate(idx_edges):
                src_init_edges = map_subg_to_g[sub_g.edges()[0][idx]].item()
                dst_init_edges = map_subg_to_g[sub_g.edges()[1][idx]].item()
                start_times[e] = datetime.now()
                if DEBUG:
                    print(f'About to submit job - decoding from edge {e}: {src_init_edges, dst_init_edges}', flush=True)
                future = executor.submit(run_greedy_both_ways, src_init_edges, dst_init_edges, heuristic_values, succs, preds, edges, visited, graph, parameters)
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
                walk_f, walk_b, visited_f, visited_b, sum_heuristic_value_f, sum_heuristic_value_b = f.result()
                if DEBUG:
                    print(f'Finished with candidate {e}: {k}\t' \
                        f'Time needed: {utils.timedelta_to_str(datetime.now() - start_times[e])}')         
                walk_it = walk_b + walk_f
                visited_iter = visited_f | visited_b
                sum_heuristic_value_it = sum_heuristic_value_f.item() + sum_heuristic_value_b.item()
                len_walk_it = len(walk_it)
                len_contig_it = get_contig_length(walk_it, graph).item()
                if k[0] == k[1]:
                    len_walk_it = 1
                
                if len_walk_it > 2:
                    mean_heuristic_value_it = sum_heuristic_value_it / (len_walk_it - 2)  # len(walk_f) - 1 + len(walk_b) - 1  <-> starting edge is neglected
                    try:
                        mean_heuristic_value_scaled_it = mean_heuristic_value_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        mean_heuristic_value_scaled_it = 0
                elif len_walk_it == 2:
                    mean_heuristic_value_it = 0.0
                    try:
                        mean_heuristic_value_scaled_it = mean_heuristic_value_it / math.sqrt(len_contig_it)
                    except ValueError:
                        print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12}')
                        print(f'Value error: something is wrong here!')
                        mean_heuristic_value_scaled_it = 0
                else:  # len_walk_it == 1 <-> SELF-LOOP!
                    len_contig_it = 0
                    sum_heuristic_value_it = 0.0
                    mean_heuristic_value_it = 0.0
                    mean_heuristic_value_scaled_it = 0.0
                    print(f'SELF-LOOP!')
                print(f'{indx:<3}: src={k[0]:<8} dst={k[1]:<8} len_walk={len_walk_it:<8} len_contig={len_contig_it:<12} ' \
                      f'sum_heuristic_value={sum_heuristic_value_it:<12.3f} mean_heuristic_value={mean_heuristic_value_it:<12.4} mean_heuristic_value_scaled={mean_heuristic_value_scaled_it:<12.4}')

                indx += 1
                all_walks.append(walk_it)
                all_visited_iter.append(visited_iter)
                all_contig_lens.append(len_contig_it)
                all_sum_heuristic_value.append(sum_heuristic_value_it)
                all_mean_heuristic_value.append(mean_heuristic_value_it)
                all_mean_heuristic_value_scaled.append(mean_heuristic_value_scaled_it)

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
        best_sum_heuristic_value = all_sum_heuristic_value[idxx]
        best_mean_heuristic_value = all_mean_heuristic_value[idxx]
        best_mean_heuristic_value_scaled = all_mean_heuristic_value_scaled[idxx]

        elapsed = utils.timedelta_to_str(datetime.now() - time_start_get_visited)
        print(f'Elapsed time (get visited): {elapsed}')

        print(f'\nChosen walk with index: {idxx}')
        print(f'len_walk={len(best_walk):<8} len_contig={best_contig_len:<12} ' \
              f'sum_heuristic_value={best_sum_heuristic_value:<12.3f} mean_heuristic_value={best_mean_heuristic_value:<12.4} mean_heuristic_value_scaled={best_mean_heuristic_value_scaled:<12.4}\n')
        
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


def inference(data_path, model_path, assembler, savedir, parameters, device='cpu', dropout=None):
    """Using a pretrained model, get walks and contigs on new data."""
    hyperparameters = get_hyperparameters()
    seed = parameters['seed'] if 'seed' in parameters else hyperparameters['seed']
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

    for idx, graph in ds:
        # Get scores
        print(f'==== Processing graph {idx} ====')
        with torch.no_grad():
            time_start_get_scores = datetime.now()
            graph = graph.to(device)
            x = graph.ndata['x'].to(device)
            e = graph.edata['e'].to(device)
            pe_in = graph.ndata['in_deg'].unsqueeze(1).to(device)
            pe_in = (pe_in - pe_in.mean()) / pe_in.std()
            pe_out = graph.ndata['out_deg'].unsqueeze(1).to(device)
            pe_out = (pe_out - pe_out.mean()) / pe_out.std()
            pe = torch.cat((pe_in, pe_out), dim=1)  # No PageRank
            
            if use_labels:  # Debugging
                print('Decoding with labels...')
                graph.edata['score'] = graph.edata['y'].clone()
            else:
                print('Decoding with model scores...')
                predicts_path = os.path.join(inference_dir, f'{idx}_predicts.pt')
                if os.path.isfile(predicts_path):
                    print(f'Loading the scores from:\n{predicts_path}\n')
                    graph.edata['score'] = torch.load(predicts_path)
                else:
                    print(f'Loading model parameters from: {model_path}')
                    model = models.SymGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, batch_norm, nb_pos_enc, dropout=dropout)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                    model.eval()
                    model.to(device)
                    print(f'Computing the scores with the model...\n')
                    edge_predictions = model(graph, x, e, pe)
                    graph.edata['score'] = edge_predictions.squeeze()
                    torch.save(graph.edata['score'], os.path.join(inference_dir, f'{idx}_predicts.pt'))

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
        graph.edata['prefix_length'] = graph.edata['prefix_length'].masked_fill(graph.edata['prefix_length']<0, 0)
        
        if strategy == 'greedy':
            walks = get_contigs_greedy(graph, succs, preds, edges, parameters, nb_paths, len_threshold, use_labels, checkpoint_dir, load_checkpoint, device='cpu', threads=threads)
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
        contigs = evaluate.walk_to_sequence(walks, graph, reads, edges)
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


def parse_args_based_on_strategy(args):
    strategy = args.strat
    parameters = {'strategy': strategy}
    exceptions = []

    def parse_seed():
        try:
            seed = int(args.seed)
        except (TypeError, ValueError):
            if args.seed is None:
                print("Defaulting to seed in hyperparameters...")
            else:
                exceptions.append(Exception("Seed provided must be an integer"))
        else:
            parameters['seed'] = seed

    if strategy == 'greedy':
        return parameters
    elif strategy == 'depth_d':
        try:
            depth = int(args.depth)
        except (TypeError, ValueError):
            raise Exception("Depth must be a positive integer")
        if depth <= 0:
            raise Exception("Depth must be a positive integer")
        parameters['depth'] = depth

    elif strategy == 'top_k':
        try:
            top_k = int(args.k)
        except (TypeError, ValueError):
            exceptions.append(Exception("Top k must be a positive integer"))
        else:
            if top_k <= 0:
                exceptions.append(Exception("Top k must be a positive integer"))
            else:
                parameters['top_k'] = top_k
        parse_seed()

    elif strategy == 'semi_random':
        try:
            random_chance = float(args.chance)
        except (TypeError, ValueError):
            exceptions.append(Exception("Chance must be between 0 and 1 (inclusive)"))
        else:
            if not 0 <= random_chance <= 1:
                exceptions.append(Exception("Chance must be between 0 and 1 (inclusive)"))
            else:
                parameters['random_chance'] = random_chance
        parse_seed()

    elif strategy == 'weighted_random':
        # for now, only polynomials (with decimal representation of coefficients) allowed!
        try:
            coeffs = list(map(float, args.coeffs.split(',')))
        except (AttributeError, TypeError, ValueError):
            exceptions.append(Exception("Coefficients must be a stream of numbers, separated by commas"))
        else:
            def func(x, coeffs):
                result = 0
                for i in range(len(coeffs)):
                    result += coeffs[i] * x ** i
                return result
            
            coeffs.reverse()
            f = lambda x: func(x, coeffs)
            parameters['heuristic_value_to_probability'] = f
        parse_seed()

    elif strategy == 'random_search':
        try:
            polynomial_degree = int(args.deg)
        except (TypeError, ValueError):
            exceptions.append(Exception("Degree must be a positive integer"))
        else: 
            if polynomial_degree <= 0:
                exceptions.append(Exception("Degree must be a positive integer"))
            else:
                parameters['polynomial_degree'] = polynomial_degree
        try:
            precision_in_decimal_places = int(args.dp)
        except (TypeError, ValueError):
            exceptions.append(Exception("Number of decimal places must be a non-negative integer"))
        else:
            if precision_in_decimal_places < 0:
                exceptions.append(Exception("Number of decimal places must be a non-negative integer"))
            else:
                parameters['precision_in_decimal_places'] = precision_in_decimal_places
        parse_seed()

    elif strategy == 'beam':
        try:
            top_b = int(args.b)
        except (TypeError, ValueError):
            exceptions.append(Exception("Top b must be a positive integer"))
        else:
            if top_b <= 0:
                exceptions.append(Exception("Top b must be a positive integer"))
            else:
                parameters['top_b'] = top_b
        try:
            top_w = int(args.w)
        except (TypeError, ValueError):
            exceptions.append(Exception("Top w must be a positive integer"))
        else:
            if top_w <= 0:
                exceptions.append(Exception("Top w must be a positive integer"))
            elif top_w < top_b:
                warnings.warn("Top w is smaller than top b. Effectively, top b is equal to top w.")
            else:
                parameters['top_w'] = top_w
        option = args.opt
        if option not in {1, 2, 3}:
            exceptions.append(Exception("Option must be 1, 2 or 3"))

    else:
        raise ValueError('Unknown strategy. Aborting decoding process...')
    
    if exceptions:
        exception_message = "\n".join(str(exception) for exception in exceptions)
        raise Exception(exception_message)
    return parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the dataset')
    parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('--out', type=str, help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Path to the model')
    parser.add_argument('--seed', type=str, default=None, help='Seed used for random processes')
    parser.add_argument('--strat', type=str, default='greedy', help='Strategy used in decoding')
    parser.add_argument('--depth', type=str, default=2, help='Depth of path search')
    parser.add_argument('--k', type=str, default=3, help='Top k edges to select randomly from')
    parser.add_argument('--chance', type=str, default=0.125, help='Probability of selecting random edge')
    parser.add_argument('--coeffs', type=str, default='1,0,0,0,0', help='Coefficients of polynomial, starting from highest power, separated by commas')
    parser.add_argument('--deg', type=str, default=4, help='Degree of polynomial')
    parser.add_argument('--dp', type=str, default=1, help='Number of decimal places in which coefficients are rounded off to')
    parser.add_argument('--b', type=str, default=2, help='Top b edges to select')
    parser.add_argument('--w', type=str, default=2, help='Top w walks to keep')
    parser.add_argument('--opt', type=str, default=2, help='Option to keep walks by')
    parser.add_argument('--hf', type=str, help='Heuristic function of an edge')
    parser.add_argument('--hr', type=str, help='Reduce function that aggregates the heuristic values')
    args = parser.parse_args()
    params = parse_args_based_on_strategy(args)
    data = args.data
    asm = args.asm
    out = args.out
    model = args.model
    if not model:
        model = 'weights/weights.pt'

    inference(data_path=data, assembler=asm, model_path=model, savedir=out, parameters=params)