import random

import heapq
import torch
import utils

from hyperparameters import get_hyperparameters

hyperparameters = get_hyperparameters()
heur_reduce_func = hyperparameters['heuristic_reduce_function']
init_heur_val = hyperparameters['initial_heuristic_value']


def greedy_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    while True:
        path.append(curr)
        visited.add(curr)
        visited.add(curr ^ 1)
        curr_neighbors = [n for n in neighbors[curr] if not (n in visited_old or n in visited)]
        if not curr_neighbors:
            break
        neighbor_edges = [edges[curr, n] for n in curr_neighbors] # a trick to find edges given nodes
        edge_heur_vals = heur_vals[neighbor_edges]
        heur_val, index = torch.topk(edge_heur_vals, k=1, dim=0) # as heur_reduce_func is strictly increasing (by design)
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index] # because neighbour_edges is constructed from curr_neighbours, they share the same index
    return path, visited, path_heur_val


def depth_d_search(start, heur_vals, neighbors, edges, visited_old, graph, parameters):
    current = start
    visited = set()
    walk = []
    sumLogProb = torch.tensor([0.0])
    walk.append(current)
    visited.add(current)
    visited.add(current ^ 1)
    while True:
        paths = []
        stack = []
        stack.append((current, []))

        while stack:
            item = stack.pop()
            node = item[0]
            path = item[1]
            masked_neighbors = None
            if node in neighbors:
                masked_neighbors = [n for n in neighbors[node] if not (n in visited_old or n in visited)]
            if path and not masked_neighbors:
                paths.append(path)
            for nbr in masked_neighbors:
                new_path = path.copy()
                new_path.append(edges[node, nbr])
                if len(new_path) >= parameters['depth']:
                    paths.append(new_path)
                else:
                    stack.append((nbr, new_path))

        if not paths:
            break
        path_p = torch.stack([torch.cat((heur_vals[path], torch.zeros(parameters['depth'] - len(path)))) for path in paths])
        path_p = torch.sum(path_p, 1)
        logProb, index = torch.topk(path_p, k=1, dim=0)
        sumLogProb += logProb

        best_path = paths[index]
        last_edge_id = best_path[-1]
        current = graph.find_edges(last_edge_id)[1][0].item()
        for edge_id in best_path:
            dst_node = graph.find_edges(edge_id)[1][0].item()
            walk.append(dst_node)
            visited.add(dst_node)
            visited.add(dst_node ^ 1)
    return walk, visited, sumLogProb


def top_k_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    while True:
        path.append(curr)
        visited.add(curr)
        visited.add(curr ^ 1)
        curr_neighbors = [n for n in neighbors[curr] if not (n in visited_old or n in visited)]
        if not curr_neighbors:
            break
        neighbor_edges = [edges[curr, n] for n in curr_neighbors]
        edge_heur_vals = heur_vals[neighbor_edges]
        if len(edge_heur_vals) < parameters['top_k']:
            rand_index = random.randint(0, len(edge_heur_vals) - 1)
            heur_val, index = edge_heur_vals[rand_index], rand_index
        else:
            top_k_heur_vals, top_k_indexes = torch.topk(edge_heur_vals, k=parameters['top_k'], dim=0)
            rand_index = random.randint(0, parameters['top_k'] - 1)
            heur_val, index = top_k_heur_vals[rand_index], top_k_indexes[rand_index]
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index]
    return path, visited, path_heur_val


def semi_random_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    while True:
        path.append(curr)
        visited.add(curr)
        visited.add(curr ^ 1)
        curr_neighbors = [n for n in neighbors[curr] if not (n in visited_old or n in visited)]
        if not curr_neighbors:
            break
        neighbor_edges = [edges[curr, n] for n in curr_neighbors]
        edge_heur_vals = heur_vals[neighbor_edges]
        rand = random.uniform(0, 1)
        if rand <= parameters['random_chance']:
            index = random.randint(0, len(neighbor_edges) - 1)
            heur_val = edge_heur_vals[index]
        else:
            heur_val, index = torch.topk(edge_heur_vals, k=1, dim=0)
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index]
    return path, visited, path_heur_val


def random_with_weights_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    f = parameters['heuristic_values_to_probability']
    while True:
        path.append(curr)
        visited.add(curr)
        visited.add(curr ^ 1)
        curr_neighbors = [n for n in neighbors[curr] if not (n in visited_old or n in visited)]
        if not curr_neighbors:
            break
        neighbor_edges = [edges[curr, n] for n in curr_neighbors]
        edge_heur_vals = torch.exp(heur_vals[neighbor_edges])
        f_edge_heur_vals = torch.tensor([f(p) for p in edge_heur_vals])
        edge_chances = f_edge_heur_vals / torch.sum(f_edge_heur_vals)
        end_intervals = torch.cumsum(edge_chances, 0) # e.g. [0.25, 0.45, 0.6, 1]
        rand = random.uniform(0, 1)
        if rand <= end_intervals[0]:
            index = 0
        else:
            for i in range(1, len(end_intervals)):
                if end_intervals[i - 1] < rand <= end_intervals[i]:
                    index = i
                    break
        # start_interval = torch.cat(torch.tensor([0]), end_interval) # 
        # start_interval.pop()

        heur_val = heur_vals[index]
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index]
    return path, visited, path_heur_val


def random_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    coeffs = []
    utils.set_seed(parameters['seed'])
    for i in range(parameters['polynomial_degree'] + 1):
        coeff = random.uniform(0, 1)
        coeff = round(coeff, parameters['precision_in_decimal_places'])
        coeffs.append(coeff)
    # coeffs[0] = 0 # to remove constant; explained later
    def func(x, coeffs):
        result = 0
        for i in range(len(coeffs)):
            result += coeffs[i] * x ** i
        return result
    f = lambda x: func(x, coeffs)
    while True:
        path.append(curr)
        visited.add(curr)
        visited.add(curr ^ 1)
        curr_neighbors = [n for n in neighbors[curr] if not (n in visited_old or n in visited)]
        if not curr_neighbors:
            break
        neighbor_edges = [edges[curr, n] for n in curr_neighbors]
        edge_heur_vals = torch.exp(heur_vals[neighbor_edges])
        f_edge_heur_vals = torch.tensor([f(p) for p in edge_heur_vals])
        edge_chances = f_edge_heur_vals / torch.sum(f_edge_heur_vals)
        end_intervals = torch.cumsum(edge_chances, 0)
        rand = random.uniform(0, 1)
        if rand <= end_intervals[0]:
            index = 0
        for i in range(1, len(end_intervals)):
            if end_intervals[i - 1] < rand <= end_intervals[i]:
                index = i
                break

        heur_val = heur_vals[index]
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index]
    return path, visited, path_heur_val


def beam_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    curr = [start]
    visited = set()
    curr_paths = [(init_heur_val, [start], {start})]
    candidate_paths = []
    path_heur_val = torch.tensor([init_heur_val])
    while True:
        curr_neighbors = [[n for n in neighbors[curr_path[1][-1]] if not (n in visited_old or n in curr_path[2])] for curr_path in curr_paths]
        flattened_neighbors = [n for curr_path in curr_paths for n in neighbors[curr_path[1][-1]] if not (n in visited_old or n in curr_path[2])]
        # flattened_neighbors = curr_neighbors.flatten() # consider aliasing if it doesnt cause unintended side effects, 
        # as it can save time and space
        if not flattened_neighbors:
            break
        neighbor_edges = [edges[curr_paths[i][1][-1], n] for i in range(len(curr_neighbors)) for n in curr_neighbors[i]]
        # for option 2
        dead_end_nodes = set()
        for i in range(len(curr_neighbors)):
            if not curr_neighbors[i]:
                dead_end_nodes.add(curr_paths[i][1][-1])
        for curr_path in curr_paths:
            if curr_path[1][-1] in dead_end_nodes:
                candidate_paths.append(curr_path)

        edge_heur_vals = heur_vals[neighbor_edges]
        if len(edge_heur_vals) < parameters['top_b']:
            top_num = len(edge_heur_vals)
        else:
            top_num = parameters['top_b']
        top_num_indexes = torch.topk(edge_heur_vals, k=top_num, dim=0)[1]
        flattened_neighbors = torch.tensor(flattened_neighbors)
        neighbor_edges = torch.tensor(neighbor_edges)
        curr = flattened_neighbors[top_num_indexes]
        curr = list(set(curr.tolist()))
        top_k_edges = neighbor_edges[top_num_indexes]

        paths_new_neighbors = {}
        for curr_path in curr_paths:
            for node in curr:
                if (curr_path[1][-1], node) in edges and edges[curr_path[1][-1], node] in top_k_edges:
                    if curr_path[1][-1] in paths_new_neighbors:
                        paths_new_neighbors[curr_path[1][-1]].append(node)
                    else:
                        paths_new_neighbors[curr_path[1][-1]] = [node]
        for node in paths_new_neighbors:
            paths_new_neighbors[node] = list(set(paths_new_neighbors[node]))

        next_curr_paths = []
        for curr_path in curr_paths:
            if curr_path[1][-1] in paths_new_neighbors:
                path_neighbors = paths_new_neighbors[curr_path[1][-1]]
                for i in range(len(path_neighbors)):
                    if path_neighbors[i] not in curr_path[1]:
                        new_edge = edges[curr_path[1][-1], path_neighbors[i]]
                        new_sumLogProb = heur_reduce_func(curr_path[0], heur_vals[new_edge])
                        if len(next_curr_paths) < parameters['top_w'] or new_sumLogProb > next_curr_paths[0][0]:
                            new_path = curr_path[1].copy()
                            new_path.append(path_neighbors[i])
                            new_visited = curr_path[2].copy()
                            new_visited.add(path_neighbors[i])
                            if len(next_curr_paths) < parameters['top_w']:
                                heapq.heappush(next_curr_paths, (new_sumLogProb, new_path, new_visited))
                            else:
                                heapq.heapreplace(next_curr_paths, (new_sumLogProb, new_path, new_visited))
            # for option 3
            # else:
                # candidate_paths.append(curr_path)
                
        curr_paths = next_curr_paths
        
    path_heur_val = -float('inf')
    path = []
    for curr_path in curr_paths:
        curr_path_heur_val = curr_path[0]
        if curr_path_heur_val > path_heur_val:
            path = curr_path[1]
            path_heur_val = curr_path_heur_val
    # for both options 2 and 3
    for curr_path in candidate_paths:
        curr_path_heur_val = curr_path[0]
        if curr_path_heur_val > path_heur_val:
            path = curr_path[1]
            path_heur_val = curr_path_heur_val
    for node in path:
        visited.add(node)
        visited.add(node ^ 1)
    path_heur_val = torch.tensor(path_heur_val)
    return path, visited, path_heur_val