import random

import heapq
import torch
import utils

from hyperparameters import get_hyperparameters

hyperparameters = get_hyperparameters()
heuristic_reduce_function = hyperparameters['heuristic_reduce_function']


def greedy_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
    while True:
        path.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        current_neighbors = [n for n in neighbors[current] if not (n in visited_old or n in visited)]
        if not current_neighbors:
            break
        neighbor_edges = [edges[current, n] for n in current_neighbors] # a trick to find edges given nodes
        edge_heuristic_values = heuristic_values[neighbor_edges]
        heuristic_value, index = torch.topk(edge_heuristic_values, k=1, dim=0) # as heuristic_reduce_function is strictly increasing (by design)
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)
        current = current_neighbors[index] # because neighbour_edges is constructed from current_neighbours, they share the same index
    return path, visited, path_heuristic_value


def depth_d_search(start, heuristic_values, neighbors, edges, visited_old, graph, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
    path.append(current)
    visited.add(current)
    visited.add(current ^ 1)
    while True:
        candidate_paths = []
        stack = []
        stack.append((current, []))

        while stack:
            item = stack.pop()
            node = item[0]
            current_path = item[1]
            masked_neighbors = None
            if node in neighbors:
                masked_neighbors = [n for n in neighbors[node] if not (n in visited_old or n in visited)]
            if path and not masked_neighbors:
                candidate_paths.append(current_path)
            for nbr in masked_neighbors:
                new_path = current_path.copy()
                new_path.append(edges[node, nbr])
                if len(new_path) >= parameters['depth']:
                    candidate_paths.append(new_path)
                else:
                    stack.append((nbr, new_path))

        if not candidate_paths:
            break
        candidate_path_heuristic_values = torch.stack([torch.cat((heuristic_values[candidate_path], torch.zeros(parameters['depth'] - len(candidate_path)))) for candidate_path in candidate_paths])
        candidate_path_heuristic_values = torch.sum(candidate_path_heuristic_values, 1)
        heuristic_value, index = torch.topk(candidate_path_heuristic_values, k=1, dim=0)
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)

        best_path = candidate_paths[index]
        last_edge_id = best_path[-1]
        current = graph.find_edges(last_edge_id)[1][0].item()
        for edge_id in best_path:
            dst_node = graph.find_edges(edge_id)[1][0].item()
            path.append(dst_node)
            visited.add(dst_node)
            visited.add(dst_node ^ 1)
    return path, visited, path_heuristic_value


def top_k_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
    while True:
        path.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        current_neighbors = [n for n in neighbors[current] if not (n in visited_old or n in visited)]
        if not current_neighbors:
            break
        neighbor_edges = [edges[current, n] for n in current_neighbors]
        edge_heuristic_values = heuristic_values[neighbor_edges]
        if len(edge_heuristic_values) < parameters['top_k']:
            rand_index = random.randint(0, len(edge_heuristic_values) - 1)
            heuristic_value, index = edge_heuristic_values[rand_index], rand_index
        else:
            top_k_heuristic_values, top_k_indexes = torch.topk(edge_heuristic_values, k=parameters['top_k'], dim=0)
            rand_index = random.randint(0, parameters['top_k'] - 1)
            heuristic_value, index = top_k_heuristic_values[rand_index], top_k_indexes[rand_index]
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)
        current = current_neighbors[index]
    return path, visited, path_heuristic_value


def semi_random_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
    while True:
        path.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        current_neighbors = [n for n in neighbors[current] if not (n in visited_old or n in visited)]
        if not current_neighbors:
            break
        neighbor_edges = [edges[current, n] for n in current_neighbors]
        edge_heuristic_values = heuristic_values[neighbor_edges]
        rand = random.uniform(0, 1)
        if rand <= parameters['random_chance']:
            index = random.randint(0, len(neighbor_edges) - 1)
            heuristic_value = edge_heuristic_values[index]
        else:
            heuristic_value, index = torch.topk(edge_heuristic_values, k=1, dim=0)
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)
        current = current_neighbors[index]
    return path, visited, path_heuristic_value


def random_with_weights_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
    f = parameters['heuristic_value_to_probability']
    while True:
        path.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        current_neighbors = [n for n in neighbors[current] if not (n in visited_old or n in visited)]
        if not current_neighbors:
            break
        neighbor_edges = [edges[current, n] for n in current_neighbors]
        edge_heuristic_values = torch.exp(heuristic_values[neighbor_edges])
        f_edge_heuristic_values = torch.tensor([f(p) for p in edge_heuristic_values])
        edge_chances = f_edge_heuristic_values / torch.sum(f_edge_heuristic_values)
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

        heuristic_value = heuristic_values[index]
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)
        current = current_neighbors[index]
    return path, visited, path_heuristic_value


def random_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = start
    visited = set()
    path = []
    path_heuristic_value = torch.tensor([0.0])
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
        path.append(current)
        visited.add(current)
        visited.add(current ^ 1)
        current_neighbors = [n for n in neighbors[current] if not (n in visited_old or n in visited)]
        if not current_neighbors:
            break
        neighbor_edges = [edges[current, n] for n in current_neighbors]
        edge_heuristic_values = torch.exp(heuristic_values[neighbor_edges])
        f_edge_heuristic_values = torch.tensor([f(p) for p in edge_heuristic_values])
        edge_chances = f_edge_heuristic_values / torch.sum(f_edge_heuristic_values)
        end_intervals = torch.cumsum(edge_chances, 0)
        rand = random.uniform(0, 1)
        if rand <= end_intervals[0]:
            index = 0
        for i in range(1, len(end_intervals)):
            if end_intervals[i - 1] < rand <= end_intervals[i]:
                index = i
                break

        heuristic_value = heuristic_values[index]
        path_heuristic_value = heuristic_reduce_function(path_heuristic_value, heuristic_value)
        current = current_neighbors[index]
    return path, visited, path_heuristic_value


def beam_search(start, heuristic_values, neighbors, edges, visited_old, parameters):
    current = [start]
    visited = set()
    current_paths = [(0, [start], {start})]
    candidate_paths = []
    path_heuristic_value = torch.tensor([0.0])
    while True:
        current_neighbors = [[n for n in neighbors[current_path[1][-1]] if not (n in visited_old or n in current_path[2])] for current_path in current_paths]
        flattened_neighbors = [n for current_path in current_paths for n in neighbors[current_path[1][-1]] if not (n in visited_old or n in current_path[2])]
        # flattened_neighbors = current_neighbors.flatten() # consider aliasing if it doesnt cause unintended side effects, 
        # as it can save time and space
        if not flattened_neighbors:
            break
        neighbor_edges = [edges[current_paths[i][1][-1], n] for i in range(len(current_neighbors)) for n in current_neighbors[i]]
        # for option 2
        dead_end_nodes = set()
        for i in range(len(current_neighbors)):
            if not current_neighbors[i]:
                dead_end_nodes.add(current_paths[i][1][-1])
        for current_path in current_paths:
            if current_path[1][-1] in dead_end_nodes:
                candidate_paths.append(current_path)

        edge_heuristic_values = heuristic_values[neighbor_edges]
        if len(edge_heuristic_values) < parameters['top_b']:
            top_num = len(edge_heuristic_values)
        else:
            top_num = parameters['top_b']
        top_num_indexes = torch.topk(edge_heuristic_values, k=top_num, dim=0)[1]
        flattened_neighbors = torch.tensor(flattened_neighbors)
        neighbor_edges = torch.tensor(neighbor_edges)
        current = flattened_neighbors[top_num_indexes]
        current = list(set(current.tolist()))
        top_k_edges = neighbor_edges[top_num_indexes]

        path_new_neighbors = {}
        for current_path in current_paths:
            for node in current:
                if (current_path[1][-1], node) in edges and edges[current_path[1][-1], node] in top_k_edges:
                    if current_path[1][-1] in path_new_neighbors:
                        path_new_neighbors[current_path[1][-1]].append(node)
                    else:
                        path_new_neighbors[current_path[1][-1]] = [node]
        for node in path_new_neighbors:
            path_new_neighbors[node] = list(set(path_new_neighbors[node]))

        next_current_paths = []
        for current_path in current_paths:
            if current_path[1][-1] in path_new_neighbors:
                path_neighbors = path_new_neighbors[current_path[1][-1]]
                for i in range(len(path_neighbors)):
                    if path_neighbors[i] not in current_path[1]:
                        new_edge = edges[current_path[1][-1], path_neighbors[i]]
                        new_sumLogProb = parameters['accumulator'](current_path[0], heuristic_values[new_edge])
                        if len(next_current_paths) < parameters['top_w'] or new_sumLogProb > next_current_paths[0][0]:
                            new_path = current_path[1].copy()
                            new_path.append(path_neighbors[i])
                            new_visited = current_path[2].copy()
                            new_visited.add(path_neighbors[i])
                            if len(next_current_paths) < parameters['top_w']:
                                heapq.heappush(next_current_paths, (new_sumLogProb, new_path, new_visited))
                            else:
                                heapq.heapreplace(next_current_paths, (new_sumLogProb, new_path, new_visited))
            # for option 3
            # else:
                # candidate_paths.append(current_path)
                
        current_paths = next_current_paths
        
    path_heuristic_value = -float('inf')
    path = []
    for current_path in current_paths:
        curr_path_heuristic_value = current_path[0]
        if curr_path_heuristic_value > path_heuristic_value:
            path = current_path[1]
            path_heuristic_value = curr_path_heuristic_value
    # for both options 2 and 3
    for current_path in candidate_paths:
        curr_path_heuristic_value = current_path[0]
        if curr_path_heuristic_value > path_heuristic_value:
            path = current_path[1]
            path_heuristic_value = curr_path_heuristic_value
    for node in path:
        visited.add(node)
        visited.add(node ^ 1)
    path_heuristic_value = torch.tensor(path_heuristic_value)
    return path, visited, path_heuristic_value