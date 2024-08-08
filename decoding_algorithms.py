from dataclasses import dataclass
import random

import heapq
from typing import List
import torch
import utils

from hyperparameters import get_hyperparameters

hyperparameters = get_hyperparameters()
heur_reduce_func = hyperparameters['heuristic_reduce_function']
init_heur_val = hyperparameters['initial_heuristic_value']


def greedy_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    """
    Selects the edge (and hence the node connected to the edge) with the highest heuristic value in each iteration.
    
    Refer to search_forwards for more information about the function parameters.
    """
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


def depth_d_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    """
    Selects the path of depth d (d-path) with the highest heuristic value in each iteration. The heuristic value of a 
    d-path is obtained by using a reduce function (with a configurable binary operator) over the d nodes.
    
    Refer to search_forwards for more information about the function parameters.

    Parameters
    ----------
    parameters: dict[str, int]
        Has the form {'depth': d}
    """
    curr = start
    visited = set()
    path = []
    path_heur_val = torch.tensor([init_heur_val])
    path.append(curr)
    visited.add(curr)
    visited.add(curr ^ 1)
    while True:
        best_d_path_with_heur = None
        stack = []
        stack.append((init_heur_val, [curr]))
        while stack:
            item = stack.pop()
            curr_heur_val = item[0]
            curr_path = item[1]
            last_node_id = curr_path[-1]
            masked_neighbors = None
            if last_node_id in neighbors:
                masked_neighbors = [n for n in neighbors[last_node_id] if not (n in visited_old or n in visited)]
            if len(curr_path) > 1 and not masked_neighbors:
                if not best_d_path_with_heur or curr_heur_val > best_d_path_with_heur[0]:
                    best_d_path_with_heur = (curr_heur_val, curr_path)
            for nbr in masked_neighbors:
                new_path = curr_path.copy()
                new_path.append(nbr)
                new_heur_val = heur_reduce_func(curr_heur_val, heur_vals[edges[last_node_id, nbr]])
                if len(new_path) > parameters['depth']:
                    if not best_d_path_with_heur or new_heur_val > best_d_path_with_heur[0]:
                        best_d_path_with_heur = (new_heur_val, new_path)
                else:
                    stack.append((new_heur_val, new_path))

        if not best_d_path_with_heur:
            break
        heur_val = best_d_path_with_heur[0]
        best_d_path = best_d_path_with_heur[1]
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        # exclude curr node to prevent double counting
        for i in range(1, len(best_d_path)):
            path.append(best_d_path[i])
            visited.add(best_d_path[i])
            visited.add(best_d_path[i] ^ 1)
        curr = path[-1]
    return path, visited, path_heur_val


def top_k_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    """
    Selects a random edge from edges with the k highest heuristic values in each iteration.
    
    Refer to search_forwards for more information about the function parameters.

    Parameters
    ----------
    parameters: dict[str, int]
        Has the form {'top_k': k}
    """
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
    """
    Selects the edge with the highest heuristic value with a certain chance. Otherwise, selects a random neighboring edge.
    
    Refer to search_forwards for more information about the function parameters.

    Parameters
    ----------
    parameters: dict[str, float]
        Has the form {'random_chance': chance}
    """
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


def weighted_random_search(start, heur_vals, f_heur_vals, neighbors, edges, visited_old, parameters):
    """
    Selects an edge with a probability, defined by the function f applied to its heuristic value, divided by the sum of
    f applied to the heuristic values of the neighbors. This allows the events of each edge being selected to be 
    mutually exclusive.
    
    Refer to search_forwards for more information about the function parameters.

    Parameters
    ----------
    parameters: dict[str, Any]
        Has the form {'heuristic_value_to_probability': f}
    """
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
        f_edge_heur_vals = f_heur_vals[neighbor_edges]
        edge_chances = f_edge_heur_vals / torch.sum(f_edge_heur_vals)
        end_intervals = torch.cumsum(edge_chances, 0) # e.g. [0.25, 0.45, 0.6, 1]
        rand = random.uniform(0, 1)
        if torch.sum(f_edge_heur_vals) == 0:
            index = random.randint(0, len(end_intervals) - 1)
        elif rand <= end_intervals[0]:
            index = 0
        else:
            for i in range(1, len(end_intervals)):
                if end_intervals[i - 1] < rand <= end_intervals[i]:
                    index = i
                    break

        heur_val = heur_vals[index]
        path_heur_val = heur_reduce_func(path_heur_val, heur_val)
        curr = curr_neighbors[index]
    return path, visited, path_heur_val


def beam_search(start, heur_vals, neighbors, edges, visited_old, parameters):
    """
    Selects edges with the b highest heuristic values in each iteration, and keep walks with the k highest heuristic 
    values. Whether a walk should be considered for the k highest heuristic values is determined by option.
    
    Refer to search_forwards for more information about the function parameters.

    Parameters
    ----------
    parameters: dict[str, int]
        Has the form {'top_b': b, 'top_w': w, 'option': opt}
    """
    @dataclass
    class PathInfo:
        heur_val: float
        path: List[int]
        visited: List[int]

        def __lt__(self, other):
            assert isinstance(other, PathInfo)
            return self.heur_val < other.heur_val
        
        def __eq__(self, other):
            assert isinstance(other, PathInfo)
            return self.heur_val == other.heur_val

        def get_last_node(self):
            assert self.path
            if self.path:
                return self.path[-1]
            
    visited = set()
    paths_info = [PathInfo(init_heur_val, [start], {start})]
    candidate_paths = []
    path_heur_val = torch.tensor([init_heur_val])
    while True:
        # if stop and stop_preds[curr] > stop_thr: break
        curr_neighbors = [[n for n in neighbors[path_info.get_last_node()] if not (n in visited_old or n in path_info.visited)] for path_info in paths_info]
        flattened_neighbors = [n for path_info in paths_info for n in neighbors[path_info.get_last_node()] if not (n in visited_old or n in path_info.visited)]
        # flattened_neighbors = curr_neighbors.flatten() # consider aliasing if it doesnt cause unintended side effects, 
        # as it can save time and space
        if not flattened_neighbors:
            break
        neighbor_edges = [edges[paths_info[i].get_last_node(), n] for i in range(len(curr_neighbors)) for n in curr_neighbors[i]]
        if parameters['option'] == 2:
            dead_end_nodes = set()
            for i in range(len(curr_neighbors)):
                if not curr_neighbors[i]:
                    dead_end_nodes.add(paths_info[i].get_last_node())
            for path_info in paths_info:
                if path_info.get_last_node() in dead_end_nodes:
                    candidate_paths.append(path_info)

        edge_heur_vals = heur_vals[neighbor_edges]
        if len(edge_heur_vals) < parameters['top_b']:
            top_num = len(edge_heur_vals)
        else:
            top_num = parameters['top_b']
        top_num_indexes = torch.topk(edge_heur_vals, k=top_num, dim=0)[1]
        flattened_neighbors = torch.tensor(flattened_neighbors)
        neighbor_edges = torch.tensor(neighbor_edges)
        next_nodes = flattened_neighbors[top_num_indexes]
        next_nodes = list(set(next_nodes.tolist()))
        top_k_edges = neighbor_edges[top_num_indexes]

        # out of the top k edges, construct the neighbors dictionary
        # this is done by checking if (last node of a path, node of next_node) is one of the top k edges, with time
        # complexity of O(wb) (no parallel edges)
        paths_new_neighbors = {}
        for path_info in paths_info:
            for node in next_nodes:
                if (path_info.get_last_node(), node) in edges and edges[path_info.get_last_node(), node] in top_k_edges:
                    if path_info.get_last_node() in paths_new_neighbors:
                        paths_new_neighbors[path_info.get_last_node()].append(node)
                    else:
                        paths_new_neighbors[path_info.get_last_node()] = [node]
        for node in paths_new_neighbors:
            paths_new_neighbors[node] = list(set(paths_new_neighbors[node]))

        # for each path, for each neighbor of the last node of the path, make a new PathInfo
        # only the top w PathInfo are kept
        next_paths_info: List[PathInfo] = []
        for path_info in paths_info:
            if path_info.get_last_node() in paths_new_neighbors:
                path_neighbors = paths_new_neighbors[path_info.get_last_node()]
                for i in range(len(path_neighbors)):
                    if path_neighbors[i] not in path_info.path:
                        new_edge = edges[path_info.get_last_node(), path_neighbors[i]]
                        new_path_heur_val = heur_reduce_func(path_info.heur_val, heur_vals[new_edge])
                        if len(next_paths_info) < parameters['top_w'] or new_path_heur_val > next_paths_info[0].heur_val:
                            new_path = path_info.path.copy()
                            new_path.append(path_neighbors[i])
                            new_visited = path_info.visited.copy()
                            new_visited.add(path_neighbors[i])
                            if len(next_paths_info) < parameters['top_w']:
                                heapq.heappush(next_paths_info, PathInfo(new_path_heur_val, new_path, new_visited))
                            else:
                                heapq.heapreplace(next_paths_info, PathInfo(new_path_heur_val, new_path, new_visited))
            elif parameters['option'] == 3:
                candidate_paths.append(path_info)
                
        paths_info = next_paths_info
        
    path_heur_val = -float('inf')
    path = []
    for path_info in paths_info:
        curr_path_heur_val = path_info.heur_val
        if curr_path_heur_val > path_heur_val:
            path = path_info.path
            path_heur_val = curr_path_heur_val
    if parameters['option'] == 2 or parameters['option'] == 3:
        for path_info in candidate_paths:
            curr_path_heur_val = path_info.heur_val
            if curr_path_heur_val > path_heur_val:
                path = path_info.path
                path_heur_val = curr_path_heur_val
    for node in path:
        visited.add(node)
        visited.add(node ^ 1)
    path_heur_val = torch.tensor(path_heur_val)
    return path, visited, path_heur_val