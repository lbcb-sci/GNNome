def assert_strand(graph, walk):
    org_strand = graph.ndata['read_strand'][walk[0]].item()
    for idx, node in enumerate(walk[1:]):
        curr_strand = graph.ndata['read_strand'][node].item()
        if curr_strand != org_strand:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'node index: {node}')


def assert_chromosome(graph, walk):
    org_chr = graph.ndata['read_chr'][walk[0]].item()
    for idx, node in enumerate(walk[1:]):
        curr_chr = graph.ndata['read_chr'][node].item()
        if curr_chr != org_chr:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'node index: {node}')


def assert_overlap(graph, walk):
    for idx, (src, dst) in enumerate(zip(walk[:-1], walk[1:])):
        src_start = graph.ndata['read_start'][src].item()
        dst_start = graph.ndata['read_start'][dst].item()
        src_end = graph.ndata['read_end'][src].item()
        dst_end = graph.ndata['read_end'][dst].item()
        src_strand = graph.ndata['read_strand'][src].item()
        dst_strand = graph.ndata['read_strand'][dst].item()
        if src_strand == dst_strand == 1 and dst_start > src_end:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_end}, start: {dst_start}')
        if src_strand == dst_strand == -1 and dst_end < src_start:
            print('-' * 20)
            print(f'walk index: {idx}')
            print(f'nodes not connected: {src}, {dst}')
            print(f'end: {src_start}, start: {dst_end}')


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