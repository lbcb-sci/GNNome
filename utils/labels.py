import networkx as nx
import dgl


def interval_union(name, root):
    graph = dgl.load_graphs(f'{root}/processed/{name}.dgl')[0][0]
    intervals = []
    for strand, start, end in zip(graph.ndata['read_strand'], graph.ndata['read_start'], graph.ndata['read_end']):
        if strand.item() == 1:
            intervals.append([start.item(), end.item()])
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)

    return result


def get_gt_for_single_strand(graph, read_start_dict, read_end_dict, positive=False):
    # New version
    # components = [] # not for gt (later used)
    all_nodes = graph.nodes()
    gt_edges = set()
    if positive:
        final_node = max(all_nodes, key=lambda x: read_end_dict[x])
        highest_node_reached = min(all_nodes, key=lambda x: read_end_dict[x])
    else:
        final_node = min(all_nodes, key=lambda x: read_start_dict[x])
        highest_node_reached = max(all_nodes, key=lambda x: read_start_dict[x])

    while all_nodes:
        if positive:
            start_node = min(all_nodes, key=lambda x: read_start_dict[x])
        else:
            start_node = max(all_nodes, key=lambda x: read_end_dict[x])

        # try finding a path and report the highest found node during the dfs
        current_graph = graph.subgraph(all_nodes)
        full_component = set(nx.dfs_postorder_nodes(current_graph, source=start_node))
        full_component.add(start_node)
        if positive:
            highest_node_in_component = max(full_component, key=lambda x: read_end_dict[x])
        else:
            highest_node_in_component = min(full_component, key=lambda x: read_start_dict[x])

        current_graph = graph.subgraph(full_component)
        component = set(nx.dfs_postorder_nodes(current_graph.reverse(copy=True), source=highest_node_in_component))
        component.add(highest_node_in_component)
        current_graph = graph.subgraph(component)

        # if the path doesnt go further then an already existing chunk - dont add any edges to gt
        not_reached_highest = (positive and (
                    read_end_dict[highest_node_in_component] < read_end_dict[highest_node_reached])) \
                              or (not positive and (
                    read_start_dict[highest_node_in_component] > read_start_dict[highest_node_reached]))
        if len(component) < 2 or not_reached_highest:  # Used to be len(component) <= 2
            all_nodes = all_nodes - full_component
            continue
        else:
            highest_node_reached = highest_node_in_component

        gt_edges = set(current_graph.edges()) | gt_edges
        # print("finish component")
        if highest_node_reached == final_node:
            break
        all_nodes = all_nodes - full_component
    return gt_edges


def create_correct_graphs(graph, read_start_dict, read_end_dict, read_strand_dict, read_chr_dict):
    # New version
    # only real connections of true overlaps
    pos_edges = []
    neg_edges = []

    for edge in graph.edges():
        src, dst = edge
        if read_start_dict[dst] < read_end_dict[src] and read_start_dict[dst] > read_start_dict[src]:
            if read_strand_dict[src] == 1 and read_strand_dict[dst] == 1 and read_chr_dict[src] == read_chr_dict[dst]:
                pos_edges.append(edge)

        if read_start_dict[src] < read_end_dict[dst] and read_start_dict[src] > read_start_dict[dst]:
            if read_strand_dict[src] == -1 and read_strand_dict[dst] == -1 and read_chr_dict[src] == read_chr_dict[dst]:
                neg_edges.append(edge)

    pos_graph = nx.DiGraph()
    pos_graph.add_edges_from(pos_edges)
    neg_graph = nx.DiGraph()
    neg_graph.add_edges_from(neg_edges)
    return pos_graph, neg_graph


def create_correct_graphs_combo(graph, read_start_dict, read_end_dict, read_strand_dict, read_chr_dict):
    # New version
    # only real connections of true overlaps

    unique_chr = set([v.item() for k, v in read_chr_dict.items()])

    pos_edges = {chr: [] for chr in unique_chr}
    neg_edges = {chr: [] for chr in unique_chr}

    pos_graphs = {}
    neg_graphs = {}

    for edge in graph.edges():
        src, dst = edge
        if read_start_dict[dst] < read_end_dict[src] and read_start_dict[dst] > read_start_dict[src]:
            if read_strand_dict[src] == 1 and read_strand_dict[dst] == 1 and read_chr_dict[src] == read_chr_dict[dst]:
                pos_edges[read_chr_dict[src].item()].append(edge)

        if read_start_dict[src] < read_end_dict[dst] and read_start_dict[src] > read_start_dict[dst]:
            if read_strand_dict[src] == -1 and read_strand_dict[dst] == -1 and read_chr_dict[src] == read_chr_dict[dst]:
                neg_edges[read_chr_dict[src].item()].append(edge)

    for chr in unique_chr:
        pos_graph = nx.DiGraph()
        pos_graph.add_edges_from(pos_edges[chr])
        pos_graphs[chr] = pos_graph
        neg_graph = nx.DiGraph()
        neg_graph.add_edges_from(neg_edges[chr])
        neg_graphs[chr] = neg_graph
    return pos_graphs, neg_graphs


def process_graph(graph):
    # New version
    read_start_dict = nx.get_node_attributes(graph, 'read_start')
    read_end_dict = nx.get_node_attributes(graph, 'read_end')
    read_strand_dict = nx.get_node_attributes(graph, 'read_strand')
    read_chr_dict = nx.get_node_attributes(graph, 'read_chr')

    pos_graph, neg_graph = create_correct_graphs(graph, read_start_dict, read_end_dict, read_strand_dict, read_chr_dict)
    pos_gt_edges = get_gt_for_single_strand(pos_graph, read_start_dict, read_end_dict, positive=True)
    neg_gt_edges = get_gt_for_single_strand(neg_graph, read_start_dict, read_end_dict, positive=False)

    gt_edges = neg_gt_edges | pos_gt_edges

    gt_dict = {}
    for e in graph.edges():
        if e in gt_edges:
            gt_dict[e] = 1.
        else:
            gt_dict[e] = 0.

    return gt_edges, gt_dict


def process_graph_combo(graph):
    # New version
    read_start_dict = nx.get_node_attributes(graph, 'read_start')
    read_end_dict = nx.get_node_attributes(graph, 'read_end')
    read_strand_dict = nx.get_node_attributes(graph, 'read_strand')
    read_chr_dict = nx.get_node_attributes(graph, 'read_chr')

    print(f'Finding correct graphs per chromosome and strand...')
    pos_graphs, neg_graphs = create_correct_graphs_combo(graph, read_start_dict, read_end_dict, read_strand_dict, read_chr_dict)
    print(f'Chromosomes found: {len(pos_graphs)}')

    gt_edges = set()
    for chr, pos_graph in pos_graphs.items():
        print(f'Processing chr{chr}...')
        pos_gt_edges = get_gt_for_single_strand(pos_graph, read_start_dict, read_end_dict, positive=True)
        gt_edges |= pos_gt_edges
    for chr, neg_graph in neg_graphs.items():
        neg_gt_edges = get_gt_for_single_strand(neg_graph, read_start_dict, read_end_dict, positive=False)
        gt_edges |= neg_gt_edges

    gt_dict = {}
    for e in graph.edges():
        if e in gt_edges:
            gt_dict[e] = 1.
        else:
            gt_dict[e] = 0.

    return gt_edges, gt_dict
