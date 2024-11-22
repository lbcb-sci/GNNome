import gzip
import re
import time
from collections import deque, defaultdict
from datetime import datetime

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
import torch
import edlib
from tqdm import tqdm

import algorithms


def get_neighbors(graph):
    """Return neighbors/successors for each node in the graph.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        A DGLGraph for which neighbors will be determined for each
        node

    Returns
    -------
    dict
        a dictionary where nodes' ordinal numbers are keys and lists
        with all the nodes' neighbors are values
    """
    neighbor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        neighbor_dict[src.item()].append(dst.item())
    return neighbor_dict


def get_predecessors(graph):
    """Return predecessors for each node in the graph.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        A DGLGraph for which predecessors will be determined for each
        node

    Returns
    -------
    dict
        a dictionary where nodes' ordinal numbers are keys and lists
        with all the nodes' predecessors are values
    """
    predecessor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        predecessor_dict[dst.item()].append(src.item())
    return predecessor_dict


def get_edges(graph):
    """Return edge index for each edge in the graph.

    Parameters
    ----------
    graph : dgl.DGLGraph
        A DGLGraph for which edge indices will be saved

    Returns
    -------
    dict
        a dictionary where keys are (source, destination) tuples of
        nodes, and corresponding edge indices are values
    """
    edges_dict = {}
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_dict[(src, dst)] = idx
    return edges_dict


def print_pairwise(graph, path):
    """Outputs the graph into a pairwise TXT format.
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        The DGLGraph which is saved to the TXT file
    path : str
        The location where to save the TXT file

    Returns
    -------
    None
    """
    with open(path, 'w') as f:
        for src, dst in zip(graph.edges()[0], graph.edges()[1]):
            f.write(f'{src}\t{dst}\n')


def calculate_similarities(edge_ids, read_seqs, overlap_lengths):
    # Make sure that read_seqs is a dict of string, not Bio.Seq objects!
    overlap_similarities = {}
    for src, dst in tqdm(edge_ids.keys(), ncols=120):
        ol_length = overlap_lengths[(src, dst)]
        read_src = read_seqs[src]
        read_dst = read_seqs[dst]
        edit_distance = edlib.align(read_src[-ol_length:], read_dst[:ol_length])['editDistance']
        overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length
    return overlap_similarities


def only_from_gfa(gfa_path, training=False, reads_path=None, get_similarities=False):
    if training:
        if reads_path is not None:
            if reads_path.endswith('gz'):
                if reads_path.endswith('fasta.gz') or reads_path.endswith('fna.gz') or reads_path.endswith('fa.gz'):
                    filetype = 'fasta'
                elif reads_path.endswith('fastq.gz') or reads_path.endswith('fnq.gz') or reads_path.endswith('fq.gz'):
                    filetype = 'fastq'
                with gzip.open(reads_path, 'rt') as handle:
                    read_headers = {read.id: read.description for read in SeqIO.parse(handle, filetype)}
            else:
                if reads_path.endswith('fasta') or reads_path.endswith('fna') or reads_path.endswith('fa'):
                    filetype = 'fasta'
                elif reads_path.endswith('fastq') or reads_path.endswith('fnq') or reads_path.endswith('fq'):
                    filetype = 'fastq'
                read_headers = {read.id: read.description for read in SeqIO.parse(reads_path, filetype)}
        else:
            print('You need to pass the reads_path with annotations')
            exit(1)

    
    graph_nx = nx.DiGraph()

    read_to_node, node_to_read = {}, {}
    read_to_node2 = {}
    edges_dict = {}
    read_lengths, read_seqs = {}, {}  # Obtained from the GFA
    read_idxs, read_strands, read_starts, read_ends, read_chrs = {}, {}, {}, {}, {}  # Obtained from the FASTA/Q headers
    edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}

    no_seqs_flag = False

    time_start = datetime.now()
    print(f'Starting to loop over GFA')
    with open(gfa_path) as f:
        node_idx = 0
        edge_idx = 0

        # -------------------------------------------------
        # We assume that the first N lines start with "S"
        # And next M lines start with "L"
        # -------------------------------------------------
        all_lines = f.readlines()
        line_idx = 0
        while line_idx < len(all_lines):
            line = all_lines[line_idx]
            line_idx += 1
            line = line.strip().split()
            if line[0] == 'S':
                tag, id, sequence, length = line[:4]
                if sequence == '*':
                    no_seqs_flag = True
                    sequence = '*'
                sequence = Seq(sequence)  # This sequence is already trimmed in raven!
                length = int(length[5:])

                real_idx = node_idx
                virt_idx = node_idx + 1
                read_to_node[id] = (real_idx, virt_idx)
                node_to_read[real_idx] = id
                node_to_read[virt_idx] = id

                graph_nx.add_node(real_idx)  # real node = original sequence
                graph_nx.add_node(virt_idx)  # virtual node = rev-comp sequence

                read_seqs[real_idx] = str(sequence)
                read_seqs[virt_idx] = str(sequence.reverse_complement())

                read_lengths[real_idx] = length
                read_lengths[virt_idx] = length

                if id.startswith('utg'):
                    line = all_lines[line_idx]
                    line = line.strip().split()
                    line_idx += 1
                    tag = line[0]
                    utg_id = line[1]
                    utg_to_read = line[4]
                    assert tag == 'A', 'Line should start with A!'
                    assert id == utg_id, 'Unitig IDs should be the same!'
                    id = utg_to_read
                    read_to_node2[id] = (real_idx, virt_idx)

                if training:
                    description = read_headers[id]
                    # desc_id, strand, start, end = description.split()
                    strand = re.findall(r'strand=(\+|\-)', description)[0]
                    strand = 1 if strand == '+' else -1
                    start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                    end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                    chromosome = int(re.findall(r'chr=(\d+)', description)[0])

                    read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                    read_starts[real_idx] = read_starts[virt_idx] = start
                    read_ends[real_idx] = read_ends[virt_idx] = end
                    read_chrs[real_idx] = read_chrs[virt_idx] = chromosome

                node_idx += 2

            if line[0] == 'L':

                if len(line) == 6:
                    # raven, normal GFA 1 standard
                    tag, id1, orient1, id2, orient2, cigar = line
                elif len(line) == 7:
                    # hifiasm GFA
                    tag, id1, orient1, id2, orient2, cigar, _ = line
                    id1 = re.findall(r'(.*):\d-\d*', id1)[0]
                    id2 = re.findall(r'(.*):\d-\d*', id2)[0]
                elif len(line) == 8:
                    # hifiasm GFA newer
                    tag, id1, orient1, id2, orient2, cigar, _, _ = line
                else:
                    raise Exception("Unknown GFA format!")

                if orient1 == '+' and orient2 == '+':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][1]
                if orient1 == '+' and orient2 == '-':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][1]
                if orient1 == '-' and orient2 == '+':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][0]
                if orient1 == '-' and orient2 == '-':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][0]        

                graph_nx.add_edge(src_real, dst_real)
                graph_nx.add_edge(src_virt, dst_virt)  # In hifiasm GFA this might be redundant, but it is necessary for raven GFA

                edge_ids[(src_real, dst_real)] = edge_idx
                edge_ids[(src_virt, dst_virt)] = edge_idx + 1
                edge_idx += 2

                # -----------------------------------------------------------------------------------
                # This enforces similarity between the edge and its "virtual pair"
                # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
                # When parsing CSV that was not necessarily so:
                # Sometimes reads would be slightly differently aligned from their RC pairs
                # Thus resulting in different overlap lengths
                # -----------------------------------------------------------------------------------

                try:
                    ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
                except ValueError:
                    print('Cannot convert CIGAR string into overlap length!')
                    raise ValueError
                
                overlap_lengths[(src_real, dst_real)] = ol_length
                overlap_lengths[(src_virt, dst_virt)] = ol_length

                prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
                prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length
                
    elapsed = (datetime.now() - time_start).seconds
    print(f'Elapsed time: {elapsed}s')
    if no_seqs_flag:
        print(f'Getting sequences from FASTA/Q file...')
        if reads_path.endswith('gz'):
            if reads_path.endswith('fasta.gz') or reads_path.endswith('fna.gz') or reads_path.endswith('fa.gz'):
                filetype = 'fasta'
            elif reads_path.endswith('fastq.gz') or reads_path.endswith('fnq.gz') or reads_path.endswith('fq.gz'):
                filetype = 'fastq'
            with gzip.open(reads_path, 'rt') as handle:
                fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(handle, filetype)}
        else:
            if reads_path.endswith('fasta') or reads_path.endswith('fna') or reads_path.endswith('fa'):
                filetype = 'fasta'
            elif reads_path.endswith('fastq') or reads_path.endswith('fnq') or reads_path.endswith('fq'):
                filetype = 'fastq'
            fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}

        print(f'Sequences successfully loaded!')
        # fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}
        for node_id in tqdm(read_seqs.keys(), ncols=120):
            read_id = node_to_read[node_id]
            seq = fastaq_seqs[read_id]
            read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
        print(f'Loaded DNA sequences!')

    elapsed = (datetime.now() - time_start).seconds
    print(f'Elapsed time: {elapsed}s')

    if get_similarities:
        print(f'Calculating similarities...')
        overlap_similarities = calculate_similarities(edge_ids, read_seqs, overlap_lengths)
        print(f'Done!')
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')

    nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
    node_attrs = ['read_length']

    nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
    nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
    edge_attrs = ['prefix_length', 'overlap_length']

    labels = None

    if training:
        nx.set_node_attributes(graph_nx, read_strands, 'read_strand')
        nx.set_node_attributes(graph_nx, read_starts, 'read_start')
        nx.set_node_attributes(graph_nx, read_ends, 'read_end')
        nx.set_node_attributes(graph_nx, read_chrs, 'read_chr')
        node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_chr'])

        unqique_chrs = set(read_chrs.values())
        if len(unqique_chrs) == 1:
            ms_pos, labels = algorithms.process_graph(graph_nx)
        else:
            ms_pos, labels = algorithms.process_graph_combo(graph_nx)
        nx.set_edge_attributes(graph_nx, labels, 'y')
        edge_attrs.append('y')

    if get_similarities:
        nx.set_edge_attributes(graph_nx, overlap_similarities, 'overlap_similarity')
        edge_attrs.append('overlap_similarity')

    # return graph_nx  # DEBUG

    # This produces vector-like features (e.g. shape=(num_nodes,))
    graph_dgl = dgl.from_networkx(graph_nx, node_attrs=node_attrs, edge_attrs=edge_attrs)
    
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)
    edges = get_edges(graph_dgl)
    
    if len(read_to_node2) != 0:
        read_to_node = read_to_node2

    return graph_dgl, predecessors, successors, read_seqs, edges, read_to_node, labels
