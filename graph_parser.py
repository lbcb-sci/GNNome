import gzip
import os
import re
import time
from collections import Counter, namedtuple
from datetime import datetime

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
import torch
import edlib
from tqdm import tqdm

import algorithms


# Overlap = namedtuple('Overlap', ['src_len', 'src_start', 'src_end', 'dst_len', 'dst_start', 'dst_end'])


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


def only_from_gfa(gfa_path, training=False, reads_path=None, get_similarities=False, paf_path=None):
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
    read_lengths, read_seqs = {}, {}  # Obtained from the GFA
    read_strands, read_starts, read_ends, read_chrs = {}, {}, {}, {}  # Obtained from the FASTA/Q headers
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
                if len(line) == 5:  # Hifiasm
                    tag, id, sequence, length, count = line
                if len(line) == 4:  # Raven
                    tag, id, sequence, length = line
                if sequence == '*':
                    no_seqs_flag = True
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
                    # The issue here is that in some cases, one unitig can consist of more than one read
                    # So this is the adapted version of the code that supports that
                    # The only things of importance here are read_to_node2 dict (not overly used)
                    # And id variable which I use for obtaining positions during training (for the labels)
                    # I don't use it for anything else, which is good
                    ids = []
                    while True:
                        line = all_lines[line_idx]
                        line = line.strip().split()
                        if line[0] != 'A':
                            break
                        line_idx += 1
                        tag = line[0]
                        utg_id = line[1]
                        read_orientation = line[3]
                        utg_to_read = line[4]
                        ids.append((utg_to_read, read_orientation))
                        read_to_node2[utg_to_read] = (real_idx, virt_idx)

                    id = ids
                    node_to_read[real_idx] = id
                    node_to_read[virt_idx] = id

                if training:

                    if type(id) != list:
                        description = read_headers[id]
                        # desc_id, strand, start, end = description.split()
                        strand = re.findall(r'strand=(\+|\-)', description)[0]
                        strand = 1 if strand == '+' else -1
                        start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                        end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                        chromosome = int(re.findall(r'chr=(\d+)', description)[0])

                    else:
                        strands = []
                        starts = []
                        ends = []
                        chromosomes = []
                        for id_r, id_o in id:
                            description = read_headers[id_r]
                            # desc_id, strand, start, end = description.split()
                            strand_fasta = re.findall(r'strand=(\+|\-)', description)[0]
                            strand_fasta = 1 if strand_fasta == '+' else -1
                            strand_gfa = 1 if id_o == '+' else -1
                            strand = strand_fasta * strand_gfa

                            strands.append(strand)
                            start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                            starts.append(start)
                            end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                            ends.append(end)
                            chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                            chromosomes.append(chromosome)

                        # What if they come from different strands but are all merged in a single unitig?
                        # Or even worse, different chromosomes? How do you handle that?
                        # I don't think you can. It's an error in the graph
                        strand = 1 if sum(strands) >= 0 else -1
                        start = min(starts)
                        end = max(ends)
                        chromosome = Counter(chromosomes).most_common()[0][0]

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

    # This produces vector-like features (e.g. shape=(num_nodes,))
    graph_dgl = dgl.from_networkx(graph_nx, node_attrs=node_attrs, edge_attrs=edge_attrs)
    
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)
    edges = get_edges(graph_dgl)
    
    if len(read_to_node2) != 0:
        read_to_node = read_to_node2

    if paf_path and os.path.isfile(paf_path):
        paf = {}
        # Parse the PAF file
        with open(paf_path) as f:
            for line in f.readlines():
                line = line.strip().split()
                src, src_len, src_start, src_end = line[:4]
                strand = line[4]
                dst, dst_len, dst_start, dst_end = line[5:9]
                paf[(src, dst)] = (src_len, src_start, src_end, strand, dst_len, dst_start, dst_end)

        edge_paf_info = {}
        n2r = node_to_read
        # Iterate over all the edges in the edge list
        for src, dst in list(edges.keys()):
            # Find the reads corresponding to the source/destination nodes (works even for collapsed unitigs)
            src_r = n2r[src]
            dst_r = n2r[dst]
            added = False
            if len(src_r) == 1 and len(dst_r) == 1:
                # Clear situation, each node is only one read
                sr, so = src_r[0]
                dr, do = dst_r[0]
                if (sr, dr) in paf:
                    edge_paf_info[(src, dst)] = paf[sr, dr], (so, do)
                    added = True
                else:
                    # Sometimes, overlaps in PAF are not symmetrical, but readB - readA overlap can be inferred from readA - readB
                    ovlp = paf[dr, sr]
                    ovlp = ovlp[4:] + ovlp[3:4] + ovlp[:3]  # Change the source-target overlap information
                    edge_paf_info[(src, dst)] = ovlp, (so, do)
                    added = True
            elif len(src_r) > 1 and len(dst_r) == 1:
                # Source node is a collapsed unitig, have to inspect which read of the source unitig is used for the overlap
                dr, do = dst_r[0]
                for sr, so in src_r:
                    if added:
                        break
                    if (sr, dr) in paf.keys():
                        edge_paf_info[(src, dst)] = paf[sr, dr], (so, do)
                        added = True
                    elif (dr, sr) in paf.keys():
                        ovlp = paf[dr, sr]
                        ovlp = ovlp[4:] + ovlp[3:4] + ovlp[:3]
                        edge_paf_info[(src, dst)] = ovlp, (so, do)
                        added = True
                    else:
                        continue
            elif len(src_r) == 1 and len(dst_r) > 1:
                # Destination node is a collapsed unitig, have to inspect which read of the destination unitig is used for the overlap
                sr, so = src_r[0]
                for dr, do in dst_r:
                    if added:
                        break
                    if (sr, dr) in paf.keys():
                        edge_paf_info[(src, dst)] = paf[sr, dr], (so, do)
                        added = True
                    elif (dr, sr) in paf.keys():
                        ovlp = paf[dr, sr]
                        ovlp = ovlp[4:] + ovlp[3:4] + ovlp[:3]
                        edge_paf_info[(src, dst)] = ovlp, (so, do)
                        added = True
                    else:
                        continue
            else:
                # Both node and destination nodes are collapsed unitigs
                for sr, so in src_r:
                    if added:
                        break
                    for dr, do in dst_r:
                        if added:
                            break
                        if (sr, dr) in paf.keys():
                            edge_paf_info[(src, dst)] = paf[sr, dr], (so, do)
                            added = True
                        elif (dr, sr) in paf.keys():
                            ovlp = paf[dr, sr]
                            ovlp = ovlp[4:] + ovlp[3:4] + ovlp[:3]
                            edge_paf_info[(src, dst)] = ovlp, (so, do)
                            added = True
                        else:
                            continue
            assert added, 'Edge not assigned PAF line!'


        edge_paf_info_new = {}
        # Create new dictionary, edge_paf_info_new, where all the PAF overlaps will be stored in a desirable src->dst format
        # This can directly be stored as start/end overlap positions for each _node_ and makes computation of overhangs as features simpler
        for (src, dst), (overlap, (so, do)) in edge_paf_info.items():
            so = 1 if so == '+' else -1  # source orientation in GFA
            do = 1 if do == '+' else -1  # destination orientation in GFA
            ss = 1 if src % 2 == 0 else -1  # source strand in FASTA
            ds = 1 if dst % 2 == 0 else -1  # destination strand in FASTA

            src_strand = ss * so
            dst_strand = ds * do

            l1, s1, e1, o, l2, s2, e2 = overlap
            l1 = int(l1)
            s1 = int(s1)
            e1 = int(e1)
            l2 = int(l2)
            s2 = int(s2)
            e2 = int(e2)
            overlap = (l1, s1, e1, o, l2, s2, e2)

            if src_strand == 1 and dst_strand == 1:
                # src=+ & dst=+ -> should result in + overlap orientation
                assert overlap[3] == '+', f'Breaking for {src} {dst}\n{overlap}'  # Make sure that the orientations are correct
                overlap_new = overlap
            elif src_strand == -1 and dst_strand == 1:
                # src=- & dst=+ -> should result in - overlap orientation
                assert overlap[3] == '-', f'Breaking for {src} {dst}\n{overlap}'
                length, start, end = overlap[:3]
                start_new = length - end
                end_new = length - start
                overlap_new = (length, start_new, end_new) + overlap[3:]
            elif src_strand == 1 and dst_strand == -1:
                # src=+ & dst=- -> should result in - overlap orientation
                assert overlap[3] == '-', f'Breaking for {src} {dst}\n{overlap}'
                length, start, end = overlap[-3:]
                start_new = length - end
                end_new = length - start
                overlap_new = overlap[:-3] + (length, start_new, end_new)
            else:
                # src=- & dst=- -> should result in + overlap orientation
                assert overlap[3] == '+', f'Breaking for {src} {dst}\n{overlap}'
                length1, start1, end1 = overlap[:3]
                length2, start2, end2 = overlap[-3:]
                sign = overlap[3]
                start1_new = length1 - end1
                end1_new = length1 - start1
                start2_new = length2 - end2
                end2_new = length2 - start2
                overlap_new = (length1, start1_new, end1_new, sign, length2, start2_new, end2_new)

            edge_paf_info_new[src, dst] = overlap_new, (so, do)

        # In some cases PAF lines for readA - readB overlap are not the same as for readB - readA overlap
        # This results in some edges getting assigned the prefix-suffix overlaps instead of suffix-prefix
        # This is a "fix" for that problem, though it relies on the sequence lengths and is not perfect
        # Ideally it would rely only on PAF entries and the graph topology
        edge_paf_info_new_new = {}
        for (src, dst), (overlap, (so, do)) in edge_paf_info_new.items():
            ss = 1 if src % 2 == 0 else -1  # source strand in FASTA
            ds = 1 if dst % 2 == 0 else -1  # destination strand in FASTA
            src_strand = ss * so
            dst_strand = ds * do
            src_len, src_start, src_end, orientation, dst_len, dst_start, dst_end = overlap
            if src_end < 0.99 * src_len or dst_start > 0.01 * dst_len:
                overlap_org, (do_org, so_org) = edge_paf_info_new[dst^1, src^1]
                src_len2, src_start2, src_end2, orientation2, dst_len2, dst_start2, dst_end2 = overlap_org
                overlap_new_new = (dst_len2, dst_len2 - dst_end2, dst_len2 - dst_start2, orientation2, src_len2, src_len2 - src_end2, src_len2 - src_start2)
                # edge_paf_info_new_new[src, dst] = (overlap_new_new, (so, do))
                # Overlaps are stored as (src_len, src_start, src_end, dst_len, dst_start, dst_end)
                edge_paf_info_new_new[src, dst] = (overlap_new_new[0], overlap_new_new[1], overlap_new_new[2], overlap_new_new[4], overlap_new_new[5], overlap_new_new[6])
            else:
                # edge_paf_info_new_new[src, dst] = (overlap, (so, do))
                # Overlaps are stored as (src_len, src_start, src_end, dst_len, dst_start, dst_end)
                edge_paf_info_new_new[src, dst] = (overlap[0], overlap[1], overlap[2], overlap[4], overlap[5], overlap[6])
        edge_paf_info = edge_paf_info_new_new

    auxiliary = {
        'pred': predecessors,
        'succ': successors,
        'reads': read_seqs,
        'edges': edges,
        'read_to_node': read_to_node,
    }

    if labels is not None:
        auxiliary['labels'] = labels
    if 'node_to_read' in locals():
        auxiliary['node_to_read'] = node_to_read
    if 'edge_paf_info' in locals():
        auxiliary['edge_paf_info'] = edge_paf_info

    return graph_dgl, auxiliary
