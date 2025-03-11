import re
import os
import pickle
import subprocess

import torch
import dgl
from Bio import SeqIO
from Bio import Seq

chr_lens = {
    'chr1' : 248387328,
    'chr2' : 242696752,
    'chr3' : 201105948,
    'chr4' : 193574945,
    'chr5' : 182045439,
    'chr6' : 172126628,
    'chr7' : 160567428,
    'chr8' : 146259331,
    'chr9' : 150617247,
    'chr10': 134758134,
    'chr11': 135127769,
    'chr12': 133324548,
    'chr13': 113566686,
    'chr14': 101161492,
    'chr15': 99753195,
    'chr16': 96330374,
    'chr17': 84276897,
    'chr18': 80542538,
    'chr19': 61707364,
    'chr20': 66210255,
    'chr21': 45090682,
    'chr22': 51324926,
    'chrX' : 154259566,
}


def walk_to_sequence(walks, graph, reads, edges):
    contigs = []
    for i, walk in enumerate(walks):
        prefixes = [(src, graph.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]
        sequences = [reads[src][:prefix] for (src, prefix) in prefixes]
        contig = Seq.Seq(''.join(sequences) + reads[walk[-1]])
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)
    return contigs


def save_assembly(contigs, save_dir, idx, suffix=''):
    assembly_path = os.path.join(save_dir, f'{idx}_assembly{suffix}.fasta')
    SeqIO.write(contigs, assembly_path, 'fasta')


def calculate_N50(contigs):
    """Calculate N50 for contigs.
    Args:
        list_of_lengths (list): List of SeqRecord objects.
    Returns:
        float: N50 value.
    """
    lengths_list = [len(c.seq) for c in contigs]
    lengths_list.sort(reverse=True)
    total_length = sum(lengths_list)
    total_bps = 0
    for length in lengths_list:
        total_bps += length
        if total_bps >= total_length / 2:
            return length
    return -1


def calculate_NG50(contigs, ref_length):
    """Calculate NG50 for contigs.
    Args:
        list_of_lengths (list): List of SeqRecord objects.
    Returns:
        int: NG50 value.
    """
    if ref_length <= 0:
        return -1
    lengths_list = [len(c.seq) for c in contigs]
    lengths_list.sort(reverse=True)
    total_bps = 0
    for length in lengths_list:
        total_bps += length
        if total_bps >= ref_length / 2:
            return length
    return -1


def quick_evaluation(contigs, chrN):
    # contigs = walk_to_sequence(walks, graph, reads, edges)
    lengths_list = [len(c.seq) for c in contigs]
    num_contigs = len(contigs)
    longest_contig = max(lengths_list)
    n50 = calculate_N50(contigs)
    if chrN:
        chr_len = chr_lens[chrN]
        reconstructed = sum(lengths_list) / chr_len
        ng50 = calculate_NG50(contigs, chr_len)
    else:
        reconstructed = ng50 = -1
    return num_contigs, longest_contig, reconstructed, n50, ng50


def txt_output(f, txt):
    print(f'{txt}')
    f.write(f'{txt}\n')


def print_summary_old(data_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50):
    reports_dir = os.path.join(data_path, 'reports')
    if not os.path.isdir(reports_dir ):
        os.mkdir(reports_dir)
    with open(f'{reports_dir}/{idx}_report.txt', 'w') as f:
        txt_output(f, f'-'*80)
        txt_output(f, f'Report for graph {idx} in {data_path}')
        txt_output(f, f'Graph created from {chrN}')
        txt_output(f, f'Num contigs:\t{num_contigs}')
        txt_output(f, f'Longest contig:\t{longest_contig}')
        txt_output(f, f'Reconstructed:\t{reconstructed * 100:2f}%')
        txt_output(f, f'N50:\t{n50}')
        txt_output(f, f'NG50:\t{ng50}')


def print_summary(data_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50):
    print(f'-'*80)
    print(f'Report for graph {idx} in {data_path}')
    # print(f'Graph created from {chrN}')
    print(f'Num contigs:\t{num_contigs}')
    print(f'Longest contig:\t{longest_contig}')
    # print(f'Reconstructed:\t{reconstructed * 100:2f}%')
    print(f'N50:\t{n50}')
    # print(f'NG50:\t{ng50}')


def run_minigraph(ref, asm, paf):
    minigraph = f'/home/vrcekl/minigraph/minigraph'
    # paftools = f'/home/vrcekl/minimap2-2.24_x64-linux/paftools.js'
    # paf = os.path.join(save_path, f'asm.paf')
    # cmdaa = f'{minigraph} -xasm -g10k -r10k --show-unmap=yes {ref} {asm} > {paf} && k8 {paftools} asmstat {idx} {paf} > {report}'
    cmd = f'{minigraph} -t32 -xasm -g10k -r10k --show-unmap=yes {ref} {asm}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p


def parse_pafs(idx, report, paf):
    paftools = f'/home/vrcekl/minimap2/paftools.js'
    # paf = os.path.join(save_path, f'asm.paf')
    cmd = f'k8 {paftools} asmstat {idx} {paf}'.split()
    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    return p


def parse_minigraph_for_chrs(save_path):
    ng50, nga50 = {}, {}
    for i in range(1, 24):
        if i == 23:
            i = 'X'
        stat_path = f'{save_path}/chr{i}/reports/0_minigraph.txt'
        try:
            with open(stat_path) as f:
                for line in f.readlines():
                    if line.startswith('NG50'):
                        try:
                            ng50_l = int(re.findall(r'NG50\s*(\d+)', line)[0])
                        except IndexError:
                            ng50_l = 0
                        ng50[f'chr{i}'] = ng50_l
                    if line.startswith('NGA50'):
                        try:
                            nga50_l = int(re.findall(r'NGA50\s*(\d+)', line)[0])
                        except IndexError:
                            nga50_l = 0
                        nga50[f'chr{i}'] = nga50_l
        except FileNotFoundError:
            print(f'Report for chr{i} at {stat_path} does not exist!')

    print('NG50')
    print(*ng50.values(), sep='\n')
    print()

    print('NGA50')
    print(*nga50.values(), sep='\n')
    print()


def parse_minigraph_for_full(report, save_path=None, directory=None, filename='0_minigraph.txt'):
    # stat_path = f'{save_path}/{directory}/reports/{filename}'
    stat_path = report
    with open(stat_path) as f:
        report = f.read()
        print(report)
