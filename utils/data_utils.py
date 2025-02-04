import os

import torch
import numpy as np
import dgl

from Bio import Seq, SeqIO
from scipy import sparse as sp 

from hyperparameters import get_hyperparameters


def extract_hifiasm_contigs(path, idx):
    gfa_path = os.path.join(path, f'{idx}_asm.bp.p_ctg.gfa')
    asm_path = os.path.join(path, f'{idx}_assembly.fasta')
    contigs = []
    with open(gfa_path) as f:
        n = 0
        for line in f.readlines():
            line = line.strip()
            if line[0] != 'S':
                continue
            seq=Seq.Seq(line.split()[2])
            ctg = SeqIO.SeqRecord(seq, description=f'contig_{n}', id=f'contig_{n}')
            contigs.append(ctg)
            n += 1
        SeqIO.write(contigs, asm_path, 'fasta')
    # subprocess.run(f'rm {path}/output.csv', shell=True)


def preprocess_graph(g):
    g = g.int()
    g.ndata['x'] = torch.ones(g.num_nodes(), 1)
    ol_len = g.edata['overlap_length'].float()
    ol_sim = g.edata['overlap_similarity']
    ol_len = (ol_len - ol_len.mean()) / ol_len.std()
    if get_hyperparameters()['use_similarities']:
        g.edata['e'] = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)
    else:
        g.edata['e'] = ol_len.unsqueeze(-1)
    return g


def add_positional_encoding(g):
    """
        Initializing positional encoding with k-RW-PE
    """

    g.ndata['in_deg'] = g.in_degrees().float()
    g.ndata['out_deg'] = g.out_degrees().float()
    
    pe_dim = get_hyperparameters()['nb_pos_enc']
    pe_type = get_hyperparameters()['type_pos_enc']
    
    if pe_dim == 0:
        return g

    if pe_type == 'RW':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A @ Dinv  
        M = RW
        # Iterate
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(pe_dim-1):
            M_power = M_power @ M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    if pe_type == 'PR':
        # k-step PageRank features
        A = g.adjacency_matrix(scipy_fmt="csr")
        D = A.sum(axis=1) # out degree
        Dinv = 1./ (D+1e-9); Dinv[D<1e-9] = 0 # take care of nodes without outgoing edges
        Dinv = sp.diags(np.squeeze(np.asarray(Dinv)), dtype=float) # D^-1 
        P = (Dinv @ A).T 
        n = A.shape[0]
        One = np.ones([n])
        x = One/ n
        PE = [] 
        alpha = 0.95 
        for _ in range(pe_dim): 
            x = alpha* P.dot(x) + (1.0-alpha)/n* One 
            PE.append(torch.from_numpy(x).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pe'] = PE  

    return g