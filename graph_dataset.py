import re
import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser
from utils import preprocess_graph, add_positional_encoding, extract_contigs


class AssemblyGraphDataset(DGLDataset):
    def __init__(self, root, nb_pos_enc, assembler, specs=None, generate=False):
        self.root = os.path.abspath(root)
        self.specs = specs
        self.assembler = assembler

        self.assembly_dir = os.path.join(self.root, self.assembler)
        print(self.assembly_dir)

        if 'raw' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'raw'", shell=True, cwd=self.root)
        if 'output' not in os.listdir(self.assembly_dir):
            subprocess.run(f"mkdir 'output'", shell=True, cwd=self.assembly_dir)
        if f'processed' not in os.listdir(self.assembly_dir):
            subprocess.run(f"mkdir 'processed'", shell=True, cwd=self.assembly_dir)
        if f'info' not in os.listdir(self.assembly_dir):
            subprocess.run(f"mkdir 'info'", shell=True, cwd=self.assembly_dir)

        raw_dir = os.path.join(self.root, 'raw')
        save_dir = os.path.join(self.assembly_dir, f'processed')
        self.output_dir = os.path.join(self.assembly_dir, f'output')
        self.info_dir = os.path.join(self.assembly_dir, f'info')
        self.raven_path = os.path.abspath('vendor/raven/build/bin/raven')
        self.hifiasm_path = os.path.abspath('vendor/hifiasm/hifiasm')
        super().__init__(name='assembly_graphs', raw_dir=raw_dir, save_dir=save_dir)

        self.graph_list = []
        if not generate:
            for file in os.listdir(self.save_dir):
                idx = int(file[:-4])
                graph = dgl.load_graphs(os.path.join(self.save_dir, file))[0][0]
                graph = preprocess_graph(graph, self.root, idx)
                if nb_pos_enc is not None:
                    graph = add_positional_encoding(graph, nb_pos_enc) 
                # graph, _ = dgl.khop_in_subgraph(graph, 390, k=20)  # DEBUG !!!!
                print(f'DGL graph idx={idx} info:\n',graph)
                self.graph_list.append((idx, graph))
            self.graph_list.sort(key=lambda x: x[0])

    def has_cache(self):
        """Check if the raw data is already processed and stored."""
        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        return len(raw_files - prc_files) == 0  # set difference

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        i, graph = self.graph_list[idx]
        return i, graph

    def process(self):
        pass


class AssemblyGraphDataset_HiFi(AssemblyGraphDataset):

    def __init__(self, root, nb_pos_enc=10, assembler='hifiasm', specs=None, generate=False):
        super().__init__(root=root, nb_pos_enc=nb_pos_enc, assembler=assembler, specs=specs, generate=generate)

    def process(self):
        """Process the raw data and save it on the disk."""
        if self.specs is None:
            threads = 32
            filter = 0.99
            out = 'assembly.fasta'
            assembler = 'hifiasm'
        else:
            threads = self.specs['threads']
            filter = self.specs['filter']
            out = self.specs['out']
            assembler = self.specs['assembler']

        assert assembler in ('raven', 'hifiasm'), 'Choose either "raven" or "hifiasm" assembler'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        print(f'====> FILTER = {filter}\n')

        n_have = len(os.listdir(self.save_dir))
        n_need = len(os.listdir(self.raw_dir))

        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        diff = raw_files - prc_files

        for cnt, idx in enumerate(diff):
            fastq = f'{idx}.fasta'
            if fastq not in os.listdir(self.raw_dir):
                fastq = f'{idx}.fastq'
            print(f'Step {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}')
            # print(f'Starting assembler at: {self.raven_path}')
            print(f'Parameters (raven only): --identity {filter} -k29 -w9 -t{threads} -p0')
            print(f'Assembly output: {out}\n')
            
            # Raven
            if assembler == 'raven':
                subprocess.run(f'{self.raven_path} --disable-checkpoints --identity {filter} -k29 -w9 -t{threads} -p0 {reads_path} > {idx}_{out}', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.csv {idx}_graph_1.csv', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_graph_1.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_graph_1.gfa')

            # Hifiasm
            elif assembler == 'hifiasm':
                # subprocess.run(f'{self.hifiasm_path} -o {idx}_asm -l0 -t{threads} {reads_path}', shell=True, cwd=self.output_dir)  # graph: {idx}_asm.unclean_moje.read.gfa
                # subprocess.run(f'mv {idx}_asm.unclean_moje.read.gfa {idx}_graph_1.gfa', shell=True, cwd=self.output_dir)
                # gfa_path = os.path.join(self.output_dir, f'{idx}_asm.unclean_moje.read.gfa')
                subprocess.run(f'/home/vrcekl/hifiasm-0.18.7-r514/hifiasm --prt-raw -o {idx}_asm -t32 -l0 {reads_path}', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv {idx}_asm.bp.raw.r_utg.gfa {idx}_graph_1.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_graph_1.gfa')
                extract_contigs(self.output_dir, idx)

            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, pred, succ, reads, edges, read_to_node, labels = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True)  # TODO: This is a mess!
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            pickle.dump(pred, open(f'{self.info_dir}/{idx}_pred.pkl', 'wb'))
            pickle.dump(succ, open(f'{self.info_dir}/{idx}_succ.pkl', 'wb'))
            pickle.dump(reads, open(f'{self.info_dir}/{idx}_reads.pkl', 'wb'))
            pickle.dump(edges, open(f'{self.info_dir}/{idx}_edges.pkl', 'wb'))
            pickle.dump(labels, open(f'{self.info_dir}/{idx}_labels.pkl', 'wb'))
            pickle.dump(read_to_node, open(f'{self.info_dir}/{idx}_read_to_node.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')


class AssemblyGraphDataset_ONT(AssemblyGraphDataset):

    def __init__(self, root, nb_pos_enc=10, assembler='raven', specs=None, generate=False):
        super().__init__(root=root, nb_pos_enc=nb_pos_enc, assembler=assembler, specs=specs, generate=generate)

    def process(self):
        """Process the raw data and save it on the disk."""
        if self.specs is None:
            threads = 32
            filter = 0.99
            out = 'assembly.fasta'
        else:
            threads = self.specs['threads']
            filter = self.specs['filter']
            out = self.specs['out']

        assembler = 'raven'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        diff = raw_files - prc_files

        for cnt, idx in enumerate(diff):
            fastq = f'{idx}.fasta'
            if fastq not in os.listdir(self.raw_dir):
                fastq = f'{idx}.fastq'
            print(f'Step {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}')
            # print(f'Starting assembler at: {self.raven_path}')
            print(f'Parameters (raven only): --identity {filter} -k29 -w9 -t{threads} -p0')
            print(f'Assembly output: {out}\n')
            
            # Raven
            if assembler == 'raven':
                subprocess.run(f'{self.raven_path} --disable-checkpoints -t{threads} -p0 {reads_path} > {idx}_{out}', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.csv {idx}_graph_1.csv', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_graph_1.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_graph_1.gfa')

            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, pred, succ, reads, edges, read_to_node, labels = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True)  # TODO: This is a mess!
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            pickle.dump(pred, open(f'{self.info_dir}/{idx}_pred.pkl', 'wb'))
            pickle.dump(succ, open(f'{self.info_dir}/{idx}_succ.pkl', 'wb'))
            pickle.dump(reads, open(f'{self.info_dir}/{idx}_reads.pkl', 'wb'))
            pickle.dump(edges, open(f'{self.info_dir}/{idx}_edges.pkl', 'wb'))
            pickle.dump(labels, open(f'{self.info_dir}/{idx}_labels.pkl', 'wb'))
            pickle.dump(read_to_node, open(f'{self.info_dir}/{idx}_read_to_node.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')
