import re
import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser
from config import get_config
from utils.data_utils import preprocess_graph, add_positional_encoding, extract_hifiasm_contigs


class AssemblyGraphDataset(DGLDataset):
    def __init__(self, root, assembler, threads=32, generate=False, n_need=0):
        self.root = os.path.abspath(root)
        self.assembler = assembler
        self.threads = threads
        self.n_need = n_need
        self.assembly_dir = os.path.join(self.root, self.assembler)

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
        
        config = get_config()
        raven_dir = config['raven_dir']
        self.raven_path = os.path.join(raven_dir, f'build/bin/raven')
        self.raven_path = os.path.abspath(self.raven_path)
        hifiasm_dir = config['hifiasm_dir']
        self.hifiasm_path = os.path.join(hifiasm_dir, f'hifiasm')
        self.hifiasm_path = os.path.abspath(self.hifiasm_path)
        
        super().__init__(name='assembly_graphs', raw_dir=raw_dir, save_dir=save_dir)

        self.graph_list = []
        if not generate:
            for file in os.listdir(self.save_dir):
                idx = int(file[:-4])
                graph = dgl.load_graphs(os.path.join(self.save_dir, file))[0][0]
                graph = preprocess_graph(graph)
                graph = add_positional_encoding(graph)
                # print(f'DGL graph idx={idx} info:\n',graph)
                self.graph_list.append((idx, graph))
            self.graph_list.sort(key=lambda x: x[0])
            print(f'Number of graphs in the dataset: {len(self.graph_list)}')

    def has_cache(self):
        """Check if the raw data is already processed and stored."""
        # raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        needed_files = {i for i in range(self.n_need)}
        return len(needed_files - prc_files) == 0  # set difference

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        i, graph = self.graph_list[idx]
        return i, graph

    def process(self):
        pass


class AssemblyGraphDataset_HiFi(AssemblyGraphDataset):

    def __init__(self, root, assembler='hifiasm', threads=1, generate=False, n_need=0):
        super().__init__(root=root, assembler=assembler, threads=threads, generate=generate, n_need=n_need)

    def process(self):
        """Process the raw data and save it on the disk."""
        assembler = 'hifiasm'
        print(f'hifiasm process')
        assert assembler in ('raven', 'hifiasm'), 'Choose either "raven" or "hifiasm" assembler'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        # raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        needed_files = {i for i in range(self.n_need)}
        diff = sorted(needed_files - prc_files)

        for cnt, idx in enumerate(diff):
            fastq = f'{idx}.fasta'
            if fastq not in os.listdir(self.raw_dir):
                fastq = f'{idx}.fastq'
            print(f'\nStep {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}\n')
            
            # Raven
            if assembler == 'raven':
                subprocess.run(f'{self.raven_path} --disable-checkpoints --identity 0.99 -k29 -w9 -t{self.threads} -p0 {reads_path} > {idx}_assembly.fasta', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_raw_graph.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_raw_graph.gfa')

            # Hifiasm
            elif assembler == 'hifiasm':
                write_paf = False  # TODO: Debugging purposes, remove
                if write_paf:
                    subprocess.run(f'{self.hifiasm_path} --prt-raw --write-paf -o {idx}_asm -t{self.threads} -l0 {reads_path}', shell=True, cwd=self.output_dir)
                    subprocess.run(f'mv {idx}_asm.ovlp.paf {idx}_ovlp.paf', shell=True, cwd=self.output_dir)
                    paf_path = os.path.join(self.output_dir, f'{idx}_ovlp.paf')
                else:
                    subprocess.run(f'{self.hifiasm_path} --prt-raw -o {idx}_asm -t{self.threads} -l0 {reads_path}', shell=True, cwd=self.output_dir)
                    paf_path = None
                subprocess.run(f'mv {idx}_asm.bp.raw.r_utg.gfa {idx}_raw_graph.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_raw_graph.gfa')
                extract_hifiasm_contigs(self.output_dir, idx)
                subprocess.run(f'rm {self.output_dir}/{idx}_asm*', shell=True)

            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, auxiliary = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True, paf_path=paf_path)
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            for name, data in auxiliary.items():
                pickle.dump(data, open(f'{self.info_dir}/{idx}_{name}.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')


class AssemblyGraphDataset_ONT(AssemblyGraphDataset):

    def __init__(self, root, assembler='raven', threads=1, generate=False, n_need=0):
        super().__init__(root=root, assembler=assembler, threads=threads, generate=generate, n_need=n_need)

    def process(self):
        """Process the raw data and save it on the disk."""
        assembler = 'raven'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        # raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        needed_files = {i for i in range(self.n_need)}
        diff = sorted(needed_files - prc_files)

        for cnt, idx in enumerate(diff):
            fastq = f'{idx}.fasta'
            if fastq not in os.listdir(self.raw_dir):
                fastq = f'{idx}.fastq'
            print(f'\nStep {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}')
            print(f'Other assemblers currently unavailable\n')
            
            # Raven
            if assembler == 'raven':
                subprocess.run(f'{self.raven_path} --disable-checkpoints -t{self.threads} -p0 {reads_path} > {idx}_assembly.fasta', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.csv {idx}_graph_1.csv', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_graph_1.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_graph_1.gfa')

            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, auxiliary = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True)
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            for name, data in auxiliary.items():
                pickle.dump(data, open(f'{self.info_dir}/{idx}_{name}.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')
