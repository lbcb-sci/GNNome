import argparse
import gzip
import os
import re
import pickle
import subprocess
import time
from datetime import datetime

import torch
from tqdm import tqdm
import requests
from Bio import SeqIO, AlignIO

import graph_dataset
from train import train
from inference import inference
import evaluate
import config
import hyperparameters


def change_description(file_path):
    new_fasta = []
    for record in SeqIO.parse(file_path, file_path[-5:]):  # 'fasta' for FASTA file, 'fastq' for FASTQ file
        des = record.description.split(",")
        id = des[0][5:]
        if des[1] == "forward":
            strand = '+'
        else:
            strand = '-'
        position = des[2][9:].split("-")
        start = position[0]
        end = position[1]
        record.id = id
        record.description = f'strand={strand} start={start} end={end}'
        new_fasta.append(record)
    SeqIO.write(new_fasta, file_path, "fasta")


def change_description2(fastq_path, maf_path, chr):
    chr = int(chr[3:])
    reads = {r.id: r for r in SeqIO.parse(fastq_path, 'fastq')}
    # print(len(reads))
    # counter = 0
    for align in AlignIO.parse(maf_path, 'maf'):
        ref, read_m = align
        start = ref.annotations['start']
        end = start + ref.annotations['size']
        strand = '+' if read_m.annotations['strand'] == 1 else '-'
        description = f'strand={strand} start={start} end={end} chr={chr}'
        reads[read_m.id].id += f'_chr{chr}'
        reads[read_m.id].name += f'_chr{chr}'
        reads[read_m.id].description = description
        # counter += 1
    # print(counter)
    fasta_path = fastq_path[:-1] + 'a'
    SeqIO.write(list(reads.values()), fasta_path, 'fasta')
    os.remove(fastq_path)
    return fasta_path


def create_chr_dirs(pth):
    for i in range(1, 24):
        if i == 23:
            i = 'X'
        subprocess.run(f'mkdir chr{i}', shell=True, cwd=pth)
        subprocess.run(f'mkdir raw raven hifiasm', shell=True, cwd=os.path.join(pth, f'chr{i}'))
        subprocess.run(f'mkdir processed info output graphia', shell=True, cwd=os.path.join(pth, f'chr{i}/raven'))
        subprocess.run(f'mkdir processed info output graphia', shell=True, cwd=os.path.join(pth, f'chr{i}/hifiasm'))


def merge_dicts(d1, d2, d3={}):
    keys = {*d1, *d2, *d3}
    merged = {key: d1.get(key, 0) + d2.get(key, 0) + d3.get(key, 0) for key in keys}
    return merged


# -1. Set up the data file structure
def file_structure_setup(data_path, refs_path):
    # TODO: Do something with this!
    return

    print(f'SETUP::filesystem:: Create directories for storing data')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if 'CHM13' not in os.listdir(refs_path):
        os.mkdir(os.path.join(refs_path, 'CHM13'))
    if 'chromosomes' not in os.listdir(refs_path):
        os.mkdir(os.path.join(refs_path, 'chromosomes'))
            
    if 'simulated_hifi' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'simulated_hifi'))
        create_chr_dirs(os.path.join(data_path, 'simulated_hifi'))
    if 'simulated_ont' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'simulated_ont'))
        create_chr_dirs(os.path.join(data_path, 'simulated_ont'))
    # if 'real' not in os.listdir(data_path):
        # subprocess.run(f'bash download_dataset.sh {data_path}', shell=True)
        # os.mkdir(os.path.join(data_path, 'real'))
        # create_chr_dirs(os.path.join(data_path, 'real'))
    if 'experiments' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'experiments'))


# 0. Download the CHM13 if necessary
def download_reference(refs_path):
    chm_path = os.path.join(refs_path, 'CHM13')
    chr_path = os.path.join(refs_path, 'chromosomes')
    chm13_url = 'https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chm13.draft_v1.1.fasta.gz'
    chm13_path = os.path.join(chm_path, 'chm13.draft_v1.1.fasta.gz')

    if len(os.listdir(chm_path)) == 0:
        # Download the CHM13 reference
        # Code for tqdm from: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        print(f'SETUP::download:: CHM13 not found! Downloading...')
        response = requests.get(chm13_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(chm13_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    if len(os.listdir(chr_path)) == 0:
        # Parse the CHM13 into individual chromosomes
        print(f'SETUP::download:: Split CHM13 per chromosome')
        with gzip.open(chm13_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                SeqIO.write(record, os.path.join(chr_path, f'{record.id}.fasta'), 'fasta')


def handle_pbsim_output(idx, chrN, chr_raw_path, combo=False):
    if combo == True:
        idx = chrN
    subprocess.run(f'mv {idx}_0001.fastq {idx}.fastq', shell=True, cwd=chr_raw_path)
    subprocess.run(f'mv {idx}_0001.maf {idx}.maf', shell=True, cwd=chr_raw_path)
    subprocess.run(f'rm {idx}_0001.ref', shell=True, cwd=chr_raw_path)
    fastq_path = os.path.join(chr_raw_path, f'{idx}.fastq')
    maf_path = os.path.join(chr_raw_path, f'{idx}.maf')
    print(f'Adding positions for training...')
    fasta_path = change_description2(fastq_path, maf_path, chr=chrN)  # Extract positional info from the MAF file
    print(f'Removing the MAF file...')
    subprocess.run(f'rm {idx}.maf', shell=True, cwd=chr_raw_path)
    if combo:
        return fasta_path
    else:
        return None


# 1. Simulate the sequences - HiFi
def simulate_reads_hifi(data_path, refs_path, chr_dict, assembler):
    print(f'SETUP::simulate')
    if 'vendor' not in os.listdir():
        os.mkdir('vendor')
        
    pbsim3_dir = f'/home/vrcekl/pbsim3'  # TODO: Put into hyperparameters/config file or into vendor!

    data_path = os.path.abspath(data_path)
    for chrN_flag, n_need in chr_dict.items():
        if chrN_flag.endswith('_r'):
            continue
        if '+' in chrN_flag:
            continue

        elif chrN_flag.endswith('_chm13'):
            chrN = chrN_flag[:-6]
            chr_path = os.path.join(refs_path, 'CHM13', 'chromosomes')
            pbsim_path = os.path.join(data_path, 'chm13_pbsim3')
            chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
            sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/chm13_subsampled.fastq'
            sample_profile_id = f'chm13'
            depth = 30
                    
        elif chrN_flag.endswith('_ncbr'):
            chrN = chrN_flag[:-5]
            chr_path = os.path.join(refs_path, 'ncoibor', 'chromosomes')
            pbsim_path = os.path.join(data_path, 'ncoibor_pbsim3')
            chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
            sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/chm13_subsampled.fastq'
            sample_profile_id = f'chm13'
            depth = 30
                    
        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            chr_path = os.path.join(refs_path, f'HG002', 'hg002_chromosomes')
            pbsim_path = os.path.join(data_path, 'hg002_pbsim3')
            chr_seq_path = os.path.join(chr_path, f'{chrN}_MATERNAL.fasta')
            sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/HG002/20kb/m64011_190830_220126.sub.fastq'
            sample_profile_id = f'20kb-m64011_190830_220126'
            depth = 30

        else:
            print('Give valid suffix!')
            raise Exception

        chr_raw_path = os.path.join(pbsim_path, f'{chrN}/raw')
        chr_processed_path = os.path.join(pbsim_path, f'{chrN}/{assembler}/processed')
        if not os.path.isdir(chr_raw_path):
            os.makedirs(chr_raw_path)
        if not os.path.isdir(chr_processed_path):
            os.makedirs(chr_processed_path)

        # TODO: Fix so that you can delete raw files
        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(chr_raw_path)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(chr_processed_path)}
        all_files = raw_files | prc_files
        n_have = max(all_files) + 1 if all_files else 0

        if n_need <= n_have:
            continue
        n_diff = n_need - n_have
        print(f'SETUP::simulate:: Simulate {n_diff} datasets for {chrN_flag} with PBSIM3')
        for i in range(n_diff):
            idx = n_have + i
            chr_save_path = os.path.join(chr_raw_path, f'{idx}.fasta')
            print(f'\nStep {i}: Simulating reads {chr_save_path}')
            # Use the CHM13/HG002 profile for all the chromosomes
            if f'sample_profile_{sample_profile_id}.fastq' not in os.listdir(pbsim3_dir):
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample {sample_file_path} '
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            else:
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            handle_pbsim_output(idx, chrN, chr_raw_path)


def simulate_reads_combo(data_path, refs_path, chr_dict, assembler):
    data_path = os.path.abspath(data_path)
    pbsim_path = os.path.join(data_path, 'combo')
    pbsim3_dir = f'/home/vrcekl/pbsim3'

    for chrN_combo, n_need in chr_dict.items():
        if '+' not in chrN_combo:
            continue
        chromosomes = chrN_combo.split('+')  # chr1_chm13+chr2_chm13+chr3_chm13

        chr_raw_path = os.path.join(pbsim_path, f'{chrN_combo}/raw')
        if not os.path.isdir(chr_raw_path):
            os.makedirs(chr_raw_path)
        chr_processed_path = os.path.join(pbsim_path, f'{chrN_combo}/{assembler}/processed')  # TODO: Fix so that you can delete raw files
        if not os.path.isdir(chr_processed_path):
            os.makedirs(chr_processed_path)

        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(chr_raw_path)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(chr_processed_path)}
        all_files = raw_files | prc_files
        if all_files:
            n_have = max(all_files) + 1
        else:
            n_have = 0

        if n_need <= n_have:
            continue
        else:
            n_diff = n_need - n_have
            print(f'SETUP::simulate:: Simulate {n_diff} datasets for {chrN_combo} with PBSIM3')

            # Simulate reads for chrN_combo n_diff times
            for i in range(n_diff):
                idx = n_have + i
                all_reads = []
                for chromosome in chromosomes:
                    if chromosome.endswith('_chm13'):
                        chrN = chromosome[:-6]
                        chr_path = os.path.join(refs_path, 'CHM13', 'chromosomes')
                        chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
                        chr_save_path = os.path.join(chr_raw_path, f'{chrN}.fasta')
                        sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/chm13_subsampled.fastq'
                        sample_profile_id = f'chm13'
                        depth = 30

                    elif chromosome.endswith('_ncbr'):
                        chrN = chromosome[:-5]
                        chr_path = os.path.join(refs_path, 'ncoibor', 'chromosomes')
                        chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
                        chr_save_path = os.path.join(chr_raw_path, f'{chrN}.fasta')
                        sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/chm13_subsampled.fastq'
                        sample_profile_id = f'chm13'
                        depth = 30

                    elif chromosome.endswith('_hg002'):
                        chrN = chromosome[:-6]
                        chr_path = os.path.join(refs_path, 'HG002', 'hg002_chromosomes')
                        chr_seq_path = os.path.join(chr_path, f'{chrN}_MATERNAL.fasta')
                        chr_save_path = os.path.join(chr_raw_path, f'{chrN}.fasta')
                        sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/HG002/20kb/m64011_190830_220126.sub.fastq'
                        sample_profile_id = f'20kb-m64011_190830_220126'
                        depth = 30

                    print(f'\nStep {i}: Simulating reads {chr_save_path}')
                    if f'sample_profile_{sample_profile_id}.fastq' not in os.listdir(pbsim3_dir):
                        subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                        f'--sample {sample_file_path} '
                                        f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{chrN}', shell=True, cwd=pbsim3_dir)
                    else:
                        subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                        f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{chrN}', shell=True, cwd=pbsim3_dir)
                        
                    fasta_path = handle_pbsim_output(idx, chrN, chr_raw_path, combo=True)  # Because it's combo we pass chrN instead of idx! We get idx.fasta later
                    # Combining individual chromosome FASTAs into a unified list
                    print(f'Appending the list of all the reads with {chromosome}...', end='\t')
                    all_reads.extend(list(SeqIO.parse(fasta_path, 'fasta')))
                    subprocess.run(f'rm {fasta_path}', shell=True)
                    print(f'Done!')

                # Saving the unified FASTA file as idx.fasta
                all_reads_path = os.path.join(chr_raw_path, f'{idx}.fasta')
                SeqIO.write(all_reads, all_reads_path, 'fasta')


def simulate_reads_ont(data_path, refs_path, chr_dict, assembler='raven'):
    print(f'SETUP::simulate')
    if 'vendor' not in os.listdir():
        os.mkdir('vendor')
    pbsim3_dir = f'/home/vrcekl/pbsim3'

    data_path = os.path.abspath(data_path)
    for chrN_flag, n_need in chr_dict.items():
        if chrN_flag.endswith('_r'):  # Training on real reads
            continue
        if '+' in chrN_flag:  # Training on combined synthetic chromosomes
            continue
        elif chrN_flag.endswith('_ncbr'):  # Training on synthetic ncoibor chromosomes
            continue

        elif chrN_flag.endswith('_chm13'):
            chrN = chrN_flag[:-6]  # chrN_chm13
            chr_path = os.path.join(refs_path, 'CHM13', 'chromosomes')
            pbsim_path = os.path.join(data_path, 'chm13_pbsim3')
            chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
            sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data-DELETE/CHM13/ONT/chm13_ont-subsampled_2M_trimmed.fastq'
            sample_profile_id = f'chm13-ont'
            depth = 120

        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]  # chrN_hg002
            chr_path = os.path.join(refs_path, 'HG002', 'hg002_chromosomes')
            pbsim_path = os.path.join(data_path, 'hg002_pbsim3')  
            chr_seq_path = os.path.join(chr_path, f'{chrN}_MATERNAL.fasta')
            sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data-DELETE/CHM13/ONT/chm13_ont-subsampled_2M_trimmed.fastq'
            sample_profile_id = f'chm13-ont'
            depth = 120

        else:
            print(f'Chromosome suffix incorrect!')
            raise Exception

        chr_raw_path = os.path.join(pbsim_path, f'{chrN}/raw')
        chr_processed_path = os.path.join(pbsim_path, f'{chrN}/{assembler}/processed')
        if not os.path.isdir(chr_raw_path):
            os.makedirs(chr_raw_path)
        if not os.path.isdir(chr_processed_path):
            os.makedirs(chr_processed_path)
        # TODO: Fix so that you can delete raw files
        raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(chr_raw_path)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(chr_processed_path)}
        all_files = raw_files | prc_files
        n_have = max(all_files) + 1 if all_files else 0

        if n_need <= n_have:
            continue
        n_diff = n_need - n_have
        print(f'SETUP::simulate:: Simulate {n_diff} datasets for {chrN} with PBSIM3')
        for i in range(n_diff):
            idx = n_have + i
            chr_save_path = os.path.join(chr_raw_path, f'{idx}.fasta')
            print(f'\nStep {i}: Simulating reads {chr_save_path}')
            if f'sample_profile_{sample_profile_id}.fastq' in os.listdir(pbsim3_dir):
                # Use the CHM13 profile for all the chromosomes
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            else:
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample {sample_file_path} ' \
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            handle_pbsim_output(idx, chrN, chr_raw_path)


# 2. Generate the graphs
def generate_graphs_hifi(data_path, chr_dict, assembler):
    print(f'SETUP::generate')

    if 'raven' not in os.listdir('vendor'):
        print(f'SETUP::generate:: Download Raven')
        subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
        subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
        subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    data_path = os.path.abspath(data_path)

    for chrN_flag, n_need in chr_dict.items():
        if '+' in chrN_flag:
            chrN = chrN_flag  # Called chrN_combo in simulate_reads_combo function
            chr_sim_path = os.path.join(data_path, 'combo', f'{chrN}')
        elif chrN_flag.endswith('_r'):
            chrN = chrN_flag[:-2]
            chr_sim_path = os.path.join(data_path, 'real', f'{chrN}')
        elif chrN_flag.endswith('_chm13'):
            chrN = chrN_flag[:-6]
            chr_sim_path = os.path.join(data_path, 'chm13_pbsim3', f'{chrN}')
        elif chrN_flag.endswith('_ncbr'):
            chrN = chrN_flag[:-5]
            chr_sim_path = os.path.join(data_path, 'ncoibor_pbsim3', f'{chrN}')
        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            chr_sim_path = os.path.join(data_path, 'hg002_pbsim3', f'{chrN}')
        else:
            print(f'Give valid suffix')
            raise Exception

        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, f'{assembler}/processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = max(0, n_raw - n_prc)
        print(f'SETUP::generate:: Generate {n_diff} graphs for {chrN}')
        specs = {
            'threads': 32,
            'filter': 0.99,
            'out': 'assembly.fasta',
            'assembler': assembler,
        }
        graph_dataset.AssemblyGraphDataset_HiFi(chr_sim_path, nb_pos_enc=None, assembler=assembler, specs=specs, generate=True)


def generate_graphs_ont(data_path, chr_dict, assembler):
    print(f'SETUP::generate')

    if 'raven' not in os.listdir('vendor'):
        print(f'SETUP::generate:: Download Raven')
        subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
        subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
        subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    data_path = os.path.abspath(data_path)

    for chrN_flag, n_need in chr_dict.items():
        if '+' in chrN_flag:
            chrN = chrN_flag  # Called chrN_combo in simulate_reads_combo function
            chr_sim_path = os.path.join(data_path, 'combo', f'{chrN}')
        elif chrN_flag.endswith('_r'):
            chrN = chrN_flag[:-2]
            chr_sim_path = os.path.join(data_path, 'real', f'{chrN}')
        elif chrN_flag.endswith('_chm13'):
            chrN = chrN_flag[:-6]
            chr_sim_path = os.path.join(data_path, 'chm13_pbsim3', f'{chrN}')
        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            chr_sim_path = os.path.join(data_path, 'hg002_pbsim3', f'{chrN}')            
        else:
            print(f'Chromosome suffix incorrect!')
            raise Exception

        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, f'{assembler}/processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = max(0, n_raw - n_prc)
        print(f'SETUP::generate:: Generate {n_diff} graphs for {chrN}')
        specs = {
            'threads': 32,
            'filter': 0.99,
            'out': 'assembly.fasta',
            'assembler': 'raven',
        }
        graph_dataset.AssemblyGraphDataset_ONT(chr_sim_path, nb_pos_enc=None, assembler='raven', specs=specs, generate=True)


# 2.5 Train-valid-test split
def train_valid_split(data_path, eval_path, temp_path, assembler, train_dict, valid_dict, test_dict={}, out=None, overfit=False):
    print(f'SETUP::split')
    data_path = os.path.abspath(data_path)
    eval_path = os.path.abspath(eval_path)
    if overfit:
        data_path = eval_path

    real_path = os.path.join(eval_path, 'real')
    # pbsim_path = os.path.join(data_path, 'pbsim3')
    ncoibor_path = os.path.join(data_path, 'ncoibor_pbsim3')
    hg002_path = os.path.join(data_path, 'hg002_pbsim3')
    combo_path = os.path.join(data_path, 'combo')
    chm13_path = os.path.join(data_path, 'chm13_pbsim3')
    arab_path = os.path.join(data_path, 'arabidopsis_pbsim3')
    zmays_path = os.path.join(data_path, 'zmays_Mo17_pbsim3')
    exp_path = temp_path

    if out is None:
        train_path = os.path.join(exp_path, f'train', assembler)
        valid_path = os.path.join(exp_path, f'valid', assembler)
        test_path  = os.path.join(exp_path, f'test', assembler)
    else:
        train_path = os.path.join(exp_path, f'train_{out}', assembler)
        valid_path = os.path.join(exp_path, f'valid_{out}', assembler)
        test_path  = os.path.join(exp_path, f'test_{out}', assembler)

    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=train_path)
    if not os.path.isdir(valid_path):
        os.makedirs(valid_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=valid_path)
    if not os.path.isdir(test_path) and len(test_dict) > 0:
        os.makedirs(test_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=test_path)
 
    train_g_to_chr = {}  # Remember chromosomes for each graph in the dataset
    train_g_to_org_g = {}  # Remember index of the graph in the master dataset for each graph in this dataset
    n_have = 0
    
    if assembler == 'both':
        assemblers = ['hifiasm', 'raven']
    else:
        assemblers = [assembler]

    for assembler in assemblers:
        for chrN_flag, n_need in train_dict.items():
            # copy n_need datasets from chrN into train dict
            if '_r' in chrN_flag and n_need > 1:
                print(f'SETUP::split::WARNING Cannot copy more than one graph for real data: {chrN_flag}')
                n_need = 1
            print(f'SETUP::split:: Copying {n_need} graphs of {chrN_flag} - {assembler} into {train_path}')
            for i in range(n_need):
                if '+' in chrN_flag:
                    chrN = chrN_flag
                    chr_sim_path = os.path.join(combo_path, chrN, assembler)
                elif chrN_flag.endswith('_r'):
                    chrN = chrN_flag[:-2]
                    chr_sim_path = os.path.join(real_path, 'chm13_chromosomes', chrN, assembler)
                # elif chrN_flag.endswith('_pbs'):
                #     chrN = chrN_flag[:-4]
                #     chr_sim_path = os.path.join(pbsim_path, chrN, assembler)
                elif chrN_flag.endswith('_ncbr'):
                    chrN = chrN_flag[:-5]
                    chr_sim_path = os.path.join(ncoibor_path, chrN, assembler)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
                elif chrN_flag.endswith('_chm13'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(chm13_path, chrN, assembler)
                elif chrN_flag.endswith('_arab'):
                    chrN = chrN_flag[:-5]
                    chr_sim_path = os.path.join(arab_path, chrN, assembler)
                elif chrN_flag.endswith('_zmays'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(zmays_path, chrN, assembler)
                else:
                    print(f'Give proper suffix!')
                    raise Exception

                train_g_to_chr[n_have] = chrN
                print(f'Copying {chr_sim_path}/processed/{i}.dgl into {train_path}/processed/{n_have}.dgl')
                subprocess.run(f'cp {chr_sim_path}/processed/{i}.dgl {train_path}/processed/{n_have}.dgl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{i}_succ.pkl {train_path}/info/{n_have}_succ.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{i}_pred.pkl {train_path}/info/{n_have}_pred.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{i}_edges.pkl {train_path}/info/{n_have}_edges.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{i}_reads.pkl {train_path}/info/{n_have}_reads.pkl', shell=True)
                train_g_to_org_g[n_have] = i
                n_have += 1
    pickle.dump(train_g_to_chr, open(f'{train_path}/info/g_to_chr.pkl', 'wb'))
    pickle.dump(train_g_to_org_g, open(f'{train_path}/info/g_to_org_g.pkl', 'wb'))

    valid_g_to_chr = {}
    valid_g_to_org_g = {}
    n_have = 0
    for assembler in assemblers:
        for chrN_flag, n_need in valid_dict.items():
            # copy n_need datasets from chrN into train dict
            if '_r' in chrN_flag and n_need > 1:
                print(f'SETUP::split::WARNING Cannot copy more than one graph for real data: {chrN_flag}')
                n_need = 1
            print(f'SETUP::split:: Copying {n_need} graphs of {chrN_flag} - {assembler} into {valid_path}')
            for i in range(n_need):
                if '+' in chrN_flag:
                    chrN = chrN_flag
                    chr_sim_path = os.path.join(combo_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_r'):
                    chrN = chrN_flag[:-2]
                    chr_sim_path = os.path.join(real_path, 'chm13_chromosomes', chrN, assembler)
                    j = 0
                # elif chrN_flag.endswith('_pbs'):
                #     chrN = chrN_flag[:-4]
                #     chr_sim_path = os.path.join(pbsim_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_ncbr'):
                    chrN = chrN_flag[:-5]
                    chr_sim_path = os.path.join(ncoibor_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_chm13'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(chm13_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_arab'):
                    chrN = chrN_flag[:-5]
                    chr_sim_path = os.path.join(arab_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_zmays'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(zmays_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                else:
                    print(f'Give proper suffix!')
                    raise Exception

                valid_g_to_chr[n_have] = chrN
                print(f'Copying {chr_sim_path}/processed/{j}.dgl into {valid_path}/processed/{n_have}.dgl')
                subprocess.run(f'cp {chr_sim_path}/processed/{j}.dgl {valid_path}/processed/{n_have}.dgl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{j}_succ.pkl {valid_path}/info/{n_have}_succ.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{j}_pred.pkl {valid_path}/info/{n_have}_pred.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{j}_edges.pkl {valid_path}/info/{n_have}_edges.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/info/{j}_reads.pkl {valid_path}/info/{n_have}_reads.pkl', shell=True)
                valid_g_to_org_g[n_have] = j
                n_have += 1
    pickle.dump(valid_g_to_chr, open(f'{valid_path}/info/g_to_chr.pkl', 'wb'))
    pickle.dump(valid_g_to_org_g, open(f'{valid_path}/info/g_to_org_g.pkl', 'wb'))

    # TODO: FIX THIS !!!!!!!!!!!!!!!!!!
    train_path = os.path.join(train_path, os.path.pardir)
    valid_path = os.path.join(valid_path, os.path.pardir)
    test_path = os.path.join(test_path, os.path.pardir)
    ###################################

    return train_path, valid_path, test_path


# def predict_baselines(test_path, assembler, out, model_path=None, device='cpu'):
#     if model_path is None:
#         model_path = os.path.abspath(f'pretrained/model_{out}.pt')
#     walks_and_contigs = inference_baselines(test_path, model_path, assembler, device)
#     walks_per_graph, contigs_per_graph = walks_and_contigs[0], walks_and_contigs[1]
#     walks_per_graph_ol_len, contigs_per_graph_ol_len = walks_and_contigs[2], walks_and_contigs[3]
#     walks_per_graph_ol_sim, contigs_per_graph_ol_sim = walks_and_contigs[4], walks_and_contigs[5]
#     g_to_chr = pickle.load(open(f'{test_path}/info/g_to_chr.pkl', 'rb'))
    
#     for idx, (contigs, contigs_ol_len, contigs_ol_sim) in enumerate(zip(contigs_per_graph, contigs_per_graph_ol_len, contigs_per_graph_ol_sim)):
#         chrN = g_to_chr[idx]
#         print(f'GNN: Scores')
#         num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
#         evaluate.print_summary(test_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50)
#         print(f'Baseline: Overlap lengths')
#         num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs_ol_len, chrN)
#         evaluate.print_summary(test_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50)
#         print(f'Baseline: Overlap similarities')
#         num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs_ol_sim, chrN)
#         evaluate.print_summary(test_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50)


def cleanup(train_path, valid_path):
    subprocess.run(f'rm -rf {train_path}', shell=True)
    subprocess.run(f'rm -rf {valid_path}', shell=True)


def evaluate_real(eval_path, assembler, model_path, asm_path, ref_path, save_dir):
    real_path = os.path.join(eval_path, 'real')
    save_dir = os.path.join(asm_path, 'real', assembler, save_dir)
    procs = []

    for i in range(1, 24):
        if i == 23:
            i = 'X'

        print(f'\nChromosome {i}')
        chr_path = os.path.join(real_path, f'chr{i}')
        save_path = os.path.join(save_dir, f'chr{i}')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            os.mkdir(os.path.join(save_path, f'assembly'))
            os.mkdir(os.path.join(save_path, f'decode'))
            os.mkdir(os.path.join(save_path, f'reports'))

        inference(chr_path, model_path, assembler, save_path)

        ref = os.path.join(ref_path, 'CHM13', 'chromosomes', f'chr{i}.fasta')
        idx = os.path.join(ref_path, 'CHM13', 'chromosomes', 'indexed', f'chr{i}.fasta.fai')
        asm = os.path.join(save_path, f'assembly', f'0_assembly.fasta')
        report = os.path.join(save_path, f'reports', '0_minigraph.txt')
        paf = os.path.join(save_path, f'asm.paf')
        p = evaluate.run_minigraph(ref, asm, paf)
        procs.append(p)

    for p in procs:
        p.wait()

    procs = []
    for i in range(1, 24):
        if i == 23:
            i = 'X'

        save_path = os.path.join(save_dir, f'chr{i}')
        idx = os.path.join(ref_path, 'indexed', f'chr{i}.fasta.fai')
        paf = os.path.join(save_path, f'asm.paf')
        report = os.path.join(save_path, f'reports', '0_minigraph.txt')
        p = evaluate.parse_pafs(idx, report, paf)
        procs.append(p)
    for p in procs:
        p.wait()

    evaluate.parse_minigraph_for_chrs(save_dir)


def evaluate_synth(eval_path, assembler, model_path, asm_path, ref_path, save_dir):
    synth_path = os.path.join(eval_path, 'synth')
    save_dir = os.path.join(asm_path, 'synth', assembler, save_dir)
    procs = []
    
    for i in range(1, 24):
        if i == 23:
            i = 'X'

        print(f'\nChromosome {i}')
        chr_path = os.path.join(synth_path, f'chr{i}')
        save_path = os.path.join(save_dir, f'chr{i}')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            os.mkdir(os.path.join(save_path, f'assembly'))
            os.mkdir(os.path.join(save_path, f'decode'))
            os.mkdir(os.path.join(save_path, f'reports'))

        inference(chr_path, model_path, assembler, save_path)

        ref = os.path.join(ref_path, 'CHM13', 'chromosomes', f'chr{i}.fasta')
        idx = os.path.join(ref_path, 'CHM13', 'chromosomes', 'indexed', f'chr{i}.fasta.fai')
        asm = os.path.join(save_path, f'assembly', f'0_assembly.fasta')
        report = os.path.join(save_path, f'reports', '0_minigraph.txt')
        paf = os.path.join(save_path, f'asm.paf')
        p = evaluate.run_minigraph(ref, asm, paf)
        procs.append(p)

    for p in procs:
        p.wait()

    procs = []
    for i in range(1, 24):
        if i == 23:
            i = 'X'

        save_path = os.path.join(save_dir, f'chr{i}')
        idx = os.path.join(ref_path, 'indexed', f'chr{i}.fasta.fai')
        paf = os.path.join(save_path, f'asm.paf')
        report = os.path.join(save_path, f'reports', '0_minigraph.txt')
        p = evaluate.parse_pafs(idx, report, paf)
        procs.append(p)
    for p in procs:
        p.wait()

    evaluate.parse_minigraph_for_chrs(save_dir)


def evaluate_genome(eval_path, assembler, model_path, asm_path, ref_path, genome, save_dir):
    real_path = os.path.join(eval_path, 'real')
    save_dir = os.path.join(asm_path, 'real', assembler, save_dir)

    print(f'New genome')
    chr_path = os.path.join(real_path, genome)
    save_path = os.path.join(save_dir, genome)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        os.mkdir(os.path.join(save_path, f'assembly'))
        os.mkdir(os.path.join(save_path, f'decode'))
        os.mkdir(os.path.join(save_path, f'reports'))

    inference(chr_path, model_path, assembler, save_path)

    ref = ref_path
    idx = ref_path + '.fai'
    asm = os.path.join(save_path, f'assembly', f'0_assembly.fasta')
    report = os.path.join(save_path, f'reports', '0_minigraph.txt')
    paf = os.path.join(save_path, f'asm.paf')
    
    p = evaluate.run_minigraph(ref, asm, paf)
    p.wait()    
    p = evaluate.parse_pafs(idx, report, paf)
    p.wait()
    evaluate.parse_minigraph_for_full(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=None, help='Output name for models')
    parser.add_argument('--overfit', action='store_true', default=False, help='Overfit on the chromosomes in the train directory')
    args = parser.parse_args()

    out = args.out
    overfit = args.overfit

    hyperparams = hyperparameters.get_hyperparameters()
    data_path = hyperparams['data_path']  # Location of the master database (storage)
    temp_path = hyperparams['temp_path']  # Location where the data will be temporarily stored for training
    eval_path = hyperparams['eval_path']  # Location where the synth and real evaluation data is stored
    refs_path = hyperparams['refs_path']  # Location where the references are stored - local because everythin else can be generated from this
    asms_path = hyperparams['asms_path']  # Where the assemblies and other inference info will be stored
    assembler = hyperparams['assembler']  # Which assembler we are using, currently: raven/hifiasm
    models_path = hyperparams['models_path']
    threads = hyperparams['num_threads']
    # dataset = hyperparams['dataset']  # Which dataset we are using, currently it's only chm13
    
    data_path_ont = hyperparams['data_path_ont']
    eval_path_ont = hyperparams['eval_path_ont']

    initials = hyperparams['initials']

    time_start = datetime.now()
    if out is None:
        timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
        out = f'{timestamp}_{initials}'
    else:
        timestamp = time_start.strftime('%y-%m-%d')
        out = f'{timestamp}_{initials}_{out}'

    # Model name must start with the date when the model was trained, in the yy-mm-dd format
    # Following is the underscore and a name of the model
    # E.g., 22-10-31_modelA
    # All the info about the training (train/valid data, hyperparameters, etc.) should be stored in the logbook
    # You can also include them in the model name, but they NEED to be stored in the logbook!
    model_name = out

    # In the inference, model_name represents the model used for evaluation
    # All the inference data (predictions, walks, assemblies, and reports)
    # Will be stored in a directory with name {model_name}_{decoding}
    # Suffix should indicate info about the decoding
    strategy = hyperparams['strategy']
    B = hyperparams['B']
    num_decoding_paths = hyperparams['num_decoding_paths']
    if strategy == 'greedy':
        save_dir = f'{model_name}_Gx{num_decoding_paths}'
    elif strategy == 'beamsearch':
        save_dir = f'{model_name}_B{B}x{num_decoding_paths}'

    dicts = config.get_config()
    train_dict = dicts['train_dict']
    valid_dict = dicts['valid_dict']
    test_dict = dicts['test_dict']
    train_dict_ont = dicts['train_dict_ont']
    valid_dict_ont = dicts['valid_dict_ont']
    test_dict_ont = {}

    specs = {
        'threads': threads,
        'filter': 0.99,
        'out': 'assembly.fasta',
        'assembler': assembler,
    }

    torch.set_num_threads(threads)

    model_path = os.path.join(models_path, f'model_{model_name}.pt')
    

    all_chr = merge_dicts(train_dict, valid_dict, test_dict)
    
    all_chr_ont = merge_dicts(train_dict_ont, valid_dict_ont)

    # file_structure_setup(data_path, refs_path)
    # download_reference(refs_path)

    # simulate_reads_hifi(data_path, refs_path, all_chr, assembler)
    # simulate_reads_combo(data_path, refs_path, all_chr, assembler)
    # generate_graphs_hifi(data_path, all_chr, assembler)
    
    # simulate_reads_ont(data_path_ont, refs_path, all_chr_ont, 'raven')
    # generate_graphs_ont(data_path_ont, all_chr_ont, 'raven')
    
    # exit(0)

    if overfit:
        train_path, valid_path, test_path = train_valid_split(data_path, eval_path, temp_path, assembler, train_dict, valid_dict, test_dict, out, overfit=True)
        train(train_path, valid_path, out, assembler, overfit)
        model_path = os.path.abspath(f'{models_path}/model_{out}.pt')
        save_dir = os.path.join(train_path, assembler)
        inference(train_path, model_path, assembler, save_dir, device='cpu')
    else:
  
        # Train: 15 x HG002 chrs 1,  3,  5,  9, 12, 18
        # Valid:  5 x HG002 chrs 6, 11, 17, 19, 20
        # Type : HiFi 60x coverage
        train_path = f'/home/vrcekl/scratch/gnnome_assembly/train/train_23-07-12_LV_t1-3-5-9-12-18_x15_hg002_60x_v6-11-17-19-20_x5_hg002_60x'
        valid_path = f'/home/vrcekl/scratch/gnnome_assembly/train/valid_23-07-12_LV_t1-3-5-9-12-18_x15_hg002_60x_v6-11-17-19-20_x5_hg002_60x'        

        dropout = '0.20'
        seed = 0
        drop_str = dropout.replace('.', '')
        out_drop = f'{out}_drop{drop_str}'
        print(f'RUN INDETIFIER:', out_drop)
        train(train_path, valid_path, out_drop, assembler, overfit, dropout=float(dropout), seed=seed)
