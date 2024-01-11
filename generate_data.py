import argparse
import os
import re
import subprocess
import time
from datetime import datetime

from Bio import SeqIO, AlignIO

import graph_dataset
import config
from paths import get_paths


def change_description_seqreq(file_path):
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


def change_description_pbsim(fastq_path, maf_path, chr):
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


def merge_dicts(d1, d2, d3={}):
    keys = {*d1, *d2, *d3}
    merged = {key: d1.get(key, 0) + d2.get(key, 0) + d3.get(key, 0) for key in keys}
    return merged


def handle_pbsim_output(idx, chrN, chr_raw_path, combo=False):
    if combo == True:
        idx = chrN
    subprocess.run(f'mv {idx}_0001.fastq {idx}.fastq', shell=True, cwd=chr_raw_path)
    subprocess.run(f'mv {idx}_0001.maf {idx}.maf', shell=True, cwd=chr_raw_path)
    subprocess.run(f'rm {idx}_0001.ref', shell=True, cwd=chr_raw_path)
    fastq_path = os.path.join(chr_raw_path, f'{idx}.fastq')
    maf_path = os.path.join(chr_raw_path, f'{idx}.maf')
    print(f'Adding positions for training...')
    fasta_path = change_description_pbsim(fastq_path, maf_path, chr=chrN)  # Extract positional info from the MAF file
    print(f'Removing the MAF file...')
    subprocess.run(f'rm {idx}.maf', shell=True, cwd=chr_raw_path)
    if combo:
        return fasta_path
    else:
        return None


# 1. Simulate the sequences - HiFi
def simulate_reads_hifi(outdir_path, chrs_path, chr_dict, assembler, pbsim3_dir, sample_profile_id, sample_file_path):
    print(f'SETUP::simulate')
    outdir_path = os.path.abspath(outdir_path)
    for chrN_flag, n_need in chr_dict.items():
        if chrN_flag.endswith('_r'):
            continue
        if '+' in chrN_flag:
            continue
        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            # chr_path = os.path.join(refs_path, f'HG002', 'hg002_chromosomes')  # TODO: redefine refs path
            chr_seq_path = os.path.join(chrs_path, f'{chrN}.fasta')
            # sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/HG002/20kb/m64011_190830_220126.sub.fastq'  # TODO: Need to provide this as an argument
            # sample_profile_id = f'20kb-m64011_190830_220126'  # TODO: Need to provide this as an argument
            depth = 60
        else:
            print('Give valid suffix!')
            raise Exception

        chr_raw_path = os.path.join(outdir_path, f'{chrN}/raw')
        chr_processed_path = os.path.join(outdir_path, f'{chrN}/{assembler}/processed')
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
                assert os.path.isfile(sample_file_path), "Sample profile ID and sample file not found! Provide either a valid sample profile ID or a sample file."
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample {sample_file_path} '
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            else:
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            handle_pbsim_output(idx, chrN, chr_raw_path)


# 2. Generate the graphs
def generate_graphs_hifi(outdir_path, chr_dict, assembler, threads):
    print(f'SETUP::generate')

    # if 'raven' not in os.listdir('vendor'):
    #     print(f'SETUP::generate:: Download Raven')
    #     subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
    #     subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
    #     subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    outdir_path = os.path.abspath(outdir_path)
    for chrN_flag, n_need in chr_dict.items():
        if chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            chr_sim_path = os.path.join(outdir_path, f'{chrN}')
        else:
            print(f'Give valid suffix')
            raise Exception  # TODO: Implement custom exception

        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, f'{assembler}/processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = max(0, n_raw - n_prc)
        print(f'SETUP::generate:: Generate {n_diff} graphs for {chrN}')
        graph_dataset.AssemblyGraphDataset_HiFi(chr_sim_path, nb_pos_enc=None, assembler=assembler, threads=threads, generate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, help='Where all the generated data is stored')
    parser.add_argument('--chrs', type=str, help='Path to directory with chromosome references')
    parser.add_argument('--asm', type=str, help='Assembler used to construct assembly graphs')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads used for running assemblers')
    args = parser.parse_args()

    chrs_path = args.chrs
    assembler = args.asm
    outdir_path = args.outdir
    threads = args.threads

    paths = get_paths()
    # TODO: What to do with the sample? Maybe a PBSIM3 config, as a special part in the paths.py? Or an actual config.py?
    pbsim3_dir = paths['pbsim3_dir']
    sample_profile_id = paths['sample_profile_id']
    assert len(sample_profile_id) > 0, "You need to specify sample_profile_id!"
    sample_file = paths['sample_file']

    dicts = config.get_config()
    train_dict = dicts['train_dict']
    valid_dict = dicts['valid_dict']
    
    all_chr = merge_dicts(train_dict, valid_dict)
    simulate_reads_hifi(outdir_path, chrs_path, all_chr, assembler, pbsim3_dir, sample_profile_id, sample_file)
    generate_graphs_hifi(outdir_path, all_chr, assembler, threads)
