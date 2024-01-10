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
from hyperparameters import get_hyperparameters
from paths import get_paths


# def change_description(file_path):
#     new_fasta = []
#     for record in SeqIO.parse(file_path, file_path[-5:]):  # 'fasta' for FASTA file, 'fastq' for FASTQ file
#         des = record.description.split(",")
#         id = des[0][5:]
#         if des[1] == "forward":
#             strand = '+'
#         else:
#             strand = '-'
#         position = des[2][9:].split("-")
#         start = position[0]
#         end = position[1]
#         record.id = id
#         record.description = f'strand={strand} start={start} end={end}'
#         new_fasta.append(record)
#     SeqIO.write(new_fasta, file_path, "fasta")


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


# def create_chr_dirs(pth):
#     for i in range(1, 24):
#         if i == 23:
#             i = 'X'
#         subprocess.run(f'mkdir chr{i}', shell=True, cwd=pth)
#         subprocess.run(f'mkdir raw raven hifiasm', shell=True, cwd=os.path.join(pth, f'chr{i}'))
#         subprocess.run(f'mkdir processed info output graphia', shell=True, cwd=os.path.join(pth, f'chr{i}/raven'))
#         subprocess.run(f'mkdir processed info output graphia', shell=True, cwd=os.path.join(pth, f'chr{i}/hifiasm'))


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
    fasta_path = change_description2(fastq_path, maf_path, chr=chrN)  # Extract positional info from the MAF file
    print(f'Removing the MAF file...')
    subprocess.run(f'rm {idx}.maf', shell=True, cwd=chr_raw_path)
    if combo:
        return fasta_path
    else:
        return None


# 1. Simulate the sequences - HiFi
def simulate_reads_hifi(data_path, refs_path, chr_dict, assembler, pbsim3_dir, sample_profile_id, sample_file_path):
    print(f'SETUP::simulate')
    # if 'vendor' not in os.listdir():
    #     os.mkdir('vendor')
    
    # pbsim3_dir = f'/home/vrcekl/pbsim3'

    data_path = os.path.abspath(data_path)
    for chrN_flag, n_need in chr_dict.items():
        if chrN_flag.endswith('_r'):
            continue
        if '+' in chrN_flag:
            continue
        elif chrN_flag.endswith('_hg002'):
            chrN = chrN_flag[:-6]
            chr_path = os.path.join(refs_path, f'HG002', 'hg002_chromosomes')  # TODO: redefine refs path
            pbsim_path = os.path.join(data_path, 'hg002_pbsim3')  # TODO: redefine data path
            chr_seq_path = os.path.join(chr_path, f'{chrN}_MATERNAL.fasta')
            # sample_file_path = f'/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/HG002/20kb/m64011_190830_220126.sub.fastq'  # TODO: Need to provide this as an argument
            # sample_profile_id = f'20kb-m64011_190830_220126'  # TODO: Need to provide this as an argument
            depth = 60
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
                assert os.path.isfile(sample_file_path), "Sample profile ID and sample file not found! Provide either a valid sample profile ID or a sample file."
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample {sample_file_path} '
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            else:
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {depth} --genome {chr_seq_path} ' \
                                f'--sample-profile-id {sample_profile_id} --prefix {chr_raw_path}/{idx}', shell=True, cwd=pbsim3_dir)
            handle_pbsim_output(idx, chrN, chr_raw_path)


# 2. Generate the graphs
def generate_graphs_hifi(data_path, chr_dict, assembler):
    print(f'SETUP::generate')

    # if 'raven' not in os.listdir('vendor'):
    #     print(f'SETUP::generate:: Download Raven')
    #     subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
    #     subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
    #     subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    data_path = os.path.abspath(data_path)

    for chrN_flag, n_need in chr_dict.items():
        if '+' in chrN_flag:
            chrN = chrN_flag  # Called chrN_combo in simulate_reads_combo function
            chr_sim_path = os.path.join(data_path, 'combo', f'{chrN}')
        # elif chrN_flag.endswith('_r'):
        #     chrN = chrN_flag[:-2]
        #     chr_sim_path = os.path.join(data_path, 'real', f'{chrN}')
        # elif chrN_flag.endswith('_chm13'):
        #     chrN = chrN_flag[:-6]
        #     chr_sim_path = os.path.join(data_path, 'chm13_pbsim3', f'{chrN}')
        # elif chrN_flag.endswith('_ncbr'):
        #     chrN = chrN_flag[:-5]
        #     chr_sim_path = os.path.join(data_path, 'ncoibor_pbsim3', f'{chrN}')
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


# 2.5 Train-valid-test split
def train_valid_split(data_path, eval_path, temp_path, assembler, train_dict, valid_dict, test_dict={}, out=None, overfit=False):
    print(f'SETUP::split')
    data_path = os.path.abspath(data_path)
    eval_path = os.path.abspath(eval_path)
    if overfit:
        data_path = eval_path

    # real_path = os.path.join(eval_path, 'real')  # DEPRECATED
    # pbsim_path = os.path.join(data_path, 'pbsim3')  # DEPRECATED
    # ncoibor_path = os.path.join(data_path, 'ncoibor_pbsim3')
    hg002_path = os.path.join(data_path, 'hg002_pbsim3')
    combo_path = os.path.join(data_path, 'combo')
    # chm13_path = os.path.join(data_path, 'chm13_pbsim3')
    # arab_path = os.path.join(data_path, 'arabidopsis_pbsim3')
    # zmays_path = os.path.join(data_path, 'zmays_Mo17_pbsim3')

    if out is None:
        train_path = os.path.join(temp_path, f'train', assembler)
        valid_path = os.path.join(temp_path, f'valid', assembler)
        # test_path  = os.path.join(temp_path, f'test', assembler)
    else:
        train_path = os.path.join(temp_path, f'train_{out}', assembler)
        valid_path = os.path.join(temp_path, f'valid_{out}', assembler)
        # test_path  = os.path.join(temp_path, f'test_{out}', assembler)

    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=train_path)
    if not os.path.isdir(valid_path):
        os.makedirs(valid_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=valid_path)
    # if not os.path.isdir(test_path) and len(test_dict) > 0:
    #     os.makedirs(test_path)
    #     subprocess.run(f'mkdir processed info', shell=True, cwd=test_path)
 
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
                # elif chrN_flag.endswith('_r'):  # DEPRECATED
                #     chrN = chrN_flag[:-2]
                #     chr_sim_path = os.path.join(real_path, 'chm13_chromosomes', chrN, assembler)
                # elif chrN_flag.endswith('_pbs'):  # DEPRECATED
                #     chrN = chrN_flag[:-4]
                #     chr_sim_path = os.path.join(pbsim_path, chrN, assembler)
                # elif chrN_flag.endswith('_ncbr'):
                #     chrN = chrN_flag[:-5]
                #     chr_sim_path = os.path.join(ncoibor_path, chrN, assembler)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
                # elif chrN_flag.endswith('_chm13'):
                #     chrN = chrN_flag[:-6]
                #     chr_sim_path = os.path.join(chm13_path, chrN, assembler)
                # elif chrN_flag.endswith('_arab'):
                #     chrN = chrN_flag[:-5]
                #     chr_sim_path = os.path.join(arab_path, chrN, assembler)
                # elif chrN_flag.endswith('_zmays'):
                #     chrN = chrN_flag[:-6]
                #     chr_sim_path = os.path.join(zmays_path, chrN, assembler)
                else:
                    print(f'Give proper suffix!')
                    raise Exception

                train_g_to_chr[n_have] = chrN
                print(f'Copying {chr_sim_path}/processed/{i}.dgl into {train_path}/processed/{n_have}.dgl')
                subprocess.run(f'cp {chr_sim_path}/processed/{i}.dgl {train_path}/processed/{n_have}.dgl', shell=True)
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
                # elif chrN_flag.endswith('_r'):  # DEPRECATED
                #     chrN = chrN_flag[:-2]
                #     chr_sim_path = os.path.join(real_path, 'chm13_chromosomes', chrN, assembler)
                #     j = 0
                # elif chrN_flag.endswith('_pbs'):  # DEPRECATED
                #     chrN = chrN_flag[:-4]
                #     chr_sim_path = os.path.join(pbsim_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                # elif chrN_flag.endswith('_ncbr'):
                #     chrN = chrN_flag[:-5]
                #     chr_sim_path = os.path.join(ncoibor_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                # elif chrN_flag.endswith('_chm13'):
                #     chrN = chrN_flag[:-6]
                #     chr_sim_path = os.path.join(chm13_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                # elif chrN_flag.endswith('_arab'):
                #     chrN = chrN_flag[:-5]
                #     chr_sim_path = os.path.join(arab_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                # elif chrN_flag.endswith('_zmays'):
                #     chrN = chrN_flag[:-6]
                #     chr_sim_path = os.path.join(zmays_path, chrN, assembler)
                #     j = i + train_dict.get(chrN_flag, 0)
                else:
                    print(f'Give proper suffix!')
                    raise Exception

                valid_g_to_chr[n_have] = chrN
                print(f'Copying {chr_sim_path}/processed/{j}.dgl into {valid_path}/processed/{n_have}.dgl')
                subprocess.run(f'cp {chr_sim_path}/processed/{j}.dgl {valid_path}/processed/{n_have}.dgl', shell=True)
                valid_g_to_org_g[n_have] = j
                n_have += 1
    pickle.dump(valid_g_to_chr, open(f'{valid_path}/info/g_to_chr.pkl', 'wb'))
    pickle.dump(valid_g_to_org_g, open(f'{valid_path}/info/g_to_org_g.pkl', 'wb'))

    # TODO: FIX THIS !!!!!!!!!!!!!!!!!!
    train_path = os.path.join(train_path, os.path.pardir)
    valid_path = os.path.join(valid_path, os.path.pardir)
    # test_path = os.path.join(test_path, os.path.pardir)
    ###################################

    return train_path, valid_path


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
    
    parser.add_argument('--data', type=str, default=None, help='Where all the generated data is stored')
    parser.add_argument('--temp', type=str, default=None, help='Where all the training data is temporarily transfered')
    parser.add_argument('--refs', type=str, default=None, help='Path to the reference genome for simulating the reads')
    # parser.add_argument('--eval', type=str, default=None, help='Where all the generated data is stored')
    
    args = parser.parse_args()

    out = args.out
    overfit = args.overfit
    data_path = args.data
    temp_path = args.temp
    refs_path = args.refs
    # eval_path = args.eval

    hyperparams = get_hyperparameters()
    paths = get_paths()
    
    if not data_path:
        data_path = paths['data_path']  # Location of the master database (storage)
    if not temp_path:
        temp_path = paths['temp_path']  # Location where the data will be temporarily stored for training
    if not refs_path:
        refs_path = paths['refs_path']  # Location where the references are stored - local because everythin else can be generated from this
    # if not eval_path:
    #     eval_path = paths['eval_path']  # Location where the synth and real evaluation data is stored
    
    # asms_path = hyperparams['asms_path']  # Where the assemblies and other inference info will be stored
    assembler = hyperparams['assembler']  # Which assembler we are using, currently: raven/hifiasm
    threads = hyperparams['num_threads']

    pbsim3_dir = paths['pbsim3_dir']
    sample_profile_id = paths['sample_profile_id']
    assert len(sample_profile_id) > 0, "You need to specify sample_profile_id!"
    sample_file = paths['sample_file']

    dicts = config.get_config()
    train_dict = dicts['train_dict']
    valid_dict = dicts['valid_dict']
    test_dict = dicts['test_dict']

    # specs = {
    #     'threads': threads,
    #     'filter': 0.99,
    #     'out': 'assembly.fasta',
    #     'assembler': assembler,
    # }

    # torch.set_num_threads(threads)
    
    all_chr = merge_dicts(train_dict, valid_dict, test_dict)
    simulate_reads_hifi(data_path, refs_path, all_chr, assembler, pbsim3_dir, sample_profile_id, sample_file)
    
    
    generate_graphs_hifi(data_path, all_chr, assembler)


    # initials = hyperparams['initials']
    # time_start = datetime.now()
    # if out is None:
    #     timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    #     out = f'{timestamp}_{initials}'
    # else:
    #     timestamp = time_start.strftime('%y-%m-%d')
    #     out = f'{timestamp}_{initials}_{out}'
    # train_path, valid_path = train_valid_split(data_path, eval_path, temp_path, assembler, train_dict, valid_dict, test_dict, out, overfit=False)