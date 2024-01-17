import argparse
import os
import pickle
import subprocess
import time
from datetime import datetime

from tqdm import tqdm

import train_valid_chrs


def train_valid_split(data_path, savedir, assembler, train_dict, valid_dict, name, overfit=False):
    print(f'SETUP::split')
    data_path = os.path.abspath(data_path)

    # hg002_path = os.path.join(data_path, 'hg002_pbsim3')
    hg002_path = data_path
    combo_path = os.path.join(data_path, 'combo')

    train_dir = os.path.join(savedir, f'train_{name}')
    valid_dir = os.path.join(savedir, f'valid_{name}')
    train_path = os.path.join(savedir, f'train_{name}', assembler)
    valid_path = os.path.join(savedir, f'valid_{name}', assembler)

    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=train_path)
    if not os.path.isdir(valid_path):
        os.makedirs(valid_path)
        subprocess.run(f'mkdir processed info', shell=True, cwd=valid_path)

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
            if '_r' in chrN_flag and n_need > 1:  # DEPRECATED
                print(f'SETUP::split::WARNING Cannot copy more than one graph for real data: {chrN_flag}')
                n_need = 1
            print(f'SETUP::split:: Copying {n_need} graphs of {chrN_flag} - {assembler} into {train_path}')
            for i in range(n_need):
                if '+' in chrN_flag:
                    chrN = chrN_flag
                    chr_sim_path = os.path.join(combo_path, chrN, assembler)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
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
            if '_r' in chrN_flag and n_need > 1:  # DEPRECATED
                print(f'SETUP::split::WARNING Cannot copy more than one graph for real data: {chrN_flag}')
                n_need = 1
            print(f'SETUP::split:: Copying {n_need} graphs of {chrN_flag} - {assembler} into {valid_path}')
            for i in range(n_need):
                if '+' in chrN_flag:
                    chrN = chrN_flag
                    chr_sim_path = os.path.join(combo_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
                elif chrN_flag.endswith('_hg002'):
                    chrN = chrN_flag[:-6]
                    chr_sim_path = os.path.join(hg002_path, chrN, assembler)
                    j = i + train_dict.get(chrN_flag, 0)
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
    return train_dir, valid_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=None, help='Where all the generated data is stored')
    parser.add_argument('--savedir', type=str, default=None, help='Where the train/valid datasets will be saved for training')
    parser.add_argument('--name', type=str, default=None, help='Output name for the train/valid datasets')
    parser.add_argument('--asm', type=str, help='Assembler used')
    
    args = parser.parse_args()
    savedir = args.savedir
    name = args.name
    assembler = args.asm
    data_path = args.datadir

    train_dict, valid_dict = train_valid_chrs.get_train_valid_chrs()
    
    time_start = datetime.now()
    if name is None:
        timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
        name = f'{timestamp}'
    else:
        timestamp = time_start.strftime('%y-%m-%d')
        name = f'{timestamp}_{name}'

    train_path, valid_path = train_valid_split(data_path, savedir, assembler, train_dict, valid_dict, name, overfit=False)
    print(f'\nTraining data saved in:   {train_path}')
    print(f'Validation data saved in: {valid_path}\n')
