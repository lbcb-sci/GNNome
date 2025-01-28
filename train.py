import argparse
from datetime import datetime
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.profiler import profile, record_function, ProfilerActivity
import dgl
import wandb

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
from config import get_config
import models
import utils
from inference import inference


def save_checkpoint(epoch, model, optimizer, loss_train, loss_valid, out, ckpt_path):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_valid': loss_valid,
    }
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(out, model, optimizer):
    ckpt_path = f'checkpoints/{out}.pt'
    checkpoint = torch.load(ckpt_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    loss_train = checkpoint['loss_train']
    loss_valid = checkpoint['loss_valid']
    return epoch, model, optimizer, loss_train, loss_valid


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


def mask_graph(g, fraction, device):
    keep_node_idx = torch.rand(g.num_nodes(), device=device) < fraction
    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    return sub_g


def mask_graph_strandwise(g, fraction, device):
    keep_node_idx_half = torch.rand(g.num_nodes() // 2, device=device) < fraction
    keep_node_idx = torch.empty(keep_node_idx_half.size(0) * 2, dtype=keep_node_idx_half.dtype)
    keep_node_idx[0::2] = keep_node_idx_half
    keep_node_idx[1::2] = keep_node_idx_half
    sub_g = dgl.node_subgraph(g, keep_node_idx, store_ids=True)
    print(f'Masking fraction: {fraction}')
    print(f'Original graph: N={g.num_nodes()}, E={g.num_edges()}')
    print(f'Subsampled graph: N={sub_g.num_nodes()}, E={sub_g.num_edges()}')
    return sub_g


def symmetry_loss(org_scores, rev_scores, labels, pos_weight=1.0, alpha=1.0):
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    BCE_org = BCE(org_scores, labels)
    BCE_rev = BCE(rev_scores, labels)
    abs_diff = torch.abs(org_scores - rev_scores)
    loss = (BCE_org + BCE_rev + alpha * abs_diff)
    loss = loss.mean()
    return loss


def get_full_ne_features(g, reverse=False):
    pe_in = g.ndata['in_deg'].unsqueeze(1)
    pe_in = (pe_in - pe_in.mean()) / pe_in.std()
    pe_out = g.ndata['out_deg'].unsqueeze(1)
    pe_out = (pe_out - pe_out.mean()) / pe_out.std()
    if reverse:
        x = torch.cat((pe_out, pe_in), dim=1)  # Reversed edges, in/out-deg also reversed
    else:
        x = torch.cat((pe_in, pe_out), dim=1)
    e = g.edata['e']
    return x, e


def get_partition_ne_features(sub_g, g, reverse=False):
    pe_in = g.ndata['in_deg'][sub_g.ndata['_ID']].unsqueeze(1)
    pe_in = (pe_in - pe_in.mean()) / pe_in.std()
    pe_out = g.ndata['out_deg'][sub_g.ndata['_ID']].unsqueeze(1)
    pe_out = (pe_out - pe_out.mean()) / pe_out.std()
    if reverse:
        x = torch.cat((pe_out, pe_in), dim=1)  # Reversed edges, in/out-deg also reversed
    else:
        x = torch.cat((pe_in, pe_out), dim=1)
    e = g.edata['e'][sub_g.edata['_ID']]
    return x, e


def get_bce_loss_full(g, model, pos_weight, device):
    x, e = get_full_ne_features(g, reverse=False)
    x, e = x.to(device), e.to(device)
    logits = model(g, x, e)
    logits = logits.squeeze(-1)
    edge_labels = g.edata['y'].to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(logits, edge_labels)
    return loss, logits


def get_bce_loss_partition(sub_g, g, model, pos_weight, device):
    sub_g = sub_g.to(device)
    x, e = get_partition_ne_features(sub_g, g, reverse=False)
    x, e = x.to(device), e.to(device)
    logits = model(sub_g, x, e) 
    logits = logits.squeeze(-1)
    edge_labels = g.edata['y'][sub_g.edata['_ID']].to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(logits, edge_labels)
    return loss, logits  # TODO: This should only return logits, loss compute outside


def get_symmetry_loss_full(g, model, pos_weight, alpha, device):
    x, e = get_full_ne_features(g, reverse=False)
    x, e = x.to(device), e.to(device)
    logits_org = model(g, x, e).squeeze(-1)
    edge_labels = g.edata['y'].to(device)
    
    g = dgl.reverse(g, True, True)
    x, e = get_full_ne_features(g, reverse=True)
    x, e = x.to(device), e.to(device)
    logits_rev = model(g, x, e).squeeze(-1)
    loss = symmetry_loss(logits_org, logits_rev, edge_labels, pos_weight, alpha=alpha)
    return loss, logits_org


def get_symmetry_loss_partition(sub_g, g, model, pos_weight, alpha, device):
    sub_g = sub_g.to(device)
    x, e = get_partition_ne_features(sub_g, g, False)
    x, e = x.to(device), e.to(device)
    logits_org = model(sub_g, x, e).squeeze(-1)
    labels = g.edata['y'][sub_g.edata['_ID']].to(device)
    
    sub_g = dgl.reverse(sub_g, True, True)
    x, e = get_partition_ne_features(sub_g, g, True)
    x, e = x.to(device), e.to(device)
    logits_rev = model(sub_g, x, e).squeeze(-1)
    loss = symmetry_loss(logits_org, logits_rev, labels, pos_weight, alpha=alpha)
    return loss, logits_org


def train(train_path, valid_path, out, assembler, overfit=False, dropout=None, seed=None, resume=False, finetune=False, ft_model=None, gpu=None):
    hyperparameters = get_hyperparameters()
    if seed is None:
        seed = hyperparameters['seed']
    num_epochs = hyperparameters['num_epochs']
    num_gnn_layers = hyperparameters['num_gnn_layers']
    hidden_features = hyperparameters['dim_latent']
    nb_pos_enc = hyperparameters['nb_pos_enc']
    patience = hyperparameters['patience']
    lr = hyperparameters['lr']
    device = hyperparameters['device']
    normalization = hyperparameters['normalization']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    hidden_edge_features = hyperparameters['hidden_edge_features']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']
    decay = hyperparameters['decay']
    wandb_mode = hyperparameters['wandb_mode']
    wandb_project = hyperparameters['wandb_project']
    num_nodes_per_cluster = hyperparameters['num_nodes_per_cluster']
    npc_lower_bound = hyperparameters['npc_lower_bound']
    npc_upper_bound = hyperparameters['npc_upper_bound']
    k_extra_hops = hyperparameters['k_extra_hops']
    masking = hyperparameters['masking']
    mask_frac_low = hyperparameters['mask_frac_low']
    mask_frac_high = hyperparameters['mask_frac_high']
    use_symmetry_loss = hyperparameters['use_symmetry_loss']
    alpha = hyperparameters['alpha']

    config = get_config()
    checkpoints_path = os.path.abspath(config['checkpoints_path'])
    models_path = os.path.abspath(config['models_path'])
    
    if gpu:
        # GPU as an option to the train.py script
        # Otherwise, take the device from hyperparameters.py
        device = f'cuda:{gpu}'
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    utils.set_seed(seed)
    
    time_start = datetime.now()
    timestamp = time_start.strftime('%Y-%b-%d-%H-%M-%S')
    
    if out is None:
        out = timestamp
    assert train_path is not None, "train_path not specified!"
    assert valid_path is not None, "valid_path not specified!"

    if not overfit:
        ds_train = AssemblyGraphDataset(train_path, assembler=assembler)
        ds_valid = AssemblyGraphDataset(valid_path, assembler=assembler)
    else:
        ds_train = ds_valid = AssemblyGraphDataset(train_path, assembler=assembler)

    pos_to_neg_ratio = sum([((torch.round(g.edata['y'])==1).sum() / (torch.round(g.edata['y'])==0).sum()).item() for _, g in ds_train]) / len(ds_train)

    model = models.SymGatedGCNModel(node_features, edge_features, hidden_features, hidden_edge_features, num_gnn_layers, hidden_edge_scores, normalization, nb_pos_enc, dropout=dropout)
    model.to(device)
    if not os.path.exists(models_path):
        print(models_path)
        os.makedirs(models_path)

    out = out + f'_seed{seed}'
    model_path = os.path.join(models_path, f'model_{out}.pt')    
    ckpt_path = f'{checkpoints_path}/ckpt_{out}.pt'

    pos_weight = torch.tensor([1 / pos_to_neg_ratio], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # TODO: Is this needed?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=patience, verbose=True)
    start_epoch = 0

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_inv_per_epoch_valid = []

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    print(f'----- TRAIN CONFIGURAION SUMMARY -----')
    print(f'Using device: {device}')
    print(f'Using seed: {seed}')
    print(f'Model path: {model_path}')
    print(f'Checkpoint path: {ckpt_path}')
    print(f'Number of network parameters: {view_model_param(model)}')
    print(f'Normalization type : {normalization}')
    print(f'--------------------------------------\n')
    
    if resume:
        # ckpt_path = f'{checkpoints_path}/ckpt_{out}.pt'  # This should be the checkpoint of the old run
        checkpoint = torch.load(ckpt_path)
        print('Loding the checkpoint from:', ckpt_path, sep='\t')
        model_path = os.path.join(models_path, f'model_{out}_resumed-{num_epochs}.pt')
        ckpt_path  = os.path.join(checkpoints_path, f'ckpt_{out}_resumed-{num_epochs}.pt')
        print('Saving the resumed model to:', model_path, sep='\t')
        print('Saving the new checkpoint to:', ckpt_path, sep='\t')
        
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch: {start_epoch}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        
        min_loss_train = checkpoint['loss_train']
        min_loss_valid = checkpoint['loss_valid']
        loss_per_epoch_train.append(min_loss_train)
        loss_per_epoch_valid.append(min_loss_valid)
        
    if finetune:
        # ckpt_path = f'{checkpoints_path}/ckpt_{out}.pt'  # This should be the checkpoint of the old run
        # checkpoint = torch.load(ckpt_path)
        # print('Loding the checkpoint from:', ckpt_path, sep='\t')
        model_path = os.path.join(models_path, f'finetune-model_{out}.pt')
        ckpt_path  = os.path.join(checkpoints_path, f'finetune-ckpt_{out}.pt')
        print('Saving the resumed model to:', model_path, sep='\t')
        print('Saving the new checkpoint to:', ckpt_path, sep='\t')
        
        start_epoch = 0
        ft_model = torch.load(ft_model)
        model.load_state_dict(ft_model)
        # optimizer.load_state_dict(checkpoint['optim_state_dict'])
        
    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
    print(f'Loading data done. Elapsed time: {elapsed}')

    try:
        with wandb.init(project=wandb_project, config=hyperparameters, mode=wandb_mode, name=out):
            wandb.watch(model, criterion, log='all', log_freq=1000)

            for epoch in range(start_epoch, num_epochs):
                
                train_loss_epoch, train_fp_rate_epoch, train_fn_rate_epoch = [], [], []
                train_acc_epoch, train_precision_epoch, train_recall_epoch, train_f1_epoch = [], [], [], []
                train_acc_inv_epoch, train_precision_inv_epoch, train_recall_inv_epoch, train_f1_inv_epoch = [], [], [], []

                print('\n===> TRAINING\n')
                random.shuffle(ds_train.graph_list)
                for data in ds_train:
                    model.train()
                    idx, g = data

                    print(f'\n(TRAIN: Epoch = {epoch:3}) NEW GRAPH: index = {idx}')

                    if masking:
                        fraction = random.randint(mask_frac_low, mask_frac_high) / 100  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
                        g = mask_graph_strandwise(g, fraction, device)

                    # Number of clusters dependant on graph size!
                    num_nodes_per_cluster_min = int(num_nodes_per_cluster * npc_lower_bound)
                    num_nodes_per_cluster_max = int(num_nodes_per_cluster * npc_upper_bound) + 1
                    num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min, num_nodes_per_cluster_max).item()
                    num_clusters = g.num_nodes() // num_nodes_for_g + 1

                    if num_nodes_for_g >= g.num_nodes(): # train with full graph
                        print(f'\nUse METIS: False')
                        print(f'Use full graph')
                        g = g.to(device)

                        if use_symmetry_loss:
                            loss, logits = get_symmetry_loss_full(g, model, pos_weight, alpha, device)
                        else:
                            loss, logits = get_bce_loss_full(g, model, pos_weight, device)

                        labels = g.edata['y'].to(device)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        TP, TN, FP, FN = utils.calculate_tfpn(logits, labels)
                        acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                        try:
                            fp_rate = FP / (FP + TN)
                        except ZeroDivisionError:
                            fp_rate = 0.0
                        try:
                            fn_rate = FN / (FN + TP)
                        except ZeroDivisionError:
                            fn_rate = 0.0
                        
                        train_loss_epoch.append(loss.item())
                        train_fp_rate_epoch.append(fp_rate)
                        train_fn_rate_epoch.append(fn_rate)

                        # elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                        # print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}')
                        # print(f'Loss: {train_loss:.4f}, fp_rate(GT=0): {train_fp_rate:.4f}, fn_rate(GT=1): {train_fn_rate:.4f}')
                        # print(f'elapsed time: {elapsed}\n\n')

                    else: # train with mini-batch
                        print(f'\nUse METIS: True')
                        print(f'Number of clusters:', num_clusters)
                        g = g.long()
                        d = dgl.metis_partition(g, num_clusters, extra_cached_hops=k_extra_hops)
                        sub_gs = list(d.values())
                        random.shuffle(sub_gs)

                        for sub_g in sub_gs:
                            if use_symmetry_loss:
                                loss, logits = get_symmetry_loss_partition(sub_g, g, model, pos_weight, alpha, device)
                            else:
                                loss, logits = get_bce_loss_partition(sub_g, g, model, pos_weight, device)

                            labels = g.edata['y'][sub_g.edata['_ID']].to(device)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # TODO: How to handle edge_predictions?
                            TP, TN, FP, FN = utils.calculate_tfpn(logits, labels)
                            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                            acc_inv, precision_inv, recall_inv, f1_inv =  utils.calculate_metrics_inverse(TP, TN, FP, FN)
                            
                            try:
                                fp_rate = FP / (FP + TN)
                            except ZeroDivisionError:
                                fp_rate = 0.0
                            try:
                                fn_rate = FN / (FN + TP)
                            except ZeroDivisionError:
                                fn_rate = 0.0
                            
                            # These are used for epoch mean = mean over all the partitions in all the graphs
                            train_loss_epoch.append(loss.item())
                            train_fp_rate_epoch.append(fp_rate)
                            train_fn_rate_epoch.append(fn_rate)
                            train_acc_epoch.append(acc)
                            train_precision_epoch.append(precision)
                            train_recall_epoch.append(recall)
                            train_f1_epoch.append(f1)
                            
                            # Inverse metrics because F1 and them are not good for dataset with mostly positive labels
                            train_acc_inv_epoch.append(acc_inv)
                            train_precision_inv_epoch.append(precision_inv)
                            train_recall_inv_epoch.append(recall_inv)
                            train_f1_inv_epoch.append(f1_inv)

                        exit(0)
                
                # Average over all the partitions in one epoch
                train_loss_epoch = np.mean(train_loss_epoch)
                train_fp_rate_epoch = np.mean(train_fp_rate_epoch)
                train_fn_rate_epoch = np.mean(train_fn_rate_epoch)
                train_acc_epoch = np.mean(train_acc_epoch)
                train_precision_epoch = np.mean(train_precision_epoch)
                train_recall_epoch = np.mean(train_recall_epoch)
                train_f1_epoch = np.mean(train_f1_epoch)
                
                train_acc_inv_epoch = np.mean(train_acc_inv_epoch)
                train_precision_inv_epoch = np.mean(train_precision_inv_epoch)
                train_recall_inv_epoch = np.mean(train_recall_inv_epoch)
                train_f1_inv_epoch = np.mean(train_f1_inv_epoch)

                loss_per_epoch_train.append(train_loss_epoch)
                lr_value = optimizer.param_groups[0]['lr']
                
                elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                print(f'\n==> TRAINING (all training graphs): Epoch = {epoch}')
                print(f'Loss: {train_loss_epoch:.4f}, fp_rate(GT=0): {train_fp_rate_epoch:.4f}, fn_rate(GT=1): {train_fn_rate_epoch:.4f}')
                print(f'Elapsed time: {elapsed}\n\n')

                if overfit:
                    if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_train) > 1 and loss_per_epoch_train[-1] < min(loss_per_epoch_train[:-1]):
                        torch.save(model.state_dict(), model_path)
                        print(f'Epoch {epoch}: Model saved!')
                    save_checkpoint(epoch, model, optimizer, loss_per_epoch_train[-1], 0.0, out, ckpt_path)
                    scheduler.step(train_loss_epoch)
                    wandb.log({'train_loss': train_loss_epoch, 'train_accuracy': train_acc_epoch, \
                               'train_precision': train_precision_epoch, 'lr_value': lr_value, \
                               'train_recall': train_recall_epoch, 'train_f1': train_f1_epoch, \
                               'train_fp-rate': train_fp_rate_epoch, 'train_fn-rate': train_fn_rate_epoch})

                    continue  # This will entirely skip the validation
                
                valid_loss_epoch, valid_fp_rate_epoch, valid_fn_rate_epoch = [], [], []
                valid_acc_epoch, valid_precision_epoch, valid_recall_epoch, valid_f1_epoch = [], [], [], []
                valid_acc_inv_epoch, valid_precision_inv_epoch, valid_recall_inv_epoch, valid_f1_inv_epoch = [], [], [], []

                with torch.no_grad():
                    print('\n===> VALIDATION\n')
                    # time_start_eval = datetime.now()
                    model.eval()
                    for data in ds_valid:
                        idx, g = data
                        
                        print(f'\n(VALID Epoch = {epoch:3}) NEW GRAPH: index = {idx}')
                        
                        if masking:
                            fraction = random.randint(mask_frac_low, mask_frac_high) / 100  # Fraction of nodes to be left in the graph (.85 -> ~30x, 1.0 -> 60x)
                            g = mask_graph_strandwise(g, fraction, device)
                        
                        # Number of clusters dependant on graph size!
                        num_nodes_per_cluster_min = int(num_nodes_per_cluster * npc_lower_bound)
                        num_nodes_per_cluster_max = int(num_nodes_per_cluster * npc_upper_bound) + 1
                        num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min, num_nodes_per_cluster_max).item() # DEBUG!!!
                        num_clusters = g.num_nodes() // num_nodes_for_g + 1
                        
                        if num_nodes_for_g >= g.num_nodes(): # full graph
                            print(f'\nUse METIS: False')
                            print(f'Use full graph')
                            g = g.to(device)

                            if use_symmetry_loss:
                                loss, logits = get_symmetry_loss_full(g, model, pos_weight, alpha, device)
                            else:
                                loss, logits = get_bce_loss_full(g, model, pos_weight, device)

                            labels = g.edata['y'].to(device)
                            TP, TN, FP, FN = utils.calculate_tfpn(logits, labels)
                            acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                            try:
                                fp_rate = FP / (FP + TN)
                            except ZeroDivisionError:
                                fp_rate = 0.0
                            try:
                                fn_rate = FN / (FN + TP)
                            except ZeroDivisionError:
                                fn_rate = 0.0
                            
                            valid_loss_epoch.append(loss.item())
                            valid_fp_rate_epoch.append(fp_rate)
                            valid_fn_rate_epoch.append(fn_rate)

                            # elapsed = utils.timedelta_to_str(datetime.now() - time_start_eval)
                            # print(f'\nVALIDATION (one validation graph): Epoch = {epoch}, Graph = {idx}')
                            # print(f'Loss: {val_loss:.4f}, fp_rate(GT=0): {val_fp_rate:.4f}, fn_rate(GT=1): {val_fn_rate:.4f}')
                            # print(f'elapsed time: {elapsed}\n\n')

                        else: # mini-batch
                            print(f'\nNum clusters:', num_clusters)
                            g = g.long()
                            d = dgl.metis_partition(g, num_clusters, extra_cached_hops=k_extra_hops)
                            sub_gs = list(d.values())
                            # g = g.to(device)
                            
                            for sub_g in sub_gs:
                                
                                if use_symmetry_loss:
                                    loss, logits = get_symmetry_loss_partition(sub_g, g, model, pos_weight, alpha, device)
                                else:
                                    loss, logits = get_bce_loss_partition(sub_g, g, model, pos_weight, device)

                                labels = g.edata['y'][sub_g.edata['_ID']].to(device)
                                TP, TN, FP, FN = utils.calculate_tfpn(logits, labels)
                                acc, precision, recall, f1 =  utils.calculate_metrics(TP, TN, FP, FN)
                                acc_inv, precision_inv, recall_inv, f1_inv =  utils.calculate_metrics_inverse(TP, TN, FP, FN)
                                
                                try:
                                    fp_rate = FP / (FP + TN)
                                except ZeroDivisionError:
                                    fp_rate = 0.0
                                try:
                                    fn_rate = FN / (FN + TP)
                                except ZeroDivisionError:
                                    fn_rate = 0.0
                                
                                # These are used for epoch mean = mean over all the partitions in all the graphs
                                valid_loss_epoch.append(loss.item())
                                valid_fp_rate_epoch.append(fp_rate)
                                valid_fn_rate_epoch.append(fn_rate)
                                valid_acc_epoch.append(acc)
                                valid_precision_epoch.append(precision)
                                valid_recall_epoch.append(recall)
                                valid_f1_epoch.append(f1)
                                
                                # Inverse metrics because F1 and them are not good for dataset with mostly positive labels
                                valid_acc_inv_epoch.append(acc_inv)
                                valid_precision_inv_epoch.append(precision_inv)
                                valid_recall_inv_epoch.append(recall_inv)
                                valid_f1_inv_epoch.append(f1_inv)
                    
                    # Average over all the partitions in one epoch
                    valid_loss_epoch = np.mean(valid_loss_epoch)
                    valid_fp_rate_epoch = np.mean(valid_fp_rate_epoch)
                    valid_fn_rate_epoch = np.mean(valid_fn_rate_epoch)
                    valid_acc_epoch = np.mean(valid_acc_epoch)
                    valid_precision_epoch = np.mean(valid_precision_epoch)
                    valid_recall_epoch = np.mean(valid_recall_epoch)
                    valid_f1_epoch = np.mean(valid_f1_epoch)

                    valid_acc_inv_epoch = np.mean(valid_acc_inv_epoch)
                    valid_precision_inv_epoch = np.mean(valid_precision_inv_epoch)
                    valid_recall_inv_epoch = np.mean(valid_recall_inv_epoch)
                    valid_f1_inv_epoch = np.mean(valid_f1_inv_epoch)

                    loss_per_epoch_valid.append(valid_loss_epoch)
                    f1_inv_per_epoch_valid.append(valid_f1_inv_epoch)

                    elapsed = utils.timedelta_to_str(datetime.now() - time_start)
                    print(f'\n==> VALIDATION (all validation graphs): Epoch = {epoch}')
                    print(f'Loss: {valid_loss_epoch:.4f}, fp_rate(GT=0): {valid_fp_rate_epoch:.4f}, fn_rate(GT=1): {valid_fn_rate_epoch:.4f}')
                    print(f'Elapsed time total: {elapsed}\n\n')

                    if not overfit:
                        if finetune:
                            if (epoch+1) % 50 == 0:
                                model_tmp_path = os.path.join(models_path, f'finetune-model-epoch{epoch}_{out}.pt')
                                torch.save(model.state_dict(), model_tmp_path)
                            # Choose the model with minimal loss on validation set
                            if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                                torch.save(model.state_dict(), model_path)
                                print(f'Epoch {epoch:3}: Model MIN-LOSS saved! -> Val Loss = {valid_loss_epoch:.6f}\tVal F1 = {valid_f1_epoch:.4f}\tVal inv-F1 = {valid_f1_inv_epoch:.4f}' \
                                    f'\tVal FPR = {valid_fp_rate_epoch:.4f}\tVal FNR = {valid_fn_rate_epoch:.4f}\t')
                            save_checkpoint(epoch, model, optimizer, min(loss_per_epoch_train), min(loss_per_epoch_valid), out, ckpt_path)  # Save the checkpoint every epoch
                            scheduler.step(valid_loss_epoch)
                        else:
                            # Choose the model with minimal loss on validation set
                            if len(loss_per_epoch_valid) == 1 or len(loss_per_epoch_valid) > 1 and loss_per_epoch_valid[-1] < min(loss_per_epoch_valid[:-1]):
                                torch.save(model.state_dict(), model_path)
                                print(f'Epoch {epoch:3}: Model MIN-LOSS saved! -> Val Loss = {valid_loss_epoch:.6f}\tVal F1 = {valid_f1_epoch:.4f}\tVal inv-F1 = {valid_f1_inv_epoch:.4f}' \
                                    f'\tVal FPR = {valid_fp_rate_epoch:.4f}\tVal FNR = {valid_fn_rate_epoch:.4f}\t')
                            save_checkpoint(epoch, model, optimizer, min(loss_per_epoch_train), min(loss_per_epoch_valid), out, ckpt_path)  # Save the checkpoint every epoch
                            scheduler.step(valid_loss_epoch)

                    # Code that evalates NGA50 during training -- only for overfitting
                    # plot_nga50_during_training = hyperparameters['plot_nga50_during_training']
                    # i = hyperparameters['chr_overfit']
                    # eval_frequency = hyperparameters['eval_frequency']
                    # if overfit and plot_nga50_during_training and (epoch+1) % eval_frequency == 0:
                    #     # call inference
                    #     refs_path = hyperparameters['refs_path']
                    #     save_dir = os.path.join(train_path, assembler)
                    #     if not os.path.isdir(save_dir):
                    #         os.makedirs(save_dir)
                    #     if not os.path.isdir(os.path.join(save_dir, f'assembly')):
                    #         os.mkdir(os.path.join(save_dir, f'assembly'))
                    #     if not os.path.isdir(os.path.join(save_dir, f'inference')):
                    #         os.mkdir(os.path.join(save_dir, f'inference'))
                    #     if not os.path.isdir(os.path.join(save_dir, f'reports')):
                    #         os.mkdir(os.path.join(save_dir, f'reports'))
                    #     inference(train_path, model_path, assembler, save_dir)
                    #     # call evaluate
                    #     ref = os.path.join(refs_path, 'chromosomes', f'chr{i}.fasta')
                    #     idx = os.path.join(refs_path, 'indexed', f'chr{i}.fasta.fai')
                    #     asm = os.path.join(save_dir, f'assembly', f'0_assembly.fasta')
                    #     report = os.path.join(save_dir, f'reports', '0_minigraph.txt')
                    #     paf = os.path.join(save_dir, f'asm.paf')
                    #     p = evaluate.run_minigraph(ref, asm, paf)
                    #     p.wait()
                    #     p = evaluate.parse_pafs(idx, report, paf)
                    #     p.wait()
                    #     with open(report) as f:
                    #         text = f.read()
                    #         ng50 = int(re.findall(r'NG50\s*(\d+)', text)[0])
                    #         nga50 = int(re.findall(r'NGA50\s*(\d+)', text)[0])
                    #         print(f'NG50: {ng50}\tNGA50: {nga50}')

                    try:
                        if 'nga50' in locals():
                            pass
                            # wandb.log({'train_loss': train_loss_epoch, 'val_loss': valid_loss_epoch, 'lr_value': lr_value, \
                            #            'train_loss': train_loss_epoch, 'train_fpr': train_fp_rate_epoch, 'train_fnr': train_fn_rate_epoch, \
                            #            'valid_loss': valid_loss_epoch, 'valid_fpr': valid_fp_rate_epoch, 'valid_fnr': valid_fn_rate_epoch, \
                            #            'train_acc': train_acc_epoch, 'train_precision': train_precision_epoch, 'train_recall': train_recall_epoch, 'train_f1': train_f1_epoch, \
                            #            'valid_acc': valid_acc_epoch, 'valid_precision': valid_precision_epoch, 'valid_recall': valid_recall_epoch, 'valid_f1': valid_f1_epoch, \
                            #            'train_precision_inv': train_precision_inv_epoch, 'train_recall_inv': train_recall_inv_epoch, 'train_f1_inv': train_f1_inv_epoch, \
                            #            'valid_precision_inv': valid_precision_inv_epoch, 'valid_recall_inv': valid_recall_inv_epoch, 'valid_f1_inv': valid_f1_inv_epoch \
                            #            'NG50': ng50, 'NGA50': nga50})
                        else:
                            wandb.log({'train_loss': train_loss_epoch, 'val_loss': valid_loss_epoch, 'lr_value': lr_value, \
                                       'train_loss': train_loss_epoch, 'train_fpr': train_fp_rate_epoch, 'train_fnr': train_fn_rate_epoch, \
                                       'valid_loss': valid_loss_epoch, 'valid_fpr': valid_fp_rate_epoch, 'valid_fnr': valid_fn_rate_epoch, \
                                       'train_acc': train_acc_epoch, 'train_precision': train_precision_epoch, 'train_recall': train_recall_epoch, 'train_f1': train_f1_epoch, \
                                       'valid_acc': valid_acc_epoch, 'valid_precision': valid_precision_epoch, 'valid_recall': valid_recall_epoch, 'valid_f1': valid_f1_epoch, \
                                       'train_precision_inv': train_precision_inv_epoch, 'train_recall_inv': train_recall_inv_epoch, 'train_f1_inv': train_f1_inv_epoch, \
                                       'valid_precision_inv': valid_precision_inv_epoch, 'valid_recall_inv': valid_recall_inv_epoch, 'valid_f1_inv': valid_f1_inv_epoch})
                    except Exception as e:
                        print(f'WandB exception occured!')
                        print(e)

    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Keyboard Interrupt...")
        print("Exiting...")

    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Path to the dataset')
    parser.add_argument('--valid', type=str, help='Path to the dataset')
    parser.add_argument('--asm', type=str, help='Assembler used')
    parser.add_argument('--name', type=str, default=None, help='Name for the model')
    parser.add_argument('--overfit', action='store_true', help='Overfit on the training data')
    parser.add_argument('--resume', action='store_true', help='Resume in case training failed')
    parser.add_argument('--finetune', action='store_true', help='Finetune a trained model')
    parser.add_argument('--ft_model', type=str, help='Path to the model for fine-tuning')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate for the model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--gpu', type=int, default=None, help='Index of a GPU to train on (unspecified = cpu)')
    # parser.add_argument('--savedir', type=str, default=None, help='Directory to save the model and the checkpoints')
    args = parser.parse_args()

    train(train_path=args.train, valid_path=args.valid, assembler=args.asm, out=args.name, overfit=args.overfit, \
          dropout=args.dropout, seed=args.seed, resume=args.resume, finetune=args.finetune, ft_model=args.ft_model, gpu=args.gpu)
