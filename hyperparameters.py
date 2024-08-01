import math
import torch

def get_hyperparameters():
    return {
        
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'seed': 1,
        'wandb_mode': 'disabled',  # switch between 'online' and 'disabled'
        'wandb_project': 'GNNome',

        'chr_overfit': 0,
        'plot_nga50_during_training': False,
        'eval_frequency': 20, 

        # Data
        'use_similarities': True,

        # Model
        'dim_latent': 64,
        'num_gnn_layers': 8,
        'node_features': 2,
        'edge_features': 2,  # Put 2 if you use similarities, 1 otherwise
        'hidden_edge_features': 16,
        'hidden_edge_scores': 64,
        'nb_pos_enc': 0,
        'type_pos_enc': 'none',
        'batch_norm': True,
        # 'dropout': 0.08,

        # Training
        'num_epochs': 200,
        'lr': 1e-4,
        'use_symmetry_loss': True,
        'alpha': 0.1,
        'num_parts_metis_train': 200,
        'num_parts_metis_eval': 200,
        'num_nodes_per_cluster': 10000,  # 2000 = max 10GB GPU memory for d=128, L=8
        'npc_lower_bound': 1,  # 0.8
        'npc_upper_bound': 1,  # 1.2
        'k_extra_hops': 1,
        'patience': 2,
        'decay': 0.95,
        'masking': True,
        'mask_frac_low': 80,   # ~ 25x
        'mask_frac_high': 100, # ~ 60x

        # Decoding
        'strategy': 'greedy',
        'num_decoding_paths': 100,
        'decode_with_labels': False,
        'load_checkpoint': True,
        'num_threads': 32,
        'B': 1,
        'len_threshold': 10,
        'heuristic_function': lambda prob, length: math.log(prob),
        'heuristic_reduce_function': lambda curr, new: curr + new,
        'initial_heuristic_value': 0.0,
        'weighted_random_function': lambda prob: -torch.sqrt(1 - prob ** 2) + 1,
    }

