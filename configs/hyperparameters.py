import torch

def get_hyperparameters():
    return {
        
        'device': 'cuda:2' if torch.cuda.is_available() else 'cpu',
        'seed': 1,
        'wandb_mode': 'online',  # switch between 'online' and 'disabled'
        'wandb_project': 'GNNome-dev',

        # Assembly during training
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
        'hidden_ne_features': 16,
        'hidden_edge_scores': 64,
        'nb_pos_enc': 0,
        'type_pos_enc': 'none',
        'normalization': 'batch',
        'dropout': 0.2,

        # Training
        'num_epochs': 5,
        'lr': 1e-4,
        'use_symmetry_loss': True,
        'alpha': 0.1,
        'num_nodes_per_cluster': 1000,  # 2000 = max 10GB GPU memory for d=128, L=8
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
        'len_threshold': 70_000,
    }

