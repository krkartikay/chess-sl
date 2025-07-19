# WandB Configuration for Chess-SL experiments

# Hyperparameter sweep configuration for WandB
SWEEP_CONFIG = {
    'method': 'grid',  # grid, random, bayes
    'name': 'chess-sl-hyperparameter-sweep',
    'metric': {
        'name': 'avg_moves',
        'goal': 'maximize'
    },
    'parameters': {
        'num_epochs': {
            'values': [20, 50]
        },
        'n_blocks': {
            'values': [0, 1, 4, 8]
        },
        'n_channels': {
            'values': [4, 16, 64]  # 128
        },
        'batch_size': {
            'values': [64, 128]  # 256
        },
        'learning_rate_multiplier': {
            'values': [1, 3]
        },
        'optimizer': {
            'values': ['SGD', 'ADAM']
        },
        'n_hidden': {
            'values': [1024, 2048]  # 4096
        },
        'num_training_examples': {
            'values': [10000, 30000, 100000]
        }
    }
}

# Development mode configuration (smaller parameter space)
DEV_SWEEP_CONFIG = {
    'method': 'grid',
    'name': 'chess-sl-dev-sweep',
    'metric': {
        'name': 'avg_moves',
        'goal': 'maximize'
    },
    'parameters': {
        'num_epochs': {
            'values': [10]
        },
        'n_blocks': {
            'values': [0, 1]
        },
        'n_channels': {
            'values': [8, 16]
        },
        'batch_size': {
            'values': [128]
        },
        'learning_rate_multiplier': {
            'values': [1]
        },
        'optimizer': {
            'values': ['ADAM']
        },
        'n_hidden': {
            'values': [1024]
        },
        'num_training_examples': {
            'values': [10000]
        }
    }
}

# Fixed configuration values
FILTER_SIZE = 3