"""
Simple test configuration for JEPA framework testing.
"""

# Test configuration
TEST_CONFIG = {
    'model': {
        'encoder_type': 'transformer',
        'encoder_dim': 64,  # Small for fast testing
        'predictor_type': 'mlp',
        'predictor_hidden_dim': 128,
        'predictor_output_dim': 64,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 2,  # Short for testing
        'warmup_epochs': 0,
        'gradient_clip_norm': 1.0,
        'save_every': 1,
        'early_stopping_patience': None,
        'log_interval': 1
    },
    'data': {
        'train_data_path': '',
        'val_data_path': '',
        'test_data_path': '',
        'num_workers': 0,  # No multiprocessing for testing
        'pin_memory': False,
        'sequence_length': 5,  # Short sequences
        'input_dim': 64
    },
    'logging': {
        'wandb': {
            'enabled': False,  # Disabled for testing
            'project': 'jepa-test',
            'entity': None,
            'name': None,
            'tags': ['test'],
            'notes': 'Test run',
            'log_model': False,
            'log_gradients': False,
            'log_freq': 1,
            'watch_model': False
        },
        'tensorboard': {
            'enabled': False,  # Disabled for testing
            'log_dir': './test_tensorboard_logs',
            'comment': 'test'
        },
        'console': {
            'enabled': True,
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': False,
            'file_level': 'DEBUG',
            'log_dir': './test_logs'
        }
    },
    'device': 'cpu',  # Always CPU for testing
    'seed': 42,
    'output_dir': './test_outputs'
}

# Test data dimensions
TEST_DIMENSIONS = {
    'batch_size': 4,
    'sequence_length': 5,
    'hidden_dim': 64,
    'num_samples': 20
}

# Expected test tolerances
TEST_TOLERANCES = {
    'rtol': 1e-5,
    'atol': 1e-6,
    'loss_threshold': 10.0  # Loss should be reasonable
}
