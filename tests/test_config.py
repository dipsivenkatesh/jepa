"""
Test suite for JEPA configuration components.

Tests the configuration classes, loading, saving, and validation.
"""

import unittest
import tempfile
import os
import yaml
from dataclasses import asdict
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.config.config import (
    ModelConfig, TrainingConfig, DataConfig, WandbConfig, 
    TensorBoardConfig, ConsoleConfig, LoggingConfig, JEPAConfig,
    load_config, save_config, create_default_config, validate_config,
    override_config_with_args, merge_configs
)


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig class."""
    
    def test_default_initialization(self):
        """Test ModelConfig default initialization."""
        config = ModelConfig()
        
        self.assertEqual(config.encoder_type, "transformer")
        self.assertEqual(config.encoder_dim, 512)
        self.assertEqual(config.predictor_type, "mlp")
        self.assertEqual(config.predictor_hidden_dim, 1024)
        self.assertEqual(config.predictor_output_dim, 512)
        self.assertEqual(config.dropout, 0.1)
    
    def test_custom_initialization(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            encoder_type="cnn",
            encoder_dim=256,
            predictor_type="attention",
            dropout=0.2
        )
        
        self.assertEqual(config.encoder_type, "cnn")
        self.assertEqual(config.encoder_dim, 256)
        self.assertEqual(config.predictor_type, "attention")
        self.assertEqual(config.dropout, 0.2)
        
        # Check that defaults are preserved for non-specified values
        self.assertEqual(config.predictor_hidden_dim, 1024)
        self.assertEqual(config.predictor_output_dim, 512)
    
    def test_to_dict(self):
        """Test converting ModelConfig to dictionary."""
        config = ModelConfig(encoder_dim=256)
        config_dict = asdict(config)
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['encoder_dim'], 256)
        self.assertIn('encoder_type', config_dict)
        self.assertIn('predictor_type', config_dict)


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig class."""
    
    def test_default_initialization(self):
        """Test TrainingConfig default initialization."""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.num_epochs, 100)
        self.assertEqual(config.warmup_epochs, 10)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.save_every, 10)
        self.assertEqual(config.early_stopping_patience, 20)
        self.assertEqual(config.log_interval, 100)
    
    def test_custom_initialization(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            batch_size=64,
            learning_rate=1e-4,
            num_epochs=50,
            early_stopping_patience=None
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_epochs, 50)
        self.assertIsNone(config.early_stopping_patience)
    
    def test_validation(self):
        """Test TrainingConfig validation."""
        # Valid config
        config = TrainingConfig(batch_size=32, learning_rate=1e-3)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.learning_rate, 0)
        
        # Test edge cases
        config_edge = TrainingConfig(batch_size=1, learning_rate=1e-10)
        self.assertEqual(config_edge.batch_size, 1)
        self.assertEqual(config_edge.learning_rate, 1e-10)


class TestDataConfig(unittest.TestCase):
    """Test cases for DataConfig class."""
    
    def test_default_initialization(self):
        """Test DataConfig default initialization."""
        config = DataConfig()
        
        self.assertEqual(config.train_data_path, "")
        self.assertEqual(config.val_data_path, "")
        self.assertEqual(config.test_data_path, "")
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.sequence_length, 10)
        self.assertEqual(config.input_dim, 784)
    
    def test_custom_initialization(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            train_data_path="/path/to/train",
            num_workers=8,
            pin_memory=False,
            sequence_length=20
        )
        
        self.assertEqual(config.train_data_path, "/path/to/train")
        self.assertEqual(config.num_workers, 8)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.sequence_length, 20)


class TestLoggingConfigs(unittest.TestCase):
    """Test cases for logging configuration classes."""
    
    def test_wandb_config(self):
        """Test WandbConfig."""
        config = WandbConfig()
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.project, "jepa")
        self.assertIsNone(config.entity)
        self.assertTrue(config.log_model)
        self.assertFalse(config.log_gradients)
        
        # Test custom values
        custom_config = WandbConfig(
            enabled=True,
            project="custom-project",
            entity="my-team"
        )
        
        self.assertTrue(custom_config.enabled)
        self.assertEqual(custom_config.project, "custom-project")
        self.assertEqual(custom_config.entity, "my-team")
    
    def test_tensorboard_config(self):
        """Test TensorBoardConfig."""
        config = TensorBoardConfig()
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.log_dir, "./tensorboard_logs")
        self.assertEqual(config.comment, "")
        
        # Test custom values
        custom_config = TensorBoardConfig(
            enabled=True,
            log_dir="/custom/logs",
            comment="experiment-1"
        )
        
        self.assertTrue(custom_config.enabled)
        self.assertEqual(custom_config.log_dir, "/custom/logs")
        self.assertEqual(custom_config.comment, "experiment-1")
    
    def test_console_config(self):
        """Test ConsoleConfig."""
        config = ConsoleConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.level, "INFO")
        self.assertFalse(config.file)
        self.assertEqual(config.file_level, "DEBUG")
        
        # Test custom values
        custom_config = ConsoleConfig(
            enabled=False,
            level="DEBUG",
            file=True
        )
        
        self.assertFalse(custom_config.enabled)
        self.assertEqual(custom_config.level, "DEBUG")
        self.assertTrue(custom_config.file)
    
    def test_logging_config(self):
        """Test LoggingConfig composite."""
        wandb_config = WandbConfig(enabled=True)
        tensorboard_config = TensorBoardConfig(enabled=True)
        console_config = ConsoleConfig(level="DEBUG")
        
        logging_config = LoggingConfig(
            wandb=wandb_config,
            tensorboard=tensorboard_config,
            console=console_config
        )
        
        self.assertTrue(logging_config.wandb.enabled)
        self.assertTrue(logging_config.tensorboard.enabled)
        self.assertEqual(logging_config.console.level, "DEBUG")


class TestJEPAConfig(unittest.TestCase):
    """Test cases for main JEPAConfig class."""
    
    def test_default_initialization(self):
        """Test JEPAConfig default initialization."""
        config = JEPAConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            logging=LoggingConfig(
                wandb=WandbConfig(),
                tensorboard=TensorBoardConfig(),
                console=ConsoleConfig()
            )
        )
        
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.output_dir, "./outputs")
    
    def test_custom_initialization(self):
        """Test JEPAConfig with custom values."""
        model_config = ModelConfig(encoder_dim=256)
        training_config = TrainingConfig(batch_size=64)
        data_config = DataConfig(num_workers=8)
        logging_config = LoggingConfig(
            wandb=WandbConfig(enabled=True),
            tensorboard=TensorBoardConfig(),
            console=ConsoleConfig()
        )
        
        config = JEPAConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            device="cuda",
            seed=123
        )
        
        self.assertEqual(config.model.encoder_dim, 256)
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.data.num_workers, 8)
        self.assertTrue(config.logging.wandb.enabled)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.seed, 123)


class TestConfigFunctions(unittest.TestCase):
    """Test cases for configuration utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'model': {
                'encoder_type': 'transformer',
                'encoder_dim': 512,
                'predictor_type': 'mlp'
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'num_epochs': 100
            },
            'data': {
                'train_data_path': '/path/to/train',
                'num_workers': 4
            },
            'device': 'auto',
            'seed': 42
        }
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(self.test_config, temp_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Load and verify contents
            with open(temp_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertEqual(loaded_config['model']['encoder_type'], 'transformer')
            self.assertEqual(loaded_config['training']['batch_size'], 32)
            self.assertEqual(loaded_config['seed'], 42)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_config(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            
            self.assertEqual(loaded_config['model']['encoder_type'], 'transformer')
            self.assertEqual(loaded_config['training']['batch_size'], 32)
            self.assertEqual(loaded_config['data']['train_data_path'], '/path/to/train')
            self.assertEqual(loaded_config['seed'], 42)
        
        finally:
            os.unlink(temp_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')
    
    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                load_config(temp_path)
        
        finally:
            os.unlink(temp_path)
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('training', config)
        self.assertIn('data', config)
        self.assertIn('logging', config)
        self.assertIn('device', config)
        self.assertIn('seed', config)
        
        # Check some default values
        self.assertEqual(config['model']['encoder_type'], 'transformer')
        self.assertEqual(config['training']['batch_size'], 32)
        self.assertEqual(config['device'], 'auto')
        self.assertEqual(config['seed'], 42)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'model': {'encoder_dim': 512, 'predictor_hidden_dim': 256},
            'training': {'batch_size': 32, 'learning_rate': 1e-3},
            'data': {'num_workers': 4}
        }
        
        is_valid, errors = validate_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid config - negative batch size
        invalid_config = {
            'training': {'batch_size': -32, 'learning_rate': 1e-3}
        }
        
        is_valid, errors = validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_override_config_with_args(self):
        """Test overriding configuration with command-line arguments."""
        base_config = {
            'model': {'encoder_dim': 512},
            'training': {'batch_size': 32, 'learning_rate': 1e-3}
        }
        
        # Mock argparse namespace
        class MockArgs:
            def __init__(self):
                self.batch_size = 64
                self.learning_rate = 1e-4
                self.encoder_dim = None  # Not provided
        
        args = MockArgs()
        overridden_config = override_config_with_args(base_config, args)
        
        # Check that values were overridden
        self.assertEqual(overridden_config['training']['batch_size'], 64)
        self.assertEqual(overridden_config['training']['learning_rate'], 1e-4)
        
        # Check that non-provided values remain unchanged
        self.assertEqual(overridden_config['model']['encoder_dim'], 512)
    
    def test_merge_configs(self):
        """Test merging multiple configurations."""
        base_config = {
            'model': {'encoder_dim': 512, 'dropout': 0.1},
            'training': {'batch_size': 32}
        }
        
        override_config = {
            'model': {'encoder_dim': 256},  # Override
            'training': {'learning_rate': 1e-3},  # New key
            'data': {'num_workers': 4}  # New section
        }
        
        merged_config = merge_configs(base_config, override_config)
        
        # Check overrides
        self.assertEqual(merged_config['model']['encoder_dim'], 256)
        
        # Check preserved values
        self.assertEqual(merged_config['model']['dropout'], 0.1)
        self.assertEqual(merged_config['training']['batch_size'], 32)
        
        # Check new values
        self.assertEqual(merged_config['training']['learning_rate'], 1e-3)
        self.assertEqual(merged_config['data']['num_workers'], 4)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration system."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # 1. Create default config
        config = create_default_config()
        
        # 2. Modify some values
        config['model']['encoder_dim'] = 256
        config['training']['batch_size'] = 64
        config['logging']['wandb']['enabled'] = True
        
        # 3. Validate config
        is_valid, errors = validate_config(config)
        self.assertTrue(is_valid, f"Config validation failed: {errors}")
        
        # 4. Save config to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            
            # 5. Load config from file
            loaded_config = load_config(temp_path)
            
            # 6. Verify loaded config
            self.assertEqual(loaded_config['model']['encoder_dim'], 256)
            self.assertEqual(loaded_config['training']['batch_size'], 64)
            self.assertTrue(loaded_config['logging']['wandb']['enabled'])
            
            # 7. Validate loaded config
            is_valid_loaded, errors_loaded = validate_config(loaded_config)
            self.assertTrue(is_valid_loaded, f"Loaded config validation failed: {errors_loaded}")
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_dataclass_conversion(self):
        """Test conversion between dict and dataclass formats."""
        # Create config dict
        config_dict = {
            'model': {
                'encoder_type': 'transformer',
                'encoder_dim': 256,
                'predictor_type': 'mlp',
                'predictor_hidden_dim': 512,
                'predictor_output_dim': 256,
                'dropout': 0.2
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 1e-4,
                'weight_decay': 1e-3,
                'num_epochs': 50,
                'warmup_epochs': 5,
                'gradient_clip_norm': 0.5,
                'save_every': 5,
                'early_stopping_patience': 10,
                'log_interval': 50
            }
        }
        
        # Convert to dataclasses
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        
        # Verify conversion
        self.assertEqual(model_config.encoder_type, 'transformer')
        self.assertEqual(model_config.encoder_dim, 256)
        self.assertEqual(training_config.batch_size, 64)
        self.assertEqual(training_config.learning_rate, 1e-4)
        
        # Convert back to dict
        model_dict = asdict(model_config)
        training_dict = asdict(training_config)
        
        # Verify round-trip
        self.assertEqual(model_dict, config_dict['model'])
        self.assertEqual(training_dict, config_dict['training'])
    
    def test_nested_config_override(self):
        """Test overriding nested configuration values."""
        base_config = {
            'model': {
                'encoder_type': 'transformer',
                'encoder_dim': 512
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-3
            },
            'logging': {
                'wandb': {
                    'enabled': False,
                    'project': 'default'
                },
                'console': {
                    'enabled': True,
                    'level': 'INFO'
                }
            }
        }
        
        override_config = {
            'model': {
                'encoder_dim': 256  # Override encoder_dim but keep encoder_type
            },
            'logging': {
                'wandb': {
                    'enabled': True  # Override wandb enabled but keep project
                }
                # Don't modify console config
            }
        }
        
        merged = merge_configs(base_config, override_config)
        
        # Check overrides
        self.assertEqual(merged['model']['encoder_dim'], 256)
        self.assertTrue(merged['logging']['wandb']['enabled'])
        
        # Check preserved values
        self.assertEqual(merged['model']['encoder_type'], 'transformer')
        self.assertEqual(merged['logging']['wandb']['project'], 'default')
        self.assertTrue(merged['logging']['console']['enabled'])
        self.assertEqual(merged['logging']['console']['level'], 'INFO')


if __name__ == '__main__':
    unittest.main()
