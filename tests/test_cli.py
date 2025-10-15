"""
Test suite for JEPA CLI components.

Tests the command-line interface including training, evaluation, and utility functions.
"""

import unittest
import tempfile
import os
import shutil
import argparse
import sys
from unittest.mock import patch, MagicMock, call
from io import StringIO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.cli.train import parse_args, main as train_main, create_model_from_config
from jepa.cli.evaluate import main as evaluate_main, parse_args as eval_parse_args
from jepa.cli.utils import (
    setup_logging, validate_config, create_output_dir,
    add_common_args, parse_overrides, merge_args_with_config
)
from jepa.cli.__main__ import main as cli_main


class TestTrainCLI(unittest.TestCase):
    """Test cases for training CLI."""
    
    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        # Mock sys.argv
        test_args = ['train.py']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
        
        # Check default values
        self.assertIsNone(args.config)
        self.assertIsNone(args.train_data)
        self.assertIsNone(args.val_data)
        self.assertIsNone(args.batch_size)
        self.assertIsNone(args.learning_rate)
        self.assertIsNone(args.num_epochs)
        self.assertIsNone(args.device)
    
    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values."""
        test_args = [
            'train.py',
            '--config', 'config.yaml',
            '--train-data', '/path/to/train',
            '--val-data', '/path/to/val',
            '--batch-size', '64',
            '--learning-rate', '1e-4',
            '--num-epochs', '50',
            '--device', 'cuda',
            '--output-dir', '/path/to/output',
            '--experiment-name', 'test-exp'
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
        
        self.assertEqual(args.config, 'config.yaml')
        self.assertEqual(args.train_data, '/path/to/train')
        self.assertEqual(args.val_data, '/path/to/val')
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.num_epochs, 50)
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.output_dir, '/path/to/output')
        self.assertEqual(args.experiment_name, 'test-exp')
    
    def test_parse_args_resume(self):
        """Test parsing resume argument."""
        test_args = ['train.py', '--resume', '/path/to/checkpoint.pth']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
        
        self.assertEqual(args.resume, '/path/to/checkpoint.pth')
    
    @patch('jepa.cli.train.load_config')
    @patch('jepa.cli.train.create_default_config')
    @patch('jepa.cli.train.create_dataloader')
    @patch('jepa.cli.train.create_trainer')
    def test_train_main_with_config(self, mock_create_trainer, mock_create_dataloader,
                                   mock_create_default_config, mock_load_config):
        """Test main training function with config file."""
        # Setup mocks
        mock_config = {
            'model': {'encoder_dim': 256},
            'training': {'batch_size': 32, 'num_epochs': 10},
            'data': {'train_data_path': '/train', 'val_data_path': '/val'}
        }
        mock_load_config.return_value = mock_config
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'train_loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
        mock_create_trainer.return_value = mock_trainer
        
        mock_dataloader = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader
        
        # Test arguments
        test_args = [
            'train.py',
            '--config', 'test_config.yaml',
            '--output-dir', '/tmp/output'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('os.makedirs'):
                train_main()
        
        # Verify mocks were called
        mock_load_config.assert_called_once_with('test_config.yaml')
        mock_create_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('jepa.cli.train.create_default_config')
    @patch('jepa.cli.train.create_dataloader')
    @patch('jepa.cli.train.create_trainer')
    def test_train_main_without_config(self, mock_create_trainer, mock_create_dataloader,
                                      mock_create_default_config):
        """Test main training function without config file."""
        # Setup mocks
        mock_config = {
            'model': {'encoder_dim': 256},
            'training': {'batch_size': 32, 'num_epochs': 10},
            'data': {'train_data_path': '', 'val_data_path': ''}
        }
        mock_create_default_config.return_value = mock_config
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'train_loss': [0.5], 'val_loss': [0.6]}
        mock_create_trainer.return_value = mock_trainer
        
        mock_dataloader = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader
        
        # Test arguments
        test_args = [
            'train.py',
            '--train-data', '/path/to/train',
            '--batch-size', '64'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('os.makedirs'):
                train_main()
        
        # Verify default config was used
        mock_create_default_config.assert_called_once()
        mock_create_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()


class TestEvaluateCLI(unittest.TestCase):
    """Test cases for evaluation CLI."""
    
    def test_eval_parse_args_defaults(self):
        """Test evaluation argument parsing with defaults."""
        test_args = ['evaluate.py']
        
        with patch.object(sys, 'argv', test_args):
            args = eval_parse_args()
        
        self.assertIsNone(args.model_path)
        self.assertIsNone(args.config)
        self.assertIsNone(args.test_data)
    
    def test_eval_parse_args_custom(self):
        """Test evaluation argument parsing with custom values."""
        test_args = [
            'evaluate.py',
            '--model-path', '/path/to/model.pth',
            '--config', 'config.yaml',
            '--test-data', '/path/to/test',
            '--batch-size', '32',
            '--output', 'results.json'
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = eval_parse_args()
        
        self.assertEqual(args.model_path, '/path/to/model.pth')
        self.assertEqual(args.config, 'config.yaml')
        self.assertEqual(args.test_data, '/path/to/test')
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.output, 'results.json')
    
    @patch('jepa.cli.evaluate.load_config')
    @patch('jepa.cli.evaluate.load_model')
    @patch('jepa.cli.evaluate.create_dataloader')
    @patch('jepa.cli.evaluate.JEPAEvaluator')
    def test_evaluate_main(self, mock_evaluator_class, mock_create_dataloader,
                          mock_load_model, mock_load_config):
        """Test main evaluation function."""
        # Setup mocks
        mock_config = {'model': {'encoder_dim': 256}, 'data': {'batch_size': 32}}
        mock_load_config.return_value = mock_config
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_dataloader = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {'test_loss': 0.5, 'test_accuracy': 0.8}
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test arguments
        test_args = [
            'evaluate.py',
            '--model-path', 'model.pth',
            '--config', 'config.yaml',
            '--test-data', '/test/data'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('builtins.open', create=True):
                evaluate_main()
        
        # Verify mocks were called
        mock_load_config.assert_called_once_with('config.yaml')
        mock_load_model.assert_called_once_with('model.pth')
        mock_create_dataloader.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()


class TestCLIUtils(unittest.TestCase):
    """Test cases for CLI utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup function."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging(level='INFO', format='cli')
            
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            self.assertIn('level', call_args.kwargs)
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        valid_config = {
            'model': {'encoder_dim': 256},
            'training': {'batch_size': 32, 'learning_rate': 1e-3},
            'data': {'num_workers': 4}
        }
        
        is_valid, errors = validate_config(valid_config)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        invalid_config = {
            'training': {'batch_size': -32}  # Invalid negative batch size
        }
        
        is_valid, errors = validate_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_create_output_dir(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'test_output')
            
            result_dir = create_output_dir(output_dir)
            
            self.assertTrue(os.path.exists(result_dir))
            self.assertEqual(result_dir, output_dir)
    
    def test_create_output_dir_with_timestamp(self):
        """Test output directory creation with timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = os.path.join(temp_dir, 'test_output')
            
            result_dir = create_output_dir(base_dir, add_timestamp=True)
            
            self.assertTrue(os.path.exists(result_dir))
            self.assertTrue(result_dir.startswith(base_dir))
            self.assertNotEqual(result_dir, base_dir)  # Should have timestamp appended
    
    def test_add_common_args(self):
        """Test adding common arguments to parser."""
        parser = argparse.ArgumentParser()
        
        add_common_args(parser)
        
        # Test that common arguments were added
        args = parser.parse_args(['--device', 'cuda', '--seed', '123'])
        
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.seed, 123)
    
    def test_parse_overrides(self):
        """Test parsing parameter overrides."""
        override_list = [
            '--model.encoder_dim=512',
            '--training.batch_size=64',
            '--training.learning_rate=1e-4'
        ]
        
        overrides = parse_overrides(override_list)
        
        expected = {
            'model': {'encoder_dim': 512},
            'training': {'batch_size': 64, 'learning_rate': 1e-4}
        }
        
        self.assertEqual(overrides, expected)
    
    def test_parse_overrides_invalid_format(self):
        """Test parsing invalid override format."""
        invalid_overrides = ['invalid_format', 'missing_equals']
        
        with self.assertRaises(ValueError):
            parse_overrides(invalid_overrides)
    
    def test_merge_args_with_config(self):
        """Test merging command-line args with config."""
        config = {
            'model': {'encoder_dim': 256},
            'training': {'batch_size': 32, 'learning_rate': 1e-3}
        }
        
        # Mock argparse namespace
        class MockArgs:
            def __init__(self):
                self.batch_size = 64
                self.learning_rate = None  # Not provided
                self.device = 'cuda'  # New value
        
        args = MockArgs()
        merged_config = merge_args_with_config(config, args)
        
        # Check overrides
        self.assertEqual(merged_config['training']['batch_size'], 64)
        
        # Check preserved values
        self.assertEqual(merged_config['training']['learning_rate'], 1e-3)
        self.assertEqual(merged_config['model']['encoder_dim'], 256)
        
        # Check new values
        self.assertEqual(merged_config['device'], 'cuda')


class TestMainCLI(unittest.TestCase):
    """Test cases for main CLI entry point."""
    
    @patch('jepa.cli.train.main')
    def test_cli_main_train(self, mock_train_main):
        """Test main CLI with train command."""
        test_args = ['cli', 'train', '--config', 'config.yaml']
        
        with patch.object(sys, 'argv', test_args):
            cli_main()
        
        mock_train_main.assert_called_once()
    
    @patch('jepa.cli.evaluate.main')
    def test_cli_main_evaluate(self, mock_eval_main):
        """Test main CLI with evaluate command."""
        test_args = ['cli', 'evaluate', '--model-path', 'model.pth']
        
        with patch.object(sys, 'argv', test_args):
            cli_main()
        
        mock_eval_main.assert_called_once()
    
    def test_cli_main_help(self):
        """Test main CLI help command."""
        test_args = ['cli', '--help']
        
        with patch.object(sys, 'argv', test_args):
            with patch('sys.exit') as mock_exit:
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    try:
                        cli_main()
                    except SystemExit:
                        pass
                
                # Check that help text was printed
                output = mock_stdout.getvalue()
                self.assertIn('usage:', output.lower())
    
    def test_cli_main_invalid_command(self):
        """Test main CLI with invalid command."""
        test_args = ['cli', 'invalid_command']
        
        with patch.object(sys, 'argv', test_args):
            with patch('sys.stderr', new_callable=StringIO):
                with self.assertRaises(SystemExit):
                    cli_main()


class TestModelCreation(unittest.TestCase):
    """Test cases for model creation from config."""
    
    @patch('jepa.cli.train.create_encoder')
    @patch('jepa.cli.train.create_predictor')
    @patch('jepa.cli.train.JEPA')
    def test_create_model_from_config(self, mock_jepa, mock_create_predictor,
                                     mock_create_encoder):
        """Test creating model from configuration."""
        config = {
            'model': {
                'encoder_type': 'transformer',
                'encoder_dim': 256,
                'predictor_type': 'mlp',
                'predictor_hidden_dim': 512
            }
        }
        
        mock_encoder = MagicMock()
        mock_predictor = MagicMock()
        mock_model = MagicMock()
        
        mock_create_encoder.return_value = mock_encoder
        mock_create_predictor.return_value = mock_predictor
        mock_jepa.return_value = mock_model
        
        result_model = create_model_from_config(config)
        
        # Verify that components were created with correct config
        mock_create_encoder.assert_called_once_with(config['model'])
        mock_create_predictor.assert_called_once_with(config['model'])
        mock_jepa.assert_called_once_with(mock_encoder, mock_predictor)
        
        self.assertEqual(result_model, mock_model)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test config file
        test_config = {
            'model': {
                'encoder_type': 'transformer',
                'encoder_dim': 64,  # Small for testing
                'predictor_type': 'mlp',
                'predictor_hidden_dim': 128
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-3,
                'num_epochs': 2
            },
            'data': {
                'train_data_path': '',
                'val_data_path': '',
                'num_workers': 0
            },
            'device': 'cpu',
            'seed': 42
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('jepa.cli.train.create_dataloader')
    @patch('jepa.cli.train.create_trainer')
    def test_full_train_workflow(self, mock_create_trainer, mock_create_dataloader):
        """Test complete training workflow through CLI."""
        # Setup mocks
        mock_dataloader = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {'train_loss': [0.5, 0.4]}
        mock_create_trainer.return_value = mock_trainer
        
        # Test training with config file
        test_args = [
            'train.py',
            '--config', self.config_file,
            '--output-dir', self.temp_dir,
            '--num-epochs', '1'  # Override config
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('os.makedirs'):
                train_main()
        
        # Verify that training was called
        mock_create_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    def test_config_file_validation(self):
        """Test config file validation in CLI workflow."""
        # Create invalid config
        invalid_config = {
            'training': {'batch_size': -32}  # Invalid
        }
        
        invalid_config_file = os.path.join(self.temp_dir, 'invalid_config.yaml')
        import yaml
        with open(invalid_config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        is_valid, errors = validate_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_output_directory_creation(self):
        """Test output directory creation in CLI workflow."""
        output_dir = os.path.join(self.temp_dir, 'new_output_dir')
        
        created_dir = create_output_dir(output_dir)
        
        self.assertTrue(os.path.exists(created_dir))
        self.assertEqual(created_dir, output_dir)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_help_message_content(self, mock_stdout):
        """Test that help messages contain expected content."""
        test_args = ['train.py', '--help']
        
        with patch.object(sys, 'argv', test_args):
            with patch('sys.exit'):
                try:
                    parse_args()
                except SystemExit:
                    pass
        
        help_output = mock_stdout.getvalue()
        
        # Check that important options are documented
        self.assertIn('--config', help_output)
        self.assertIn('--batch-size', help_output)
        self.assertIn('--learning-rate', help_output)
        self.assertIn('--num-epochs', help_output)


if __name__ == '__main__':
    unittest.main()
