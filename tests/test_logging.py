"""
Test suite for JEPA logging components.

Tests the logging system including different logger types and the multi-logger.
"""

import unittest
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock, call
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.loggers.base_logger import BaseLogger, LoggerRegistry, LogLevel
from jepa.loggers.console_logger import ConsoleLogger
from jepa.loggers.tensorboard_logger import TensorBoardLogger
from jepa.loggers.wandb_logger import WandbLogger
from jepa.loggers.multi_logger import MultiLogger, create_logger


class TestBaseLogger(unittest.TestCase):
    """Test cases for BaseLogger class."""
    
    def test_abstract_methods(self):
        """Test that BaseLogger abstract methods raise NotImplementedError."""
        logger = BaseLogger({'enabled': True})
        
        with self.assertRaises(NotImplementedError):
            logger.log_metrics({'loss': 0.5}, step=1)
        
        with self.assertRaises(NotImplementedError):
            logger.log_hyperparameters({'lr': 1e-3})
        
        with self.assertRaises(NotImplementedError):
            logger.save_artifact('path/to/file')
        
        with self.assertRaises(NotImplementedError):
            logger.finish()
    
    def test_initialization(self):
        """Test BaseLogger initialization."""
        config = {'enabled': True, 'project': 'test'}
        logger = BaseLogger(config)
        
        self.assertEqual(logger.config, config)
        self.assertTrue(logger.enabled)
        
        # Test disabled logger
        disabled_config = {'enabled': False}
        disabled_logger = BaseLogger(disabled_config)
        self.assertFalse(disabled_logger.enabled)


class TestLoggerRegistry(unittest.TestCase):
    """Test cases for LoggerRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear registry for clean tests
        LoggerRegistry._loggers = {}
    
    def test_register_logger(self):
        """Test registering logger types."""
        class TestLogger(BaseLogger):
            def log_metrics(self, metrics, step=None):
                pass
            
            def log_hyperparameters(self, hyperparams):
                pass
            
            def save_artifact(self, file_path, artifact_name=None):
                pass
            
            def finish(self):
                pass
        
        LoggerRegistry.register('test', TestLogger)
        
        self.assertIn('test', LoggerRegistry._loggers)
        self.assertEqual(LoggerRegistry._loggers['test'], TestLogger)
    
    def test_create_logger(self):
        """Test creating logger from registry."""
        class TestLogger(BaseLogger):
            def log_metrics(self, metrics, step=None):
                pass
            
            def log_hyperparameters(self, hyperparams):
                pass
            
            def save_artifact(self, file_path, artifact_name=None):
                pass
            
            def finish(self):
                pass
        
        LoggerRegistry.register('test', TestLogger)
        
        config = {'enabled': True}
        logger = LoggerRegistry.create('test', config)
        
        self.assertIsInstance(logger, TestLogger)
        self.assertTrue(logger.enabled)
    
    def test_create_unknown_logger(self):
        """Test creating unknown logger type."""
        with self.assertRaises(ValueError):
            LoggerRegistry.create('unknown', {'enabled': True})
    
    def test_list_available_loggers(self):
        """Test listing available logger types."""
        class TestLogger1(BaseLogger):
            pass
        
        class TestLogger2(BaseLogger):
            pass
        
        LoggerRegistry.register('test1', TestLogger1)
        LoggerRegistry.register('test2', TestLogger2)
        
        available = LoggerRegistry.list_available()
        self.assertIn('test1', available)
        self.assertIn('test2', available)


class TestLogLevel(unittest.TestCase):
    """Test cases for LogLevel enum."""
    
    def test_log_levels(self):
        """Test log level values."""
        self.assertEqual(LogLevel.DEBUG.value, 'DEBUG')
        self.assertEqual(LogLevel.INFO.value, 'INFO')
        self.assertEqual(LogLevel.WARNING.value, 'WARNING')
        self.assertEqual(LogLevel.ERROR.value, 'ERROR')
        self.assertEqual(LogLevel.CRITICAL.value, 'CRITICAL')


class TestConsoleLogger(unittest.TestCase):
    """Test cases for ConsoleLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'level': 'INFO',
            'format': '%(message)s',
            'file': False
        }
    
    def test_initialization(self):
        """Test ConsoleLogger initialization."""
        logger = ConsoleLogger(self.config)
        
        self.assertTrue(logger.enabled)
        self.assertEqual(logger.level, 'INFO')
        self.assertFalse(logger.file_logging)
    
    @patch('logging.getLogger')
    def test_log_metrics(self, mock_get_logger):
        """Test logging metrics."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConsoleLogger(self.config)
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        
        logger.log_metrics(metrics, step=100)
        
        # Check that logger.info was called
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn('Step 100', call_args)
        self.assertIn('loss: 0.5', call_args)
        self.assertIn('accuracy: 0.8', call_args)
    
    @patch('logging.getLogger')
    def test_log_hyperparameters(self, mock_get_logger):
        """Test logging hyperparameters."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = ConsoleLogger(self.config)
        hyperparams = {'learning_rate': 1e-3, 'batch_size': 32}
        
        logger.log_hyperparameters(hyperparams)
        
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn('Hyperparameters:', call_args)
    
    def test_disabled_logger(self):
        """Test disabled console logger."""
        config = {'enabled': False}
        logger = ConsoleLogger(config)
        
        # Should not raise any exceptions
        logger.log_metrics({'loss': 0.5})
        logger.log_hyperparameters({'lr': 1e-3})
        logger.save_artifact('dummy_path')
        logger.finish()
    
    def test_file_logging(self):
        """Test file logging functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                'enabled': True,
                'file': True,
                'log_dir': tmp_dir,
                'level': 'INFO'
            }
            
            logger = ConsoleLogger(config)
            logger.log_metrics({'loss': 0.5}, step=1)
            
            # Check that log file was created
            log_files = [f for f in os.listdir(tmp_dir) if f.endswith('.log')]
            self.assertGreater(len(log_files), 0)


class TestTensorBoardLogger(unittest.TestCase):
    """Test cases for TensorBoardLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'enabled': True,
            'log_dir': self.temp_dir,
            'comment': 'test'
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_initialization(self, mock_writer):
        """Test TensorBoardLogger initialization."""
        logger = TensorBoardLogger(self.config)
        
        self.assertTrue(logger.enabled)
        self.assertEqual(logger.log_dir, self.temp_dir)
        self.assertEqual(logger.comment, 'test')
        
        # Check that SummaryWriter was created
        mock_writer.assert_called_once_with(
            log_dir=self.temp_dir,
            comment='test'
        )
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_log_metrics(self, mock_writer):
        """Test logging metrics to TensorBoard."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger(self.config)
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        
        logger.log_metrics(metrics, step=100)
        
        # Check that scalar values were logged
        expected_calls = [
            call('loss', 0.5, 100),
            call('accuracy', 0.8, 100)
        ]
        mock_writer_instance.add_scalar.assert_has_calls(expected_calls, any_order=True)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_log_hyperparameters(self, mock_writer):
        """Test logging hyperparameters to TensorBoard."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger(self.config)
        hyperparams = {'learning_rate': 1e-3, 'batch_size': 32}
        
        logger.log_hyperparameters(hyperparams)
        
        mock_writer_instance.add_hparams.assert_called_once_with(hyperparams, {})
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_save_artifact(self, mock_writer):
        """Test saving artifacts to TensorBoard."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger(self.config)
        
        # Create dummy file
        dummy_file = os.path.join(self.temp_dir, 'test.txt')
        with open(dummy_file, 'w') as f:
            f.write('test content')
        
        logger.save_artifact(dummy_file, 'test_artifact')
        
        # TensorBoard doesn't have direct artifact saving, so check it doesn't crash
        # In real implementation, this might save as text or other format
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_finish(self, mock_writer):
        """Test finishing TensorBoard logger."""
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger(self.config)
        logger.finish()
        
        mock_writer_instance.close.assert_called_once()
    
    def test_disabled_logger(self):
        """Test disabled TensorBoard logger."""
        config = {'enabled': False}
        logger = TensorBoardLogger(config)
        
        # Should not create writer or raise exceptions
        logger.log_metrics({'loss': 0.5})
        logger.log_hyperparameters({'lr': 1e-3})
        logger.save_artifact('dummy_path')
        logger.finish()


class TestWandbLogger(unittest.TestCase):
    """Test cases for WandbLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'project': 'test-project',
            'entity': 'test-entity',
            'name': 'test-run',
            'tags': ['test', 'unit'],
            'notes': 'Test run'
        }
    
    @patch('wandb.init')
    def test_initialization(self, mock_wandb_init):
        """Test WandbLogger initialization."""
        logger = WandbLogger(self.config)
        
        self.assertTrue(logger.enabled)
        
        # Check that wandb.init was called with correct parameters
        mock_wandb_init.assert_called_once_with(
            project='test-project',
            entity='test-entity',
            name='test-run',
            tags=['test', 'unit'],
            notes='Test run'
        )
    
    @patch('wandb.init')
    @patch('wandb.log')
    def test_log_metrics(self, mock_wandb_log, mock_wandb_init):
        """Test logging metrics to W&B."""
        logger = WandbLogger(self.config)
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        
        logger.log_metrics(metrics, step=100)
        
        mock_wandb_log.assert_called_once_with(metrics, step=100)
    
    @patch('wandb.init')
    @patch('wandb.config')
    def test_log_hyperparameters(self, mock_wandb_config, mock_wandb_init):
        """Test logging hyperparameters to W&B."""
        logger = WandbLogger(self.config)
        hyperparams = {'learning_rate': 1e-3, 'batch_size': 32}
        
        logger.log_hyperparameters(hyperparams)
        
        mock_wandb_config.update.assert_called_once_with(hyperparams)
    
    @patch('wandb.init')
    @patch('wandb.save')
    def test_save_artifact(self, mock_wandb_save, mock_wandb_init):
        """Test saving artifacts to W&B."""
        logger = WandbLogger(self.config)
        
        logger.save_artifact('/path/to/file', 'test_artifact')
        
        mock_wandb_save.assert_called_once_with('/path/to/file')
    
    @patch('wandb.init')
    @patch('wandb.finish')
    def test_finish(self, mock_wandb_finish, mock_wandb_init):
        """Test finishing W&B logger."""
        logger = WandbLogger(self.config)
        logger.finish()
        
        mock_wandb_finish.assert_called_once()
    
    @patch('wandb.init')
    @patch('wandb.watch')
    def test_watch_model(self, mock_wandb_watch, mock_wandb_init):
        """Test watching model with W&B."""
        logger = WandbLogger(self.config)
        model = nn.Linear(10, 5)
        
        logger.watch_model(model)
        
        mock_wandb_watch.assert_called_once_with(model)
    
    def test_disabled_logger(self):
        """Test disabled W&B logger."""
        config = {'enabled': False}
        logger = WandbLogger(config)
        
        # Should not initialize wandb or raise exceptions
        logger.log_metrics({'loss': 0.5})
        logger.log_hyperparameters({'lr': 1e-3})
        logger.save_artifact('dummy_path')
        logger.finish()


class TestMultiLogger(unittest.TestCase):
    """Test cases for MultiLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger1 = MagicMock()
        self.mock_logger2 = MagicMock()
        
        self.multi_logger = MultiLogger([self.mock_logger1, self.mock_logger2])
    
    def test_initialization(self):
        """Test MultiLogger initialization."""
        self.assertEqual(len(self.multi_logger.loggers), 2)
        self.assertIn(self.mock_logger1, self.multi_logger.loggers)
        self.assertIn(self.mock_logger2, self.multi_logger.loggers)
    
    def test_log_metrics(self):
        """Test logging metrics to all loggers."""
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        step = 100
        
        self.multi_logger.log_metrics(metrics, step)
        
        # Check that all loggers received the call
        self.mock_logger1.log_metrics.assert_called_once_with(metrics, step)
        self.mock_logger2.log_metrics.assert_called_once_with(metrics, step)
    
    def test_log_hyperparameters(self):
        """Test logging hyperparameters to all loggers."""
        hyperparams = {'learning_rate': 1e-3, 'batch_size': 32}
        
        self.multi_logger.log_hyperparameters(hyperparams)
        
        self.mock_logger1.log_hyperparameters.assert_called_once_with(hyperparams)
        self.mock_logger2.log_hyperparameters.assert_called_once_with(hyperparams)
    
    def test_save_artifact(self):
        """Test saving artifacts to all loggers."""
        file_path = '/path/to/file'
        artifact_name = 'test_artifact'
        
        self.multi_logger.save_artifact(file_path, artifact_name)
        
        self.mock_logger1.save_artifact.assert_called_once_with(file_path, artifact_name)
        self.mock_logger2.save_artifact.assert_called_once_with(file_path, artifact_name)
    
    def test_finish(self):
        """Test finishing all loggers."""
        self.multi_logger.finish()
        
        self.mock_logger1.finish.assert_called_once()
        self.mock_logger2.finish.assert_called_once()
    
    def test_watch_model(self):
        """Test watching model with all loggers."""
        model = nn.Linear(10, 5)
        
        self.multi_logger.watch_model(model)
        
        self.mock_logger1.watch_model.assert_called_once_with(model)
        self.mock_logger2.watch_model.assert_called_once_with(model)
    
    def test_error_handling(self):
        """Test error handling when one logger fails."""
        # Make one logger raise an exception
        self.mock_logger1.log_metrics.side_effect = Exception("Logger 1 failed")
        
        metrics = {'loss': 0.5}
        
        # Should not raise exception, but continue with other loggers
        self.multi_logger.log_metrics(metrics)
        
        # Second logger should still be called
        self.mock_logger2.log_metrics.assert_called_once_with(metrics, None)


class TestCreateLogger(unittest.TestCase):
    """Test cases for create_logger factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('wandb.init')
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_create_single_logger(self, mock_tb_writer, mock_wandb_init):
        """Test creating single logger."""
        config = {
            'console': {'enabled': True, 'level': 'INFO'}
        }
        
        logger = create_logger(config)
        
        self.assertIsInstance(logger, ConsoleLogger)
        self.assertTrue(logger.enabled)
    
    @patch('wandb.init')
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_create_multi_logger(self, mock_tb_writer, mock_wandb_init):
        """Test creating multi-logger with multiple backends."""
        config = {
            'console': {'enabled': True, 'level': 'INFO'},
            'tensorboard': {'enabled': True, 'log_dir': self.temp_dir},
            'wandb': {'enabled': True, 'project': 'test'}
        }
        
        logger = create_logger(config)
        
        self.assertIsInstance(logger, MultiLogger)
        self.assertEqual(len(logger.loggers), 3)
    
    def test_create_no_enabled_loggers(self):
        """Test creating logger when no loggers are enabled."""
        config = {
            'console': {'enabled': False},
            'tensorboard': {'enabled': False},
            'wandb': {'enabled': False}
        }
        
        logger = create_logger(config)
        
        # Should return a disabled logger or None
        self.assertIsNone(logger)
    
    @patch('wandb.init')
    def test_create_logger_wandb_only(self, mock_wandb_init):
        """Test creating W&B logger only."""
        config = {
            'wandb': {
                'enabled': True,
                'project': 'test-project',
                'entity': 'test-entity'
            }
        }
        
        logger = create_logger(config)
        
        self.assertIsInstance(logger, WandbLogger)
        mock_wandb_init.assert_called_once()
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_create_logger_tensorboard_only(self, mock_tb_writer):
        """Test creating TensorBoard logger only."""
        config = {
            'tensorboard': {
                'enabled': True,
                'log_dir': self.temp_dir,
                'comment': 'test'
            }
        }
        
        logger = create_logger(config)
        
        self.assertIsInstance(logger, TensorBoardLogger)
        mock_tb_writer.assert_called_once()


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logger components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_console_file_logging_integration(self):
        """Test console logger with file logging."""
        config = {
            'enabled': True,
            'level': 'INFO',
            'file': True,
            'log_dir': self.temp_dir
        }
        
        logger = ConsoleLogger(config)
        
        # Log some metrics
        logger.log_metrics({'loss': 0.5, 'accuracy': 0.8}, step=1)
        logger.log_hyperparameters({'lr': 1e-3, 'batch_size': 32})
        
        # Check that log file was created
        log_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.log')]
        self.assertGreater(len(log_files), 0)
        
        # Check log file contents
        log_file_path = os.path.join(self.temp_dir, log_files[0])
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        self.assertIn('loss', log_content)
        self.assertIn('accuracy', log_content)
    
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.config')
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_multi_logger_integration(self, mock_tb_writer, mock_wandb_config, 
                                     mock_wandb_log, mock_wandb_init):
        """Test multi-logger integration."""
        # Setup mocks
        mock_tb_instance = MagicMock()
        mock_tb_writer.return_value = mock_tb_instance
        
        config = {
            'console': {'enabled': True, 'level': 'INFO'},
            'tensorboard': {'enabled': True, 'log_dir': self.temp_dir},
            'wandb': {'enabled': True, 'project': 'test'}
        }
        
        logger = create_logger(config)
        
        # Test logging metrics
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        logger.log_metrics(metrics, step=100)
        
        # Check that all backends received the metrics
        mock_wandb_log.assert_called_with(metrics, step=100)
        mock_tb_instance.add_scalar.assert_any_call('loss', 0.5, 100)
        mock_tb_instance.add_scalar.assert_any_call('accuracy', 0.8, 100)
        
        # Test logging hyperparameters
        hyperparams = {'learning_rate': 1e-3, 'batch_size': 32}
        logger.log_hyperparameters(hyperparams)
        
        mock_wandb_config.update.assert_called_with(hyperparams)
        mock_tb_instance.add_hparams.assert_called_with(hyperparams, {})
    
    def test_logger_error_resilience(self):
        """Test that logger system is resilient to individual logger failures."""
        # Create a logger that will fail
        failing_logger = MagicMock()
        failing_logger.log_metrics.side_effect = Exception("Simulated failure")
        
        # Create a working logger
        working_logger = MagicMock()
        
        # Create multi-logger
        multi_logger = MultiLogger([failing_logger, working_logger])
        
        # Log metrics - should not raise exception
        metrics = {'loss': 0.5}
        multi_logger.log_metrics(metrics)
        
        # Working logger should still receive the call
        working_logger.log_metrics.assert_called_once_with(metrics, None)
        
        # Failing logger should have been called but failed
        failing_logger.log_metrics.assert_called_once_with(metrics, None)


if __name__ == '__main__':
    unittest.main()
