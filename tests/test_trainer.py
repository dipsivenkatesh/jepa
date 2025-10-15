"""
Test suite for JEPA trainer components.

Tests the trainer classes, evaluation, and training utilities.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock, call
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.trainer.trainer import JEPATrainer
from jepa.trainer.eval import JEPAEvaluator
from jepa.trainer.utils import create_optimizer, create_scheduler, setup_training
from jepa.models.jepa import JEPA
from jepa.models.encoder import Encoder
from jepa.models.predictor import Predictor
from jepa.loggers.console_logger import ConsoleLogger


class TestJEPATrainer(unittest.TestCase):
    """Test cases for JEPATrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple model
        self.encoder = Encoder(64)
        self.predictor = Predictor(64)
        self.model = JEPA(self.encoder, self.predictor)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Create temporary save directory
        self.save_dir = tempfile.mkdtemp()
        
        # Create trainer
        self.trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            device="cpu",
            save_dir=self.save_dir,
            log_interval=1
        )
        
        # Create dummy dataset
        self.create_dummy_dataset()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.save_dir, ignore_errors=True)
    
    def create_dummy_dataset(self):
        """Create dummy dataset for testing."""
        # Create dummy data
        seq_len, batch_size, hidden_dim = 10, 4, 64
        data_t = torch.randn(20, seq_len, hidden_dim)  # 20 samples
        data_t1 = torch.randn(20, seq_len, hidden_dim)
        
        dataset = TensorDataset(data_t, data_t1)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Create smaller validation set
        val_data_t = torch.randn(8, seq_len, hidden_dim)
        val_data_t1 = torch.randn(8, seq_len, hidden_dim)
        val_dataset = TensorDataset(val_data_t, val_data_t1)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    def test_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertEqual(self.trainer.device, torch.device("cpu"))
        self.assertEqual(self.trainer.save_dir, self.save_dir)
        self.assertEqual(self.trainer.log_interval, 1)
        self.assertEqual(self.trainer.current_epoch, 0)
        self.assertEqual(self.trainer.global_step, 0)
        self.assertEqual(self.trainer.best_loss, float('inf'))
        
        # Check that save directory was created
        self.assertTrue(os.path.exists(self.save_dir))
    
    def test_device_auto_detection(self):
        """Test automatic device detection."""
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            device="auto"
        )
        
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(trainer.device, expected_device)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Train one epoch
        metrics = self.trainer.train_epoch(self.dataloader)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertIn('num_batches', metrics)
        self.assertGreater(metrics['loss'], 0)
        self.assertGreater(metrics['num_batches'], 0)
        
        # Check that parameters have changed
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.allclose(initial, current, atol=1e-6))
        
        # Check that global step was updated
        self.assertGreater(self.trainer.global_step, 0)
    
    def test_validate(self):
        """Test validation."""
        metrics = self.trainer.validate(self.val_dataloader)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('val_loss', metrics)
        self.assertIn('val_num_batches', metrics)
        self.assertGreater(metrics['val_loss'], 0)
        self.assertGreater(metrics['val_num_batches'], 0)
        
        # Model should be in eval mode during validation
        # But will be reset to original mode after
        self.assertFalse(self.model.training)  # Should be back to original mode
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        self.trainer.current_epoch = 5
        self.trainer.global_step = 100
        self.trainer.best_loss = 0.5
        
        checkpoint_path = self.trainer.save_checkpoint()
        
        # Check that checkpoint file was created
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Load checkpoint and verify contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        expected_keys = [
            'epoch', 'global_step', 'model_state_dict', 
            'optimizer_state_dict', 'best_loss'
        ]
        for key in expected_keys:
            self.assertIn(key, checkpoint)
        
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['global_step'], 100)
        self.assertEqual(checkpoint['best_loss'], 0.5)
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        # First save a checkpoint
        self.trainer.current_epoch = 3
        self.trainer.global_step = 50
        self.trainer.best_loss = 0.3
        checkpoint_path = self.trainer.save_checkpoint()
        
        # Create new trainer and load checkpoint
        new_trainer = JEPATrainer(
            model=JEPA(Encoder(64), Predictor(64)),
            optimizer=optim.Adam(self.model.parameters(), lr=1e-3),
            device="cpu"
        )
        
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check that state was loaded correctly
        self.assertEqual(new_trainer.current_epoch, 3)
        self.assertEqual(new_trainer.global_step, 50)
        self.assertEqual(new_trainer.best_loss, 0.3)
    
    def test_train_full_training_loop(self):
        """Test full training loop."""
        # Train for 2 epochs
        history = self.trainer.train(
            train_dataloader=self.dataloader,
            val_dataloader=self.val_dataloader,
            num_epochs=2
        )
        
        # Check that history is returned
        self.assertIsInstance(history, dict)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        
        # Check that we trained for 2 epochs
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        self.assertEqual(self.trainer.current_epoch, 2)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Create trainer with early stopping
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            device="cpu",
            save_dir=self.save_dir
        )
        
        # Mock validation that doesn't improve
        with patch.object(trainer, 'validate') as mock_validate:
            mock_validate.return_value = {'val_loss': 1.0, 'val_num_batches': 1}
            
            history = trainer.train(
                train_dataloader=self.dataloader,
                val_dataloader=self.val_dataloader,
                num_epochs=10,
                early_stopping_patience=2
            )
            
            # Should stop early due to no improvement
            self.assertLess(len(history['train_loss']), 10)
    
    def test_scheduler_integration(self):
        """Test learning rate scheduler integration."""
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            device="cpu"
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Train one epoch
        trainer.train(
            train_dataloader=self.dataloader,
            num_epochs=1
        )
        
        # Learning rate should have been updated
        current_lr = self.optimizer.param_groups[0]['lr']
        self.assertLess(current_lr, initial_lr)
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            gradient_clip_norm=1.0,
            device="cpu"
        )
        
        # Mock large gradients
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            trainer.train_epoch(self.dataloader)
            
            # Check that gradient clipping was called
            mock_clip.assert_called()
    
    def test_logger_integration(self):
        """Test logger integration."""
        logger = MagicMock()
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            device="cpu",
            logger=logger
        )
        
        # Train one epoch
        trainer.train_epoch(self.dataloader)
        
        # Check that logger methods were called
        logger.log_metrics.assert_called()
    
    def test_device_movement(self):
        """Test that data is moved to correct device."""
        # Test with CPU
        trainer = JEPATrainer(
            model=self.model,
            optimizer=self.optimizer,
            device="cpu"
        )
        
        # All model parameters should be on CPU
        for param in trainer.model.parameters():
            self.assertEqual(param.device, torch.device("cpu"))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            trainer_cuda = JEPATrainer(
                model=JEPA(Encoder(64), Predictor(64)),
                optimizer=optim.Adam(self.model.parameters(), lr=1e-3),
                device="cuda"
            )
            
            for param in trainer_cuda.model.parameters():
                self.assertTrue(param.device.type == "cuda")


class TestJEPAEvaluator(unittest.TestCase):
    """Test cases for JEPAEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create model
        encoder = Encoder(64)
        predictor = Predictor(64)
        self.model = JEPA(encoder, predictor)
        
        # Create evaluator
        self.evaluator = JEPAEvaluator(self.model, device="cpu")
        
        # Create dummy test data
        seq_len, batch_size, hidden_dim = 10, 4, 64
        test_data_t = torch.randn(16, seq_len, hidden_dim)
        test_data_t1 = torch.randn(16, seq_len, hidden_dim)
        test_dataset = TensorDataset(test_data_t, test_data_t1)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.model, self.model)
        self.assertEqual(self.evaluator.device, torch.device("cpu"))
    
    def test_evaluate(self):
        """Test model evaluation."""
        metrics = self.evaluator.evaluate(self.test_dataloader)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_loss', metrics)
        self.assertIn('test_num_samples', metrics)
        self.assertGreater(metrics['test_loss'], 0)
        self.assertGreater(metrics['test_num_samples'], 0)
    
    def test_evaluate_with_metrics(self):
        """Test evaluation with additional metrics."""
        def mse_metric(pred, target):
            return torch.mean((pred - target) ** 2)
        
        def mae_metric(pred, target):
            return torch.mean(torch.abs(pred - target))
        
        additional_metrics = {
            'mse': mse_metric,
            'mae': mae_metric
        }
        
        metrics = self.evaluator.evaluate(
            self.test_dataloader, 
            additional_metrics=additional_metrics
        )
        
        # Check that additional metrics are included
        self.assertIn('test_mse', metrics)
        self.assertIn('test_mae', metrics)
        self.assertGreater(metrics['test_mse'], 0)
        self.assertGreater(metrics['test_mae'], 0)
    
    def test_model_in_eval_mode(self):
        """Test that model is set to eval mode during evaluation."""
        # Set model to training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Evaluate
        self.evaluator.evaluate(self.test_dataloader)
        
        # Model should be in eval mode
        self.assertFalse(self.model.training)


class TestTrainerUtils(unittest.TestCase):
    """Test cases for trainer utility functions."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = nn.Linear(10, 5)
        
        # Test Adam optimizer
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type="adam",
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        
        self.assertIsInstance(optimizer, optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-3)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)
        
        # Test SGD optimizer
        optimizer_sgd = create_optimizer(
            model.parameters(),
            optimizer_type="sgd",
            learning_rate=1e-2,
            momentum=0.9
        )
        
        self.assertIsInstance(optimizer_sgd, optim.SGD)
        self.assertEqual(optimizer_sgd.param_groups[0]['lr'], 1e-2)
        self.assertEqual(optimizer_sgd.param_groups[0]['momentum'], 0.9)
        
        # Test AdamW optimizer
        optimizer_adamw = create_optimizer(
            model.parameters(),
            optimizer_type="adamw",
            learning_rate=1e-3
        )
        
        self.assertIsInstance(optimizer_adamw, optim.AdamW)
    
    def test_create_optimizer_invalid_type(self):
        """Test optimizer creation with invalid type."""
        model = nn.Linear(10, 5)
        
        with self.assertRaises(ValueError):
            create_optimizer(
                model.parameters(),
                optimizer_type="invalid_optimizer"
            )
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters())
        
        # Test StepLR scheduler
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="step",
            step_size=10,
            gamma=0.1
        )
        
        self.assertIsInstance(scheduler, optim.lr_scheduler.StepLR)
        
        # Test CosineAnnealingLR scheduler
        scheduler_cosine = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            T_max=100
        )
        
        self.assertIsInstance(scheduler_cosine, optim.lr_scheduler.CosineAnnealingLR)
        
        # Test ReduceLROnPlateau scheduler
        scheduler_plateau = create_scheduler(
            optimizer,
            scheduler_type="plateau",
            patience=10,
            factor=0.5
        )
        
        self.assertIsInstance(scheduler_plateau, optim.lr_scheduler.ReduceLROnPlateau)
    
    def test_create_scheduler_invalid_type(self):
        """Test scheduler creation with invalid type."""
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters())
        
        with self.assertRaises(ValueError):
            create_scheduler(optimizer, scheduler_type="invalid_scheduler")
    
    def test_setup_training(self):
        """Test training setup utility."""
        # Mock config
        config = {
            'model': {
                'encoder_dim': 64,
                'predictor_hidden_dim': 128
            },
            'training': {
                'learning_rate': 1e-3,
                'optimizer': 'adam'
            },
            'device': 'cpu'
        }
        
        device, model, optimizer = setup_training(config)
        
        # Check that correct objects are returned
        self.assertEqual(device, torch.device('cpu'))
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(optimizer, optim.Optimizer)


class TestTrainerIntegration(unittest.TestCase):
    """Integration tests for trainer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.save_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.save_dir, ignore_errors=True)
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        # Create model
        encoder = Encoder(32)  # Smaller for faster testing
        predictor = Predictor(32)
        model = JEPA(encoder, predictor)
        
        # Create optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Create trainer
        trainer = JEPATrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            save_dir=self.save_dir,
            log_interval=1
        )
        
        # Create dummy data
        seq_len, batch_size, hidden_dim = 5, 2, 32
        train_data_t = torch.randn(8, seq_len, hidden_dim)
        train_data_t1 = torch.randn(8, seq_len, hidden_dim)
        train_dataset = TensorDataset(train_data_t, train_data_t1)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        
        val_data_t = torch.randn(4, seq_len, hidden_dim)
        val_data_t1 = torch.randn(4, seq_len, hidden_dim)
        val_dataset = TensorDataset(val_data_t, val_data_t1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2,
            save_every=1
        )
        
        # Check training results
        self.assertIsInstance(history, dict)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        
        # Check that checkpoints were saved
        checkpoint_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        self.assertGreater(len(checkpoint_files), 0)
        
        # Test evaluation
        evaluator = JEPAEvaluator(model, device="cpu")
        test_metrics = evaluator.evaluate(val_dataloader)
        
        self.assertIn('test_loss', test_metrics)
        self.assertGreater(test_metrics['test_loss'], 0)
    
    def test_resume_training(self):
        """Test resuming training from checkpoint."""
        # Create and train initial model
        model = JEPA(Encoder(32), Predictor(32))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = JEPATrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            save_dir=self.save_dir
        )
        
        # Create dummy data
        seq_len, batch_size, hidden_dim = 5, 2, 32
        data_t = torch.randn(4, seq_len, hidden_dim)
        data_t1 = torch.randn(4, seq_len, hidden_dim)
        dataset = TensorDataset(data_t, data_t1)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Train for 1 epoch and save
        trainer.train(dataloader, num_epochs=1, save_every=1)
        initial_epoch = trainer.current_epoch
        
        # Create new trainer and resume
        new_model = JEPA(Encoder(32), Predictor(32))
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
        new_trainer = JEPATrainer(
            model=new_model,
            optimizer=new_optimizer,
            device="cpu",
            save_dir=self.save_dir
        )
        
        # Find and load checkpoint
        checkpoint_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        self.assertGreater(len(checkpoint_files), 0)
        
        checkpoint_path = os.path.join(self.save_dir, checkpoint_files[0])
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check that state was restored
        self.assertEqual(new_trainer.current_epoch, initial_epoch)


if __name__ == '__main__':
    unittest.main()
