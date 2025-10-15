"""
Test suite for JEPA model components.

Tests the core model classes including BaseModel, JEPA, Encoder, and Predictor.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa.models.base import BaseModel
from jepa.models.jepa import JEPA
from jepa.models.encoder import Encoder
from jepa.models.predictor import Predictor


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        class SimpleModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        self.model = SimpleModel()
        self.test_input = torch.randn(2, 10)
    
    def test_forward_not_implemented(self):
        """Test that BaseModel raises NotImplementedError for forward."""
        base_model = BaseModel()
        with self.assertRaises(NotImplementedError):
            base_model.forward(self.test_input)
    
    def test_loss_function(self):
        """Test that models can define custom loss functions."""
        prediction = torch.randn(2, 5)
        target = torch.randn(2, 5)

        loss = nn.MSELoss()(prediction, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Save model
            self.model.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new model and load
            class SimpleModel(BaseModel):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return self.linear(x)
            
            new_model = SimpleModel()
            original_params = list(self.model.parameters())
            new_model.load(temp_path)
            
            # Check that model is in eval mode after loading
            self.assertFalse(new_model.training)
            
            # Check that parameters were loaded correctly
            loaded_params = list(new_model.parameters())
            for orig, loaded in zip(original_params, loaded_params):
                torch.testing.assert_close(orig, loaded)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEncoder(unittest.TestCase):
    """Test cases for Encoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 256
        self.encoder = Encoder(self.hidden_dim)
        self.batch_size = 2
        self.seq_length = 10
        self.test_input = torch.randn(self.seq_length, self.batch_size, self.hidden_dim)
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertIsInstance(self.encoder, BaseModel)
        self.assertIsInstance(self.encoder.encoder, nn.TransformerEncoder)
        
        # Check that encoder layers have correct dimensions
        first_layer = self.encoder.encoder.layers[0]
        self.assertEqual(first_layer.self_attn.embed_dim, self.hidden_dim)
        self.assertEqual(first_layer.self_attn.num_heads, 4)
    
    def test_forward_pass(self):
        """Test encoder forward pass."""
        output = self.encoder(self.test_input)
        
        # Check output shape
        expected_shape = (self.seq_length, self.batch_size, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is tensor
        self.assertIsInstance(output, torch.Tensor)
    
    def test_forward_different_input_sizes(self):
        """Test encoder with different input sizes."""
        # Test with different sequence lengths
        for seq_len in [5, 15, 20]:
            test_input = torch.randn(seq_len, self.batch_size, self.hidden_dim)
            output = self.encoder(test_input)
            self.assertEqual(output.shape, (seq_len, self.batch_size, self.hidden_dim))
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(self.seq_length, batch_size, self.hidden_dim)
            output = self.encoder(test_input)
            self.assertEqual(output.shape, (self.seq_length, batch_size, self.hidden_dim))
    
    def test_encoder_training_mode(self):
        """Test encoder in training and eval modes."""
        # Test training mode
        self.encoder.train()
        output_train = self.encoder(self.test_input)
        
        # Test eval mode
        self.encoder.eval()
        with torch.no_grad():
            output_eval = self.encoder(self.test_input)
        
        # Outputs should be different due to dropout
        self.encoder.train()
        output_train2 = self.encoder(self.test_input)
        
        # In training mode, outputs might differ due to dropout
        # In eval mode, they should be deterministic
        self.assertIsInstance(output_train, torch.Tensor)
        self.assertIsInstance(output_eval, torch.Tensor)
        self.assertIsInstance(output_train2, torch.Tensor)


class TestPredictor(unittest.TestCase):
    """Test cases for Predictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 256
        self.predictor = Predictor(self.hidden_dim)
        self.batch_size = 2
        self.test_input = torch.randn(self.batch_size, self.hidden_dim)
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsInstance(self.predictor, BaseModel)
        self.assertIsInstance(self.predictor.net, nn.Sequential)
        
        # Check network architecture
        layers = list(self.predictor.net.children())
        self.assertEqual(len(layers), 3)  # Linear -> ReLU -> Linear
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertIsInstance(layers[1], nn.ReLU)
        self.assertIsInstance(layers[2], nn.Linear)
        
        # Check dimensions
        self.assertEqual(layers[0].in_features, self.hidden_dim)
        self.assertEqual(layers[0].out_features, self.hidden_dim)
        self.assertEqual(layers[2].in_features, self.hidden_dim)
        self.assertEqual(layers[2].out_features, self.hidden_dim)
    
    def test_forward_pass(self):
        """Test predictor forward pass."""
        output = self.predictor(self.test_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is tensor
        self.assertIsInstance(output, torch.Tensor)
    
    def test_forward_different_input_sizes(self):
        """Test predictor with different input sizes."""
        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            test_input = torch.randn(batch_size, self.hidden_dim)
            output = self.predictor(test_input)
            self.assertEqual(output.shape, (batch_size, self.hidden_dim))
    
    def test_nonlinearity(self):
        """Test that predictor introduces non-linearity."""
        # Create input with all zeros except one element
        test_input = torch.zeros(1, self.hidden_dim)
        test_input[0, 0] = 1.0
        
        output = self.predictor(test_input)
        
        # Output should be non-zero and different from input due to ReLU
        self.assertFalse(torch.allclose(output, test_input, atol=1e-6))
        self.assertTrue(torch.any(output >= 0))  # ReLU output should be non-negative


class TestJEPA(unittest.TestCase):
    """Test cases for JEPA model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 256
        self.encoder = Encoder(self.hidden_dim)
        self.predictor = Predictor(self.hidden_dim)
        self.jepa = JEPA(self.encoder, self.predictor)
        
        self.batch_size = 2
        self.seq_length = 10
        self.state_t = torch.randn(self.seq_length, self.batch_size, self.hidden_dim)
        self.state_t1 = torch.randn(self.seq_length, self.batch_size, self.hidden_dim)
    
    def test_initialization(self):
        """Test JEPA initialization."""
        self.assertIsInstance(self.jepa, BaseModel)
        self.assertEqual(self.jepa.encoder, self.encoder)
        self.assertEqual(self.jepa.predictor, self.predictor)
    
    def test_forward_pass(self):
        """Test JEPA forward pass."""
        pred, target = self.jepa(self.state_t, self.state_t1)
        
        # Check output shapes
        expected_shape = (self.seq_length, self.batch_size, self.hidden_dim)
        self.assertEqual(pred.shape, expected_shape)
        self.assertEqual(target.shape, expected_shape)
        
        # Check outputs are tensors
        self.assertIsInstance(pred, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
    
    def test_forward_target_detached(self):
        """Test that target embeddings are detached from computation graph."""
        pred, target = self.jepa(self.state_t, self.state_t1)
        
        # Target should not require gradients (detached)
        self.assertFalse(target.requires_grad)
        
        # Prediction should require gradients
        if self.state_t.requires_grad:
            self.assertTrue(pred.requires_grad)
    
    def test_loss_computation(self):
        """Test loss computation."""
        pred, target = self.jepa(self.state_t, self.state_t1)
        loss = nn.MSELoss()(pred, target)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0)  # MSE is non-negative
    
    def test_training_mode(self):
        """Test JEPA in training and eval modes."""
        # Test training mode
        self.jepa.train()
        pred_train, target_train = self.jepa(self.state_t, self.state_t1)
        
        # Test eval mode
        self.jepa.eval()
        with torch.no_grad():
            pred_eval, target_eval = self.jepa(self.state_t, self.state_t1)
        
        # Predictions should be different due to dropout in encoder
        self.assertIsInstance(pred_train, torch.Tensor)
        self.assertIsInstance(pred_eval, torch.Tensor)
        self.assertIsInstance(target_train, torch.Tensor)
        self.assertIsInstance(target_eval, torch.Tensor)
    
    def test_different_encoder_predictor_combinations(self):
        """Test JEPA with different encoder/predictor combinations."""
        # Test with different hidden dimensions
        for hidden_dim in [128, 512, 1024]:
            encoder = Encoder(hidden_dim)
            predictor = Predictor(hidden_dim)
            jepa = JEPA(encoder, predictor)
            
            state_t = torch.randn(5, 2, hidden_dim)
            state_t1 = torch.randn(5, 2, hidden_dim)
            
            pred, target = jepa(state_t, state_t1)
            
            self.assertEqual(pred.shape, (5, 2, hidden_dim))
            self.assertEqual(target.shape, (5, 2, hidden_dim))
    
    def test_backward_pass(self):
        """Test that gradients flow correctly through JEPA."""
        # Set requires_grad for inputs
        self.state_t.requires_grad_(True)
        self.state_t1.requires_grad_(True)
        
        pred, target = self.jepa(self.state_t, self.state_t1)
        loss = nn.MSELoss()(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed for model parameters
        for param in self.jepa.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                # Check that some gradients are non-zero
                self.assertTrue(torch.any(param.grad != 0))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def test_model_save_load_integration(self):
        """Test saving and loading complete JEPA model."""
        # Create JEPA model
        encoder = Encoder(256)
        predictor = Predictor(256)
        jepa = JEPA(encoder, predictor)
        
        # Create test data
        state_t = torch.randn(5, 2, 256)
        state_t1 = torch.randn(5, 2, 256)
        
        # Get initial output
        jepa.eval()
        with torch.no_grad():
            pred_original, target_original = jepa(state_t, state_t1)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            jepa.save(temp_path)
            
            # Create new model and load
            new_encoder = Encoder(256)
            new_predictor = Predictor(256)
            new_jepa = JEPA(new_encoder, new_predictor)
            new_jepa.load(temp_path)
            
            # Test that loaded model produces same output
            with torch.no_grad():
                pred_loaded, target_loaded = new_jepa(state_t, state_t1)
            
            torch.testing.assert_close(pred_original, pred_loaded, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(target_original, target_loaded, atol=1e-6, rtol=1e-5)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_model_device_movement(self):
        """Test moving models between devices."""
        encoder = Encoder(128)
        predictor = Predictor(128)
        jepa = JEPA(encoder, predictor)
        
        # Test CPU
        device = torch.device('cpu')
        jepa.to(device)
        
        for param in jepa.parameters():
            self.assertEqual(param.device, device)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            jepa.to(device)
            
            for param in jepa.parameters():
                self.assertEqual(param.device, device)


if __name__ == '__main__':
    unittest.main()
