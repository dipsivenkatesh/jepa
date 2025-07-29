"""
Simple working test to demonstrate the test framework.

This test only tests basic Python functionality without requiring external dependencies.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ModelConfig, TrainingConfig, DataConfig


class TestBasicConfig(unittest.TestCase):
    """Test basic configuration classes that don't require external dependencies."""
    
    def test_model_config_creation(self):
        """Test ModelConfig can be created with defaults."""
        config = ModelConfig()
        
        self.assertEqual(config.encoder_type, "transformer")
        self.assertEqual(config.encoder_dim, 512)
        self.assertEqual(config.predictor_type, "mlp")
        self.assertEqual(config.dropout, 0.1)
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            encoder_type="cnn",
            encoder_dim=256,
            dropout=0.2
        )
        
        self.assertEqual(config.encoder_type, "cnn")
        self.assertEqual(config.encoder_dim, 256)
        self.assertEqual(config.dropout, 0.2)
    
    def test_training_config_creation(self):
        """Test TrainingConfig can be created with defaults."""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.num_epochs, 100)
    
    def test_data_config_creation(self):
        """Test DataConfig can be created with defaults."""
        config = DataConfig()
        
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.sequence_length, 10)


class TestPythonBasics(unittest.TestCase):
    """Test basic Python functionality."""
    
    def test_python_version(self):
        """Test Python version is 3.8+."""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        if version.major == 3:
            self.assertGreaterEqual(version.minor, 8)
    
    def test_imports(self):
        """Test basic imports work."""
        import json
        import yaml
        import os
        import sys
        
        # Basic functionality
        self.assertTrue(hasattr(json, 'dumps'))
        self.assertTrue(hasattr(yaml, 'safe_load'))
        self.assertTrue(hasattr(os, 'path'))
        self.assertTrue(hasattr(sys, 'version'))
    
    def test_file_operations(self):
        """Test basic file operations."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_path = f.name
        
        try:
            # Read the file
            with open(temp_path, 'r') as f:
                content = f.read()
            
            self.assertEqual(content, 'test content')
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
