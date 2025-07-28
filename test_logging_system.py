#!/usr/bin/env python3
"""
Test script for the centralized logging system.
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logging.multi_logger import MultiLogger
from logging.wandb_logger import WandbLogger
from logging.tensorboard_logger import TensorBoardLogger
from logging.console_logger import ConsoleLogger
from config.config import create_default_config


def test_console_logger():
    """Test console logger functionality."""
    print("Testing Console Logger...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "test.log")
        logger = ConsoleLogger(log_file=log_file, level="INFO")
        
        logger.log_metric("test_metric", 0.5, step=1)
        logger.log_scalar("loss", 0.1, step=1)
        logger.log_hyperparameters({"lr": 0.001, "batch_size": 32})
        
        # Check log file exists
        assert os.path.exists(log_file), "Log file was not created"
        print("✓ Console logger works!")


def test_tensorboard_logger():
    """Test TensorBoard logger functionality."""
    print("Testing TensorBoard Logger...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = TensorBoardLogger(log_dir=tmp_dir)
        
        logger.log_metric("accuracy", 0.95, step=10)
        logger.log_scalar("loss", 0.05, step=10)
        logger.log_hyperparameters({"lr": 0.001})
        
        logger.close()
        print("✓ TensorBoard logger works!")


def test_multi_logger():
    """Test multi-logger functionality."""
    print("Testing Multi Logger...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create individual loggers
        console_logger = ConsoleLogger(
            log_file=os.path.join(tmp_dir, "multi_test.log"),
            level="INFO"
        )
        tensorboard_logger = TensorBoardLogger(
            log_dir=os.path.join(tmp_dir, "tensorboard")
        )
        
        # Create multi-logger
        multi_logger = MultiLogger([console_logger, tensorboard_logger])
        
        # Test logging
        multi_logger.log_metric("combined_metric", 0.8, step=5)
        multi_logger.log_scalar("combined_loss", 0.2, step=5)
        multi_logger.log_hyperparameters({"model": "jepa", "hidden_dim": 512})
        
        multi_logger.close()
        print("✓ Multi-logger works!")


def test_config_integration():
    """Test integration with configuration system."""
    print("Testing Config Integration...")
    
    # Create default config
    config = create_default_config("vision")
    
    # Verify logging structure
    assert hasattr(config, 'logging'), "Config missing logging section"
    assert hasattr(config.logging, 'wandb'), "Config missing wandb logging"
    assert hasattr(config.logging, 'tensorboard'), "Config missing tensorboard logging"
    assert hasattr(config.logging, 'console'), "Config missing console logging"
    
    print("✓ Config integration works!")


def main():
    """Run all tests."""
    print("🧪 Testing JEPA Centralized Logging System\n")
    
    try:
        test_console_logger()
        test_tensorboard_logger()
        test_multi_logger()
        test_config_integration()
        
        print("\n✅ All tests passed! Centralized logging system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
