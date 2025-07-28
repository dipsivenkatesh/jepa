#!/usr/bin/env python3
"""
Example usage of JEPA CLI

This script demonstrates how to use the JEPA CLI for training and evaluation.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR:", e.stderr)
        return False


def main():
    """Demonstrate JEPA CLI usage."""
    
    # Step 1: Generate a default configuration
    print("Step 1: Generate default configuration")
    run_command([
        sys.executable, "-m", "jepa.cli", "generate-config",
        "--output", "example_config.yaml",
        "--template", "default"
    ], "Generate default configuration")
    
    # Step 2: Show training command (without actually running)
    print("\n" + "="*50)
    print("Step 2: Training command example")
    print("="*50)
    print("To train a model, you would run:")
    print("python -m jepa.cli train \\")
    print("    --config config/default_config.yaml \\")
    print("    --train-data /path/to/train_data.npy \\")
    print("    --val-data /path/to/val_data.npy \\")
    print("    --experiment-name my_experiment \\")
    print("    --num-epochs 50")
    
    # Step 3: Show evaluation command (without actually running)
    print("\n" + "="*50)
    print("Step 3: Evaluation command example")
    print("="*50)
    print("To evaluate a trained model, you would run:")
    print("python -m jepa.cli evaluate \\")
    print("    --model-path checkpoints/my_experiment/best_model.pt \\")
    print("    --test-data /path/to/test_data.npy \\")
    print("    --visualize \\")
    print("    --compute-embeddings \\")
    print("    --analyze-latent")
    
    # Step 4: Show other available templates
    print("\n" + "="*50)
    print("Step 4: Other configuration templates")
    print("="*50)
    
    templates = ["vision", "nlp", "timeseries"]
    for template in templates:
        print(f"\nGenerating {template} template:")
        run_command([
            sys.executable, "-m", "jepa.cli", "generate-config",
            "--output", f"config/{template}_config.yaml",
            "--template", template
        ], f"Generate {template} configuration")
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50)
    print("Configuration files generated:")
    for config_file in ["example_config.yaml", "config/vision_config.yaml", "config/nlp_config.yaml", "config/timeseries_config.yaml"]:
        if os.path.exists(config_file):
            print(f"  - {config_file}")
    
    print("\nNext steps:")
    print("1. Edit the configuration file for your specific use case")
    print("2. Prepare your data in the expected format")
    print("3. Run training with the CLI")
    print("4. Evaluate your trained model")


if __name__ == "__main__":
    main()
