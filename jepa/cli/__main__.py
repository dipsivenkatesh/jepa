#!/usr/bin/env python3
"""
JEPA CLI - Main entry point for JEPA training and evaluation.

Usage:
    python -m jepa.cli train --config config.yaml --train-data data/train.npy
    python -m jepa.cli evaluate --model-path checkpoints/best_model.pt --test-data data/test.npy
    python -m jepa.cli generate-config --output config.yaml
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JEPA - Joint Embedding Predictive Architecture CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Train a JEPA model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained JEPA model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Generate config subcommand
    config_parser = subparsers.add_parser(
        'generate-config',
        help='Generate a default configuration file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments for train subcommand
    train_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    train_parser.add_argument(
        '--train-data',
        type=str,
        help='Path to training data'
    )
    train_parser.add_argument(
        '--val-data',
        type=str,
        help='Path to validation data'
    )
    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for training'
    )
    train_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory to save checkpoints'
    )
    train_parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name of the experiment'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    train_parser.add_argument(
        '--generate-config',
        type=str,
        help='Generate a default configuration file at specified path'
    )
    
    # Add arguments for evaluate subcommand
    eval_parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    eval_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    eval_parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data'
    )
    eval_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    eval_parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to run evaluation on'
    )
    eval_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./eval_results',
        help='Output directory for evaluation results'
    )
    eval_parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions to file'
    )
    eval_parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    eval_parser.add_argument(
        '--compute-embeddings',
        action='store_true',
        help='Compute and save embeddings for all test samples'
    )
    eval_parser.add_argument(
        '--analyze-latent',
        action='store_true',
        help='Perform latent space analysis'
    )
    
    # Add arguments for generate-config subcommand
    config_parser.add_argument(
        '--output', '-o',
        type=str,
        default='config.yaml',
        help='Output path for configuration file'
    )
    config_parser.add_argument(
        '--template',
        type=str,
        choices=['default', 'vision', 'nlp', 'timeseries'],
        default='default',
        help='Configuration template to use'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Import and run the appropriate command
    if args.command == 'train':
        # Convert args to format expected by train.py
        import cli.train as train_module
        
        # Override sys.argv to pass arguments to train script
        train_args = ['train.py']
        if args.config:
            train_args.extend(['--config', args.config])
        if args.train_data:
            train_args.extend(['--train-data', args.train_data])
        if args.val_data:
            train_args.extend(['--val-data', args.val_data])
        if args.batch_size:
            train_args.extend(['--batch-size', str(args.batch_size)])
        if args.learning_rate:
            train_args.extend(['--learning-rate', str(args.learning_rate)])
        if args.num_epochs:
            train_args.extend(['--num-epochs', str(args.num_epochs)])
        if args.device:
            train_args.extend(['--device', args.device])
        if args.output_dir:
            train_args.extend(['--output-dir', args.output_dir])
        if args.checkpoint_dir:
            train_args.extend(['--checkpoint-dir', args.checkpoint_dir])
        if args.experiment_name:
            train_args.extend(['--experiment-name', args.experiment_name])
        if args.resume:
            train_args.extend(['--resume', args.resume])
        if args.generate_config:
            train_args.extend(['--generate-config', args.generate_config])
        
        # Temporarily override sys.argv
        original_argv = sys.argv
        sys.argv = train_args
        try:
            train_module.main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'evaluate':
        import cli.evaluate as eval_module
        
        # Convert args to format expected by evaluate.py
        eval_args = ['evaluate.py', '--model-path', args.model_path]
        if args.config:
            eval_args.extend(['--config', args.config])
        if args.test_data:
            eval_args.extend(['--test-data', args.test_data])
        eval_args.extend(['--batch-size', str(args.batch_size)])
        eval_args.extend(['--device', args.device])
        eval_args.extend(['--output-dir', args.output_dir])
        if args.save_predictions:
            eval_args.append('--save-predictions')
        if args.visualize:
            eval_args.append('--visualize')
        if args.compute_embeddings:
            eval_args.append('--compute-embeddings')
        if args.analyze_latent:
            eval_args.append('--analyze-latent')
        
        # Temporarily override sys.argv
        original_argv = sys.argv
        sys.argv = eval_args
        try:
            eval_module.main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'generate-config':
        from ..config.config import save_config, create_default_config
        
        print(f"Generating {args.template} configuration template...")
        
        config = create_default_config()
        
        # Modify config based on template
        if args.template == 'vision':
            config.model.encoder_type = "cnn"
            config.model.encoder_dim = 256
            config.data.input_dim = [3, 224, 224]  # RGB images
            config.training.batch_size = 64
            config.experiment_name = "vision_jepa"
        
        elif args.template == 'nlp':
            config.model.encoder_type = "transformer"
            config.model.encoder_dim = 768
            config.model.predictor_type = "transformer"
            config.data.input_dim = 50000  # Vocabulary size
            config.data.sequence_length = 512
            config.training.batch_size = 16
            config.training.learning_rate = 0.0001
            config.experiment_name = "nlp_jepa"
        
        elif args.template == 'timeseries':
            config.model.encoder_type = "transformer"
            config.model.encoder_dim = 128
            config.data.input_dim = 32  # Number of features
            config.data.sequence_length = 100
            config.training.batch_size = 128
            config.training.num_epochs = 200
            config.model.dropout = 0.2
            config.experiment_name = "timeseries_jepa"
        
        save_config(config, args.output)
        print(f"Configuration saved to: {args.output}")
        print(f"Edit the file to customize for your specific use case.")


if __name__ == "__main__":
    main()
