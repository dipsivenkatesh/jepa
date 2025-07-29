"""
JEPA Evaluation CLI

Command-line interface for evaluating trained JEPA models.
"""

import argparse
import os
import sys
import torch
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

# Package imports - use relative imports for proper package structure
from ..config.config import load_config
from ..models.jepa import JEPA
from ..models.encoder import Encoder
from ..models.predictor import Predictor
from ..trainer.trainer import JEPATrainer
from ..data.dataset import create_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Config and data
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file (if not provided, will look for config.yaml in model directory)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data (if not provided, will use test_data_path from config)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to run evaluation on'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./eval_results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions to file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    # Analysis options
    parser.add_argument(
        '--compute-embeddings',
        action='store_true',
        help='Compute and save embeddings for all test samples'
    )
    parser.add_argument(
        '--analyze-latent',
        action='store_true',
        help='Perform latent space analysis'
    )
    
    return parser.parse_args()


def load_model_and_config(model_path: str, config_path: Optional[str] = None):
    """Load model and configuration."""
    # Try to find config file if not provided
    if config_path is None:
        model_dir = os.path.dirname(model_path)
        potential_configs = [
            os.path.join(model_dir, "config.yaml"),
            os.path.join(os.path.dirname(model_dir), "config.yaml")
        ]
        
        config_path = None
        for cfg_path in potential_configs:
            if os.path.exists(cfg_path):
                config_path = cfg_path
                break
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        config = create_default_config()
        print("Using default configuration (no config file found)")
    
    # Create model
    encoder = create_encoder(
        encoder_type=config.model.encoder_type,
        input_dim=config.data.input_dim,
        hidden_dim=config.model.encoder_dim,
        dropout=config.model.dropout
    )
    
    predictor = create_predictor(
        predictor_type=config.model.predictor_type,
        input_dim=config.model.encoder_dim,
        hidden_dim=config.model.predictor_hidden_dim,
        output_dim=config.model.predictor_output_dim,
        dropout=config.model.dropout
    )
    
    model = JEPA(encoder=encoder, predictor=predictor)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    return model, config


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    embeddings = []
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (state_t, state_t1) in enumerate(dataloader):
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            # Forward pass
            prediction, target = model(state_t, state_t1)
            loss = model.loss(prediction, target)
            
            # Collect results
            total_loss += loss.item() * state_t.size(0)
            total_samples += state_t.size(0)
            
            # Store predictions and targets
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            
            # Store embeddings (encoded states)
            z_t = model.encoder(state_t)
            embeddings.append(z_t.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / total_samples
    
    # Concatenate all results
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    embeddings = np.concatenate(embeddings, axis=0)
    
    return {
        'loss': avg_loss,
        'predictions': predictions,
        'targets': targets,
        'embeddings': embeddings,
        'num_samples': total_samples
    }


def compute_metrics(predictions, targets):
    """Compute evaluation metrics."""
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Cosine similarity (if applicable)
    cos_sim = np.mean([
        np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t))
        for p, t in zip(predictions, targets)
        if np.linalg.norm(p) > 0 and np.linalg.norm(t) > 0
    ])
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'cosine_similarity': cos_sim
    }


def analyze_latent_space(embeddings, output_dir):
    """Analyze the latent space representation."""
    print("Analyzing latent space...")
    
    # Basic statistics
    stats = {
        'mean': np.mean(embeddings, axis=0).tolist(),
        'std': np.std(embeddings, axis=0).tolist(),
        'min': np.min(embeddings, axis=0).tolist(),
        'max': np.max(embeddings, axis=0).tolist()
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'latent_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # PCA analysis if dimensionality is high
    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Save PCA results
        np.save(os.path.join(output_dir, 'embeddings_pca.npy'), embeddings_pca)
        np.save(os.path.join(output_dir, 'pca_components.npy'), pca.components_)
        
        pca_stats = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        with open(os.path.join(output_dir, 'pca_stats.json'), 'w') as f:
            json.dump(pca_stats, f, indent=2)
    
    return stats


def create_visualizations(results, output_dir):
    """Create visualization plots."""
    print("Creating visualizations...")
    
    predictions = results['predictions']
    targets = results['targets']
    embeddings = results['embeddings']
    
    # Prediction vs Target scatter plot
    plt.figure(figsize=(10, 8))
    
    # Flatten arrays for plotting
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Sample points if too many
    if len(pred_flat) > 10000:
        idx = np.random.choice(len(pred_flat), 10000, replace=False)
        pred_flat = pred_flat[idx]
        target_flat = target_flat[idx]
    
    plt.subplot(2, 2, 1)
    plt.scatter(target_flat, pred_flat, alpha=0.5)
    plt.plot([target_flat.min(), target_flat.max()], [target_flat.min(), target_flat.max()], 'r--')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title('Predictions vs Targets')
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = pred_flat - target_flat
    plt.scatter(target_flat, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Target')
    plt.ylabel('Residual')
    plt.title('Residuals')
    
    # Embedding distribution (first dimension)
    plt.subplot(2, 2, 3)
    plt.hist(embeddings[:, 0], bins=50, alpha=0.7)
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.title('Embedding Distribution (Dim 0)')
    
    # Loss histogram
    plt.subplot(2, 2, 4)
    losses = np.mean((predictions - targets) ** 2, axis=tuple(range(1, predictions.ndim)))
    plt.hist(losses, bins=50, alpha=0.7)
    plt.xlabel('Sample Loss')
    plt.ylabel('Frequency')
    plt.title('Per-Sample Loss Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Embedding visualization (if PCA was performed)
    pca_file = os.path.join(output_dir, 'embeddings_pca.npy')
    if os.path.exists(pca_file):
        embeddings_pca = np.load(pca_file)
        
        plt.figure(figsize=(12, 4))
        
        # 2D PCA plot
        plt.subplot(1, 3, 1)
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Embeddings in PC Space')
        
        # Explained variance
        with open(os.path.join(output_dir, 'pca_stats.json'), 'r') as f:
            pca_stats = json.load(f)
        
        plt.subplot(1, 3, 2)
        plt.plot(pca_stats['explained_variance_ratio'][:20])
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        
        plt.subplot(1, 3, 3)
        plt.plot(pca_stats['cumulative_variance_ratio'][:20])
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Ratio')
        plt.title('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and configuration
    print("Loading model and configuration...")
    model, config = load_model_and_config(args.model_path, args.config)
    
    # Override test data path if provided
    test_data_path = args.test_data or config.data.test_data_path
    if not test_data_path:
        raise ValueError("Test data path must be provided via --test-data or config file")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    print(f"Test data: {test_data_path}")
    
    # Create data loader
    print("Creating data loader...")
    test_loader = create_dataset(
        data_path=test_data_path,
        batch_size=args.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle=False
    )
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results['predictions'], results['targets'])
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average Loss: {results['loss']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print("="*50)
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    
    # Save metrics
    eval_results = {
        'model_path': args.model_path,
        'test_data_path': test_data_path,
        'num_samples': results['num_samples'],
        'loss': results['loss'],
        'metrics': metrics,
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
    }
    
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    # Save predictions if requested
    if args.save_predictions:
        np.save(os.path.join(args.output_dir, 'predictions.npy'), results['predictions'])
        np.save(os.path.join(args.output_dir, 'targets.npy'), results['targets'])
        print("Predictions saved")
    
    # Save embeddings if requested
    if args.compute_embeddings:
        np.save(os.path.join(args.output_dir, 'embeddings.npy'), results['embeddings'])
        print("Embeddings saved")
    
    # Analyze latent space if requested
    if args.analyze_latent:
        analyze_latent_space(results['embeddings'], args.output_dir)
        print("Latent space analysis completed")
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(results, args.output_dir)
        print("Visualizations created")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
