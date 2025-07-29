"""
Evaluation utilities for JEPA models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import numpy as np


class JEPAEvaluator:
    """
    Evaluator for JEPA models with various metrics and analysis tools.
    """
    
    def __init__(self, model: nn.Module, device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            model: JEPA model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        mse_losses = []
        prediction_norms = []
        target_norms = []
        cosine_similarities = []
        
        num_samples = 0
        
        with torch.no_grad():
            for state_t, state_t1 in dataloader:
                state_t = state_t.to(self.device)
                state_t1 = state_t1.to(self.device)
                
                # Forward pass
                prediction, target = self.model(state_t, state_t1)
                loss = self.model.loss(prediction, target)
                
                # Accumulate metrics
                batch_size = prediction.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
                # Per-sample MSE
                mse_per_sample = torch.mean((prediction - target) ** 2, dim=-1)
                mse_losses.extend(mse_per_sample.cpu().numpy())
                
                # Norms
                pred_norms = torch.norm(prediction, dim=-1)
                target_norms = torch.norm(target, dim=-1)
                prediction_norms.extend(pred_norms.cpu().numpy())
                target_norms.extend(target_norms.cpu().numpy())
                
                # Cosine similarity
                cos_sim = torch.cosine_similarity(prediction, target, dim=-1)
                cosine_similarities.extend(cos_sim.cpu().numpy())
        
        # Compute final metrics
        avg_loss = total_loss / num_samples
        mse_losses = np.array(mse_losses)
        prediction_norms = np.array(prediction_norms)
        target_norms = np.array(target_norms)
        cosine_similarities = np.array(cosine_similarities)
        
        metrics = {
            "loss": avg_loss,
            "mse_mean": np.mean(mse_losses),
            "mse_std": np.std(mse_losses),
            "prediction_norm_mean": np.mean(prediction_norms),
            "prediction_norm_std": np.std(prediction_norms),
            "target_norm_mean": np.mean(target_norms),
            "target_norm_std": np.std(target_norms),
            "cosine_similarity_mean": np.mean(cosine_similarities),
            "cosine_similarity_std": np.std(cosine_similarities),
            "num_samples": num_samples
        }
        
        return metrics
    
    def get_representations(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Extract encoder representations from data.
        
        Args:
            dataloader: DataLoader with data
            
        Returns:
            Dictionary with representations and targets
        """
        self.model.eval()
        
        representations_t = []
        representations_t1 = []
        predictions = []
        
        with torch.no_grad():
            for state_t, state_t1 in dataloader:
                state_t = state_t.to(self.device)
                state_t1 = state_t1.to(self.device)
                
                # Get representations
                z_t = self.model.encoder(state_t)
                z_t1 = self.model.encoder(state_t1)
                pred = self.model.predictor(z_t)
                
                representations_t.append(z_t.cpu().numpy())
                representations_t1.append(z_t1.cpu().numpy())
                predictions.append(pred.cpu().numpy())
        
        return {
            "representations_t": np.concatenate(representations_t, axis=0),
            "representations_t1": np.concatenate(representations_t1, axis=0),
            "predictions": np.concatenate(predictions, axis=0)
        }
    
    def representation_analysis(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Analyze the quality of learned representations.
        
        Args:
            dataloader: DataLoader with data
            
        Returns:
            Dictionary with analysis results
        """
        data = self.get_representations(dataloader)
        
        reps_t = data["representations_t"]
        reps_t1 = data["representations_t1"]
        predictions = data["predictions"]
        
        # Representation statistics
        analysis = {
            "representation_dim": reps_t.shape[-1],
            "num_samples": reps_t.shape[0],
            
            # Mean and std of representations
            "rep_t_mean": np.mean(reps_t, axis=0),
            "rep_t_std": np.std(reps_t, axis=0),
            "rep_t1_mean": np.mean(reps_t1, axis=0),
            "rep_t1_std": np.std(reps_t1, axis=0),
            
            # Prediction quality
            "prediction_error": np.mean(np.linalg.norm(predictions - reps_t1, axis=-1)),
            
            # Representation diversity (average pairwise distance)
            "rep_diversity_t": self._compute_diversity(reps_t),
            "rep_diversity_t1": self._compute_diversity(reps_t1),
        }
        
        return analysis
    
    def _compute_diversity(self, representations: np.ndarray, max_samples: int = 1000) -> float:
        """
        Compute average pairwise distance as a measure of representation diversity.
        
        Args:
            representations: Array of representations [N, D]
            max_samples: Maximum number of samples to use for efficiency
            
        Returns:
            Average pairwise distance
        """
        n_samples = min(representations.shape[0], max_samples)
        indices = np.random.choice(representations.shape[0], n_samples, replace=False)
        sample_reps = representations[indices]
        
        # Compute pairwise distances
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(sample_reps[i] - sample_reps[j])
                distances.append(dist)
        
        return np.mean(distances)


def quick_evaluate(model: nn.Module, dataloader: DataLoader, device: str = "auto") -> Dict[str, float]:
    """
    Quick evaluation function for convenience.
    
    Args:
        model: JEPA model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run on
        
    Returns:
        Dictionary with basic evaluation metrics
    """
    evaluator = JEPAEvaluator(model, device)
    return evaluator.evaluate(dataloader)
