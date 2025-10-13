"""
Evaluation utilities for JEPA models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Callable
import numpy as np

from ..loss_functions import mse_loss

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class JEPAEvaluator:
    """
    Evaluator for JEPA models with various metrics and analysis tools.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        loss_fn: LossFunction = mse_loss,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: JEPA model to evaluate
            device: Device to run evaluation on
            loss_fn: Callable that computes loss from prediction and target tensors
        """
        self.model = model
        self.loss_fn = loss_fn
        
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
        mse_losses: List[float] = []
        prediction_norms: List[float] = []
        target_norms: List[float] = []
        cosine_similarities: List[float] = []
        
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    state_t, action_t, state_t1 = batch
                    state_t = state_t.to(self.device)
                    action_t = action_t.to(self.device)
                    state_t1 = state_t1.to(self.device)
                    prediction, target = self.model(state_t, action_t, state_t1)
                else:
                    state_t, state_t1 = batch
                    state_t = state_t.to(self.device)
                    state_t1 = state_t1.to(self.device)
                    prediction, target = self.model(state_t, state_t1)

                loss = self.loss_fn(prediction, target)
                
                batch_size = prediction.shape[0]
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                
                mse_per_sample = torch.mean((prediction - target) ** 2, dim=-1)
                mse_losses.extend(mse_per_sample.reshape(-1).cpu().numpy())
                
                pred_norms = torch.norm(prediction, dim=-1)
                target_norms_tensor = torch.norm(target, dim=-1)
                prediction_norms.extend(pred_norms.reshape(-1).cpu().numpy())
                target_norms.extend(target_norms_tensor.reshape(-1).cpu().numpy())
                
                cos_sim = torch.cosine_similarity(prediction, target, dim=-1)
                cosine_similarities.extend(cos_sim.reshape(-1).cpu().numpy())
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('nan')
        mse_losses_arr = np.array(mse_losses)
        prediction_norms_arr = np.array(prediction_norms)
        target_norms_arr = np.array(target_norms)
        cosine_similarities_arr = np.array(cosine_similarities)
        
        metrics = {
            "loss": avg_loss,
            "mse_mean": float(np.mean(mse_losses_arr)) if mse_losses_arr.size else float('nan'),
            "mse_std": float(np.std(mse_losses_arr)) if mse_losses_arr.size else float('nan'),
            "prediction_norm_mean": float(np.mean(prediction_norms_arr)) if prediction_norms_arr.size else float('nan'),
            "prediction_norm_std": float(np.std(prediction_norms_arr)) if prediction_norms_arr.size else float('nan'),
            "target_norm_mean": float(np.mean(target_norms_arr)) if target_norms_arr.size else float('nan'),
            "target_norm_std": float(np.std(target_norms_arr)) if target_norms_arr.size else float('nan'),
            "cosine_similarity_mean": float(np.mean(cosine_similarities_arr)) if cosine_similarities_arr.size else float('nan'),
            "cosine_similarity_std": float(np.std(cosine_similarities_arr)) if cosine_similarities_arr.size else float('nan'),
            "num_samples": num_samples,
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

