"""
TensorBoard logger implementation.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

from .base_logger import BaseLogger

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TensorBoard logger.
        
        Args:
            config: TensorBoard configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.writer = None
        self._initialized = False
        
        if self.is_available() and config.get('enabled', False):
            self._initialize()
    
    def _initialize(self):
        """Initialize TensorBoard writer."""
        try:
            log_dir = self.config.get('log_dir', './tensorboard_logs')
            experiment_name = self.config.get('experiment_name', 'jepa_experiment')
            
            # Create full log directory path
            full_log_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(full_log_dir, exist_ok=True)
            
            self.writer = SummaryWriter(
                log_dir=full_log_dir,
                comment=self.config.get('comment', '')
            )
            self._initialized = True
            self.logger.info(f"TensorBoard initialized: {full_log_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            self._initialized = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to TensorBoard."""
        if not self._initialized:
            return
        
        try:
            for key, value in metrics.items():
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, value, global_step=step)
            
            self.writer.flush()
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to TensorBoard: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        if not self._initialized:
            return
        
        try:
            # Convert all values to strings for TensorBoard
            hparams = {k: str(v) for k, v in params.items()}
            self.writer.add_hparams(hparams, {})
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifact to TensorBoard (limited support)."""
        if not self._initialized:
            return
        
        # TensorBoard has limited artifact support, so we just log the path
        try:
            self.writer.add_text(
                f"artifacts/{artifact_type}",
                f"{name}: {file_path}",
                global_step=0
            )
            self.logger.info(f"Artifact path logged to TensorBoard: {name}")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact to TensorBoard: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model in TensorBoard."""
        if not self._initialized:
            return
        
        try:
            # Add model graph to TensorBoard
            # Note: This requires a sample input, which we don't have here
            # In practice, this would be called from the trainer with sample data
            self.logger.info("Model watching setup for TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to watch model in TensorBoard: {e}")
    
    def log_model_graph(self, model, input_sample):
        """Log model graph to TensorBoard."""
        if not self._initialized:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
            self.logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to log model graph to TensorBoard: {e}")
    
    def finish(self):
        """Finish TensorBoard session."""
        if self._initialized and self.writer:
            try:
                self.writer.close()
                self.logger.info("TensorBoard session finished")
            except Exception as e:
                self.logger.warning(f"Failed to finish TensorBoard session: {e}")
    
    def is_available(self) -> bool:
        """Check if TensorBoard is available."""
        return TENSORBOARD_AVAILABLE
