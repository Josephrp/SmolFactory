"""
Trackio Module Interface for TRL Library
Provides the interface expected by TRL library while integrating with our custom monitoring system
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import our custom monitoring
from monitoring import SmolLM3Monitor

logger = logging.getLogger(__name__)

# Global monitor instance
_monitor = None

def init(
    project_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Initialize trackio experiment (TRL interface)
    
    Args:
        project_name: Name of the project (optional, defaults to 'smollm3_experiment')
        experiment_name: Name of the experiment (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Experiment ID
    """
    global _monitor
    
    try:
        # Provide default project name if not provided
        if project_name is None:
            project_name = os.environ.get('EXPERIMENT_NAME', 'smollm3_experiment')
        
        # Extract configuration from kwargs
        # Accept both TRACKIO_URL (full URL or org/space) and TRACKIO_SPACE_ID
        trackio_url = (
            kwargs.get('trackio_url')
            or os.environ.get('TRACKIO_URL')
            or os.environ.get('TRACKIO_SPACE_ID')
        )
        trackio_token = kwargs.get('trackio_token') or os.environ.get('TRACKIO_TOKEN')
        hf_token = kwargs.get('hf_token') or os.environ.get('HF_TOKEN')
        dataset_repo = kwargs.get('dataset_repo') or os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        
        # Use experiment_name if provided, otherwise use project_name
        exp_name = experiment_name or project_name
        
        # Create monitor instance
        _monitor = SmolLM3Monitor(
            experiment_name=exp_name,
            trackio_url=trackio_url,
            trackio_token=trackio_token,
            enable_tracking=True,
            log_artifacts=True,
            log_metrics=True,
            log_config=True,
            hf_token=hf_token,
            dataset_repo=dataset_repo
        )
        # The monitor constructor creates the experiment remotely and sets
        # `experiment_id`. Do NOT overwrite it with a locally generated ID.
        experiment_id = getattr(_monitor, "experiment_id", None)
        logger.info(f"Trackio initialized for experiment: {exp_name}")
        logger.info(f"Experiment ID: {experiment_id}")
        return experiment_id or f"exp_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    except Exception as e:
        logger.error(f"Failed to initialize trackio: {e}")
        # Return a fallback experiment ID - use the same format as our monitoring system
        return f"exp_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def log(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    **kwargs
):
    """
    Log metrics to trackio (TRL interface)
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        **kwargs: Additional parameters
    """
    global _monitor
    
    try:
        if _monitor is None:
            logger.warning("Trackio not initialized, skipping log")
            return
        
        # Log metrics using our custom monitor
        _monitor.log_metrics(metrics, step)
        
        # Also log system metrics if available
        _monitor.log_system_metrics(step)
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

def finish():
    """
    Finish trackio experiment (TRL interface)
    """
    global _monitor
    
    try:
        if _monitor is None:
            logger.warning("Trackio not initialized, skipping finish")
            return
        
        # Close the monitoring session
        _monitor.close()
        
        logger.info("Trackio experiment finished")
        
    except Exception as e:
        logger.error(f"Failed to finish trackio experiment: {e}")

def set_monitor(monitor: SmolLM3Monitor) -> None:
    """Set the shared monitor instance used by this module.

    This allows external code (e.g., our trainer) to create a
    `SmolLM3Monitor` once and have `trackio.log/finish` operate on
    the exact same object, preventing mismatched experiment IDs.
    """
    global _monitor
    _monitor = monitor
    try:
        logger.info("trackio monitor set: experiment_id=%s", getattr(monitor, "experiment_id", None))
    except Exception:
        pass

def log_config(config: Dict[str, Any]):
    """
    Log configuration to trackio (TRL interface)
    
    Args:
        config: Configuration dictionary to log
    """
    global _monitor
    
    try:
        if _monitor is None:
            logger.warning("Trackio not initialized, skipping config log")
            return
        
        # Log configuration using our custom monitor
        _monitor.log_configuration(config)
        
    except Exception as e:
        logger.error(f"Failed to log config: {e}")

def log_checkpoint(checkpoint_path: str, step: Optional[int] = None):
    """
    Log checkpoint to trackio (TRL interface)
    
    Args:
        checkpoint_path: Path to the checkpoint file
        step: Current training step
    """
    global _monitor
    
    try:
        if _monitor is None:
            logger.warning("Trackio not initialized, skipping checkpoint log")
            return
        
        # Log checkpoint using our custom monitor
        _monitor.log_model_checkpoint(checkpoint_path, step)
        
    except Exception as e:
        logger.error(f"Failed to log checkpoint: {e}")

def log_evaluation_results(results: Dict[str, Any], step: Optional[int] = None):
    """
    Log evaluation results to trackio (TRL interface)
    
    Args:
        results: Evaluation results dictionary
        step: Current training step
    """
    global _monitor
    
    try:
        if _monitor is None:
            logger.warning("Trackio not initialized, skipping evaluation log")
            return
        
        # Log evaluation results using our custom monitor
        _monitor.log_evaluation_results(results, step)
        
    except Exception as e:
        logger.error(f"Failed to log evaluation results: {e}")

# Additional utility functions for TRL compatibility
def get_experiment_url() -> Optional[str]:
    """Get the URL to view the experiment"""
    global _monitor
    
    if _monitor is not None:
        return _monitor.get_experiment_url()
    return None

def is_available() -> bool:
    """Check if trackio is available and initialized"""
    return _monitor is not None and _monitor.enable_tracking

def get_monitor():
    """Get the current monitor instance (for advanced usage)"""
    return _monitor

# Add config attribute for TRL compatibility
class TrackioConfig:
    """Configuration class for trackio (TRL compatibility)"""
    
    def __init__(self):
        self.project_name = os.environ.get('EXPERIMENT_NAME', 'smollm3_experiment')
        self.experiment_name = os.environ.get('EXPERIMENT_NAME', 'smollm3_experiment')
        self.trackio_url = os.environ.get('TRACKIO_URL')
        self.trackio_token = os.environ.get('TRACKIO_TOKEN')
        self.hf_token = os.environ.get('HF_TOKEN')
        self.dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
    
    def update(self, config_dict: Dict[str, Any] = None, **kwargs):
        """
        Update configuration with new values (TRL compatibility)
        
        Args:
            config_dict: Dictionary of configuration values to update (optional)
            **kwargs: Additional configuration values to update
        """
        # Handle both dictionary and keyword arguments
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    # Add new attributes dynamically
                    setattr(self, key, value)
        
        # Handle keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Add new attributes dynamically
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to configuration values
        
        Args:
            key: Configuration key to access
            
        Returns:
            Configuration value
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Configuration key '{key}' not found")
    
    def __setitem__(self, key: str, value: Any):
        """
        Dictionary-style assignment to configuration values
        
        Args:
            key: Configuration key to set
            value: Value to assign
        """
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if configuration key exists
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default
        
        Args:
            key: Configuration key to access
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default
    
    def keys(self):
        """
        Get all configuration keys
        
        Returns:
            List of configuration keys
        """
        # Use __dict__ to avoid recursion with dir()
        return list(self.__dict__.keys())
    
    def items(self):
        """
        Get all configuration key-value pairs
        
        Returns:
            List of (key, value) tuples
        """
        # Use __dict__ to avoid recursion
        return list(self.__dict__.items())
    
    def __repr__(self):
        """String representation of configuration"""
        # Use __dict__ to avoid recursion
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={repr(value)}")
        return f"TrackioConfig({', '.join(attrs)})"

# Create config instance
config = TrackioConfig() 