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
        trackio_url = kwargs.get('trackio_url') or os.environ.get('TRACKIO_URL')
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
        
        # Generate experiment ID
        experiment_id = f"trl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _monitor.experiment_id = experiment_id
        
        logger.info(f"Trackio initialized for experiment: {exp_name}")
        logger.info(f"Experiment ID: {experiment_id}")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Failed to initialize trackio: {e}")
        # Return a fallback experiment ID
        return f"trl_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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