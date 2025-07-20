"""
Trackio Monitoring Integration for SmolLM3 Fine-tuning
Provides comprehensive experiment tracking and monitoring capabilities
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
from pathlib import Path

# Import the real API client
try:
    from trackio_api_client import TrackioAPIClient
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("Warning: Trackio API client not available. Install with: pip install requests")

logger = logging.getLogger(__name__)

class SmolLM3Monitor:
    """Monitoring and tracking for SmolLM3 fine-tuning experiments"""
    
    def __init__(
        self,
        experiment_name: str,
        trackio_url: Optional[str] = None,
        trackio_token: Optional[str] = None,
        enable_tracking: bool = True,
        log_artifacts: bool = True,
        log_metrics: bool = True,
        log_config: bool = True
    ):
        self.experiment_name = experiment_name
        self.enable_tracking = enable_tracking and TRACKIO_AVAILABLE
        self.log_artifacts = log_artifacts
        self.log_metrics = log_metrics
        self.log_config = log_config
        
        # Initialize experiment metadata first
        self.experiment_id = None
        self.start_time = datetime.now()
        self.metrics_history = []
        self.artifacts = []
        
        # Initialize Trackio API client
        self.trackio_client = None
        if self.enable_tracking:
            self._setup_trackio(trackio_url, trackio_token)
        
        logger.info(f"Initialized monitoring for experiment: {experiment_name}")
    
    def _setup_trackio(self, trackio_url: Optional[str], trackio_token: Optional[str]):
        """Setup Trackio API client"""
        try:
            # Get Trackio configuration from environment or parameters
            url = trackio_url or os.getenv('TRACKIO_URL')
            
            if not url:
                logger.warning("Trackio URL not provided. Set TRACKIO_URL environment variable.")
                self.enable_tracking = False
                return
            
            self.trackio_client = TrackioAPIClient(url)
            
            # Create experiment
            create_result = self.trackio_client.create_experiment(
                name=self.experiment_name,
                description=f"SmolLM3 fine-tuning experiment started at {self.start_time}"
            )
            
            if "success" in create_result:
                # Extract experiment ID from response
                import re
                response_text = create_result['data']
                match = re.search(r'exp_\d{8}_\d{6}', response_text)
                if match:
                    self.experiment_id = match.group()
                    logger.info(f"Trackio API client initialized. Experiment ID: {self.experiment_id}")
                else:
                    logger.error("Could not extract experiment ID from response")
                    self.enable_tracking = False
            else:
                logger.error(f"Failed to create experiment: {create_result}")
                self.enable_tracking = False
            
        except Exception as e:
            logger.error(f"Failed to initialize Trackio API: {e}")
            self.enable_tracking = False
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        if not self.enable_tracking or not self.log_config:
            return
        
        try:
            # Log configuration as parameters
            result = self.trackio_client.log_parameters(
                experiment_id=self.experiment_id,
                parameters=config
            )
            
            if "success" in result:
                # Also save config locally
                config_path = f"config_{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
                
                self.artifacts.append(config_path)
                logger.info(f"Configuration logged to Trackio and saved to {config_path}")
            else:
                logger.error(f"Failed to log configuration: {result}")
            
        except Exception as e:
            logger.error(f"Failed to log configuration: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training metrics"""
        if not self.enable_tracking or not self.log_metrics:
            return
        
        try:
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            if step is not None:
                metrics['step'] = step
            
            # Log to Trackio
            result = self.trackio_client.log_metrics(
                experiment_id=self.experiment_id,
                metrics=metrics,
                step=step
            )
            
            if "success" in result:
                # Store locally
                self.metrics_history.append(metrics)
                logger.debug(f"Metrics logged: {metrics}")
            else:
                logger.error(f"Failed to log metrics: {result}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model_checkpoint(self, checkpoint_path: str, step: Optional[int] = None):
        """Log model checkpoint"""
        if not self.enable_tracking or not self.log_artifacts:
            return
        
        try:
            # For now, just log the checkpoint path as a parameter
            # The actual file upload would need additional API endpoints
            checkpoint_info = {
                "checkpoint_path": checkpoint_path,
                "checkpoint_step": step,
                "checkpoint_size": os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0
            }
            
            result = self.trackio_client.log_parameters(
                experiment_id=self.experiment_id,
                parameters=checkpoint_info
            )
            
            if "success" in result:
                self.artifacts.append(checkpoint_path)
                logger.info(f"Checkpoint logged: {checkpoint_path}")
            else:
                logger.error(f"Failed to log checkpoint: {result}")
            
        except Exception as e:
            logger.error(f"Failed to log checkpoint: {e}")
    
    def log_evaluation_results(self, results: Dict[str, Any], step: Optional[int] = None):
        """Log evaluation results"""
        if not self.enable_tracking:
            return
        
        try:
            # Add evaluation prefix to metrics
            eval_metrics = {f"eval_{k}": v for k, v in results.items()}
            
            self.log_metrics(eval_metrics, step)
            
            # Save evaluation results locally
            eval_path = f"eval_results_step_{step}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.artifacts.append(eval_path)
            logger.info(f"Evaluation results logged and saved to {eval_path}")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")
    
    def log_system_metrics(self, step: Optional[int] = None):
        """Log system metrics (GPU, memory, etc.)"""
        if not self.enable_tracking:
            return
        
        try:
            system_metrics = {}
            
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    system_metrics[f'gpu_{i}_memory_allocated'] = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    system_metrics[f'gpu_{i}_memory_reserved'] = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    system_metrics[f'gpu_{i}_utilization'] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            
            # CPU and memory metrics (basic)
            import psutil
            system_metrics['cpu_percent'] = psutil.cpu_percent()
            system_metrics['memory_percent'] = psutil.virtual_memory().percent
            
            self.log_metrics(system_metrics, step)
            
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")
    
    def log_training_summary(self, summary: Dict[str, Any]):
        """Log training summary at the end"""
        if not self.enable_tracking:
            return
        
        try:
            # Add experiment duration
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            summary['experiment_duration_seconds'] = duration
            summary['experiment_duration_hours'] = duration / 3600
            
            # Log final summary
            result = self.trackio_client.log_parameters(
                experiment_id=self.experiment_id,
                parameters=summary
            )
            
            if "success" in result:
                # Save summary locally
                summary_path = f"training_summary_{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                self.artifacts.append(summary_path)
                logger.info(f"Training summary logged and saved to {summary_path}")
            else:
                logger.error(f"Failed to log training summary: {result}")
            
        except Exception as e:
            logger.error(f"Failed to log training summary: {e}")
    
    def create_monitoring_callback(self):
        """Create a callback for integration with Hugging Face Trainer"""
        if not self.enable_tracking:
            return None
        
        class TrackioCallback:
            def __init__(self, monitor):
                self.monitor = monitor
            
            def on_init_end(self, args, state, control, **kwargs):
                """Called when training initialization is complete"""
                try:
                    logger.info("Training initialization completed")
                except Exception as e:
                    logger.error(f"Error in on_init_end: {e}")
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Called when logs are created"""
                try:
                    if logs and isinstance(logs, dict):
                        self.monitor.log_metrics(logs, state.global_step)
                        self.monitor.log_system_metrics(state.global_step)
                except Exception as e:
                    logger.error(f"Error in on_log: {e}")
            
            def on_save(self, args, state, control, **kwargs):
                """Called when a checkpoint is saved"""
                try:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    if os.path.exists(checkpoint_path):
                        self.monitor.log_model_checkpoint(checkpoint_path, state.global_step)
                except Exception as e:
                    logger.error(f"Error in on_save: {e}")
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Called when evaluation is performed"""
                try:
                    if metrics and isinstance(metrics, dict):
                        self.monitor.log_evaluation_results(metrics, state.global_step)
                except Exception as e:
                    logger.error(f"Error in on_evaluate: {e}")
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Called when training begins"""
                try:
                    logger.info("Training started")
                except Exception as e:
                    logger.error(f"Error in on_train_begin: {e}")
            
            def on_train_end(self, args, state, control, **kwargs):
                """Called when training ends"""
                try:
                    logger.info("Training completed")
                    if self.monitor:
                        self.monitor.close()
                except Exception as e:
                    logger.error(f"Error in on_train_end: {e}")
            
            def __call__(self, *args, **kwargs):
                """Make the callback callable to avoid any issues"""
                return self
        
        return TrackioCallback(self)
    
    def get_experiment_url(self) -> Optional[str]:
        """Get the URL to view the experiment in Trackio"""
        if self.trackio_client and self.experiment_id:
            return f"{self.trackio_client.space_url}?tab=view_experiments"
        return None
    
    def close(self):
        """Close the monitoring session"""
        if self.enable_tracking and self.trackio_client:
            try:
                # Mark experiment as completed
                result = self.trackio_client.update_experiment_status(
                    experiment_id=self.experiment_id,
                    status="completed"
                )
                if "success" in result:
                    logger.info("Monitoring session closed")
                else:
                    logger.error(f"Failed to close monitoring session: {result}")
            except Exception as e:
                logger.error(f"Failed to close monitoring session: {e}")

# Utility function to create monitor from config
def create_monitor_from_config(config, experiment_name: Optional[str] = None) -> SmolLM3Monitor:
    """Create a monitor instance from configuration"""
    if experiment_name is None:
        experiment_name = getattr(config, 'experiment_name', 'smollm3_experiment')
    
    return SmolLM3Monitor(
        experiment_name=experiment_name,
        trackio_url=getattr(config, 'trackio_url', None),
        trackio_token=getattr(config, 'trackio_token', None),
        enable_tracking=getattr(config, 'enable_tracking', True),
        log_artifacts=getattr(config, 'log_artifacts', True),
        log_metrics=getattr(config, 'log_metrics', True),
        log_config=getattr(config, 'log_config', True)
    ) 