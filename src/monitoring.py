"""
Trackio Monitoring Integration for SmolLM3 Fine-tuning
Provides comprehensive experiment tracking and monitoring capabilities with HF Datasets support
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
    from scripts.trackio_tonic.trackio_api_client import TrackioAPIClient
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("Warning: Trackio API client not available. Install with: pip install requests")

logger = logging.getLogger(__name__)

class SmolLM3Monitor:
    """Monitoring and tracking for SmolLM3 fine-tuning experiments with HF Datasets support"""
    
    def __init__(
        self,
        experiment_name: str,
        trackio_url: Optional[str] = None,
        trackio_token: Optional[str] = None,
        enable_tracking: bool = True,
        log_artifacts: bool = True,
        log_metrics: bool = True,
        log_config: bool = True,
        hf_token: Optional[str] = None,
        dataset_repo: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.enable_tracking = enable_tracking and TRACKIO_AVAILABLE
        self.log_artifacts = log_artifacts
        self.log_metrics_enabled = log_metrics  # Rename to avoid conflict
        self.log_config_enabled = log_config  # Rename to avoid conflict
        
        # HF Datasets configuration
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.dataset_repo = dataset_repo or os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        
        # Initialize experiment metadata first
        self.experiment_id = None
        self.start_time = datetime.now()
        self.metrics_history = []
        self.artifacts = []
        
        # Initialize Trackio API client
        self.trackio_client = None
        if self.enable_tracking:
            self._setup_trackio(trackio_url, trackio_token)
        
        # Initialize HF Datasets client
        self.hf_dataset_client = None
        if self.hf_token:
            self._setup_hf_datasets()
        
        logger.info("Initialized monitoring for experiment: %s", experiment_name)
        logger.info("Dataset repository: %s", self.dataset_repo)
    
    def _setup_hf_datasets(self):
        """Setup HF Datasets client for persistent storage"""
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
            
            self.hf_dataset_client = {
                'Dataset': Dataset,
                'HfApi': HfApi,
                'api': HfApi(token=self.hf_token)
            }
            logger.info("✅ HF Datasets client initialized for %s", self.dataset_repo)
            
        except ImportError:
            logger.warning("⚠️ datasets or huggingface-hub not available. Install with: pip install datasets huggingface-hub")
            self.hf_dataset_client = None
        except Exception as e:
            logger.error("Failed to initialize HF Datasets client: %s", e)
            self.hf_dataset_client = None
    
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
                description="SmolLM3 fine-tuning experiment started at {}".format(self.start_time)
            )
            
            if "success" in create_result:
                # Extract experiment ID from response
                import re
                response_text = create_result['data']
                match = re.search(r'exp_\d{8}_\d{6}', response_text)
                if match:
                    self.experiment_id = match.group()
                    logger.info("Trackio API client initialized. Experiment ID: %s", self.experiment_id)
                else:
                    logger.error("Could not extract experiment ID from response")
                    self.enable_tracking = False
            else:
                logger.error("Failed to create experiment: %s", create_result)
                self.enable_tracking = False
            
        except Exception as e:
            logger.error("Failed to initialize Trackio API: %s", e)
            self.enable_tracking = False
    
    def _save_to_hf_dataset(self, experiment_data: Dict[str, Any]):
        """Save experiment data to HF Dataset"""
        if not self.hf_dataset_client:
            return False
        
        try:
            # Convert experiment data to dataset format
            dataset_data = [{
                'experiment_id': self.experiment_id or "exp_{}".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                'name': self.experiment_name,
                'description': "SmolLM3 fine-tuning experiment",
                'created_at': self.start_time.isoformat(),
                'status': 'running',
                'metrics': json.dumps(self.metrics_history),
                'parameters': json.dumps(experiment_data),
                'artifacts': json.dumps(self.artifacts),
                'logs': json.dumps([]),
                'last_updated': datetime.now().isoformat()
            }]
            
            # Create dataset
            Dataset = self.hf_dataset_client['Dataset']
            dataset = Dataset.from_list(dataset_data)
            
            # Push to HF Hub
            dataset.push_to_hub(
                self.dataset_repo,
                token=self.hf_token,
                private=True
            )
            
            logger.info("✅ Saved experiment data to %s", self.dataset_repo)
            return True
            
        except Exception as e:
            logger.error("Failed to save to HF Dataset: %s", e)
            return False
    
    def log_configuration(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        if not self.enable_tracking or not self.log_config_enabled:
            return
        
        try:
            # Log configuration as parameters
            if self.trackio_client:
                result = self.trackio_client.log_parameters(
                    experiment_id=self.experiment_id,
                    parameters=config
                )
                
                if "success" in result:
                    logger.info("Configuration logged to Trackio")
                else:
                    logger.error("Failed to log configuration: %s", result)
            
            # Save to HF Dataset
            self._save_to_hf_dataset(config)
            
            # Also save config locally
            config_path = "config_{}_{}.json".format(
                self.experiment_name, 
                self.start_time.strftime('%Y%m%d_%H%M%S')
            )
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.artifacts.append(config_path)
            logger.info("Configuration saved to %s", config_path)
            
        except Exception as e:
            logger.error("Failed to log configuration: %s", e)
    
    def log_config(self, config: Dict[str, Any]):
        """Alias for log_configuration for backward compatibility"""
        return self.log_configuration(config)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log training metrics"""
        if not self.enable_tracking or not self.log_metrics_enabled:
            return
        
        try:
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            if step is not None:
                metrics['step'] = step
            
            # Log to Trackio
            if self.trackio_client:
                result = self.trackio_client.log_metrics(
                    experiment_id=self.experiment_id,
                    metrics=metrics,
                    step=step
                )
                
                if "success" in result:
                    logger.debug("Metrics logged to Trackio")
                else:
                    logger.error("Failed to log metrics to Trackio: %s", result)
            
            # Store locally
            self.metrics_history.append(metrics)
            
            # Save to HF Dataset periodically
            if len(self.metrics_history) % 10 == 0:  # Save every 10 metrics
                self._save_to_hf_dataset({'metrics': self.metrics_history})
            
            logger.debug("Metrics logged: %s", metrics)
            
        except Exception as e:
            logger.error("Failed to log metrics: %s", e)
    
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
            
            if self.trackio_client:
                result = self.trackio_client.log_parameters(
                    experiment_id=self.experiment_id,
                    parameters=checkpoint_info
                )
                
                if "success" in result:
                    logger.info("Checkpoint logged to Trackio")
                else:
                    logger.error("Failed to log checkpoint to Trackio: %s", result)
            
            self.artifacts.append(checkpoint_path)
            logger.info("Checkpoint logged: %s", checkpoint_path)
            
        except Exception as e:
            logger.error("Failed to log checkpoint: %s", e)
    
    def log_evaluation_results(self, results: Dict[str, Any], step: Optional[int] = None):
        """Log evaluation results"""
        if not self.enable_tracking:
            return
        
        try:
            # Add evaluation prefix to metrics
            eval_metrics = {f"eval_{k}": v for k, v in results.items()}
            
            self.log_metrics(eval_metrics, step)
            
            # Save evaluation results locally
            eval_path = "eval_results_step_{}_{}.json".format(
                step or "unknown",
                self.start_time.strftime('%Y%m%d_%H%M%S')
            )
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.artifacts.append(eval_path)
            logger.info("Evaluation results logged and saved to %s", eval_path)
            
        except Exception as e:
            logger.error("Failed to log evaluation results: %s", e)
    
    def log_system_metrics(self, step: Optional[int] = None):
        """Log system metrics (GPU, memory, etc.)"""
        if not self.enable_tracking:
            return
        
        try:
            system_metrics = {}
            
            # GPU metrics
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    system_metrics['gpu_{}_memory_allocated'.format(i)] = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    system_metrics['gpu_{}_memory_reserved'.format(i)] = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    system_metrics['gpu_{}_utilization'.format(i)] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            
            # CPU and memory metrics (basic)
            try:
                import psutil
                system_metrics['cpu_percent'] = psutil.cpu_percent()
                system_metrics['memory_percent'] = psutil.virtual_memory().percent
            except ImportError:
                logger.warning("psutil not available, skipping CPU/memory metrics")
            
            self.log_metrics(system_metrics, step)
            
        except Exception as e:
            logger.error("Failed to log system metrics: %s", e)
    
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
            
            # Log final summary to Trackio
            if self.trackio_client:
                result = self.trackio_client.log_parameters(
                    experiment_id=self.experiment_id,
                    parameters=summary
                )
                
                if "success" in result:
                    logger.info("Training summary logged to Trackio")
                else:
                    logger.error("Failed to log training summary to Trackio: %s", result)
            
            # Save to HF Dataset
            self._save_to_hf_dataset(summary)
            
            # Save summary locally
            summary_path = "training_summary_{}_{}.json".format(
                self.experiment_name,
                self.start_time.strftime('%Y%m%d_%H%M%S')
            )
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.artifacts.append(summary_path)
            logger.info("Training summary logged and saved to %s", summary_path)
            
        except Exception as e:
            logger.error("Failed to log training summary: %s", e)
    
    def create_monitoring_callback(self):
        """Create a callback for integration with Hugging Face Trainer"""
        from transformers import TrainerCallback
        
        class TrackioCallback(TrainerCallback):
            def __init__(self, monitor):
                super().__init__()
                self.monitor = monitor
                logger.info("TrackioCallback initialized")
            
            def on_init_end(self, args, state, control, **kwargs):
                """Called when training initialization is complete"""
                try:
                    logger.info("Training initialization completed")
                except Exception as e:
                    logger.error("Error in on_init_end: %s", e)
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Called when logs are created"""
                try:
                    if logs and isinstance(logs, dict):
                        step = getattr(state, 'global_step', None)
                        self.monitor.log_metrics(logs, step)
                        self.monitor.log_system_metrics(step)
                except Exception as e:
                    logger.error("Error in on_log: %s", e)
            
            def on_save(self, args, state, control, **kwargs):
                """Called when a checkpoint is saved"""
                try:
                    step = getattr(state, 'global_step', None)
                    if step is not None:
                        checkpoint_path = os.path.join(args.output_dir, "checkpoint-{}".format(step))
                        if os.path.exists(checkpoint_path):
                            self.monitor.log_model_checkpoint(checkpoint_path, step)
                except Exception as e:
                    logger.error("Error in on_save: %s", e)
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Called when evaluation is performed"""
                try:
                    if metrics and isinstance(metrics, dict):
                        step = getattr(state, 'global_step', None)
                        self.monitor.log_evaluation_results(metrics, step)
                except Exception as e:
                    logger.error("Error in on_evaluate: %s", e)
            
            def on_train_begin(self, args, state, control, **kwargs):
                """Called when training begins"""
                try:
                    logger.info("Training started")
                except Exception as e:
                    logger.error("Error in on_train_begin: %s", e)
            
            def on_train_end(self, args, state, control, **kwargs):
                """Called when training ends"""
                try:
                    logger.info("Training completed")
                    if self.monitor:
                        self.monitor.close()
                except Exception as e:
                    logger.error("Error in on_train_end: %s", e)
        
        callback = TrackioCallback(self)
        logger.info("TrackioCallback created successfully")
        return callback
    
    def get_experiment_url(self) -> Optional[str]:
        """Get the URL to view the experiment in Trackio"""
        if self.trackio_client and self.experiment_id:
            return "{}?tab=view_experiments".format(self.trackio_client.space_url)
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
                    logger.error("Failed to close monitoring session: %s", result)
            except Exception as e:
                logger.error("Failed to close monitoring session: %s", e)
        
        # Final save to HF Dataset
        if self.hf_dataset_client:
            self._save_to_hf_dataset({'status': 'completed'})

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
        log_config=getattr(config, 'log_config', True),
        hf_token=getattr(config, 'hf_token', None),
        dataset_repo=getattr(config, 'dataset_repo', None)
    ) 