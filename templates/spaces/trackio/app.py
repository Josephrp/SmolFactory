"""
Trackio Deployment on Hugging Face Spaces
A Gradio interface for experiment tracking and monitoring
"""

import gradio as gr
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.io as pio
pio.templates.default = "plotly_white"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackioSpace:
    """Trackio deployment for Hugging Face Spaces using HF Datasets"""
    
    def __init__(self, hf_token: Optional[str] = None, dataset_repo: Optional[str] = None):
        self.experiments = {}
        self.current_experiment = None
        self.using_backup_data = False
        
        # Get dataset repository and HF token from parameters or environment variables
        self.dataset_repo = dataset_repo or os.environ.get('TRACKIO_DATASET_REPO', 'Tonic/trackio-experiments')
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        
        # Initialize dataset manager for safe operations
        self.dataset_manager = None
        if self.hf_token and self.dataset_repo:
            try:
                # Prefer local dataset_utils in Space repo
                from dataset_utils import TrackioDatasetManager  # type: ignore
                self.dataset_manager = TrackioDatasetManager(self.dataset_repo, self.hf_token)
                logger.info("✅ Dataset manager initialized for safe operations (local)")
            except Exception as local_e:
                try:
                    # Fallback: try project src layout if present
                    import sys
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
                    from dataset_utils import TrackioDatasetManager  # type: ignore
                    self.dataset_manager = TrackioDatasetManager(self.dataset_repo, self.hf_token)
                    logger.info("✅ Dataset manager initialized for safe operations (fallback src)")
                except Exception as e:
                    logger.warning(f"⚠️ Dataset manager not available, using legacy data handling: {local_e or e}")
        
        logger.info(f"🔧 Using dataset repository: {self.dataset_repo}")
        
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not found. Some features may not work.")
        
        self._load_experiments()
        
    def _load_experiments(self):
        """Load experiments from HF Dataset with data preservation support"""
        try:
            # Try using dataset manager first for safe operations
            if self.dataset_manager:
                logger.info("🔍 Loading experiments using dataset manager")
                experiments_list = self.dataset_manager.load_existing_experiments()
                
                # Convert list to dict format expected by the interface
                self.experiments = {}
                for exp_data in experiments_list:
                    exp_id = exp_data.get('experiment_id')
                    if exp_id:
                        converted_experiment = self._convert_dataset_row_to_experiment(exp_data)
                        if converted_experiment:
                            self.experiments[exp_id] = converted_experiment
                
                logger.info(f"✅ Loaded {len(self.experiments)} experiments using dataset manager")
                
                # Sort experiments by creation date (newest first)
                self.experiments = dict(sorted(
                    self.experiments.items(), 
                    key=lambda x: x[1].get('created_at', ''), 
                    reverse=True
                ))
                
                # If no experiments found, use backup but mark backup mode to avoid accidental writes
                if not self.experiments:
                    logger.info("📊 No experiments found in dataset, using backup data")
                    self._load_backup_experiments()
                    self.using_backup_data = True
                
                return
            
            # Fallback to direct dataset loading if dataset manager not available
            if self.hf_token:
                success = self._load_experiments_direct()
                if success:
                    self.using_backup_data = False
                    return
            
            # Final fallback to backup data
            logger.info("🔄 Using backup data")
            self._load_backup_experiments()
            self.using_backup_data = True
                
        except Exception as e:
            logger.error(f"❌ Failed to load experiments: {e}")
            self._load_backup_experiments()
            self.using_backup_data = True
    
    def _load_experiments_direct(self) -> bool:
        """Load experiments directly from HF Dataset without dataset manager"""
        try:
            from datasets import load_dataset
            
            logger.info(f"🔍 Loading experiments directly from {self.dataset_repo}")
            dataset = load_dataset(self.dataset_repo, token=self.hf_token)
            logger.info(f"✅ Successfully loaded dataset from {self.dataset_repo}")
            
            # Convert dataset to experiments dict
            self.experiments = {}
            if 'train' in dataset:
                for row in dataset['train']:
                    exp_id = row.get('experiment_id')
                    if exp_id:
                        converted_experiment = self._convert_dataset_row_to_experiment(row)
                        if converted_experiment:
                            self.experiments[exp_id] = converted_experiment
            
            logger.info(f"📊 Successfully loaded {len(self.experiments)} experiments from dataset")
            
            # Sort experiments by creation date (newest first)
            self.experiments = dict(sorted(
                self.experiments.items(), 
                key=lambda x: x[1].get('created_at', ''), 
                reverse=True
            ))
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load from dataset directly: {e}")
            return False
    
    def _convert_dataset_row_to_experiment(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a dataset row to experiment format, handling JSON parsing safely"""
        try:
            exp_id = row.get('experiment_id')
            if not exp_id:
                return None
            
            # Parse JSON fields safely
            try:
                metrics_raw = row.get('metrics', '[]')
                if isinstance(metrics_raw, str):
                    metrics = json.loads(metrics_raw) if metrics_raw else []
                else:
                    metrics = metrics_raw if metrics_raw else []
                    
                parameters_raw = row.get('parameters', '{}')
                if isinstance(parameters_raw, str):
                    parameters = json.loads(parameters_raw) if parameters_raw else {}
                else:
                    parameters = parameters_raw if parameters_raw else {}
                    
                artifacts_raw = row.get('artifacts', '[]')
                if isinstance(artifacts_raw, str):
                    artifacts = json.loads(artifacts_raw) if artifacts_raw else []
                else:
                    artifacts = artifacts_raw if artifacts_raw else []
                    
                logs_raw = row.get('logs', '[]')
                if isinstance(logs_raw, str):
                    logs = json.loads(logs_raw) if logs_raw else []
                else:
                    logs = logs_raw if logs_raw else []
                    
            except json.JSONDecodeError as json_err:
                logger.warning(f"JSON decode error for experiment {exp_id}: {json_err}")
                metrics, parameters, artifacts, logs = [], {}, [], []
            
            return {
                'id': exp_id,
                'name': row.get('name', ''),
                'description': row.get('description', ''),
                'created_at': row.get('created_at', ''),
                'status': row.get('status', 'running'),
                'metrics': metrics,
                'parameters': parameters,
                'artifacts': artifacts,
                'logs': logs,
                'last_updated': row.get('last_updated', '')
            }
            
        except Exception as e:
            logger.warning(f"Failed to convert dataset row to experiment: {e}")
            return None
    
    def _load_backup_experiments(self):
        """Load backup experiments when dataset is not available"""
        logger.info("🔄 Loading backup experiments...")
        
        backup_experiments = {
            'exp_20250720_130853': {
                'id': 'exp_20250720_130853',
                'name': 'petite-elle-l-aime-3',
                'description': 'SmolLM3 fine-tuning experiment',
                'created_at': '2025-07-20T11:20:01.780908',
                'status': 'running',
                'metrics': [
                    {
                        'timestamp': '2025-07-20T11:20:01.780908',
                        'step': 25,
                        'metrics': {
                            'loss': 1.1659,
                            'grad_norm': 10.3125,
                            'learning_rate': 7e-08,
                            'num_tokens': 1642080.0,
                            'mean_token_accuracy': 0.75923578992486,
                            'epoch': 0.004851130919895701
                        }
                    },
                    {
                        'timestamp': '2025-07-20T11:26:39.042155',
                        'step': 50,
                        'metrics': {
                            'loss': 1.165,
                            'grad_norm': 10.75,
                            'learning_rate': 1.4291666666666667e-07,
                            'num_tokens': 3324682.0,
                            'mean_token_accuracy': 0.7577659255266189,
                            'epoch': 0.009702261839791402
                        }
                    },
                    {
                        'timestamp': '2025-07-20T11:33:16.203045',
                        'step': 75,
                        'metrics': {
                            'loss': 1.1639,
                            'grad_norm': 10.6875,
                            'learning_rate': 2.1583333333333334e-07,
                            'num_tokens': 4987941.0,
                            'mean_token_accuracy': 0.7581205774843692,
                            'epoch': 0.014553392759687101
                        }
                    },
                    {
                        'timestamp': '2025-07-20T11:39:53.453917',
                        'step': 100,
                        'metrics': {
                            'loss': 1.1528,
                            'grad_norm': 10.75,
                            'learning_rate': 2.8875e-07,
                            'num_tokens': 6630190.0,
                            'mean_token_accuracy': 0.7614579878747463,
                            'epoch': 0.019404523679582803
                        }
                    }
                ],
                'parameters': {
                    'model_name': 'HuggingFaceTB/SmolLM3-3B',
                    'max_seq_length': 12288,
                    'use_flash_attention': True,
                    'use_gradient_checkpointing': False,
                    'batch_size': 8,
                    'gradient_accumulation_steps': 16,
                    'learning_rate': 3.5e-06,
                    'weight_decay': 0.01,
                    'warmup_steps': 1200,
                    'max_iters': 18000,
                    'eval_interval': 1000,
                    'log_interval': 25,
                    'save_interval': 2000,
                    'optimizer': 'adamw_torch',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-08,
                    'scheduler': 'cosine',
                    'min_lr': 3.5e-07,
                    'fp16': False,
                    'bf16': True,
                    'ddp_backend': 'nccl',
                    'ddp_find_unused_parameters': False,
                    'save_steps': 2000,
                    'eval_steps': 1000,
                    'logging_steps': 25,
                    'save_total_limit': 5,
                    'eval_strategy': 'steps',
                    'metric_for_best_model': 'eval_loss',
                    'greater_is_better': False,
                    'load_best_model_at_end': True,
                    'data_dir': None,
                    'train_file': None,
                    'validation_file': None,
                    'test_file': None,
                    'use_chat_template': True,
                    'chat_template_kwargs': {'add_generation_prompt': True, 'no_think_system_message': True},
                    'enable_tracking': True,
                    'trackio_url': 'https://tonic-test-trackio-test.hf.space',
                    'trackio_token': None,
                    'log_artifacts': True,
                    'log_metrics': True,
                    'log_config': True,
                    'experiment_name': 'petite-elle-l-aime-3',
                    'dataset_name': 'legmlai/openhermes-fr',
                    'dataset_split': 'train',
                    'input_field': 'prompt',
                    'target_field': 'accepted_completion',
                    'filter_bad_entries': True,
                    'bad_entry_field': 'bad_entry',
                    'packing': False,
                    'max_prompt_length': 12288,
                    'max_completion_length': 8192,
                    'truncation': True,
                    'dataloader_num_workers': 10,
                    'dataloader_pin_memory': True,
                    'dataloader_prefetch_factor': 3,
                    'max_grad_norm': 1.0,
                    'group_by_length': True
                },
                'artifacts': [],
                'logs': []
            },
            'exp_20250720_134319': {
                'id': 'exp_20250720_134319',
                'name': 'petite-elle-l-aime-3-1',
                'description': 'SmolLM3 fine-tuning experiment',
                'created_at': '2025-07-20T11:54:31.993219',
                'status': 'running',
                'metrics': [
                    {
                        'timestamp': '2025-07-20T11:54:31.993219',
                        'step': 25,
                        'metrics': {
                            'loss': 1.166,
                            'grad_norm': 10.375,
                            'learning_rate': 7e-08,
                            'num_tokens': 1642080.0,
                            'mean_token_accuracy': 0.7590958896279335,
                            'epoch': 0.004851130919895701,
                            'gpu_0_memory_allocated': 17.202261447906494,
                            'gpu_0_memory_reserved': 75.474609375,
                            'gpu_0_utilization': 0,
                            'cpu_percent': 2.7,
                            'memory_percent': 10.1
                        }
                    },
                    {
                        'timestamp': '2025-07-20T11:54:33.589487',
                        'step': 25,
                        'metrics': {
                            'gpu_0_memory_allocated': 17.202261447906494,
                            'gpu_0_memory_reserved': 75.474609375,
                            'gpu_0_utilization': 0,
                            'cpu_percent': 2.7,
                            'memory_percent': 10.1
                        }
                    }
                ],
                'parameters': {
                    'model_name': 'HuggingFaceTB/SmolLM3-3B',
                    'max_seq_length': 12288,
                    'use_flash_attention': True,
                    'use_gradient_checkpointing': False,
                    'batch_size': 8,
                    'gradient_accumulation_steps': 16,
                    'learning_rate': 3.5e-06,
                    'weight_decay': 0.01,
                    'warmup_steps': 1200,
                    'max_iters': 18000,
                    'eval_interval': 1000,
                    'log_interval': 25,
                    'save_interval': 2000,
                    'optimizer': 'adamw_torch',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-08,
                    'scheduler': 'cosine',
                    'min_lr': 3.5e-07,
                    'fp16': False,
                    'bf16': True,
                    'ddp_backend': 'nccl',
                    'ddp_find_unused_parameters': False,
                    'save_steps': 2000,
                    'eval_steps': 1000,
                    'logging_steps': 25,
                    'save_total_limit': 5,
                    'eval_strategy': 'steps',
                    'metric_for_best_model': 'eval_loss',
                    'greater_is_better': False,
                    'load_best_model_at_end': True,
                    'data_dir': None,
                    'train_file': None,
                    'validation_file': None,
                    'test_file': None,
                    'use_chat_template': True,
                    'chat_template_kwargs': {'add_generation_prompt': True, 'no_think_system_message': True},
                    'enable_tracking': True,
                    'trackio_url': 'https://tonic-test-trackio-test.hf.space',
                    'trackio_token': None,
                    'log_artifacts': True,
                    'log_metrics': True,
                    'log_config': True,
                    'experiment_name': 'petite-elle-l-aime-3-1',
                    'dataset_name': 'legmlai/openhermes-fr',
                    'dataset_split': 'train',
                    'input_field': 'prompt',
                    'target_field': 'accepted_completion',
                    'filter_bad_entries': True,
                    'bad_entry_field': 'bad_entry',
                    'packing': False,
                    'max_prompt_length': 12288,
                    'max_completion_length': 8192,
                    'truncation': True,
                    'dataloader_num_workers': 10,
                    'dataloader_pin_memory': True,
                    'dataloader_prefetch_factor': 3,
                    'max_grad_norm': 1.0,
                    'group_by_length': True
                },
                'artifacts': [],
                'logs': []
            }
        }
        
        self.experiments = backup_experiments
        self.current_experiment = 'exp_20250720_134319'
        logger.info(f"✅ Loaded {len(backup_experiments)} backup experiments")
    
    def _save_experiments(self):
        """Save experiments to HF Dataset with data preservation

        Note: This saves the full in-memory set. Prefer per-operation upsert via
        dataset manager when available to reduce overwrite risk.
        """
        try:
            if self.using_backup_data:
                logger.warning("⚠️ Using backup data; skip saving to dataset to avoid overwriting with demo values")
                return
            # Use dataset manager for safe operations if available
            if self.dataset_manager:
                logger.info("💾 Saving experiments using dataset manager (data preservation)")
                
                # Convert current experiments to dataset format
                experiments_to_save = []
                for exp_id, exp_data in self.experiments.items():
                    experiment_entry = {
                        'experiment_id': exp_id,
                        'name': exp_data.get('name', ''),
                        'description': exp_data.get('description', ''),
                        'created_at': exp_data.get('created_at', ''),
                        'status': exp_data.get('status', 'running'),
                        'metrics': json.dumps(exp_data.get('metrics', []), default=str),
                        'parameters': json.dumps(exp_data.get('parameters', {}), default=str),
                        'artifacts': json.dumps(exp_data.get('artifacts', []), default=str),
                        'logs': json.dumps(exp_data.get('logs', []), default=str),
                        'last_updated': datetime.now().isoformat()
                    }
                    experiments_to_save.append(experiment_entry)
                
                # Use dataset manager to save with data preservation
                success = self.dataset_manager.save_experiments(
                    experiments_to_save,
                    f"Update experiments from Trackio Space ({len(experiments_to_save)} total experiments)"
                )
                
                if success:
                    logger.info(f"✅ Successfully saved {len(experiments_to_save)} experiments with data preservation")
                else:
                    logger.error("❌ Failed to save experiments using dataset manager")
                    # Fallback to legacy method
                    self._save_experiments_legacy()
                
                return
            
            # Fallback to legacy method if dataset manager not available
            self._save_experiments_legacy()
                
        except Exception as e:
            logger.error(f"❌ Failed to save experiments: {e}")
            # Fallback to legacy method
            self._save_experiments_legacy()

    def _upsert_experiment(self, experiment_id: str):
        """Non-destructive upsert of a single experiment when dataset manager is available."""
        try:
            if not self.dataset_manager:
                # Fallback to legacy save of full set
                self._save_experiments()
                return
            exp = self.experiments.get(experiment_id)
            if not exp:
                return
            payload = {
                'experiment_id': experiment_id,
                'name': exp.get('name', ''),
                'description': exp.get('description', ''),
                'created_at': exp.get('created_at', ''),
                'status': exp.get('status', 'running'),
                'metrics': json.dumps(exp.get('metrics', []), default=str),
                'parameters': json.dumps(exp.get('parameters', {}), default=str),
                'artifacts': json.dumps(exp.get('artifacts', []), default=str),
                'logs': json.dumps(exp.get('logs', []), default=str),
                'last_updated': datetime.now().isoformat()
            }
            self.dataset_manager.upsert_experiment(payload)
        except Exception as e:
            logger.warning(f"⚠️ Upsert failed, falling back to legacy save: {e}")
            self._save_experiments()
    
    def _save_experiments_legacy(self):
        """Legacy save method without data preservation (fallback only)"""
        try:
            if self.hf_token:
                from datasets import Dataset
                from huggingface_hub import HfApi
                
                logger.warning("⚠️ Using legacy save method - data preservation not guaranteed")
                
                # Convert experiments to dataset format
                dataset_data = []
                for exp_id, exp_data in self.experiments.items():
                    dataset_data.append({
                        'experiment_id': exp_id,
                        'name': exp_data.get('name', ''),
                        'description': exp_data.get('description', ''),
                        'created_at': exp_data.get('created_at', ''),
                        'status': exp_data.get('status', 'running'),
                        'metrics': json.dumps(exp_data.get('metrics', []), default=str),
                        'parameters': json.dumps(exp_data.get('parameters', {}), default=str),
                        'artifacts': json.dumps(exp_data.get('artifacts', []), default=str),
                        'logs': json.dumps(exp_data.get('logs', []), default=str),
                        'last_updated': datetime.now().isoformat()
                    })
                
                # Create dataset
                dataset = Dataset.from_list(dataset_data)
                
                # Push to HF Hub
                api = HfApi(token=self.hf_token)
                dataset.push_to_hub(
                    self.dataset_repo,
                    token=self.hf_token,
                    private=True,
                    commit_message=f"Legacy update: {len(dataset_data)} experiments"
                )
                
                logger.info(f"✅ Saved {len(dataset_data)} experiments to {self.dataset_repo} (legacy method)")
                
            else:
                logger.warning("⚠️ No HF_TOKEN available, experiments not saved to dataset")
                
        except Exception as e:
            logger.error(f"❌ Failed to save experiments with legacy method: {e}")
            # Fall back to local file for backup
            try:
                data = {
                    'experiments': self.experiments,
                    'current_experiment': self.current_experiment,
                    'last_updated': datetime.now().isoformat()
                }
                with open("trackio_experiments_backup.json", 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info("✅ Saved backup to local file")
            except Exception as backup_e:
                logger.error(f"❌ Failed to save backup: {backup_e}")
    
    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'metrics': [],
            'parameters': {},
            'artifacts': [],
            'logs': []
        }
        
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        self._upsert_experiment(experiment_id)
        
        logger.info(f"Created experiment: {experiment_id} - {name}")
        return experiment
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        metric_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'metrics': metrics
        }
        
        self.experiments[experiment_id]['metrics'].append(metric_entry)
        self._upsert_experiment(experiment_id)
        logger.info(f"Logged metrics for experiment {experiment_id}: {metrics}")
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]):
        """Log parameters for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.experiments[experiment_id]['parameters'].update(parameters)
        self._upsert_experiment(experiment_id)
        logger.info(f"Logged parameters for experiment {experiment_id}: {parameters}")
    
    def log_artifact(self, experiment_id: str, artifact_name: str, artifact_data: str):
        """Log an artifact for an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        artifact_entry = {
            'name': artifact_name,
            'timestamp': datetime.now().isoformat(),
            'data': artifact_data
        }
        
        self.experiments[experiment_id]['artifacts'].append(artifact_entry)
        self._save_experiments()
        logger.info(f"Logged artifact for experiment {experiment_id}: {artifact_name}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments"""
        return {
            'experiments': list(self.experiments.keys()),
            'current_experiment': self.current_experiment,
            'total_experiments': len(self.experiments)
        }
    
    def update_experiment_status(self, experiment_id: str, status: str):
        """Update experiment status"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = status
            self._upsert_experiment(experiment_id)
            logger.info(f"Updated experiment {experiment_id} status to {status}")
    
    def get_metrics_dataframe(self, experiment_id: str) -> pd.DataFrame:
        """Get metrics as a pandas DataFrame for plotting"""
        if experiment_id not in self.experiments:
            return pd.DataFrame()
        
        experiment = self.experiments[experiment_id]
        if not experiment['metrics']:
            return pd.DataFrame()
        
        # Convert metrics to DataFrame
        data = []
        for metric_entry in experiment['metrics']:
            step = metric_entry.get('step', 0)
            timestamp = metric_entry.get('timestamp', '')
            metrics = metric_entry.get('metrics', {})
            
            row = {'step': step, 'timestamp': timestamp}
            row.update(metrics)
            data.append(row)
        
        return pd.DataFrame(data)

# Global instance
trackio_space = TrackioSpace()

def update_trackio_config(hf_token: str, dataset_repo: str) -> str:
    """Update TrackioSpace configuration with new HF token and dataset repository"""
    global trackio_space
    
    try:
        # Create new instance with updated configuration
        trackio_space = TrackioSpace(hf_token=hf_token if hf_token.strip() else None, 
                                   dataset_repo=dataset_repo if dataset_repo.strip() else None)
        
        # Reload experiments with new configuration
        trackio_space._load_experiments()
        
        # Check if dataset manager is available
        manager_status = "✅ Available (data preservation enabled)" if trackio_space.dataset_manager else "⚠️ Not available (legacy mode)"
        
        return f"✅ Configuration updated successfully!\n📊 Dataset: {trackio_space.dataset_repo}\n🔑 HF Token: {'Set' if trackio_space.hf_token else 'Not set'}\n🛡️ Data Manager: {manager_status}\n📈 Loaded {len(trackio_space.experiments)} experiments"
        
    except Exception as e:
        return f"❌ Failed to update configuration: {str(e)}"

def test_dataset_connection(hf_token: str, dataset_repo: str) -> str:
    """Test connection to HF Dataset repository"""
    try:
        if not hf_token.strip():
            return "❌ Please provide a Hugging Face token"
        
        if not dataset_repo.strip():
            return "❌ Please provide a dataset repository"
        
        from datasets import load_dataset
        
        # Test loading the dataset
        dataset = load_dataset(dataset_repo, token=hf_token)
        
        # Count experiments and analyze structure
        experiment_count = len(dataset['train']) if 'train' in dataset else 0
        
        # Get column information
        columns = list(dataset['train'].column_names) if 'train' in dataset else []
        
        # Sample first few experiment IDs
        sample_experiments = []
        if 'train' in dataset and experiment_count > 0:
            for i, row in enumerate(dataset['train']):
                if i >= 3:  # Only show first 3
                    break
                sample_experiments.append(row.get('experiment_id', 'unknown'))
        
        result = f"✅ Connection successful!\n📊 Dataset: {dataset_repo}\n📈 Found {experiment_count} experiments\n🔗 Dataset URL: https://huggingface.co/datasets/{dataset_repo}\n\n"
        result += f"📋 Dataset Columns: {', '.join(columns)}\n"
        if sample_experiments:
            result += f"🔬 Sample Experiments: {', '.join(sample_experiments)}\n"
        
        # Test parsing one experiment if available
        if 'train' in dataset and experiment_count > 0:
            first_row = dataset['train'][0]
            exp_id = first_row.get('experiment_id', 'unknown')
            metrics_raw = first_row.get('metrics', '[]')
            
            try:
                if isinstance(metrics_raw, str):
                    metrics = json.loads(metrics_raw)
                    metrics_count = len(metrics) if isinstance(metrics, list) else 0
                    result += f"📊 First experiment ({exp_id}) metrics: {metrics_count} entries\n"
                else:
                    result += f"📊 First experiment ({exp_id}) metrics: Non-string format\n"
            except json.JSONDecodeError as e:
                result += f"⚠️ JSON parse error in first experiment: {e}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Connection failed: {str(e)}\n\n💡 Troubleshooting:\n1. Check your HF token is correct\n2. Verify the dataset repository exists\n3. Ensure your token has read access to the dataset"

def create_dataset_repository(hf_token: str, dataset_repo: str) -> str:
    """Create HF Dataset repository if it doesn't exist"""
    try:
        if not hf_token.strip():
            return "❌ Please provide a Hugging Face token"
        
        if not dataset_repo.strip():
            return "❌ Please provide a dataset repository"
        
        from datasets import Dataset
        from huggingface_hub import HfApi
        
        # Parse username and dataset name
        if '/' not in dataset_repo:
            return "❌ Dataset repository must be in format: username/dataset-name"
        
        username, dataset_name = dataset_repo.split('/', 1)
        
        # Create API client
        api = HfApi(token=hf_token)
        
        # Check if dataset exists
        try:
            api.dataset_info(dataset_repo)
            return f"✅ Dataset {dataset_repo} already exists!\n🛡️ Data preservation is enabled for existing datasets\n🔗 View at: https://huggingface.co/datasets/{dataset_repo}"
        except:
            # Dataset doesn't exist, create it
            pass
        
        # Try to initialize dataset manager to use its repository creation
        try:
            # Import dataset manager
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
            from dataset_utils import TrackioDatasetManager
            
            # Create dataset manager instance
            dataset_manager = TrackioDatasetManager(dataset_repo, hf_token)
            
            # Check if dataset exists using the manager
            exists = dataset_manager.check_dataset_exists()
            if exists:
                return f"✅ Dataset {dataset_repo} already exists!\n🛡️ Data preservation is enabled\n🔗 View at: https://huggingface.co/datasets/{dataset_repo}"
            
        except ImportError:
            # Dataset manager not available, use legacy method
            pass
        except Exception as e:
            # Dataset manager failed, use legacy method
            logger.warning(f"Dataset manager failed: {e}, using legacy method")
        
        # Create empty dataset with proper structure
        empty_dataset = Dataset.from_dict({
            'experiment_id': [],
            'name': [],
            'description': [],
            'created_at': [],
            'status': [],
            'metrics': [],
            'parameters': [],
            'artifacts': [],
            'logs': [],
            'last_updated': []
        })
        
        # Push to hub
        empty_dataset.push_to_hub(
            dataset_repo,
            token=hf_token,
            private=True,
            commit_message="Create Trackio experiment dataset with data preservation support"
        )
        
        return f"✅ Dataset {dataset_repo} created successfully!\n🛡️ Data preservation is now enabled\n🔗 View at: https://huggingface.co/datasets/{dataset_repo}\n📊 Ready to store experiments safely"
        
    except Exception as e:
        return f"❌ Failed to create dataset: {str(e)}\n\n💡 Troubleshooting:\n1. Check your HF token has write permissions\n2. Verify the username in the repository name\n3. Ensure the dataset name is valid\n4. Check internet connectivity"

"""
Initialize API client for remote data. We do not hardcode a default test URL to avoid
overwriting dataset content with demo data. The API client will only be initialized
when TRACKIO_URL or TRACKIO_SPACE_ID is present.
"""
api_client = None
try:
    from trackio_api_client import TrackioAPIClient
    # Resolve Trackio space from environment
    trackio_url_env = os.environ.get('TRACKIO_URL') or os.environ.get('TRACKIO_SPACE_ID')
    if trackio_url_env:
        # Clean up URL to avoid double protocol issues
        trackio_url = trackio_url_env
        if trackio_url.startswith('https://https://'):
            trackio_url = trackio_url.replace('https://https://', 'https://')
        elif trackio_url.startswith('http://http://'):
            trackio_url = trackio_url.replace('http://http://', 'http://')
        api_client = TrackioAPIClient(trackio_url)
        logger.info(f"✅ API client initialized for remote data access: {trackio_url}")
    else:
        logger.info("No TRACKIO_URL/TRACKIO_SPACE_ID set; remote API client disabled")
except ImportError:
    logger.warning("⚠️ API client not available, using local data only")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize API client: {e}, using local data only")

# Add Hugging Face Spaces compatibility
def is_huggingface_spaces():
    """Check if running on Hugging Face Spaces"""
    return os.environ.get('SPACE_ID') is not None

def get_persistent_data_path():
    """Get a persistent data path for Hugging Face Spaces"""
    if is_huggingface_spaces():
        # Use a path that might persist better on HF Spaces
        return "/tmp/trackio_experiments.json"
    else:
        return "trackio_experiments.json"

# Override the data file path for HF Spaces
if is_huggingface_spaces():
    logger.info("🚀 Running on Hugging Face Spaces - using persistent storage")
    trackio_space.data_file = get_persistent_data_path()

def get_remote_experiment_data(experiment_id: str) -> Dict[str, Any]:
    """Get experiment data from remote API"""
    if api_client is None:
        return None
    
    try:
        # Get experiment details from API
        details_result = api_client.get_experiment_details(experiment_id)
        if "success" in details_result:
            return {"remote": True, "data": details_result["data"]}
        else:
            logger.warning(f"Failed to get remote data for {experiment_id}: {details_result}")
            return None
    except Exception as e:
        logger.error(f"Error getting remote data: {e}")
        return None

def parse_remote_metrics_data(experiment_details: str) -> pd.DataFrame:
    """Parse metrics data from remote experiment details"""
    try:
        # Look for metrics in the experiment details
        lines = experiment_details.split('\n')
        metrics_data = []
        
        # First try to parse the new format with structured experiment details
        for line in lines:
            if 'Step:' in line and 'Metrics:' in line:
                # Extract step and metrics from the line
                try:
                    # Parse step number
                    step_part = line.split('Step:')[1].split('Metrics:')[0].strip()
                    step = int(step_part)
                    
                    # Parse metrics JSON
                    metrics_part = line.split('Metrics:')[1].strip()
                    metrics = json.loads(metrics_part)
                    
                    # Add timestamp
                    row = {'step': step, 'timestamp': datetime.now().isoformat()}
                    row.update(metrics)
                    metrics_data.append(row)
                    
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse metrics line: {line} - {e}")
                    continue
        
        # If no metrics found in text format, try to parse from the dataset directly
        if not metrics_data:
            logger.info("No metrics found in text format, trying to parse from experiment structure")
            # This will be handled by the updated get_remote_experiment_data function
            
        if metrics_data:
            return pd.DataFrame(metrics_data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error parsing remote metrics: {e}")
        return pd.DataFrame()

def get_metrics_dataframe(experiment_id: str) -> pd.DataFrame:
    """Get metrics as a pandas DataFrame for plotting - tries dataset first, then local backup"""
    try:
        # First try to get data directly from the dataset using the dataset manager
        if trackio_space.dataset_manager:
            logger.info(f"Getting metrics for {experiment_id} from dataset")
            experiment_data = trackio_space.dataset_manager.get_experiment_by_id(experiment_id)
            
            if experiment_data:
                # Parse metrics from the dataset
                metrics_json = experiment_data.get('metrics', '[]')
                if isinstance(metrics_json, str):
                    try:
                        metrics_list = json.loads(metrics_json)
                        
                        # Convert to DataFrame format
                        df_data = []
                        for metric_entry in metrics_list:
                            if isinstance(metric_entry, dict):
                                step = metric_entry.get('step', 0)
                                timestamp = metric_entry.get('timestamp', '')
                                metrics = metric_entry.get('metrics', {})
                                
                                row = {'step': step, 'timestamp': timestamp}
                                row.update(metrics)
                                df_data.append(row)
                        
                        if df_data:
                            logger.info(f"Found {len(df_data)} metrics entries from dataset for {experiment_id}")
                            return pd.DataFrame(df_data)
                        else:
                            logger.warning(f"No valid metrics found in dataset for {experiment_id}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse metrics JSON for {experiment_id}: {e}")
                else:
                    logger.warning(f"Metrics data is not a JSON string for {experiment_id}")
            else:
                logger.warning(f"Experiment {experiment_id} not found in dataset")
        
        # Try legacy remote data approach
        remote_data = get_remote_experiment_data(experiment_id)
        if remote_data:
            logger.info(f"Using remote API data for {experiment_id}")
            # Parse the remote experiment details to extract metrics
            df = parse_remote_metrics_data(remote_data["data"])
            if not df.empty:
                logger.info(f"Found {len(df)} metrics entries from remote API")
                return df
            else:
                logger.warning(f"No metrics found in remote API data for {experiment_id}")
        
        # Fall back to local data
        logger.info(f"Using local backup data for {experiment_id}")
        return trackio_space.get_metrics_dataframe(experiment_id)
        
    except Exception as e:
        logger.error(f"Error getting metrics dataframe for {experiment_id}: {e}")
        # Fall back to local data
        logger.info(f"Falling back to local data for {experiment_id}")
        return trackio_space.get_metrics_dataframe(experiment_id)

def create_experiment_interface(name: str, description: str):
    """Create a new experiment"""
    try:
        experiment = trackio_space.create_experiment(name, description)
        # Return both the status message and a refreshed dropdown
        msg = f"✅ Experiment created successfully!\nID: {experiment['id']}\nName: {experiment['name']}\nStatus: {experiment['status']}"
        dropdown = gr.Dropdown(choices=get_experiment_dropdown_choices(), value=experiment['id'])
        return msg, dropdown
    except Exception as e:
        dropdown = gr.Dropdown(choices=get_experiment_dropdown_choices(), value=None)
        return f"❌ Error creating experiment: {str(e)}", dropdown

def log_metrics_interface(experiment_id: str, metrics_json: str, step: str) -> str:
    """Log metrics for an experiment"""
    try:
        metrics = json.loads(metrics_json)
        step_int = int(step) if step else None
        trackio_space.log_metrics(experiment_id, metrics, step_int)
        return f"✅ Metrics logged successfully for experiment {experiment_id}\nStep: {step_int}\nMetrics: {json.dumps(metrics, indent=2)}"
    except Exception as e:
        return f"❌ Error logging metrics: {str(e)}"

def log_parameters_interface(experiment_id: str, parameters_json: str) -> str:
    """Log parameters for an experiment"""
    try:
        parameters = json.loads(parameters_json)
        trackio_space.log_parameters(experiment_id, parameters)
        return f"✅ Parameters logged successfully for experiment {experiment_id}\nParameters: {json.dumps(parameters, indent=2)}"
    except Exception as e:
        return f"❌ Error logging parameters: {str(e)}"

def get_experiment_details(experiment_id: str) -> str:
    """Get experiment details"""
    try:
        experiment = trackio_space.get_experiment(experiment_id)
        if experiment:
            # Format the output nicely
            details = f"""
📊 EXPERIMENT DETAILS
====================
ID: {experiment['id']}
Name: {experiment['name']}
Description: {experiment['description']}
Status: {experiment['status']}
Created: {experiment['created_at']}
📈 METRICS COUNT: {len(experiment['metrics'])}
📋 PARAMETERS COUNT: {len(experiment['parameters'])}
📦 ARTIFACTS COUNT: {len(experiment['artifacts'])}
🔧 PARAMETERS:
{json.dumps(experiment['parameters'], indent=2)}
📊 LATEST METRICS:
"""
            if experiment['metrics']:
                latest_metrics = experiment['metrics'][-1]
                details += f"Step: {latest_metrics.get('step', 'N/A')}\n"
                details += f"Timestamp: {latest_metrics.get('timestamp', 'N/A')}\n"
                details += f"Metrics: {json.dumps(latest_metrics.get('metrics', {}), indent=2)}"
            else:
                details += "No metrics logged yet."
            
            return details
        else:
            return f"❌ Experiment {experiment_id} not found"
    except Exception as e:
        return f"❌ Error getting experiment details: {str(e)}"

def list_experiments_interface() -> str:
    """List all experiments with details"""
    try:
        experiments_info = trackio_space.list_experiments()
        experiments = trackio_space.experiments
        
        if not experiments:
            return "📭 No experiments found. Create one first!"
        
        result = f"📋 EXPERIMENTS OVERVIEW\n{'='*50}\n"
        result += f"Total Experiments: {len(experiments)}\n"
        result += f"Current Experiment: {experiments_info['current_experiment']}\n\n"
        
        for exp_id, exp_data in experiments.items():
            status_emoji = {
                'running': '🟢',
                'completed': '✅',
                'failed': '❌',
                'paused': '⏸️'
            }.get(exp_data['status'], '❓')
            
            result += f"{status_emoji} {exp_id}\n"
            result += f"   Name: {exp_data['name']}\n"
            result += f"   Status: {exp_data['status']}\n"
            result += f"   Created: {exp_data['created_at']}\n"
            result += f"   Metrics: {len(exp_data['metrics'])} entries\n"
            result += f"   Parameters: {len(exp_data['parameters'])} entries\n"
            result += f"   Artifacts: {len(exp_data['artifacts'])} entries\n\n"
        
        return result
    except Exception as e:
        return f"❌ Error listing experiments: {str(e)}"

def update_experiment_status_interface(experiment_id: str, status: str) -> str:
    """Update experiment status"""
    try:
        trackio_space.update_experiment_status(experiment_id, status)
        return f"✅ Experiment {experiment_id} status updated to {status}"
    except Exception as e:
        return f"❌ Error updating experiment status: {str(e)}"

def create_metrics_plot(experiment_id: str, metric_name: str = "loss") -> go.Figure:
    """Create a plot for a specific metric (supports all logged metrics, including new ones)"""
    try:
        df = get_metrics_dataframe(experiment_id)
        if df.empty:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        if metric_name not in df.columns:
            # Show available metrics
            available_metrics = [col for col in df.columns if col not in ['step', 'timestamp']]
            fig = go.Figure()
            fig.add_annotation(
                text=f"Available metrics: {', '.join(available_metrics)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = px.line(df, x='step', y=metric_name, title=f'{metric_name} over time')
        fig.update_layout(
            xaxis_title="Training Step",
            yaxis_title=metric_name.title(),
            hovermode='x unified'
        )
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_experiment_comparison(experiment_ids: str) -> go.Figure:
    """Compare multiple experiments"""
    try:
        exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(',')]
        
        fig = go.Figure()
        
        for exp_id in exp_ids:
            df = get_metrics_dataframe(exp_id)
            if not df.empty and 'loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['step'],
                    y=df['loss'],
                    mode='lines+markers',
                    name=f"{exp_id} - Loss",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Experiment Comparison - Loss",
            xaxis_title="Training Step",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating comparison: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def simulate_training_data(experiment_id: str):
    """Simulate training data for demonstration"""
    try:
        import random
        import time
        last_time = time.time()
        for step in range(0, 1000, 50):
            # Simulate loss decreasing over time
            loss = 2.0 * np.exp(-step / 500) + 0.1 * np.random.random()
            accuracy = 0.3 + 0.6 * (1 - np.exp(-step / 300)) + 0.05 * np.random.random()
            lr = 3.5e-6 * (0.9 ** (step // 200))
            batch_size = 8
            seq_len = 2048
            total_tokens = batch_size * seq_len
            padding_tokens = random.randint(0, batch_size * 32)
            truncated_tokens = random.randint(0, batch_size * 8)
            now = time.time()
            step_time = random.uniform(0.4, 0.7)
            throughput = total_tokens / step_time
            token_acc = accuracy
            gate_ortho = random.uniform(0.01, 0.05)
            center = random.uniform(0.01, 0.05)
            metrics = {
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "learning_rate": round(lr, 8),
                "gpu_memory": round(20 + 5 * np.random.random(), 2),
                "training_time": round(0.5 + 0.2 * np.random.random(), 3),
                "total_tokens": total_tokens,
                "padding_tokens": padding_tokens,
                "truncated_tokens": truncated_tokens,
                "throughput": throughput,
                "step_time": step_time,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "token_acc": token_acc,
                "train/gate_ortho": gate_ortho,
                "train/center": center
            }
            trackio_space.log_metrics(experiment_id, metrics, step)
            last_time = now
        return f"✅ Simulated training data for experiment {experiment_id}\nAdded 20 metric entries (steps 0-950)"
    except Exception as e:
        return f"❌ Error simulating data: {str(e)}"

def create_demo_experiment():
    """Create a demo experiment with training data"""
    try:
        # Create demo experiment
        experiment = trackio_space.create_experiment(
            "demo_smollm3_training",
            "Demo experiment with simulated training data"
        )
        
        experiment_id = experiment['id']
        
        # Add some demo parameters
        parameters = {
            "model_name": "HuggingFaceTB/SmolLM3-3B",
            "batch_size": 8,
            "learning_rate": 3.5e-6,
            "max_iters": 18000,
            "mixed_precision": "bf16",
            "dataset": "legmlai/openhermes-fr"
        }
        trackio_space.log_parameters(experiment_id, parameters)
        
        # Add demo training data
        simulate_training_data(experiment_id)
        
        return f"✅ Demo experiment created: {experiment_id}\nYou can now test the visualization with this experiment!"
    except Exception as e:
        return f"❌ Error creating demo experiment: {str(e)}"


# Helper functions for the new interface
def get_experiment_dropdown_choices() -> list:
    """Get the list of experiments for the dropdown"""
    experiments = list(trackio_space.experiments.keys())
    if not experiments:
        return ["No experiments available"]
    return experiments

def refresh_experiment_dropdown() -> tuple:
    """Refresh the experiment dropdown and return current choices"""
    choices = get_experiment_dropdown_choices()
    current_value = choices[0] if choices and choices[0] != "No experiments available" else None
    return gr.Dropdown(choices=choices, value=current_value)

def get_available_metrics_for_experiments(experiment_ids: list) -> list:
    """Get all available metrics across selected experiments"""
    try:
        all_metrics = set()
        for exp_id in experiment_ids:
            df = get_metrics_dataframe(exp_id)
            if not df.empty:
                # Get numeric columns (excluding step and timestamp)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col not in ['step']]
                all_metrics.update(numeric_cols)
        
        return sorted(list(all_metrics))
    except Exception as e:
        logger.error(f"Error getting available metrics: {str(e)}")
        return ["loss", "accuracy"]

def create_test_plot() -> go.Figure:
    """Create a simple test plot to verify plotly rendering works"""
    try:
        # Create simple test data
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 2, 3, 5]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name='Test Data',
            line=dict(width=2, color='blue'),
            marker=dict(size=5, color='red'),
            connectgaps=True,
            hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Test Plot - If you can see this, plotly is working!",
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        logger.info("Test plot created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating test plot: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Test plot error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def get_experiment_status_summary(experiment_id: str) -> str:
    """Get a formatted summary of experiment status and metadata"""
    try:
        experiment = trackio_space.get_experiment(experiment_id)
        if not experiment:
            return f"Experiment {experiment_id} not found."
        
        summary = f"📋 EXPERIMENT STATUS SUMMARY\n{'='*50}\n"
        summary += f"ID: {experiment['id']}\n"
        summary += f"Name: {experiment['name']}\n"
        summary += f"Description: {experiment['description']}\n"
        summary += f"Status: {experiment['status']}\n"
        summary += f"Created: {experiment['created_at']}\n"
        summary += f"Metrics entries: {len(experiment['metrics'])}\n"
        summary += f"Parameters: {len(experiment['parameters'])}\n"
        summary += f"Artifacts: {len(experiment['artifacts'])}\n"
        summary += f"Logs: {len(experiment['logs'])}\n"
        
        # Add latest metrics if available
        if experiment['metrics']:
            latest = experiment['metrics'][-1]
            summary += f"\n📈 LATEST METRICS (Step {latest.get('step', 'N/A')}):\n"
            for k, v in latest.get('metrics', {}).items():
                summary += f"  {k}: {v}\n"
        
        return summary
    except Exception as e:
        return f"Error generating status summary: {str(e)}"

def get_experiment_parameters_summary(experiment_id: str) -> str:
    """Get a formatted summary of experiment parameters"""
    try:
        experiment = trackio_space.get_experiment(experiment_id)
        if not experiment:
            return f"Experiment {experiment_id} not found."
        
        params = experiment.get('parameters', {})
        if not params:
            return "No parameters logged for this experiment."
        
        summary = f"🔧 PARAMETERS FOR {experiment_id}\n{'='*50}\n"
        
        # Group parameters by category
        model_params = {k: v for k, v in params.items() if 'model' in k.lower() or 'name' in k.lower()}
        training_params = {k: v for k, v in params.items() if any(x in k.lower() for x in ['learning', 'batch', 'epoch', 'step', 'iter', 'optimizer'])}
        data_params = {k: v for k, v in params.items() if any(x in k.lower() for x in ['data', 'dataset', 'file', 'split'])}
        other_params = {k: v for k, v in params.items() if k not in model_params and k not in training_params and k not in data_params}
        
        if model_params:
            summary += "🤖 MODEL PARAMETERS:\n"
            for k, v in model_params.items():
                summary += f"  {k}: {v}\n"
            summary += "\n"
        
        if training_params:
            summary += "🏃 TRAINING PARAMETERS:\n"
            for k, v in training_params.items():
                summary += f"  {k}: {v}\n"
            summary += "\n"
        
        if data_params:
            summary += "📁 DATA PARAMETERS:\n"
            for k, v in data_params.items():
                summary += f"  {k}: {v}\n"
            summary += "\n"
        
        if other_params:
            summary += "⚙️ OTHER PARAMETERS:\n"
            for k, v in other_params.items():
                summary += f"  {k}: {v}\n"
        
        return summary
    except Exception as e:
        return f"Error generating parameters summary: {str(e)}"

def get_experiment_metrics_summary(experiment_id: str) -> str:
    """Get a summary of all metrics for an experiment"""
    try:
        df = get_metrics_dataframe(experiment_id)
        if df.empty:
            return "No metrics data available for this experiment.\n\n💡 This could mean:\n• The experiment hasn't started logging metrics yet\n• The experiment is using a different data format\n• No training has been performed on this experiment"
        
        # Get numeric columns (excluding step and timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['step']]
        
        if not numeric_cols:
            return "No numeric metrics found for this experiment.\n\n💡 This could mean:\n• Only timestamp data is available\n• Metrics are stored in a different format\n• The experiment hasn't logged any numeric metrics yet"
        
        summary = f"📊 METRICS SUMMARY FOR {experiment_id}\n{'='*50}\n"
        summary += f"Total data points: {len(df)}\n"
        summary += f"Steps range: {df['step'].min()} - {df['step'].max()}\n"
        summary += f"Available metrics: {', '.join(numeric_cols)}\n\n"
        
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    summary += f"{col}:\n"
                    summary += f"  Min: {values.min():.6f}\n"
                    summary += f"  Max: {values.max():.6f}\n"
                    summary += f"  Mean: {values.mean():.6f}\n"
                    summary += f"  Latest: {values.iloc[-1]:.6f}\n\n"
        
        return summary
    except Exception as e:
        return f"Error generating metrics summary: {str(e)}"

def create_combined_metrics_plot(experiment_id: str) -> go.Figure:
    """Create a combined plot showing all metrics for an experiment"""
    try:
        if not experiment_id:
            fig = go.Figure()
            fig.add_annotation(
                text="No experiment selected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Select an Experiment",
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
            
        df = get_metrics_dataframe(experiment_id)
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available for this experiment",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="No Data Available",
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
        
        # Get numeric columns (excluding step and timestamp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['step']]
        
        if not numeric_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric metrics found for this experiment",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="No Metrics Found",
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
        
        # Create subplots for multiple metrics
        from plotly.subplots import make_subplots
        
        # Determine number of rows and columns for subplots
        n_metrics = len(numeric_cols)
        n_cols = min(3, n_metrics)  # Max 3 columns
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.05,
            horizontal_spacing=0.1
        )
        
        # Define colors for different metrics
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i, metric in enumerate(numeric_cols):
            if metric in df.columns and not df[metric].isna().all():
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['step'].tolist(),
                        y=df[metric].tolist(),
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2, color=color),
                        marker=dict(size=4, color=color),
                        showlegend=False,
                        connectgaps=True
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"All Metrics for Experiment {experiment_id}",
            height=350 * n_rows,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update all subplot axes
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    zeroline=True, zerolinecolor='black',
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True, gridwidth=1, gridcolor='lightgray',
                    zeroline=True, zerolinecolor='black',
                    row=i, col=j
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating combined metrics plot: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating combined plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def update_dashboard(experiment_id: str) -> tuple:
    """Update all dashboard components for a selected experiment"""
    try:
        if not experiment_id or experiment_id == "No experiments available":
            return (
                "Please select an experiment from the dropdown.",
                "No experiment selected.",
                "No experiment selected.",
                create_combined_metrics_plot(""),
                "No experiment selected."
            )
        
        # Get all the dashboard components
        status_summary = get_experiment_status_summary(experiment_id)
        parameters_summary = get_experiment_parameters_summary(experiment_id)
        metrics_summary = get_experiment_metrics_summary(experiment_id)
        combined_plot = create_combined_metrics_plot(experiment_id)
        
        # Create a combined summary
        combined_summary = f"{status_summary}\n\n{parameters_summary}\n\n{metrics_summary}"
        
        return (
            status_summary,
            parameters_summary,
            metrics_summary,
            combined_plot,
            combined_summary
        )
    except Exception as e:
        error_msg = f"Error updating dashboard: {str(e)}"
        return (error_msg, error_msg, error_msg, create_combined_metrics_plot(""), error_msg)

def update_dashboard_metric_plot(experiment_id: str, metric_name: str = "loss") -> go.Figure:
    """Update the dashboard metric plot for a selected experiment and metric"""
    try:
        if not experiment_id or experiment_id == "No experiments available":
            return create_metrics_plot("", metric_name)
        
        return create_metrics_plot(experiment_id, metric_name)
    except Exception as e:
        logger.error(f"Error updating dashboard metric plot: {str(e)}")
        return create_metrics_plot("", metric_name)

def create_experiment_comparison_from_selection(selected_experiments: list, selected_metrics: list) -> go.Figure:
    """Create experiment comparison from checkbox selections"""
    try:
        if not selected_experiments:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select at least one experiment to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="No Experiments Selected",
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
        
        if not selected_metrics:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select at least one metric to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            fig.update_layout(
                title="No Metrics Selected",
                plot_bgcolor='white', paper_bgcolor='white'
            )
            return fig
        
        # Use the existing comparison function with comma-separated IDs
        experiment_ids_str = ",".join(selected_experiments)
        return create_experiment_comparison(experiment_ids_str)
        
    except Exception as e:
        logger.error(f"Error creating comparison from selection: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating comparison: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def refresh_comparison_options() -> tuple:
    """Refresh the experiment and metric options for comparison"""
    try:
        # Get updated experiment choices
        experiment_choices = get_experiment_dropdown_choices()
        if experiment_choices == ["No experiments available"]:
            experiment_choices = []
        
        # Get available metrics from all experiments
        all_experiments = list(trackio_space.experiments.keys())
        available_metrics = get_available_metrics_for_experiments(all_experiments)
        
        # Default to common metrics if available
        default_metrics = []
        common_metrics = ["loss", "accuracy", "learning_rate", "gpu_memory"]
        for metric in common_metrics:
            if metric in available_metrics:
                default_metrics.append(metric)
        
        # If no common metrics, use first few available
        if not default_metrics and available_metrics:
            default_metrics = available_metrics[:2]
        
        return gr.CheckboxGroup(choices=experiment_choices, value=[]), gr.CheckboxGroup(choices=available_metrics, value=default_metrics)
    except Exception as e:
        logger.error(f"Error refreshing comparison options: {str(e)}")
        return gr.CheckboxGroup(choices=[], value=[]), gr.CheckboxGroup(choices=["loss", "accuracy"], value=[])

# Create Gradio interface
with gr.Blocks(title="Trackio - Experiment Tracking", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Trackio Experiment Tracking & Monitoring")
    gr.Markdown("Monitor and track your ML experiments with real-time visualization!")
    
    with gr.Tabs():
        # Dashboard Tab (NEW)
        with gr.Tab("📊 Dashboard"):
            gr.Markdown("### Comprehensive Experiment Dashboard")
            gr.Markdown("Select an experiment to view all its data, plots, and information in one place.")
            
            # Row 1: Experiment Selection
            with gr.Row():
                with gr.Column(scale=3):
                    # Experiment selection dropdown
                    experiment_dropdown = gr.Dropdown(
                        label="Select Experiment",
                        choices=get_experiment_dropdown_choices(),
                        value=get_experiment_dropdown_choices()[0] if get_experiment_dropdown_choices() and get_experiment_dropdown_choices()[0] != "No experiments available" else None,
                        info="Choose an experiment to view its dashboard"
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        refresh_dropdown_btn = gr.Button("🔄 Refresh List", variant="secondary", size="sm")
                        refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", size="sm")
            
            # Row 2: All Metrics Plots
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Markdown("### 📈 All Metrics Plots")
                    with gr.Row():
                        with gr.Column(scale=3):    
                            dashboard_plots = gr.Plot(
                                label="Training Metrics", 
                                container=True, 
                                show_label=True,
                                elem_classes=["plot-container"]
                            )
                
            # Row 3: Training Metrics Visualization Accordion
            with gr.Row():
                with gr.Accordion("📈 Training Metrics Visualization", open=False):
                    with gr.Row():
                        with gr.Column():
                            metric_dropdown = gr.Dropdown(
                                label="Metric to Plot",
                                choices=[
                                    "loss", "accuracy", "learning_rate", "gpu_memory", "training_time",
                                    "total_tokens", "truncated_tokens", "padding_tokens", "throughput", "step_time",
                                    "batch_size", "seq_len", "token_acc", "train/gate_ortho", "train/center"
                                ],
                                value="loss"
                            )
                            plot_btn = gr.Button("Create Plot", variant="primary")
                            test_plot_btn = gr.Button("Test Plot Rendering", variant="secondary")
                        
                    with gr.Row():
                        dashboard_metric_plot = gr.Plot(
                            label="Training Metrics", 
                            container=True, 
                            show_label=True,
                            elem_classes=["plot-container"]
                        )
                    
                    plot_btn.click(
                        create_metrics_plot,
                        inputs=[experiment_dropdown, metric_dropdown],
                        outputs=dashboard_metric_plot
                    )
                    
                    test_plot_btn.click(
                        create_test_plot,
                        inputs=[],
                        outputs=dashboard_metric_plot
                    )
            
            # Row 4: Accordion with Detailed Information
            with gr.Row():
                with gr.Accordion("📋 Experiment Details", open=False):
                    with gr.Tabs():
                        with gr.Tab("📋 Status"):
                            dashboard_status = gr.Textbox(
                                label="Experiment Status",
                                lines=8,
                                interactive=False
                            )
                        
                        with gr.Tab("🔧 Parameters"):
                            dashboard_parameters = gr.Textbox(
                                label="Experiment Parameters",
                                lines=12,
                                interactive=False
                            )
                        
                        with gr.Tab("📊 Metrics Summary"):
                            dashboard_metrics = gr.Textbox(
                                label="Metrics Summary",
                                lines=12,
                                interactive=False
                            )
                        
                        with gr.Tab("📋 Complete Summary"):
                            dashboard_summary = gr.Textbox(
                                label="Full Experiment Summary",
                                lines=20,
                                interactive=False
                            )
            
            # Connect the dashboard update function
            experiment_dropdown.change(
                update_dashboard,
                inputs=[experiment_dropdown],
                outputs=[dashboard_status, dashboard_parameters, dashboard_metrics, dashboard_plots, dashboard_summary]
            )
            
            refresh_dashboard_btn.click(
                update_dashboard,
                inputs=[experiment_dropdown],
                outputs=[dashboard_status, dashboard_parameters, dashboard_metrics, dashboard_plots, dashboard_summary]
            )
            
            # Connect the metric plot update function
            metric_dropdown.change(
                update_dashboard_metric_plot,
                inputs=[experiment_dropdown, metric_dropdown],
                outputs=[dashboard_metric_plot]
            )
            
            refresh_dropdown_btn.click(
                refresh_experiment_dropdown,
                inputs=[],
                outputs=[experiment_dropdown]
            )
        

        # Experiment Comparison Tab
        with gr.Tab("📊 Experiment Comparison"):
            gr.Markdown("### Compare Multiple Experiments")
            gr.Markdown("Select experiments and metrics to compare from the available options below.")
            
            # Selection controls
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Available Experiments")
                    experiment_checkboxes = gr.CheckboxGroup(
                        label="Select Experiments to Compare",
                        choices=get_experiment_dropdown_choices(),
                        value=[],
                        info="Choose experiments to include in the comparison"
                    )
                    
                    gr.Markdown("### Available Metrics")
                    metric_checkboxes = gr.CheckboxGroup(
                        label="Select Metrics to Compare",
                        choices=get_available_metrics_for_experiments(list(trackio_space.experiments.keys())),
                        value=["loss", "accuracy"],
                        info="Choose metrics to include in the comparison"
                    )
                    
                    with gr.Row():
                        comparison_btn = gr.Button("Compare Selected", variant="primary")
                        refresh_options_btn = gr.Button("🔄 Refresh Options", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Comparison Results")
                    gr.Markdown("The comparison will show subplots for the selected metrics across the selected experiments.")
            
            # Comparison plots as subplots
            comparison_plot = gr.Plot(
                label="Experiment Comparison Dashboard", 
                container=True, 
                show_label=True,
                elem_classes=["plot-container"]
            )
            
            comparison_btn.click(
                create_experiment_comparison_from_selection,
                inputs=[experiment_checkboxes, metric_checkboxes],
                outputs=comparison_plot
            )
            
            refresh_options_btn.click(
                refresh_comparison_options,
                inputs=[],
                outputs=[experiment_checkboxes, metric_checkboxes]
            )

        # Configuration Tab
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("### Configure HF Datasets Connection")
            gr.Markdown("Set your Hugging Face token and dataset repository for persistent experiment storage.")
            
            with gr.Row():
                with gr.Column():
                    hf_token_input = gr.Textbox(
                        label="Hugging Face Token",
                        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                        type="password",
                        info="Your HF token for dataset access (optional - will use environment variable if not set)"
                    )
                    dataset_repo_input = gr.Textbox(
                        label="Dataset Repository",
                        placeholder="your-username/your-dataset-name",
                        value="Tonic/trackio-experiments",
                        info="HF Dataset repository for experiment storage"
                    )
                    
                    with gr.Row():
                        update_config_btn = gr.Button("Update Configuration", variant="primary")
                        test_connection_btn = gr.Button("Test Connection", variant="secondary")
                        create_repo_btn = gr.Button("Create Dataset", variant="success")
                    
                    gr.Markdown("### Current Configuration")
                    current_config_output = gr.Textbox(
                        label="Status",
                        lines=10,
                        interactive=False,
                        value=f"📊 Dataset: {trackio_space.dataset_repo}\n🔑 HF Token: {'Set' if trackio_space.hf_token else 'Not set'}\n🛡️ Data Preservation: {'✅ Enabled' if trackio_space.dataset_manager else '⚠️ Legacy Mode'}\n📈 Experiments: {len(trackio_space.experiments)}\n📋 Available Experiments: {', '.join(list(trackio_space.experiments.keys())[:3])}{'...' if len(trackio_space.experiments) > 3 else ''}"
                    )
                
                with gr.Column():
                    gr.Markdown("### Configuration Help")
                    gr.Markdown("""
                    **Getting Your HF Token:**
                    1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
                    2. Click "New token"
                    3. Give it a name (e.g., "Trackio Access")
                    4. Select "Write" permissions
                    5. Copy the token and paste it above
                    
                    **Dataset Repository:**
                    - Format: `username/dataset-name`
                    - Examples: `tonic/trackio-experiments`, `your-username/my-experiments`
                    - Use "Create Dataset" button to create a new repository
                    
                    **Environment Variables:**
                    You can also set these as environment variables:
                    - `HF_TOKEN`: Your Hugging Face token
                    - `TRACKIO_DATASET_REPO`: Dataset repository
                    
                    **Data Preservation:**
                    - ✅ **Enabled**: All experiment data is preserved when adding/updating experiments
                    - ⚠️ **Legacy Mode**: Data preservation not guaranteed (fallback mode)
                    - Data preservation requires the dataset management utilities to be available
                    
                    **Actions:**
                    - **Update Configuration**: Apply new settings and reload experiments
                    - **Test Connection**: Verify access to the dataset repository
                    - **Create Dataset**: Create a new dataset repository if it doesn't exist
                    """)
            
            # Experiment Management Accordion
            with gr.Accordion("🔧 Experiment Management", open=False):
                with gr.Tabs():
                    # Create Experiment Tab
                    with gr.Tab("Create Experiment"):
                        gr.Markdown("### Create a New Experiment")
                        with gr.Row():
                            with gr.Column():
                                create_exp_name = gr.Textbox(
                                    label="Experiment Name",
                                    placeholder="my_smollm3_finetune",
                                    value="smollm3_finetune"
                                )
                                create_exp_description = gr.Textbox(
                                    label="Description",
                                    placeholder="Fine-tuning SmolLM3 model on custom dataset",
                                    value="SmolLM3 fine-tuning experiment"
                                )
                                create_exp_btn = gr.Button("Create Experiment", variant="primary")
                            
                            with gr.Column():
                                create_exp_output = gr.Textbox(
                                    label="Result",
                                    lines=5,
                                    interactive=False
                                )
                        
                        create_exp_btn.click(
                            create_experiment_interface,
                            inputs=[create_exp_name, create_exp_description],
                            outputs=[create_exp_output, experiment_dropdown]
                        )
                    
                    # Log Metrics Tab
                    with gr.Tab("Log Metrics"):
                        gr.Markdown("### Log Training Metrics")
                        with gr.Row():
                            with gr.Column():
                                log_metrics_exp_id = gr.Textbox(
                                    label="Experiment ID",
                                    placeholder="exp_20231201_143022"
                                )
                                log_metrics_json = gr.Textbox(
                                    label="Metrics (JSON)",
                                    placeholder='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5}',
                                    value='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5, "gpu_memory": 22.5}'
                                )
                                log_metrics_step = gr.Textbox(
                                    label="Step (optional)",
                                    placeholder="100"
                                )
                                log_metrics_btn = gr.Button("Log Metrics", variant="primary")
                            
                            with gr.Column():
                                log_metrics_output = gr.Textbox(
                                    label="Result",
                                    lines=5,
                                    interactive=False
                                )
                        
                        log_metrics_btn.click(
                            log_metrics_interface,
                            inputs=[log_metrics_exp_id, log_metrics_json, log_metrics_step],
                            outputs=log_metrics_output
                        )
                    
                    # Log Parameters Tab
                    with gr.Tab("Log Parameters"):
                        gr.Markdown("### Log Experiment Parameters")
                        with gr.Row():
                            with gr.Column():
                                log_params_exp_id = gr.Textbox(
                                    label="Experiment ID",
                                    placeholder="exp_20231201_143022"
                                )
                                log_params_json = gr.Textbox(
                                    label="Parameters (JSON)",
                                    placeholder='{"learning_rate": 2e-5, "batch_size": 4}',
                                    value='{"learning_rate": 3.5e-6, "batch_size": 8, "model_name": "HuggingFaceTB/SmolLM3-3B", "max_iters": 18000, "mixed_precision": "bf16"}'
                                )
                                log_params_btn = gr.Button("Log Parameters", variant="primary")
                            
                            with gr.Column():
                                log_params_output = gr.Textbox(
                                    label="Result",
                                    lines=5,
                                    interactive=False
                                )
                        
                        log_params_btn.click(
                            log_parameters_interface,
                            inputs=[log_params_exp_id, log_params_json],
                            outputs=log_params_output
                        )
                    
                    # View Experiments Tab
                    with gr.Tab("View Experiments"):
                        gr.Markdown("### View Experiment Details")
                        with gr.Row():
                            with gr.Column():
                                view_exp_id = gr.Textbox(
                                    label="Experiment ID",
                                    placeholder="exp_20231201_143022"
                                )
                                view_btn = gr.Button("View Experiment", variant="primary")
                                list_btn = gr.Button("List All Experiments", variant="secondary")
                            
                            with gr.Column():
                                view_output = gr.Textbox(
                                    label="Experiment Details",
                                    lines=20,
                                    interactive=False
                                )
                        
                        view_btn.click(
                            get_experiment_details,
                            inputs=[view_exp_id],
                            outputs=view_output
                        )
                        
                        list_btn.click(
                            list_experiments_interface,
                            inputs=[],
                            outputs=view_output
                        )
                    
                    # Update Status Tab
                    with gr.Tab("Update Status"):
                        gr.Markdown("### Update Experiment Status")
                        with gr.Row():
                            with gr.Column():
                                status_exp_id = gr.Textbox(
                                    label="Experiment ID",
                                    placeholder="exp_20231201_143022"
                                )
                                status_dropdown = gr.Dropdown(
                                    label="Status",
                                    choices=["running", "completed", "failed", "paused"],
                                    value="running"
                                )
                                update_status_btn = gr.Button("Update Status", variant="primary")
                            
                            with gr.Column():
                                status_output = gr.Textbox(
                                    label="Result",
                                    lines=3,
                                    interactive=False
                                )
                        
                        update_status_btn.click(
                            update_experiment_status_interface,
                            inputs=[status_exp_id, status_dropdown],
                            outputs=status_output
                        )
                    
                    # Demo Data Tab
                    with gr.Tab("Demo Data"):
                        gr.Markdown("### Generate Demo Training Data")
                        gr.Markdown("Use this to simulate training data for testing the interface")
                        with gr.Row():
                            with gr.Column():
                                demo_exp_id = gr.Textbox(
                                    label="Experiment ID",
                                    placeholder="exp_20231201_143022"
                                )
                                demo_btn = gr.Button("Generate Demo Data", variant="primary")
                                create_demo_btn = gr.Button("Create Demo Experiment", variant="secondary")
                            
                            with gr.Column():
                                demo_output = gr.Textbox(
                                    label="Result",
                                    lines=5,
                                    interactive=False
                                )
                        
                        demo_btn.click(
                            simulate_training_data,
                            inputs=[demo_exp_id],
                            outputs=[demo_output, dashboard_status, dashboard_parameters, dashboard_metrics, dashboard_plots, dashboard_summary]
                        )
                        
                        create_demo_btn.click(
                            create_demo_experiment,
                            inputs=[],
                            outputs=[demo_output, experiment_dropdown]
                        )
            
            update_config_btn.click(
                update_trackio_config,
                inputs=[hf_token_input, dataset_repo_input],
                outputs=current_config_output
            )
            
            test_connection_btn.click(
                test_dataset_connection,
                inputs=[hf_token_input, dataset_repo_input],
                outputs=current_config_output
            )
            
            create_repo_btn.click(
                create_dataset_repository,
                inputs=[hf_token_input, dataset_repo_input],
                outputs=current_config_output
            )
    
        


# Launch the app
if __name__ == "__main__":
    demo.launch() 