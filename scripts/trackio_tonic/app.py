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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackioSpace:
    """Trackio deployment for Hugging Face Spaces using HF Datasets"""
    
    def __init__(self, hf_token: Optional[str] = None, dataset_repo: Optional[str] = None):
        self.experiments = {}
        self.current_experiment = None
        self.backup_mode = False
        self.dataset_manager = None
        
        # Get dataset repository and HF token from parameters or environment variables
        # Respect explicit values; avoid hardcoded defaults that might point to test repos
        default_dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        self.dataset_repo = dataset_repo or default_dataset_repo
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        
        logger.info(f"🔧 Using dataset repository: {self.dataset_repo}")
        
        if not self.hf_token:
            logger.warning("⚠️ HF_TOKEN not found. Some features may not work.")
        
        # Initialize dataset manager for safe, non-destructive operations
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
            from dataset_utils import TrackioDatasetManager  # type: ignore
            if self.hf_token and self.dataset_repo:
                self.dataset_manager = TrackioDatasetManager(self.dataset_repo, self.hf_token)
                logger.info("✅ Dataset manager initialized (data preservation enabled)")
        except Exception as e:
            logger.warning(f"⚠️ Dataset manager not available, using legacy save mode: {e}")
        
        self._load_experiments()
        
    def _load_experiments(self):
        """Load experiments from HF Dataset"""
        try:
            if self.hf_token:
                from datasets import load_dataset
                
                # Try to load the dataset
                try:
                    dataset = load_dataset(self.dataset_repo, token=self.hf_token)
                    logger.info(f"✅ Loaded experiments from {self.dataset_repo}")
                    
                    # Convert dataset to experiments dict
                    self.experiments = {}
                    if 'train' in dataset:
                        for row in dataset['train']:
                            exp_id = row.get('experiment_id')
                            if exp_id:
                                self.experiments[exp_id] = {
                                    'id': exp_id,
                                    'name': row.get('name', ''),
                                    'description': row.get('description', ''),
                                    'created_at': row.get('created_at', ''),
                                    'status': row.get('status', 'running'),
                                    'metrics': json.loads(row.get('metrics', '[]')),
                                    'parameters': json.loads(row.get('parameters', '{}')),
                                    'artifacts': json.loads(row.get('artifacts', '[]')),
                                    'logs': json.loads(row.get('logs', '[]'))
                                }
                    
                    logger.info(f"📊 Loaded {len(self.experiments)} experiments from dataset")
                    
                except Exception as e:
                    logger.warning(f"Failed to load from dataset: {e}")
                    # Fall back to backup data
                    self._load_backup_experiments()
            else:
                # No HF token, use backup data but do not allow saving to dataset from backup
                self._load_backup_experiments()
                self.backup_mode = True
                
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
            self._load_backup_experiments()
            self.backup_mode = True
    
    def _load_backup_experiments(self):
        """Load backup experiments when dataset is not available"""
        logger.info("🔄 Loading backup experiments...")
        
        # Get dynamic trackio URL from environment or use a placeholder
        trackio_url = os.environ.get('TRACKIO_URL', 'https://your-trackio-space.hf.space')
        
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
                    'trackio_url': trackio_url,
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
                            'epoch': 0.004851130919895701
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
                    'trackio_url': trackio_url,
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
    
    def _upsert_experiment(self, experiment_id: str):
        """Non-destructive upsert of a single experiment to the dataset if manager available."""
        try:
            if not self.dataset_manager or not self.hf_token:
                # Fallback to legacy save method
                self._save_experiments()
                return
            exp = self.experiments.get(experiment_id)
            if not exp:
                return
            # Build dataset row with JSON-encoded fields
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
    
    def _save_experiments(self):
        """Save experiments to HF Dataset (legacy fallback).

        Prefer using dataset manager upserts in per-operation paths. This method is
        retained as a fallback when the manager isn't available.
        """
        try:
            if self.backup_mode:
                logger.warning("⚠️ Backup mode active; skipping dataset save to avoid overwriting real data with demo values")
                return
            if self.hf_token and not self.dataset_manager:
                from datasets import Dataset
                from huggingface_hub import HfApi
                
                # Convert experiments to dataset format
                dataset_data = []
                for exp_id, exp_data in self.experiments.items():
                    dataset_data.append({
                        'experiment_id': exp_id,
                        'name': exp_data.get('name', ''),
                        'description': exp_data.get('description', ''),
                        'created_at': exp_data.get('created_at', ''),
                        'status': exp_data.get('status', 'running'),
                        'metrics': json.dumps(exp_data.get('metrics', [])),
                        'parameters': json.dumps(exp_data.get('parameters', {})),
                        'artifacts': json.dumps(exp_data.get('artifacts', [])),
                        'logs': json.dumps(exp_data.get('logs', [])),
                        'last_updated': datetime.now().isoformat()
                    })
                
                # Create dataset
                dataset = Dataset.from_list(dataset_data)
                
                # Push to HF Hub
                api = HfApi(token=self.hf_token)
                dataset.push_to_hub(
                    self.dataset_repo,
                    token=self.hf_token,
                    private=True  # Make it private for security
                )
                
                logger.info(f"✅ Saved {len(dataset_data)} experiments to {self.dataset_repo} (legacy mode)")
                
            else:
                logger.warning("⚠️ No dataset manager and/or HF_TOKEN available, experiments not saved to dataset")
                
        except Exception as e:
            logger.error(f"Failed to save experiments to dataset: {e}")
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
                logger.error(f"Failed to save backup: {backup_e}")
    
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
        # Prefer non-destructive upsert
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
        self._upsert_experiment(experiment_id)
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
        
        return f"✅ Configuration updated successfully!\n📊 Dataset: {trackio_space.dataset_repo}\n🔑 HF Token: {'Set' if trackio_space.hf_token else 'Not set'}\n📈 Loaded {len(trackio_space.experiments)} experiments"
        
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
        
        # Count experiments
        experiment_count = len(dataset['train']) if 'train' in dataset else 0
        
        return f"✅ Connection successful!\n📊 Dataset: {dataset_repo}\n📈 Found {experiment_count} experiments\n🔗 Dataset URL: https://huggingface.co/datasets/{dataset_repo}"
        
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
            return f"✅ Dataset {dataset_repo} already exists!"
        except:
            # Dataset doesn't exist, create it
            pass
        
        # Create empty dataset
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
            private=True
        )
        
        return f"✅ Dataset {dataset_repo} created successfully!\n🔗 View at: https://huggingface.co/datasets/{dataset_repo}\n📊 Ready to store experiments"
        
    except Exception as e:
        return f"❌ Failed to create dataset: {str(e)}\n\n💡 Troubleshooting:\n1. Check your HF token has write permissions\n2. Verify the username in the repository name\n3. Ensure the dataset name is valid"

# Initialize API client for remote data if environment provides a space id/url
api_client = None
try:
    from trackio_api_client import TrackioAPIClient
    space_id = os.environ.get('TRACKIO_URL') or os.environ.get('TRACKIO_SPACE_ID')
    if space_id:
        api_client = TrackioAPIClient(space_id, os.environ.get('HF_TOKEN'))
        logger.info("✅ API client initialized for remote data access")
    else:
        logger.info("No TRACKIO_URL/TRACKIO_SPACE_ID set; remote API client disabled")
except ImportError:
    logger.warning("⚠️ API client not available, using local data only")
except Exception as e:
    logger.warning(f"⚠️ Could not initialize API client: {e}")

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

# Override the data file path for HF Spaces if attribute exists
if is_huggingface_spaces() and hasattr(trackio_space, 'data_file'):
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
        
        if metrics_data:
            return pd.DataFrame(metrics_data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error parsing remote metrics: {e}")
        return pd.DataFrame()

def get_metrics_dataframe(experiment_id: str) -> pd.DataFrame:
    """Get metrics as a pandas DataFrame for plotting - tries remote first, then local"""
    # Try to get remote data first
    remote_data = get_remote_experiment_data(experiment_id)
    if remote_data:
        logger.info(f"Using remote data for {experiment_id}")
        # Parse the remote experiment details to extract metrics
        df = parse_remote_metrics_data(remote_data["data"])
        if not df.empty:
            logger.info(f"Found {len(df)} metrics entries from remote data")
            return df
        else:
            logger.warning(f"No metrics found in remote data for {experiment_id}")
    
    # Fall back to local data
    logger.info(f"Using local data for {experiment_id}")
    return trackio_space.get_metrics_dataframe(experiment_id)

def create_experiment_interface(name: str, description: str) -> str:
    """Create a new experiment"""
    try:
        experiment = trackio_space.create_experiment(name, description)
        return f"✅ Experiment created successfully!\nID: {experiment['id']}\nName: {experiment['name']}\nStatus: {experiment['status']}"
    except Exception as e:
        return f"❌ Error creating experiment: {str(e)}"

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
    """Create a plot for a specific metric"""
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
        # Simulate some realistic training metrics
        for step in range(0, 1000, 50):
            # Simulate loss decreasing over time
            loss = 2.0 * np.exp(-step / 500) + 0.1 * np.random.random()
            accuracy = 0.3 + 0.6 * (1 - np.exp(-step / 300)) + 0.05 * np.random.random()
            lr = 3.5e-6 * (0.9 ** (step // 200))
            
            metrics = {
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "learning_rate": round(lr, 8),
                "gpu_memory": round(20 + 5 * np.random.random(), 2),
                "training_time": round(0.5 + 0.2 * np.random.random(), 3)
            }
            
            trackio_space.log_metrics(experiment_id, metrics, step)
        
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

# Create Gradio interface
with gr.Blocks(title="Trackio - Experiment Tracking", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Trackio Experiment Tracking & Monitoring")
    gr.Markdown("Monitor and track your ML experiments with real-time visualization!")
    
    with gr.Tabs():
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
                        value=os.environ.get('TRACKIO_DATASET_REPO', 'trackio-experiments'),
                        info="HF Dataset repository for experiment storage"
                    )
                    
                    with gr.Row():
                        update_config_btn = gr.Button("Update Configuration", variant="primary")
                        test_connection_btn = gr.Button("Test Connection", variant="secondary")
                        create_repo_btn = gr.Button("Create Dataset", variant="success")
                    
                    gr.Markdown("### Current Configuration")
                    current_config_output = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                        value=f"📊 Dataset: {trackio_space.dataset_repo}\n🔑 HF Token: {'Set' if trackio_space.hf_token else 'Not set'}\n📈 Experiments: {len(trackio_space.experiments)}"
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
                    
                    **Actions:**
                    - **Update Configuration**: Apply new settings and reload experiments
                    - **Test Connection**: Verify access to the dataset repository
                    - **Create Dataset**: Create a new dataset repository if it doesn't exist
                    """)
            
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
        
        # Create Experiment Tab
        with gr.Tab("Create Experiment"):
            gr.Markdown("### Create a New Experiment")
            with gr.Row():
                with gr.Column():
                    experiment_name = gr.Textbox(
                        label="Experiment Name",
                        placeholder="my_smollm3_finetune",
                        value="smollm3_finetune"
                    )
                    experiment_description = gr.Textbox(
                        label="Description",
                        placeholder="Fine-tuning SmolLM3 model on custom dataset",
                        value="SmolLM3 fine-tuning experiment"
                    )
                    create_btn = gr.Button("Create Experiment", variant="primary")
                
                with gr.Column():
                    create_output = gr.Textbox(
                        label="Result",
                        lines=5,
                        interactive=False
                    )
            
            create_btn.click(
                create_experiment_interface,
                inputs=[experiment_name, experiment_description],
                outputs=create_output
            )
        
        # Log Metrics Tab
        with gr.Tab("Log Metrics"):
            gr.Markdown("### Log Training Metrics")
            with gr.Row():
                with gr.Column():
                    metrics_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    metrics_json = gr.Textbox(
                        label="Metrics (JSON)",
                        placeholder='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5}',
                        value='{"loss": 0.5, "accuracy": 0.85, "learning_rate": 2e-5, "gpu_memory": 22.5}'
                    )
                    metrics_step = gr.Textbox(
                        label="Step (optional)",
                        placeholder="100"
                    )
                    log_metrics_btn = gr.Button("Log Metrics", variant="primary")
                
                with gr.Column():
                    metrics_output = gr.Textbox(
                        label="Result",
                        lines=5,
                        interactive=False
                    )
            
            log_metrics_btn.click(
                log_metrics_interface,
                inputs=[metrics_exp_id, metrics_json, metrics_step],
                outputs=metrics_output
            )
        
        # Log Parameters Tab
        with gr.Tab("Log Parameters"):
            gr.Markdown("### Log Experiment Parameters")
            with gr.Row():
                with gr.Column():
                    params_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    parameters_json = gr.Textbox(
                        label="Parameters (JSON)",
                        placeholder='{"learning_rate": 2e-5, "batch_size": 4}',
                        value='{"learning_rate": 3.5e-6, "batch_size": 8, "model_name": "HuggingFaceTB/SmolLM3-3B", "max_iters": 18000, "mixed_precision": "bf16"}'
                    )
                    log_params_btn = gr.Button("Log Parameters", variant="primary")
                
                with gr.Column():
                    params_output = gr.Textbox(
                        label="Result",
                        lines=5,
                        interactive=False
                    )
            
            log_params_btn.click(
                log_parameters_interface,
                inputs=[params_exp_id, parameters_json],
                outputs=params_output
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
        
        # Visualization Tab
        with gr.Tab("📊 Visualizations"):
            gr.Markdown("### Training Metrics Visualization")
            with gr.Row():
                with gr.Column():
                    plot_exp_id = gr.Textbox(
                        label="Experiment ID",
                        placeholder="exp_20231201_143022"
                    )
                    metric_dropdown = gr.Dropdown(
                        label="Metric to Plot",
                        choices=["loss", "accuracy", "learning_rate", "gpu_memory", "training_time"],
                        value="loss"
                    )
                    plot_btn = gr.Button("Create Plot", variant="primary")
                
                with gr.Column():
                    plot_output = gr.Plot(label="Training Metrics")
            
            plot_btn.click(
                create_metrics_plot,
                inputs=[plot_exp_id, metric_dropdown],
                outputs=plot_output
            )
            
            gr.Markdown("### Experiment Comparison")
            with gr.Row():
                with gr.Column():
                    comparison_exp_ids = gr.Textbox(
                        label="Experiment IDs (comma-separated)",
                        placeholder="exp_1,exp_2,exp_3"
                    )
                    comparison_btn = gr.Button("Compare Experiments", variant="primary")
                
                with gr.Column():
                    comparison_plot = gr.Plot(label="Experiment Comparison")
            
            comparison_btn.click(
                create_experiment_comparison,
                inputs=[comparison_exp_ids],
                outputs=comparison_plot
            )
        
        # Demo Data Tab
        with gr.Tab("🎯 Demo Data"):
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
                outputs=demo_output
            )
            
            create_demo_btn.click(
                create_demo_experiment,
                inputs=[],
                outputs=demo_output
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

# Launch the app
if __name__ == "__main__":
    demo.launch() 