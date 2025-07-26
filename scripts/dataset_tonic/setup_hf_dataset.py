#!/usr/bin/env python3
"""
Setup script for Hugging Face Dataset repository for Trackio experiments
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, create_repo
import subprocess

def get_username_from_token(token: str) -> Optional[str]:
    """
    Get username from HF token using the API.
    
    Args:
        token (str): Hugging Face token
        
    Returns:
        Optional[str]: Username if successful, None otherwise
    """
    try:
        # Create API client with token directly
        api = HfApi(token=token)
        
        # Get user info
        user_info = api.whoami()
        username = user_info.get("name", user_info.get("username"))
        
        return username
    except Exception as e:
        print(f"❌ Error getting username from token: {e}")
        return None

def create_dataset_repository(username: str, dataset_name: str = "trackio-experiments", token: str = None) -> str:
    """
    Create a dataset repository on Hugging Face.
    
    Args:
        username (str): HF username
        dataset_name (str): Name for the dataset repository
        token (str): HF token for authentication
        
    Returns:
        str: Full repository name (username/dataset_name)
    """
    repo_id = f"{username}/{dataset_name}"
    
    try:
        # Create the dataset repository
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False  # Public dataset for easier sharing
        )
        
        print(f"✅ Successfully created dataset repository: {repo_id}")
        return repo_id
        
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Dataset repository already exists: {repo_id}")
            return repo_id
        else:
            print(f"❌ Error creating dataset repository: {e}")
            return None

def setup_trackio_dataset(dataset_name: str = None) -> bool:
    """
    Set up Trackio dataset repository automatically.
    
    Args:
        dataset_name (str): Optional custom dataset name (default: trackio-experiments)
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("🚀 Setting up Trackio Dataset Repository")
    print("=" * 50)
    
    # Get token from environment or command line
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    # If no token in environment, try command line argument
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
    
    if not token:
        print("❌ No HF token found. Please set HUGGING_FACE_HUB_TOKEN environment variable or provide as argument.")
        return False
    
    # Get username from token
    print("🔍 Getting username from token...")
    username = get_username_from_token(token)
    if not username:
        print("❌ Could not determine username from token. Please check your token.")
        return False
    
    print(f"✅ Authenticated as: {username}")
    
    # Use provided dataset name or default
    if not dataset_name:
        dataset_name = "trackio-experiments"
    
    # Create dataset repository
    print(f"🔧 Creating dataset repository: {username}/{dataset_name}")
    repo_id = create_dataset_repository(username, dataset_name, token)
    
    if not repo_id:
        print("❌ Failed to create dataset repository")
        return False
    
    # Set environment variable for other scripts
    os.environ['TRACKIO_DATASET_REPO'] = repo_id
    print(f"✅ Set TRACKIO_DATASET_REPO={repo_id}")
    
    # Add initial experiment data
    print("📊 Adding initial experiment data...")
    if add_initial_experiment_data(repo_id, token):
        print("✅ Successfully added initial experiment data")
    else:
        print("⚠️  Could not add initial experiment data (this is optional)")
    
    print(f"\n🎉 Dataset setup complete!")
    print(f"📊 Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"🔧 Repository ID: {repo_id}")
    
    return True

def add_initial_experiment_data(repo_id: str, token: str = None) -> bool:
    """
    Add initial experiment data to the dataset.
    
    Args:
        repo_id (str): Dataset repository ID
        token (str): HF token for authentication
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get token from parameter or environment
        if not token:
            token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        
        if not token:
            print("⚠️  No token available for uploading data")
            return False
        
        # Initial experiment data
        initial_experiments = [
            {
                'experiment_id': f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'name': 'smollm3-finetune-demo',
                'description': 'SmolLM3 fine-tuning experiment demo with comprehensive metrics tracking',
                'created_at': datetime.now().isoformat(),
                'status': 'completed',
                'metrics': json.dumps([
                    {
                        'timestamp': datetime.now().isoformat(),
                        'step': 100,
                        'metrics': {
                            'loss': 1.15,
                            'grad_norm': 10.5,
                            'learning_rate': 5e-6,
                            'num_tokens': 1000000.0,
                            'mean_token_accuracy': 0.76,
                            'epoch': 0.1,
                            'total_tokens': 1000000.0,
                            'throughput': 2000000.0,
                            'step_time': 0.5,
                            'batch_size': 2,
                            'seq_len': 4096,
                            'token_acc': 0.76,
                            'gpu_memory_allocated': 15.2,
                            'gpu_memory_reserved': 70.1,
                            'gpu_utilization': 85.2,
                            'cpu_percent': 2.7,
                            'memory_percent': 10.1
                        }
                    }
                ]),
                'parameters': json.dumps({
                    'model_name': 'HuggingFaceTB/SmolLM3-3B',
                    'max_seq_length': 4096,
                    'batch_size': 2,
                    'learning_rate': 5e-6,
                    'epochs': 3,
                    'dataset': 'OpenHermes-FR',
                    'trainer_type': 'SFTTrainer',
                    'hardware': 'GPU (H100/A100)',
                    'mixed_precision': True,
                    'gradient_checkpointing': True,
                    'flash_attention': True
                }),
                'artifacts': json.dumps([]),
                'logs': json.dumps([
                    {
                        'timestamp': datetime.now().isoformat(),
                        'level': 'INFO',
                        'message': 'Training started successfully'
                    },
                    {
                        'timestamp': datetime.now().isoformat(),
                        'level': 'INFO',
                        'message': 'Model loaded and configured'
                    },
                    {
                        'timestamp': datetime.now().isoformat(),
                        'level': 'INFO',
                        'message': 'Dataset loaded and preprocessed'
                    }
                ]),
                'last_updated': datetime.now().isoformat()
            }
        ]
        
        # Create dataset and upload
        from datasets import Dataset
        
        # Create dataset from the initial experiments
        dataset = Dataset.from_list(initial_experiments)
        
        # Push to hub
        dataset.push_to_hub(
            repo_id,
            token=token,
            private=False,
            commit_message="Add initial experiment data"
        )
        
        print(f"✅ Successfully uploaded initial experiment data to {repo_id}")
        
        # Add README template
        add_dataset_readme(repo_id, token)
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not add initial experiment data: {e}")
        return False

def add_dataset_readme(repo_id: str, token: str) -> bool:
    """
    Add README template to the dataset repository.
    
    Args:
        repo_id (str): Dataset repository ID
        token (str): HF token
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the README template
        template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'templates', 'datasets', 'readme.md')
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
        else:
            # Create a basic README if template doesn't exist
            readme_content = f"""---
dataset_info:
  features:
  - name: experiment_id
    dtype: string
  - name: name
    dtype: string
  - name: description
    dtype: string
  - name: created_at
    dtype: string
  - name: status
    dtype: string
  - name: metrics
    dtype: string
  - name: parameters
    dtype: string
  - name: artifacts
    dtype: string
  - name: logs
    dtype: string
  - name: last_updated
    dtype: string
tags:
- trackio
- experiment tracking
- smollm3
- fine-tuning
---

# Trackio Experiments Dataset

This dataset stores experiment tracking data for ML training runs, particularly focused on SmolLM3 fine-tuning experiments with comprehensive metrics tracking.

## Dataset Structure

The dataset contains the following columns:

- **experiment_id**: Unique identifier for each experiment
- **name**: Human-readable name for the experiment
- **description**: Detailed description of the experiment
- **created_at**: Timestamp when the experiment was created
- **status**: Current status (running, completed, failed, paused)
- **metrics**: JSON string containing training metrics over time
- **parameters**: JSON string containing experiment configuration
- **artifacts**: JSON string containing experiment artifacts
- **logs**: JSON string containing experiment logs
- **last_updated**: Timestamp of last update

## Usage

This dataset is automatically used by the Trackio monitoring system to store and retrieve experiment data. It provides persistent storage for experiment tracking across different training runs.

## Integration

The dataset is used by:
- Trackio Spaces for experiment visualization
- Training scripts for logging metrics and parameters
- Monitoring systems for experiment tracking
- SmolLM3 fine-tuning pipeline for comprehensive metrics capture

## Privacy

This dataset is public by default for easier sharing and collaboration. Only non-sensitive experiment data is stored.

## Examples

### Sample Experiment Entry
```json
{{
  "experiment_id": "exp_20250720_130853",
  "name": "smollm3_finetune",
  "description": "SmolLM3 fine-tuning experiment with comprehensive metrics",
  "created_at": "2025-07-20T11:20:01.780908",
  "status": "running",
  "metrics": "[{{\"timestamp\": \"2025-07-20T11:20:01.780908\", \"step\": 25, \"metrics\": {{\"loss\": 1.1659, \"accuracy\": 0.759, \"total_tokens\": 1642080.0, \"throughput\": 3284160.0, \"train/gate_ortho\": 0.0234, \"train/center\": 0.0156}}}}]",
  "parameters": "{{\"model_name\": \"HuggingFaceTB/SmolLM3-3B\", \"batch_size\": 8, \"learning_rate\": 3.5e-06, \"max_seq_length\": 12288}}",
  "artifacts": "[]",
  "logs": "[]",
  "last_updated": "2025-07-20T11:20:01.780908"
}}
```

## License

This dataset is part of the Trackio experiment tracking system and follows the same license as the main project.
"""
        
        # Upload README to the dataset repository
        from huggingface_hub import upload_file
        
        # Create a temporary file with the README content
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(readme_content)
            temp_file = f.name
        
        try:
            upload_file(
                path_or_fileobj=temp_file,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Add dataset README"
            )
            print(f"✅ Successfully added README to {repo_id}")
            return True
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
    except Exception as e:
        print(f"⚠️  Could not add README to dataset: {e}")
        return False

def main():
    """Main function to set up the dataset."""
    
    # Get token from environment first
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    # If no token in environment, try command line argument
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
    
    if not token:
        print("❌ No HF token found. Please set HUGGING_FACE_HUB_TOKEN environment variable or provide as argument.")
        sys.exit(1)
    
    # Get dataset name from command line or use default
    dataset_name = None
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]
    
    success = setup_trackio_dataset(dataset_name)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 