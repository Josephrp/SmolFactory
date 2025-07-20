#!/usr/bin/env python3
"""
Setup script for Hugging Face Dataset repository for Trackio experiments
"""

import os
import json
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

def setup_trackio_dataset():
    """Set up the Trackio experiments dataset on Hugging Face Hub"""
    
    # Configuration - get from environment variables with fallbacks
    dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
    hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        print("❌ HF_TOKEN not found. Please set the HF_TOKEN environment variable.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        return False
    
    print(f"🚀 Setting up Trackio dataset: {dataset_repo}")
    print(f"🔧 Using dataset repository: {dataset_repo}")
    
    # Initial experiment data
    initial_experiments = [
        {
            'experiment_id': 'exp_20250720_130853',
            'name': 'petite-elle-l-aime-3',
            'description': 'SmolLM3 fine-tuning experiment',
            'created_at': '2025-07-20T11:20:01.780908',
            'status': 'running',
            'metrics': json.dumps([
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
            ]),
            'parameters': json.dumps({
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
            }),
            'artifacts': json.dumps([]),
            'logs': json.dumps([]),
            'last_updated': datetime.now().isoformat()
        },
        {
            'experiment_id': 'exp_20250720_134319',
            'name': 'petite-elle-l-aime-3-1',
            'description': 'SmolLM3 fine-tuning experiment',
            'created_at': '2025-07-20T11:54:31.993219',
            'status': 'running',
            'metrics': json.dumps([
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
            ]),
            'parameters': json.dumps({
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
            }),
            'artifacts': json.dumps([]),
            'logs': json.dumps([]),
            'last_updated': datetime.now().isoformat()
        }
    ]
    
    try:
        # Create dataset
        dataset = Dataset.from_list(initial_experiments)
        
        # Get the project root directory (2 levels up from this script)
        project_root = Path(__file__).parent.parent.parent
        templates_dir = project_root / "templates" / "datasets"
        readme_path = templates_dir / "readme.md"
        
        # Read README content if it exists
        readme_content = None
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            print(f"✅ Found README template: {readme_path}")
        
        # Push to HF Hub with README
        api = HfApi(token=hf_token)
        dataset.push_to_hub(
            dataset_repo,
            token=hf_token,
            private=True,  # Make it private for security
            readme_content=readme_content  # Include README if available
        )
        
        print(f"✅ Successfully created dataset: {dataset_repo}")
        print(f"📊 Added {len(initial_experiments)} experiments")
        if readme_content:
            print("📝 Included README from templates")
        print("🔒 Dataset is private (only accessible with your token)")
        print("\n🎯 Next steps:")
        print("1. Set HF_TOKEN in your Hugging Face Space environment")
        print("2. Deploy the updated app.py to your Space")
        print("3. The app will now load experiments from the dataset")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False

if __name__ == "__main__":
    setup_trackio_dataset() 