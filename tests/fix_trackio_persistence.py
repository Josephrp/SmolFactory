#!/usr/bin/env python3
"""
Fix script to manually add missing experiments to trackio_experiments.json
"""

import json
import os
from datetime import datetime

def add_missing_experiments():
    """Add the missing experiments from the logs to the data file"""
    
    data_file = "trackio_experiments.json"
    
    # Load existing data
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'experiments': {},
            'current_experiment': None,
            'last_updated': datetime.now().isoformat()
        }
    
    # Add the missing experiments based on the logs
    experiments = data['experiments']
    
    # Experiment 1: exp_20250720_130853
    experiments['exp_20250720_130853'] = {
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
    }
    
    # Experiment 2: exp_20250720_134319
    experiments['exp_20250720_134319'] = {
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
    
    # Update metadata
    data['current_experiment'] = 'exp_20250720_134319'
    data['last_updated'] = datetime.now().isoformat()
    
    # Save the updated data
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Added missing experiments to trackio_experiments.json")
    print(f"📊 Total experiments: {len(experiments)}")
    print("🔬 Experiments added:")
    print("   - exp_20250720_130853 (petite-elle-l-aime-3)")
    print("   - exp_20250720_134319 (petite-elle-l-aime-3-1)")
    print("\n🎯 You can now view these experiments in the Trackio interface!")

if __name__ == "__main__":
    add_missing_experiments() 