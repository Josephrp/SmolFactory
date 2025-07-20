#!/usr/bin/env python3
"""
SmolLM3 Fine-tuning Script for FlexAI Console
Based on the nanoGPT structure but adapted for SmolLM3 model
"""

import os
import sys
import argparse
import json
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path for config imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import get_config
except ImportError:
    # Fallback: try direct import
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from config import get_config
from model import SmolLM3Model
from data import SmolLM3Dataset
from trainer import SmolLM3Trainer
from monitoring import create_monitor_from_config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SmolLM3 Fine-tuning Script')
    
    # Configuration file
    parser.add_argument('config', type=str, help='Path to configuration file')
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, default='my_dataset',
                       help='Path to dataset directory within /input')
    
    # Checkpoint arguments
    parser.add_argument('--out_dir', type=str, default='/output-checkpoint',
                       help='Output directory for checkpoints')
    parser.add_argument('--init_from', type=str, default='scratch',
                       choices=['scratch', 'resume', 'pretrained'],
                       help='Initialization method')
    
    # Training arguments
    parser.add_argument('--max_iters', type=int, default=None,
                       help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                       help='Gradient accumulation steps')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='HuggingFaceTB/SmolLM3-3B',
                       help='Model name or path')
    parser.add_argument('--max_seq_length', type=int, default=4096,
                       help='Maximum sequence length')
    
    # Logging and saving
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    
    # Trackio monitoring arguments
    parser.add_argument('--enable_tracking', action='store_true', default=True,
                       help='Enable Trackio experiment tracking')
    parser.add_argument('--trackio_url', type=str, default=None,
                       help='Trackio server URL')
    parser.add_argument('--trackio_token', type=str, default=None,
                       help='Trackio authentication token')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name for tracking')
    
    # HF Datasets arguments
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face token for dataset access')
    parser.add_argument('--dataset_repo', type=str, default=None,
                       help='HF Dataset repository for experiment storage')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting SmolLM3 fine-tuning...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Override Trackio configuration
    if args.enable_tracking is not None:
        config.enable_tracking = args.enable_tracking
    if args.trackio_url is not None:
        config.trackio_url = args.trackio_url
    if args.trackio_token is not None:
        config.trackio_token = args.trackio_token
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    
    # Override HF Datasets configuration
    if args.hf_token is not None:
        os.environ['HF_TOKEN'] = args.hf_token
    if args.dataset_repo is not None:
        os.environ['TRACKIO_DATASET_REPO'] = args.dataset_repo
    
    # Setup paths
    output_path = args.out_dir
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Output path: {output_path}")
    
    # Initialize monitoring
    monitor = None
    if config.enable_tracking:
        try:
            monitor = create_monitor_from_config(config, args.experiment_name)
            logger.info(f"✅ Monitoring initialized for experiment: {monitor.experiment_name}")
            logger.info(f"📊 Dataset repository: {monitor.dataset_repo}")
            
            # Log configuration
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            monitor.log_configuration(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            logger.warning("Continuing without monitoring...")
    
    # Initialize model
    model = SmolLM3Model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        config=config
    )
    
    # Determine dataset path
    if hasattr(config, 'dataset_name') and config.dataset_name:
        # Use Hugging Face dataset
        dataset_path = config.dataset_name
        logger.info(f"Using Hugging Face dataset: {dataset_path}")
    else:
        # Use local dataset
        dataset_path = os.path.join('/input', args.dataset_dir)
        logger.info(f"Using local dataset: {dataset_path}")
    
    # Load dataset with filtering options
    dataset = SmolLM3Dataset(
        data_path=dataset_path,
        tokenizer=model.tokenizer,
        max_seq_length=args.max_seq_length,
        filter_bad_entries=getattr(config, 'filter_bad_entries', False),
        bad_entry_field=getattr(config, 'bad_entry_field', 'bad_entry')
    )
    
    # Initialize trainer
    trainer = SmolLM3Trainer(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_path,
        init_from=args.init_from
    )
    
    # Add monitoring callback if available
    if monitor:
        try:
            callback = monitor.create_monitoring_callback()
            trainer.add_callback(callback)
            logger.info("✅ Monitoring callback added to trainer")
        except Exception as e:
            logger.error(f"Failed to add monitoring callback: {e}")
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Log training summary
        if monitor:
            try:
                summary = {
                    'final_loss': getattr(trainer, 'final_loss', None),
                    'total_steps': getattr(trainer, 'total_steps', None),
                    'training_duration': getattr(trainer, 'training_duration', None),
                    'model_path': output_path,
                    'config_file': args.config
                }
                monitor.log_training_summary(summary)
                logger.info("✅ Training summary logged")
            except Exception as e:
                logger.error(f"Failed to log training summary: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        
        # Log error to monitoring
        if monitor:
            try:
                error_summary = {
                    'error': str(e),
                    'status': 'failed',
                    'model_path': output_path,
                    'config_file': args.config
                }
                monitor.log_training_summary(error_summary)
            except Exception as log_error:
                logger.error(f"Failed to log error to monitoring: {log_error}")
        
        raise
    finally:
        # Close monitoring
        if monitor:
            try:
                monitor.close()
                logger.info("✅ Monitoring session closed")
            except Exception as e:
                logger.error(f"Failed to close monitoring: {e}")

if __name__ == '__main__':
    main() 