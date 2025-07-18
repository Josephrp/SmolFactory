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

from config import get_config
from model import SmolLM3Model
from data import SmolLM3Dataset
from trainer import SmolLM3Trainer

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
    
    # Setup paths
    dataset_path = os.path.join('/input', args.dataset_dir)
    output_path = args.out_dir
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output path: {output_path}")
    
    # Initialize model
    model = SmolLM3Model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        config=config
    )
    
    # Load dataset
    dataset = SmolLM3Dataset(
        data_path=dataset_path,
        tokenizer=model.tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # Initialize trainer
    trainer = SmolLM3Trainer(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_path,
        init_from=args.init_from
    )
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main() 