#!/usr/bin/env python3
"""
Script to run A100 large-scale experiments on OpenHermes-FR dataset
Supports multiple configurations for different training scenarios
"""

import argparse
import os
import sys
from pathlib import Path

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Run A100 large-scale experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/train_smollm3_openhermes_fr_a100_large.py",
        help="Configuration file to use"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name for tracking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without starting training"
    )
    parser.add_argument(
        "--trackio-url",
        "--trackio_url",
        type=str,
        help="Trackio URL for experiment tracking"
    )
    parser.add_argument(
        "--trackio-token",
        "--trackio_token",
        type=str,
        help="Trackio token for authentication"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="my_dataset",
        help="Dataset directory path"
    )
    
    args = parser.parse_args()
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Import the configuration
    try:
        # Import all available configurations
        from config.train_smollm3_openhermes_fr_a100_large import get_config as get_large_config
        from config.train_smollm3_openhermes_fr_a100_multiple_passes import get_config as get_multiple_passes_config
        from config.train_smollm3_h100_lightweight import get_config as get_h100_lightweight_config
        
        # Map config files to their respective functions
        config_map = {
            "config/train_smollm3_openhermes_fr_a100_large.py": get_large_config,
            "config/train_smollm3_openhermes_fr_a100_multiple_passes.py": get_multiple_passes_config,
            "config/train_smollm3_h100_lightweight.py": get_h100_lightweight_config,
        }
        
        if args.config in config_map:
            config = config_map[args.config](args.config)
        else:
            # Try to load from the specified config file
            config = get_large_config(args.config)
            
    except ImportError as e:
        print(f"Error importing configuration: {e}")
        print("Available configurations:")
        print("  - config/train_smollm3_openhermes_fr_a100_large.py (Large batch, 1.3 passes)")
        print("  - config/train_smollm3_openhermes_fr_a100_multiple_passes.py (Multiple passes, 4 epochs)")
        print("  - config/train_smollm3_h100_lightweight.py (H100 lightweight, 80K samples)")
        return 1
    
    # Override experiment name if provided
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Override Trackio settings if provided
    if args.trackio_url:
        config.trackio_url = args.trackio_url
    if args.trackio_token:
        config.trackio_token = args.trackio_token
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"Config file: {args.config}")
    print(f"Experiment name: {config.experiment_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max iterations: {config.max_iters}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Mixed precision: {'bf16' if config.bf16 else 'fp16'}")
    if hasattr(config, 'dataset_name') and config.dataset_name:
        print(f"Dataset: {config.dataset_name}")
        if hasattr(config, 'sample_size') and config.sample_size:
            print(f"Sample size: {config.sample_size}")
    else:
        print(f"Dataset directory: {config.data_dir}")
        print(f"Training file: {config.train_file}")
        if config.validation_file:
            print(f"Validation file: {config.validation_file}")
    if config.trackio_url:
        print(f"Trackio URL: {config.trackio_url}")
    if config.trackio_token:
        print(f"Trackio Token: {'*' * len(config.trackio_token)}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN - Configuration printed above. Use without --dry-run to start training.")
        return 0
    
    # Import and run training
    try:
        # Add src directory to path
        src_path = str(project_root / "src")
        sys.path.insert(0, src_path)
        from train import main as train_main
        
        # Set up training arguments - config is positional, not --config
        train_args = [
            args.config,  # Config file as positional argument
            "--out_dir", args.output_dir,
        ]
        
        if args.resume:
            train_args.extend(["--init_from", "resume"])
        
        # Add Trackio arguments if provided
        if args.trackio_url:
            train_args.extend(["--trackio_url", args.trackio_url])
        if args.trackio_token:
            train_args.extend(["--trackio_token", args.trackio_token])
        if args.experiment_name:
            train_args.extend(["--experiment_name", args.experiment_name])
        
        # Add dataset directory argument
        train_args.extend(["--dataset_dir", args.dataset_dir])
        
        # Override sys.argv for the training script
        original_argv = sys.argv
        sys.argv = ["train.py"] + train_args
        
        # Run training
        train_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure train.py is available in the current directory.")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 