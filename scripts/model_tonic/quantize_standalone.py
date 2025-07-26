#!/usr/bin/env python3
"""
Standalone Model Quantization Script
Quick quantization of trained models using torchao
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.model_tonic.quantize_model import ModelQuantizer

def main():
    """Standalone quantization script"""
    parser = argparse.ArgumentParser(description="Quantize a trained model using torchao")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("repo_name", help="Hugging Face repository name for quantized model")
    parser.add_argument("--quant-type", choices=["int8_weight_only", "int4_weight_only", "int8_dynamic"], 
                       default="int8_weight_only", help="Quantization type")
    parser.add_argument("--device", default="auto", help="Device for quantization (auto, cpu, cuda)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--trackio-url", help="Trackio URL for monitoring")
    parser.add_argument("--experiment-name", help="Experiment name for tracking")
    parser.add_argument("--dataset-repo", help="HF Dataset repository")
    parser.add_argument("--save-only", action="store_true", help="Save quantized model locally without pushing to HF")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Model Quantization")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print(f"Quantization: {args.quant_type}")
    print(f"Device: {args.device}")
    print(f"Repository: {args.repo_name}")
    print(f"Save only: {args.save_only}")
    print("=" * 40)
    
    # Initialize quantizer
    quantizer = ModelQuantizer(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        trackio_url=args.trackio_url,
        experiment_name=args.experiment_name,
        dataset_repo=args.dataset_repo
    )
    
    if args.save_only:
        # Just quantize and save locally
        print("💾 Quantizing and saving locally...")
        quantized_path = quantizer.quantize_model(
            quant_type=args.quant_type,
            device=args.device,
            group_size=args.group_size
        )
        
        if quantized_path:
            print(f"✅ Quantized model saved to: {quantized_path}")
            print(f"📁 You can find the quantized model in: {quantized_path}")
        else:
            print("❌ Quantization failed")
            return 1
    else:
        # Full quantization and push workflow
        success = quantizer.quantize_and_push(
            quant_type=args.quant_type,
            device=args.device,
            group_size=args.group_size
        )
        
        if not success:
            print("❌ Quantization and push failed")
            return 1
    
    print("🎉 Quantization completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 