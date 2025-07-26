#!/usr/bin/env python3
"""
Test script to verify mixed precision configuration
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_mixed_precision():
    """Test mixed precision configuration"""
    print("Testing mixed precision configuration...")
    
    # Test 1: Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Test 2: Test model loading with different dtypes
    try:
        from src.model import SmolLM3Model
        from config.train_smollm3_h100_lightweight import SmolLM3ConfigH100Lightweight
        
        config = SmolLM3ConfigH100Lightweight()
        print(f"Config fp16: {config.fp16}")
        print(f"Config bf16: {config.bf16}")
        
        # Test model loading
        model = SmolLM3Model(
            model_name="HuggingFaceTB/SmolLM3-3B",
            max_seq_length=4096,
            config=config
        )
        
        print(f"Model dtype: {model.torch_dtype}")
        print(f"Model device map: {model.device_map}")
        print("✅ Model loading successful!")
        
        # Test training arguments
        training_args = model.get_training_arguments("/tmp/test")
        print(f"Training args fp16: {training_args.fp16}")
        print(f"Training args bf16: {training_args.bf16}")
        print("✅ Training arguments created successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_mixed_precision()
    if success:
        print("\n🎉 Mixed precision test passed!")
    else:
        print("\n❌ Mixed precision test failed!")
        sys.exit(1) 