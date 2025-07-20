#!/usr/bin/env python3
"""
Test script to verify H100 lightweight configuration loads correctly
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_h100_lightweight_config():
    """Test the H100 lightweight configuration"""
    try:
        from config.train_smollm3_h100_lightweight import config
        
        print("✅ H100 Lightweight configuration loaded successfully!")
        print(f"Model: {config.model_name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Sample size: {config.sample_size}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max sequence length: {config.max_seq_length}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading H100 lightweight configuration: {e}")
        return False

def test_training_script_import():
    """Test that the training script can import the configuration"""
    try:
        from scripts.training.train import main
        print("✅ Training script imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Error importing training script: {e}")
        return False

if __name__ == "__main__":
    print("Testing H100 Lightweight Configuration...")
    print("=" * 50)
    
    success = True
    success &= test_h100_lightweight_config()
    success &= test_training_script_import()
    
    if success:
        print("\n🎉 All tests passed! Configuration is ready for training.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1) 