#!/usr/bin/env python3
"""
Quick test script to verify pipeline components
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from src.config import get_config
        print("✅ src.config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.config: {e}")
        return False
    
    try:
        from src.model import SmolLM3Model
        print("✅ src.model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.model: {e}")
        return False
    
    try:
        from src.data import SmolLM3Dataset
        print("✅ src.data imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.data: {e}")
        return False
    
    try:
        from src.trainer import SmolLM3Trainer
        print("✅ src.trainer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.trainer: {e}")
        return False
    
    try:
        from src.monitoring import create_monitor_from_config
        print("✅ src.monitoring imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.monitoring: {e}")
        return False
    
    return True

def test_config_loading():
    """Test that configuration files can be loaded"""
    print("\n🔍 Testing config loading...")
    
    config_files = [
        "config/train_smollm3_h100_lightweight.py",
        "config/train_smollm3_openhermes_fr_a100_large.py",
        "config/train_smollm3.py"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                config = get_config(config_file)
                print(f"✅ {config_file} loaded successfully")
                print(f"   Model: {config.model_name}")
                print(f"   Batch size: {config.batch_size}")
                if hasattr(config, 'sample_size') and config.sample_size:
                    print(f"   Sample size: {config.sample_size}")
            except Exception as e:
                print(f"❌ Failed to load {config_file}: {e}")
                return False
        else:
            print(f"⚠️  {config_file} not found")
    
    return True

def test_dataset_sampling():
    """Test dataset sampling functionality"""
    print("\n🔍 Testing dataset sampling...")
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Load a small test dataset
        print("Loading test dataset...")
        dataset = load_dataset("legmlai/openhermes-fr", split="train[:100]")
        print(f"Loaded {len(dataset)} samples")
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        print("✅ Tokenizer loaded successfully")
        
        # Test dataset with sampling
        from src.data import SmolLM3Dataset
        
        dataset_handler = SmolLM3Dataset(
            data_path="legmlai/openhermes-fr",
            tokenizer=tokenizer,
            max_seq_length=1024,
            sample_size=50,  # Sample 50 from the 100 we loaded
            sample_seed=42
        )
        
        train_dataset = dataset_handler.get_train_dataset()
        print(f"✅ Dataset sampling works: {len(train_dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset sampling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SmolLM3 Pipeline Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_dataset_sampling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to run.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        return 1

if __name__ == "__main__":
    exit(main()) 