#!/usr/bin/env python3
"""
Simple test script for the simplified pipeline approach
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_simple_training_script():
    """Test the simplified training script"""
    print("🔍 Testing simplified training script...")
    
    try:
        # Test that the training script can be imported
        from scripts.training.train import main as train_main
        print("✅ Training script imported successfully")
        
        # Test config loading
        from config.train_smollm3_h100_lightweight import config as h100_config
        print("✅ H100 lightweight config loaded successfully")
        print(f"   Model: {h100_config.model_name}")
        print(f"   Batch size: {h100_config.batch_size}")
        print(f"   Sample size: {h100_config.sample_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def test_config_files():
    """Test that all required config files exist"""
    print("\n🔍 Testing config files...")
    
    config_files = [
        "config/train_smollm3_h100_lightweight.py",
        "config/train_smollm3_openhermes_fr_a100_large.py",
        "config/train_smollm3_openhermes_fr_a100_multiple_passes.py"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} not found")
            return False
    
    return True

def test_scripts():
    """Test that all required scripts exist"""
    print("\n🔍 Testing scripts...")
    
    script_files = [
        "scripts/training/train.py",
        "scripts/trackio_tonic/deploy_trackio_space.py",
        "scripts/trackio_tonic/configure_trackio.py",
        "scripts/dataset_tonic/setup_hf_dataset.py",
        "scripts/model_tonic/push_to_huggingface.py"
    ]
    
    for script_file in script_files:
        if os.path.exists(script_file):
            print(f"✅ {script_file}")
        else:
            print(f"❌ {script_file} not found")
            return False
    
    return True

def test_launch_script():
    """Test that the launch script exists and is executable"""
    print("\n🔍 Testing launch script...")
    
    launch_script = "launch.sh"
    if os.path.exists(launch_script):
        print(f"✅ {launch_script} exists")
        
        # Check if it's executable
        if os.access(launch_script, os.X_OK):
            print(f"✅ {launch_script} is executable")
        else:
            print(f"⚠️  {launch_script} is not executable (run: chmod +x launch.sh)")
        
        return True
    else:
        print(f"❌ {launch_script} not found")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Simplified SmolLM3 Pipeline")
    print("=" * 50)
    
    tests = [
        test_simple_training_script,
        test_config_files,
        test_scripts,
        test_launch_script
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
        print("🎉 All tests passed! Simplified pipeline is ready to run.")
        print("\n🚀 To run the pipeline:")
        print("1. chmod +x launch.sh")
        print("2. ./launch.sh")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        return 1

if __name__ == "__main__":
    exit(main()) 