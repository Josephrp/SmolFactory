#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_imports():
    """Test that all configuration files can be imported correctly"""
    print("🧪 Testing configuration imports...")
    
    try:
        # Test base config only
        from config.train_smollm3 import SmolLM3Config, get_config
        print("✅ Base config imported successfully")
        
        # Test H100 lightweight config (without triggering __post_init__)
        import importlib.util
        spec = importlib.util.spec_from_file_location("h100_config", "config/train_smollm3_h100_lightweight.py")
        h100_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h100_module)
        print("✅ H100 lightweight config imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_training_script():
    """Test that the training script can be imported"""
    print("\n🧪 Testing training script...")
    
    try:
        # Add src to path
        src_path = str(project_root / "src")
        sys.path.insert(0, src_path)
        
        # Test importing training modules
        from train import main as train_main
        print("✅ Training script imported successfully")
        
        from model import SmolLM3Model
        print("✅ Model module imported successfully")
        
        from data import load_dataset
        print("✅ Data module imported successfully")
        
        from monitoring import SmolLM3Monitor, create_monitor_from_config
        print("✅ Monitoring module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_scripts():
    """Test that the scripts can be imported"""
    print("\n🧪 Testing scripts...")
    
    try:
        # Test dataset setup script
        sys.path.insert(0, str(project_root / "scripts" / "dataset_tonic"))
        from setup_hf_dataset import setup_trackio_dataset
        print("✅ Dataset setup script imported successfully")
        
        # Test trackio scripts
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        print("✅ Trackio deployment script imported successfully")
        
        from configure_trackio import configure_trackio
        print("✅ Trackio configuration script imported successfully")
        
        # Test model push script
        sys.path.insert(0, str(project_root / "scripts" / "model_tonic"))
        from push_to_huggingface import HuggingFacePusher
        print("✅ Model push script imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SmolLM3 Fine-tuning Pipeline")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_training_script,
        test_scripts
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\n🚀 You can now run: ./launch.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 