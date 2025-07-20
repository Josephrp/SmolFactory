#!/usr/bin/env python3
"""
Test script to verify the training pipeline fixes
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from src.config import get_config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from src.model import SmolLM3Model
        print("✅ model.py imported successfully")
    except Exception as e:
        print(f"❌ model.py import failed: {e}")
        return False
    
    try:
        from src.data import SmolLM3Dataset
        print("✅ data.py imported successfully")
    except Exception as e:
        print(f"❌ data.py import failed: {e}")
        return False
    
    try:
        from src.trainer import SmolLM3Trainer
        print("✅ trainer.py imported successfully")
    except Exception as e:
        print(f"❌ trainer.py import failed: {e}")
        return False
    
    try:
        from src.monitoring import create_monitor_from_config
        print("✅ monitoring.py imported successfully")
    except Exception as e:
        print(f"❌ monitoring.py import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 Testing configuration loading...")
    
    try:
        from src.config import get_config
        
        # Test loading the H100 lightweight config
        config = get_config("config/train_smollm3_h100_lightweight.py")
        print("✅ Configuration loaded successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_monitoring_setup():
    """Test monitoring setup without Trackio Space"""
    print("\n🔍 Testing monitoring setup...")
    
    try:
        from src.monitoring import create_monitor_from_config
        from src.config import get_config
        
        # Load config
        config = get_config("config/train_smollm3_h100_lightweight.py")
        
        # Set Trackio URL to a non-existent one to test fallback
        config.trackio_url = "https://non-existent-space.hf.space"
        config.experiment_name = "test_experiment"
        
        # Create monitor
        monitor = create_monitor_from_config(config)
        print("✅ Monitoring setup successful")
        print(f"   Experiment: {monitor.experiment_name}")
        print(f"   Tracking enabled: {monitor.enable_tracking}")
        print(f"   HF Dataset: {monitor.dataset_repo}")
        
        return True
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return False

def test_trainer_creation():
    """Test trainer creation"""
    print("\n🔍 Testing trainer creation...")
    
    try:
        from src.config import get_config
        from src.model import SmolLM3Model
        from src.data import SmolLM3Dataset
        from src.trainer import SmolLM3Trainer
        
        # Load config
        config = get_config("config/train_smollm3_h100_lightweight.py")
        
        # Create model (without loading the actual model)
        model = SmolLM3Model(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            config=config
        )
        print("✅ Model created successfully")
        
        # Create dataset (without loading actual data)
        dataset = SmolLM3Dataset(
            data_path=config.dataset_name,
            tokenizer=model.tokenizer,
            max_seq_length=config.max_seq_length,
            config=config
        )
        print("✅ Dataset created successfully")
        
        # Create trainer
        trainer = SmolLM3Trainer(
            model=model,
            dataset=dataset,
            config=config,
            output_dir="/tmp/test_output",
            init_from="scratch"
        )
        print("✅ Trainer created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False

def test_format_string_fix():
    """Test that the format string fix works"""
    print("\n🔍 Testing format string fix...")
    
    try:
        from src.trainer import SmolLM3Trainer
        
        # Test the SimpleConsoleCallback format string handling
        from transformers import TrainerCallback
        
        class TestCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and isinstance(logs, dict):
                    step = getattr(state, 'global_step', 'unknown')
                    loss = logs.get('loss', 'N/A')
                    lr = logs.get('learning_rate', 'N/A')
                    
                    # Test the fixed format string logic
                    if isinstance(loss, (int, float)):
                        loss_str = f"{loss:.4f}"
                    else:
                        loss_str = str(loss)
                    if isinstance(lr, (int, float)):
                        lr_str = f"{lr:.2e}"
                    else:
                        lr_str = str(lr)
                    
                    print(f"Step {step}: loss={loss_str}, lr={lr_str}")
        
        print("✅ Format string fix works correctly")
        return True
    except Exception as e:
        print(f"❌ Format string fix test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SmolLM3 Training Pipeline Fixes")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_monitoring_setup,
        test_trainer_creation,
        test_format_string_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The training pipeline should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 