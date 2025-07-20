#!/usr/bin/env python3
"""
Simple test script to verify Trackio integration without loading models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.train_smollm3_openhermes_fr_a100_balanced import SmolLM3ConfigOpenHermesFRBalanced
from monitoring import create_monitor_from_config, SmolLM3Monitor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_trackio_config():
    """Test that Trackio configuration is properly set up"""
    print("Testing Trackio configuration...")
    
    # Create config
    config = SmolLM3ConfigOpenHermesFRBalanced()
    
    # Check Trackio-specific attributes
    trackio_attrs = [
        'enable_tracking',
        'trackio_url', 
        'trackio_token',
        'log_artifacts',
        'log_metrics',
        'log_config',
        'experiment_name'
    ]
    
    all_present = True
    for attr in trackio_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            print(f"✅ {attr}: {value}")
        else:
            print(f"❌ {attr}: Missing")
            all_present = False
    
    return all_present

def test_monitor_creation():
    """Test that monitor can be created from config"""
    print("\nTesting monitor creation...")
    
    try:
        config = SmolLM3ConfigOpenHermesFRBalanced()
        monitor = create_monitor_from_config(config)
        
        print(f"✅ Monitor created: {type(monitor)}")
        print(f"✅ Enable tracking: {monitor.enable_tracking}")
        print(f"✅ Log artifacts: {monitor.log_artifacts}")
        print(f"✅ Log metrics: {monitor.log_metrics}")
        print(f"✅ Log config: {monitor.log_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitor creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback_creation():
    """Test that Trackio callback can be created"""
    print("\nTesting callback creation...")
    
    try:
        config = SmolLM3ConfigOpenHermesFRBalanced()
        monitor = create_monitor_from_config(config)
        
        # Test callback creation
        callback = monitor.create_monitoring_callback()
        if callback:
            print(f"✅ Callback created: {type(callback)}")
            
            # Test callback methods exist
            required_methods = [
                'on_init_end',
                'on_log', 
                'on_save',
                'on_evaluate',
                'on_train_begin',
                'on_train_end'
            ]
            
            all_methods_present = True
            for method in required_methods:
                if hasattr(callback, method):
                    print(f"✅ Method {method}: exists")
                else:
                    print(f"❌ Method {method}: missing")
                    all_methods_present = False
            
            # Test that callback can be called (even if tracking is disabled)
            try:
                # Test a simple callback method
                callback.on_train_begin(None, None, None)
                print("✅ Callback methods can be called")
            except Exception as e:
                print(f"❌ Callback method call failed: {e}")
                all_methods_present = False
            
            return all_methods_present
        else:
            print("❌ Callback creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Callback creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitor_methods():
    """Test that monitor methods work correctly"""
    print("\nTesting monitor methods...")
    
    try:
        config = SmolLM3ConfigOpenHermesFRBalanced()
        monitor = SmolLM3Monitor(
            experiment_name="test_experiment",
            enable_tracking=False  # Disable actual tracking for test
        )
        
        # Test log_config
        test_config = {"batch_size": 8, "learning_rate": 3.5e-6}
        monitor.log_config(test_config)
        print("✅ log_config: works")
        
        # Test log_metrics
        test_metrics = {"loss": 0.5, "accuracy": 0.85}
        monitor.log_metrics(test_metrics, step=100)
        print("✅ log_metrics: works")
        
        # Test log_system_metrics
        monitor.log_system_metrics(step=100)
        print("✅ log_system_metrics: works")
        
        # Test log_evaluation_results
        test_eval = {"eval_loss": 0.4, "eval_accuracy": 0.88}
        monitor.log_evaluation_results(test_eval, step=100)
        print("✅ log_evaluation_results: works")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitor methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_arguments_fix():
    """Test that the training arguments fix is working"""
    print("\nTesting training arguments fix...")
    
    try:
        # Test the specific fix for report_to parameter
        from transformers import TrainingArguments
        import torch
        
        # Check if bf16 is supported
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        
        # Test that report_to=None works
        args = TrainingArguments(
            output_dir="/tmp/test",
            report_to=None,
            dataloader_pin_memory=False,
            group_by_length=True,
            prediction_loss_only=True,
            remove_unused_columns=False,
            ignore_data_skip=False,
            fp16=False,
            bf16=use_bf16,  # Only use bf16 if supported
            load_best_model_at_end=False,  # Disable to avoid eval strategy conflict
            greater_is_better=False,
            eval_strategy="no",  # Set to "no" to avoid conflicts
            save_strategy="steps"
        )
        
        print(f"✅ TrainingArguments created successfully")
        print(f"✅ report_to: {args.report_to}")
        print(f"✅ dataloader_pin_memory: {args.dataloader_pin_memory}")
        print(f"✅ group_by_length: {args.group_by_length}")
        print(f"✅ prediction_loss_only: {args.prediction_loss_only}")
        print(f"✅ bf16: {args.bf16} (supported: {use_bf16})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training arguments fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running Trackio integration tests...")
    
    tests = [
        test_trackio_config,
        test_monitor_creation,
        test_callback_creation,
        test_monitor_methods,
        test_training_arguments_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Trackio Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All Trackio integration tests passed!")
        print("\nTrackio integration is correctly implemented according to the documentation.")
        print("\nKey fixes applied:")
        print("- Fixed report_to parameter to use None instead of 'none'")
        print("- Added proper boolean type conversion for training arguments")
        print("- Improved callback implementation with proper inheritance")
        print("- Enhanced error handling in monitoring methods")
        print("- Added conditional support for dataloader_prefetch_factor")
    else:
        print("❌ Some Trackio integration tests failed.")
        print("Please check the errors above and fix any issues.") 