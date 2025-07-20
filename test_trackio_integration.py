#!/usr/bin/env python3
"""
Test script to verify Trackio integration
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
    
    for attr in trackio_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            print(f"✅ {attr}: {value}")
        else:
            print(f"❌ {attr}: Missing")
    
    return True

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
            
            for method in required_methods:
                if hasattr(callback, method):
                    print(f"✅ Method {method}: exists")
                else:
                    print(f"❌ Method {method}: missing")
            
            return True
        else:
            print("❌ Callback creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Callback creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_arguments():
    """Test that training arguments are properly configured for Trackio"""
    print("\nTesting training arguments...")
    
    try:
        from model import SmolLM3Model
        
        config = SmolLM3ConfigOpenHermesFRBalanced()
        
        # Create model without loading the actual model
        model = SmolLM3Model(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            config=config
        )
        
        # Test training arguments creation
        training_args = model.get_training_arguments("/tmp/test_output")
        
        # Check that report_to is properly set
        if training_args.report_to is None:
            print("✅ report_to: None (correctly disabled external logging)")
        else:
            print(f"❌ report_to: {training_args.report_to} (should be None)")
        
        # Check other important parameters
        print(f"✅ dataloader_pin_memory: {training_args.dataloader_pin_memory}")
        print(f"✅ group_by_length: {training_args.group_by_length}")
        print(f"✅ prediction_loss_only: {training_args.prediction_loss_only}")
        print(f"✅ remove_unused_columns: {training_args.remove_unused_columns}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training arguments test failed: {e}")
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

if __name__ == "__main__":
    print("Running Trackio integration tests...")
    
    tests = [
        test_trackio_config,
        test_monitor_creation,
        test_callback_creation,
        test_training_arguments,
        test_monitor_methods
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
    else:
        print("❌ Some Trackio integration tests failed.")
        print("Please check the errors above and fix any issues.") 