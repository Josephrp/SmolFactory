#!/usr/bin/env python3
"""
Test script to verify that training arguments are properly created
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.train_smollm3_openhermes_fr_a100_balanced import SmolLM3ConfigOpenHermesFRBalanced
from model import SmolLM3Model
from trainer import SmolLM3Trainer
from data import SmolLM3Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_training_arguments():
    """Test that training arguments are properly created"""
    print("Testing training arguments creation...")
    
    # Create config
    config = SmolLM3ConfigOpenHermesFRBalanced()
    print(f"Config created: {type(config)}")
    
    # Create model (without actually loading the model)
    try:
        model = SmolLM3Model(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            config=config
        )
        print("Model created successfully")
        
        # Test training arguments creation
        training_args = model.get_training_arguments("/tmp/test_output")
        print(f"Training arguments created: {type(training_args)}")
        print(f"Training arguments keys: {list(training_args.__dict__.keys())}")
        
        # Test specific parameters that might cause issues
        print(f"report_to: {training_args.report_to}")
        print(f"dataloader_pin_memory: {training_args.dataloader_pin_memory}")
        print(f"group_by_length: {training_args.group_by_length}")
        print(f"prediction_loss_only: {training_args.prediction_loss_only}")
        print(f"ignore_data_skip: {training_args.ignore_data_skip}")
        print(f"remove_unused_columns: {training_args.remove_unused_columns}")
        print(f"fp16: {training_args.fp16}")
        print(f"bf16: {training_args.bf16}")
        print(f"load_best_model_at_end: {training_args.load_best_model_at_end}")
        print(f"greater_is_better: {training_args.greater_is_better}")
        
        print("✅ Training arguments test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training arguments test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_callback_creation():
    """Test that callbacks are properly created"""
    print("\nTesting callback creation...")
    
    try:
        from monitoring import create_monitor_from_config
        from config.train_smollm3_openhermes_fr_a100_balanced import SmolLM3ConfigOpenHermesFRBalanced
        
        config = SmolLM3ConfigOpenHermesFRBalanced()
        monitor = create_monitor_from_config(config)
        
        # Test callback creation
        callback = monitor.create_monitoring_callback()
        if callback:
            print(f"✅ Callback created successfully: {type(callback)}")
            return True
        else:
            print("❌ Callback creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Callback creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running training fixes tests...")
    
    test1_passed = test_training_arguments()
    test2_passed = test_callback_creation()
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed! The fixes should work.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 