#!/usr/bin/env python3
"""
Test monitoring integration for real experiment
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_monitoring_setup():
    """Test that monitoring is correctly configured"""
    
    print("🔍 Testing Monitoring Integration")
    print("=" * 50)
    
    # Test 1: Check if monitoring module can be imported
    try:
        from monitoring import SmolLM3Monitor, create_monitor_from_config
        print("✅ Monitoring module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import monitoring module: {e}")
        return False
    
    # Test 2: Check if API client can be imported
    try:
        from trackio_api_client import TrackioAPIClient
        print("✅ Trackio API client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Trackio API client: {e}")
        return False
    
    # Test 3: Test configuration loading
    try:
        from config.train_smollm3_openhermes_fr_a100_balanced import get_config
        config = get_config("config/train_smollm3_openhermes_fr_a100_balanced.py")
        print("✅ Configuration loaded successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Max iterations: {config.max_iters}")
        print(f"   Enable tracking: {config.enable_tracking}")
        print(f"   Trackio URL: {config.trackio_url}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test 4: Test monitor creation
    try:
        # Set the Trackio URL for testing
        config.trackio_url = "https://tonic-test-trackio-test.hf.space"
        config.experiment_name = "test_monitoring_integration"
        
        monitor = create_monitor_from_config(config)
        print("✅ Monitor created successfully")
        print(f"   Experiment name: {monitor.experiment_name}")
        print(f"   Enable tracking: {monitor.enable_tracking}")
        print(f"   Log metrics: {monitor.log_metrics}")
        print(f"   Log artifacts: {monitor.log_artifacts}")
        
        if monitor.enable_tracking and monitor.trackio_client:
            print("✅ Trackio client initialized")
            if monitor.experiment_id:
                print(f"   Experiment ID: {monitor.experiment_id}")
            else:
                print("   ⚠️ No experiment ID (will be created during training)")
        else:
            print("   ⚠️ Trackio client not initialized")
            
    except Exception as e:
        print(f"❌ Failed to create monitor: {e}")
        return False
    
    # Test 5: Test callback creation
    try:
        callback = monitor.create_monitoring_callback()
        if callback:
            print("✅ Monitoring callback created successfully")
        else:
            print("   ⚠️ No monitoring callback (tracking disabled)")
    except Exception as e:
        print(f"❌ Failed to create callback: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 Monitoring Integration Test Complete")
    print("=" * 50)
    
    return True

def test_real_experiment_command():
    """Test the real experiment command"""
    
    print("\n🚀 Testing Real Experiment Command")
    print("=" * 50)
    
    # Build the command
    cmd = [
        "python", "run_a100_large_experiment.py",
        "--config", "config/train_smollm3_openhermes_fr_a100_balanced.py",
        "--experiment-name", "petit-elle-l-aime-3-balanced-real",
        "--output-dir", "./outputs/balanced-real",
        "--trackio-url", "https://tonic-test-trackio-test.hf.space"
    ]
    
    print("Command to run:")
    print(" ".join(cmd))
    
    print("\nThis command will:")
    print("✅ Load the balanced A100 configuration")
    print("✅ Create a real experiment in Trackio")
    print("✅ Log real training metrics every 25 steps")
    print("✅ Save checkpoints every 2000 steps")
    print("✅ Monitor progress in real-time")
    
    print("\nExpected training parameters:")
    print("   Model: HuggingFaceTB/SmolLM3-3B")
    print("   Batch size: 8")
    print("   Gradient accumulation: 16")
    print("   Effective batch size: 128")
    print("   Learning rate: 3.5e-6")
    print("   Max iterations: 18000")
    print("   Mixed precision: bf16")
    print("   Max sequence length: 12288")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to run real experiment!")
    print("=" * 50)

if __name__ == "__main__":
    # Test monitoring integration
    if test_monitoring_setup():
        # Show real experiment command
        test_real_experiment_command()
    else:
        print("\n❌ Monitoring integration test failed. Please fix issues before running real experiment.") 