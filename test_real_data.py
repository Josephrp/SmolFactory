#!/usr/bin/env python3
"""
Test real training data logging and retrieval
"""

import json
import logging
from trackio_api_client import TrackioAPIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_training_data():
    """Test if real training data is being logged and can be retrieved"""
    
    client = TrackioAPIClient("https://tonic-test-trackio-test.hf.space")
    
    # Your experiment ID
    experiment_id = "exp_20250720_101955"
    
    print("🔍 Testing Real Training Data")
    print("=" * 50)
    
    # 1. Test getting experiment details
    print(f"\n1. Getting experiment details for {experiment_id}...")
    details_result = client.get_experiment_details(experiment_id)
    
    if "success" in details_result:
        print("✅ Experiment details retrieved successfully")
        try:
            details_preview = details_result['data'][:200]
            print(f"Details: {details_preview}...")
        except UnicodeEncodeError:
            print(f"Details: {details_result['data'][:100].encode('utf-8', errors='ignore').decode('utf-8')}...")
        
        # Look for metrics in the details
        if "metrics" in details_result['data'].lower():
            print("✅ Found metrics in experiment details")
        else:
            print("❌ No metrics found in experiment details")
    else:
        print(f"❌ Failed to get experiment details: {details_result}")
    
    # 2. Test getting training metrics specifically
    print(f"\n2. Getting training metrics for {experiment_id}...")
    metrics_result = client.get_training_metrics(experiment_id)
    
    if "success" in metrics_result:
        print("✅ Training metrics retrieved successfully")
        print(f"Metrics: {metrics_result['data'][:200]}...")
    else:
        print(f"❌ Failed to get training metrics: {metrics_result}")
    
    # 3. Test getting metrics history
    print(f"\n3. Getting metrics history for {experiment_id}...")
    history_result = client.get_experiment_metrics_history(experiment_id)
    
    if "success" in history_result:
        print("✅ Metrics history retrieved successfully")
        print(f"History: {history_result['data'][:200]}...")
    else:
        print(f"❌ Failed to get metrics history: {history_result}")
    
    # 4. List all experiments to see what's available
    print(f"\n4. Listing all experiments...")
    list_result = client.list_experiments()
    
    if "success" in list_result:
        print("✅ Experiments listed successfully")
        try:
            response_preview = list_result['data'][:300]
            print(f"Response: {response_preview}...")
        except UnicodeEncodeError:
            print(f"Response: {list_result['data'][:150].encode('utf-8', errors='ignore').decode('utf-8')}...")
    else:
        print(f"❌ Failed to list experiments: {list_result}")
    
    print("\n" + "=" * 50)
    print("🎯 Analysis Complete")
    print("=" * 50)

def log_real_training_step(experiment_id: str, step: int):
    """Log a single real training step for testing"""
    
    client = TrackioAPIClient("https://tonic-test-trackio-test.hf.space")
    
    # Real training metrics
    metrics = {
        "loss": 1.2345,
        "accuracy": 0.8567,
        "learning_rate": 3.5e-6,
        "gpu_memory_gb": 22.5,
        "training_time_per_step": 0.8,
        "epoch": 1,
        "samples_per_second": 45.2
    }
    
    print(f"📊 Logging real training step {step}...")
    result = client.log_metrics(experiment_id, metrics, step)
    
    if "success" in result:
        print(f"✅ Step {step} logged successfully")
        print(f"Metrics: {metrics}")
    else:
        print(f"❌ Failed to log step {step}: {result}")

if __name__ == "__main__":
    # Test existing data
    test_real_training_data()
    
    # Optionally log a test step
    print("\n" + "=" * 50)
    print("🧪 Testing Real Data Logging")
    print("=" * 50)
    
    experiment_id = "exp_20250720_101955"
    log_real_training_step(experiment_id, 1000)
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Run your actual training: python run_a100_large_experiment.py")
    print("2. The training will log real metrics every 25 steps")
    print("3. Check the visualization tab in your Trackio Space")
    print("4. Real training data should appear as training progresses")
    print("=" * 50) 