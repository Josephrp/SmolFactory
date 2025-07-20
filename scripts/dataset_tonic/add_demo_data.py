#!/usr/bin/env python3
"""
Add demo training data to an existing experiment
This will populate the experiment with realistic training metrics for visualization
"""

import json
import logging
import numpy as np
from datetime import datetime
from trackio_api_client import TrackioAPIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_demo_training_data(experiment_id: str, num_steps: int = 50):
    """Add realistic demo training data to an experiment"""
    
    client = TrackioAPIClient("https://tonic-test-trackio-test.hf.space")
    
    print(f"🎯 Adding demo training data to experiment: {experiment_id}")
    print(f"📊 Will add {num_steps} metric entries...")
    
    # Simulate realistic training metrics
    for step in range(0, num_steps * 25, 25):  # Every 25 steps
        # Simulate loss decreasing over time with some noise
        base_loss = 2.0 * np.exp(-step / 500)
        noise = 0.1 * np.random.random()
        loss = max(0.1, base_loss + noise)
        
        # Simulate accuracy increasing over time
        base_accuracy = 0.3 + 0.6 * (1 - np.exp(-step / 300))
        accuracy = min(0.95, base_accuracy + 0.05 * np.random.random())
        
        # Simulate learning rate decay
        lr = 3.5e-6 * (0.9 ** (step // 200))
        
        # Simulate GPU memory usage
        gpu_memory = 20 + 5 * np.random.random()
        
        # Simulate training time per step
        training_time = 0.5 + 0.2 * np.random.random()
        
        metrics = {
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "learning_rate": round(lr, 8),
            "gpu_memory_gb": round(gpu_memory, 2),
            "training_time_per_step": round(training_time, 3),
            "epoch": step // 100 + 1,
            "samples_per_second": round(50 + 20 * np.random.random(), 1)
        }
        
        # Log metrics to the experiment
        result = client.log_metrics(experiment_id, metrics, step)
        
        if "success" in result:
            print(f"✅ Step {step}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        else:
            print(f"❌ Step {step}: Failed to log metrics - {result}")
    
    print(f"\n🎉 Demo data added successfully!")
    print(f"📊 Total steps logged: {num_steps}")
    print(f"🔗 View in Trackio Space: https://tonic-test-trackio-test.hf.space")
    print(f"📈 Go to 'Visualizations' tab and select experiment: {experiment_id}")

def main():
    """Main function"""
    print("🚀 Trackio Demo Data Generator")
    print("=" * 50)
    
    # Your experiment ID from the logs
    experiment_id = "exp_20250720_101955"  # petit-elle-l-aime-3-balanced
    
    print(f"📋 Target experiment: {experiment_id}")
    print(f"📝 Experiment name: petit-elle-l-aime-3-balanced")
    
    # Add demo data
    add_demo_training_data(experiment_id, num_steps=50)
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Go to https://tonic-test-trackio-test.hf.space")
    print("2. Click on '📊 Visualizations' tab")
    print("3. Enter your experiment ID: exp_20250720_101955")
    print("4. Select a metric (loss, accuracy, etc.)")
    print("5. Click 'Create Plot' to see the training curves!")
    print("=" * 50)

if __name__ == "__main__":
    main() 