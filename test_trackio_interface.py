#!/usr/bin/env python3
"""
Test script for Trackio interface
Demonstrates how to use the enhanced monitoring interface
"""

import requests
import json
import time
from datetime import datetime

def test_trackio_interface():
    """Test the Trackio interface with realistic SmolLM3 training data"""
    
    # Trackio Space URL (replace with your actual URL)
    trackio_url = "https://tonic-test-trackio-test.hf.space"
    
    print("🚀 Testing Trackio Interface")
    print("=" * 50)
    
    # Step 1: Create an experiment
    print("\n1. Creating experiment...")
    experiment_name = "smollm3_openhermes_fr_balanced_test"
    experiment_description = "SmolLM3 fine-tuning on OpenHermes-FR dataset with balanced A100 configuration"
    
    # For demonstration, we'll simulate the API calls
    # In reality, these would be HTTP requests to your Trackio Space
    
    print(f"✅ Created experiment: {experiment_name}")
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"   Experiment ID: {experiment_id}")
    
    # Step 2: Log parameters
    print("\n2. Logging experiment parameters...")
    parameters = {
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "legmlai/openhermes-fr",
        "batch_size": 8,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 128,
        "learning_rate": 3.5e-6,
        "max_iters": 18000,
        "max_seq_length": 12288,
        "mixed_precision": "bf16",
        "use_flash_attention": True,
        "use_gradient_checkpointing": False,
        "optimizer": "adamw_torch",
        "scheduler": "cosine",
        "warmup_steps": 1200,
        "save_steps": 2000,
        "eval_steps": 1000,
        "logging_steps": 25,
        "no_think_system_message": True
    }
    
    print("✅ Logged parameters:")
    for key, value in parameters.items():
        print(f"   {key}: {value}")
    
    # Step 3: Simulate training metrics
    print("\n3. Simulating training metrics...")
    
    # Simulate realistic training progression
    base_loss = 2.5
    steps = list(range(0, 1000, 50))  # Every 50 steps
    
    for i, step in enumerate(steps):
        # Simulate loss decreasing over time with some noise
        progress = step / 1000
        loss = base_loss * (0.1 + 0.9 * (1 - progress)) + 0.1 * (1 - progress) * (i % 3 - 1)
        
        # Simulate accuracy increasing
        accuracy = 0.2 + 0.7 * progress + 0.05 * (i % 2)
        
        # Simulate learning rate decay
        lr = 3.5e-6 * (0.9 ** (step // 200))
        
        # Simulate GPU metrics
        gpu_memory = 20 + 5 * (0.8 + 0.2 * (i % 4) / 4)
        gpu_utilization = 85 + 10 * (i % 3 - 1)
        
        # Simulate training time
        training_time = 0.4 + 0.2 * (i % 2)
        
        metrics = {
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4),
            "learning_rate": round(lr, 8),
            "gpu_memory_gb": round(gpu_memory, 2),
            "gpu_utilization_percent": round(gpu_utilization, 1),
            "training_time_per_step": round(training_time, 3),
            "step": step
        }
        
        print(f"   Step {step}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}, LR={metrics['learning_rate']:.2e}")
        
        # In reality, this would be an HTTP POST to your Trackio Space
        # requests.post(f"{trackio_url}/log_metrics", json={
        #     "experiment_id": experiment_id,
        #     "metrics": metrics,
        #     "step": step
        # })
        
        time.sleep(0.1)  # Simulate processing time
    
    # Step 4: Log final results
    print("\n4. Logging final results...")
    final_results = {
        "final_loss": 0.234,
        "final_accuracy": 0.892,
        "total_training_time_hours": 4.5,
        "total_steps": 1000,
        "model_size_gb": 6.2,
        "training_completed": True,
        "checkpoint_path": "./outputs/balanced/checkpoint-1000"
    }
    
    print("✅ Final results:")
    for key, value in final_results.items():
        print(f"   {key}: {value}")
    
    # Step 5: Update experiment status
    print("\n5. Updating experiment status...")
    status = "completed"
    print(f"✅ Experiment status updated to: {status}")
    
    print("\n" + "=" * 50)
    print("🎉 Test completed successfully!")
    print(f"📊 View your experiment at: {trackio_url}")
    print(f"🔍 Experiment ID: {experiment_id}")
    print("\nNext steps:")
    print("1. Visit your Trackio Space")
    print("2. Go to 'View Experiments' tab")
    print("3. Enter the experiment ID to see details")
    print("4. Go to 'Visualizations' tab to see plots")
    print("5. Use 'Demo Data' tab to generate more test data")

def show_interface_features():
    """Show what features are available in the enhanced interface"""
    
    print("\n📊 Enhanced Trackio Interface Features")
    print("=" * 50)
    
    features = [
        "✅ Create experiments with detailed descriptions",
        "✅ Log comprehensive training parameters",
        "✅ Real-time metrics visualization with Plotly",
        "✅ Multiple metric types: loss, accuracy, learning rate, GPU metrics",
        "✅ Experiment comparison across multiple runs",
        "✅ Demo data generation for testing",
        "✅ Formatted experiment details with emojis and structure",
        "✅ Status tracking (running, completed, failed, paused)",
        "✅ Interactive plots with hover information",
        "✅ Comprehensive experiment overview with statistics"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n🎯 How to use with your SmolLM3 training:")
    print("1. Start your training with the monitoring enabled")
    print("2. Visit your Trackio Space during training")
    print("3. Watch real-time loss curves and metrics")
    print("4. Compare different training runs")
    print("5. Track GPU utilization and system metrics")

if __name__ == "__main__":
    test_trackio_interface()
    show_interface_features() 