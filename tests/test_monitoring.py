#!/usr/bin/env python3
"""
Quick Start Script for Trackio Integration
Tests the monitoring functionality without full training
"""

import os
import json
import logging
from datetime import datetime
from monitoring import SmolLM3Monitor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_trackio_integration():
    """Test Trackio integration with sample data"""
    logger = setup_logging()
    
    print("🚀 Testing Trackio Integration")
    print("=" * 40)
    
    # Get Trackio URL from user or environment
    trackio_url = os.getenv('TRACKIO_URL')
    if not trackio_url:
        trackio_url = input("Enter your Trackio Space URL (or press Enter to skip): ").strip()
        if not trackio_url:
            print("⚠️  No Trackio URL provided. Running in local mode only.")
            trackio_url = None
    
    # Initialize monitor
    experiment_name = f"test_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    monitor = SmolLM3Monitor(
        experiment_name=experiment_name,
        trackio_url=trackio_url,
        enable_tracking=trackio_url is not None,
        log_artifacts=True,
        log_metrics=True,
        log_config=True
    )
    
    print(f"✅ Monitor initialized for experiment: {experiment_name}")
    
    # Test configuration logging
    sample_config = {
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "batch_size": 4,
        "learning_rate": 2e-5,
        "max_iters": 1000,
        "max_seq_length": 4096,
        "test_mode": True
    }
    
    print("📝 Logging configuration...")
    monitor.log_config(sample_config)
    
    # Test metrics logging
    print("📊 Logging sample metrics...")
    for step in range(0, 100, 10):
        metrics = {
            "loss": 2.0 - (step * 0.015),  # Simulate decreasing loss
            "accuracy": 0.5 + (step * 0.004),  # Simulate increasing accuracy
            "learning_rate": 2e-5,
            "step": step
        }
        monitor.log_metrics(metrics, step=step)
        print(f"   Step {step}: loss={metrics['loss']:.3f}, accuracy={metrics['accuracy']:.3f}")
    
    # Test system metrics
    print("💻 Logging system metrics...")
    monitor.log_system_metrics(step=50)
    
    # Test evaluation results
    print("📈 Logging evaluation results...")
    eval_results = {
        "eval_loss": 1.2,
        "eval_accuracy": 0.85,
        "perplexity": 3.3,
        "bleu_score": 0.72
    }
    monitor.log_evaluation_results(eval_results, step=100)
    
    # Test training summary
    print("📋 Logging training summary...")
    summary = {
        "final_loss": 0.5,
        "final_accuracy": 0.89,
        "total_steps": 100,
        "training_time_hours": 2.5,
        "model_size_gb": 6.2,
        "test_mode": True
    }
    monitor.log_training_summary(summary)
    
    # Close monitoring
    monitor.close()
    
    print("✅ Trackio integration test completed!")
    
    if trackio_url:
        experiment_url = monitor.get_experiment_url()
        if experiment_url:
            print(f"🌐 View your experiment at: {experiment_url}")
    
    return True

def test_local_monitoring():
    """Test local monitoring without Trackio"""
    logger = setup_logging()
    
    print("🔧 Testing Local Monitoring")
    print("=" * 30)
    
    # Initialize monitor without Trackio
    experiment_name = f"local_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    monitor = SmolLM3Monitor(
        experiment_name=experiment_name,
        enable_tracking=False,  # Disable Trackio
        log_artifacts=True,
        log_metrics=True,
        log_config=True
    )
    
    print(f"✅ Local monitor initialized for experiment: {experiment_name}")
    
    # Test local logging
    sample_config = {
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "batch_size": 4,
        "learning_rate": 2e-5,
        "local_test": True
    }
    
    print("📝 Logging configuration locally...")
    monitor.log_config(sample_config)
    
    # Test local metrics
    print("📊 Logging sample metrics locally...")
    for step in range(0, 50, 10):
        metrics = {
            "loss": 1.8 - (step * 0.02),
            "accuracy": 0.6 + (step * 0.005),
            "step": step
        }
        monitor.log_metrics(metrics, step=step)
        print(f"   Step {step}: loss={metrics['loss']:.3f}, accuracy={metrics['accuracy']:.3f}")
    
    print("✅ Local monitoring test completed!")
    return True

def main():
    """Main function"""
    print("Trackio Integration Quick Start")
    print("=" * 40)
    
    # Test local monitoring first
    test_local_monitoring()
    print()
    
    # Test Trackio integration if available
    try:
        test_trackio_integration()
    except Exception as e:
        print(f"❌ Trackio integration test failed: {e}")
        print("💡 Make sure you have a valid Trackio Space URL")
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Deploy Trackio to Hugging Face Spaces (see DEPLOYMENT_GUIDE.md)")
    print("2. Update your training script with Trackio integration")
    print("3. Run your first monitored training session")

if __name__ == "__main__":
    main() 