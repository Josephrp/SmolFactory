#!/usr/bin/env python3
"""
Test script to verify that both monitoring systems use the same experiment ID format
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring import SmolLM3Monitor
from src.trackio import init as trackio_init

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_experiment_id_consistency():
    """Test that both monitoring systems use the same experiment ID format"""
    print("🔧 Testing experiment ID consistency...")
    
    # Test 1: SmolLM3Monitor experiment ID format
    print("\n1️⃣ Testing SmolLM3Monitor experiment ID format...")
    monitor = SmolLM3Monitor(
        experiment_name="test_experiment_id_consistency",
        enable_tracking=True
    )
    
    print(f"SmolLM3Monitor experiment ID: {monitor.experiment_id}")
    
    if monitor.experiment_id and monitor.experiment_id.startswith('exp_'):
        print("✅ SmolLM3Monitor uses correct experiment ID format (exp_)")
    else:
        print("❌ SmolLM3Monitor uses incorrect experiment ID format")
        return False
    
    # Test 2: Trackio experiment ID format
    print("\n2️⃣ Testing Trackio experiment ID format...")
    trackio_experiment_id = trackio_init(
        project_name="test_experiment_id_consistency",
        experiment_name="test_experiment_id_consistency"
    )
    
    print(f"Trackio experiment ID: {trackio_experiment_id}")
    
    if trackio_experiment_id and trackio_experiment_id.startswith('exp_'):
        print("✅ Trackio uses correct experiment ID format (exp_)")
    else:
        print("❌ Trackio uses incorrect experiment ID format")
        return False
    
    # Test 3: Verify both use the same format
    print("\n3️⃣ Testing experiment ID format consistency...")
    if monitor.experiment_id.startswith('exp_') and trackio_experiment_id.startswith('exp_'):
        print("✅ Both monitoring systems use the same experiment ID format")
        return True
    else:
        print("❌ Monitoring systems use different experiment ID formats")
        return False

def test_monitoring_integration():
    """Test that both monitoring systems can work together"""
    print("\n🔧 Testing monitoring integration...")
    
    try:
        # Create monitor
        monitor = SmolLM3Monitor(
            experiment_name="test_monitoring_integration",
            enable_tracking=True
        )
        
        print(f"✅ Monitor created with experiment ID: {monitor.experiment_id}")
        
        # Initialize trackio with the same experiment ID
        trackio_experiment_id = trackio_init(
            project_name="test_monitoring_integration",
            experiment_name="test_monitoring_integration"
        )
        
        print(f"✅ Trackio initialized with experiment ID: {trackio_experiment_id}")
        
        # Test logging metrics to both systems
        metrics = {"loss": 1.234, "accuracy": 0.85}
        
        # Log to monitor
        monitor.log_metrics(metrics, step=100)
        print("✅ Metrics logged to monitor")
        
        # Log to trackio
        from src.trackio import log as trackio_log
        trackio_log(metrics, step=100)
        print("✅ Metrics logged to trackio")
        
        print("🎉 Monitoring integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Experiment ID Consistency Tests")
    print("=" * 60)
    
    # Test 1: Experiment ID format consistency
    format_consistency = test_experiment_id_consistency()
    
    # Test 2: Monitoring integration
    integration_success = test_monitoring_integration()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"Experiment ID Format Consistency: {'✅ PASSED' if format_consistency else '❌ FAILED'}")
    print(f"Monitoring Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if format_consistency and integration_success:
        print("\n🎉 All tests passed! Experiment ID conflict is resolved.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 