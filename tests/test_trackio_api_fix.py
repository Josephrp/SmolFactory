#!/usr/bin/env python3
"""
Test script for the fixed Trackio API client
Verifies connection to the deployed Trackio Space with automatic URL resolution
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.trackio_tonic.trackio_api_client import TrackioAPIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trackio_connection():
    """Test connection to Trackio Space"""
    print("🔧 Testing Trackio API Client with automatic URL resolution...")
    
    # Initialize the API client with Space ID
    space_id = "Tonic/trackio-monitoring-20250727"
    client = TrackioAPIClient(space_id)
    
    # Test 1: Space info
    print("\n1️⃣ Testing Space info resolution...")
    space_info = client.get_space_info()
    print(f"Space info result: {space_info}")
    
    if space_info.get('error'):
        print("❌ Space info failed")
        return False
    
    print("✅ Space info successful!")
    
    # Test 2: Connection test
    print("\n2️⃣ Testing connection...")
    connection_result = client.test_connection()
    print(f"Connection result: {connection_result}")
    
    if connection_result.get('error'):
        print("❌ Connection failed")
        return False
    
    print("✅ Connection successful!")
    
    # Test 3: List experiments
    print("\n3️⃣ Testing list experiments...")
    list_result = client.list_experiments()
    print(f"List experiments result: {list_result}")
    
    if list_result.get('error'):
        print("❌ List experiments failed")
        return False
    
    print("✅ List experiments successful!")
    
    # Test 4: Create a test experiment
    print("\n4️⃣ Testing create experiment...")
    create_result = client.create_experiment(
        name="test_experiment_auto_resolve",
        description="Test experiment with automatic URL resolution"
    )
    print(f"Create experiment result: {create_result}")
    
    if create_result.get('error'):
        print("❌ Create experiment failed")
        return False
    
    print("✅ Create experiment successful!")
    
    # Test 5: Log metrics
    print("\n5️⃣ Testing log metrics...")
    metrics = {
        "loss": 1.234,
        "accuracy": 0.85,
        "learning_rate": 2e-5,
        "gpu_memory": 22.5
    }
    
    log_metrics_result = client.log_metrics(
        experiment_id="test_experiment_auto_resolve",
        metrics=metrics,
        step=100
    )
    print(f"Log metrics result: {log_metrics_result}")
    
    if log_metrics_result.get('error'):
        print("❌ Log metrics failed")
        return False
    
    print("✅ Log metrics successful!")
    
    # Test 6: Log parameters
    print("\n6️⃣ Testing log parameters...")
    parameters = {
        "learning_rate": 2e-5,
        "batch_size": 8,
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "max_iters": 18000,
        "mixed_precision": "bf16"
    }
    
    log_params_result = client.log_parameters(
        experiment_id="test_experiment_auto_resolve",
        parameters=parameters
    )
    print(f"Log parameters result: {log_params_result}")
    
    if log_params_result.get('error'):
        print("❌ Log parameters failed")
        return False
    
    print("✅ Log parameters successful!")
    
    # Test 7: Get experiment details
    print("\n7️⃣ Testing get experiment details...")
    details_result = client.get_experiment_details("test_experiment_auto_resolve")
    print(f"Get experiment details result: {details_result}")
    
    if details_result.get('error'):
        print("❌ Get experiment details failed")
        return False
    
    print("✅ Get experiment details successful!")
    
    print("\n🎉 All tests passed! Trackio API client with automatic URL resolution is working correctly.")
    return True

def test_monitoring_integration():
    """Test the monitoring integration with the fixed API client"""
    print("\n🔧 Testing monitoring integration...")
    
    try:
        from src.monitoring import SmolLM3Monitor
        
        # Create a monitor instance
        monitor = SmolLM3Monitor(
            experiment_name="test_monitoring_auto_resolve",
            enable_tracking=True,
            log_metrics=True,
            log_config=True
        )
        
        print("✅ Monitor created successfully")
        
        # Test logging metrics
        metrics = {
            "loss": 1.123,
            "accuracy": 0.87,
            "learning_rate": 2e-5
        }
        
        monitor.log_metrics(metrics, step=50)
        print("✅ Metrics logged successfully")
        
        # Test logging configuration
        config = {
            "model_name": "HuggingFaceTB/SmolLM3-3B",
            "batch_size": 8,
            "learning_rate": 2e-5
        }
        
        monitor.log_config(config)
        print("✅ Configuration logged successfully")
        
        print("🎉 Monitoring integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        return False

def test_space_url_resolution():
    """Test automatic Space URL resolution"""
    print("\n🔧 Testing Space URL resolution...")
    
    try:
        from huggingface_hub import HfApi
        
        # Test Space info retrieval
        api = HfApi()
        space_id = "Tonic/trackio-monitoring-20250727"
        
        space_info = api.space_info(space_id)
        print(f"✅ Space info retrieved: {space_info}")
        
        if hasattr(space_info, 'host'):
            space_url = f"https://{space_info.host}"
            print(f"✅ Resolved Space URL: {space_url}")
        else:
            print("⚠️ Space host not available, using fallback")
            space_url = f"https://{space_id.replace('/', '-')}.hf.space"
            print(f"✅ Fallback Space URL: {space_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Space URL resolution failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Trackio API Client Tests with Automatic URL Resolution")
    print("=" * 70)
    
    # Test 1: Space URL Resolution
    url_resolution_success = test_space_url_resolution()
    
    # Test 2: API Client
    api_success = test_trackio_connection()
    
    # Test 3: Monitoring Integration
    monitoring_success = test_monitoring_integration()
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"Space URL Resolution: {'✅ PASSED' if url_resolution_success else '❌ FAILED'}")
    print(f"API Client Test: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"Monitoring Integration: {'✅ PASSED' if monitoring_success else '❌ FAILED'}")
    
    if url_resolution_success and api_success and monitoring_success:
        print("\n🎉 All tests passed! The Trackio integration with automatic URL resolution is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 