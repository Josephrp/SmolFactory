#!/usr/bin/env python3
"""
Test script for monitoring integration with HF Datasets
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_monitoring_import():
    """Test that monitoring can be imported"""
    try:
        from monitoring import SmolLM3Monitor, create_monitor_from_config
        logger.info("✅ Monitoring module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import monitoring: {e}")
        return False

def test_monitor_creation():
    """Test monitor creation with environment variables"""
    try:
        from monitoring import SmolLM3Monitor
        
        # Test with environment variables
        hf_token = os.environ.get('HF_TOKEN')
        dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
        
        logger.info(f"🔧 Testing monitor creation...")
        logger.info(f"   HF_TOKEN: {'Set' if hf_token else 'Not set'}")
        logger.info(f"   Dataset repo: {dataset_repo}")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_experiment",
            enable_tracking=False,  # Disable Trackio for testing
            hf_token=hf_token,
            dataset_repo=dataset_repo
        )
        
        logger.info(f"✅ Monitor created successfully")
        logger.info(f"   Experiment name: {monitor.experiment_name}")
        logger.info(f"   Dataset repo: {monitor.dataset_repo}")
        logger.info(f"   HF client: {'Available' if monitor.hf_dataset_client else 'Not available'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create monitor: {e}")
        return False

def test_config_creation():
    """Test monitor creation from config"""
    try:
        from monitoring import create_monitor_from_config
        
        # Create a simple config object
        class TestConfig:
            enable_tracking = True
            experiment_name = "test_config_experiment"
            trackio_url = None
            trackio_token = None
            log_artifacts = True
            log_metrics = True
            log_config = True
        
        config = TestConfig()
        
        logger.info(f"🔧 Testing monitor creation from config...")
        
        monitor = create_monitor_from_config(config)
        
        logger.info(f"✅ Monitor created from config successfully")
        logger.info(f"   Experiment name: {monitor.experiment_name}")
        logger.info(f"   Dataset repo: {monitor.dataset_repo}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create monitor from config: {e}")
        return False

def test_metrics_logging():
    """Test metrics logging functionality"""
    try:
        from monitoring import SmolLM3Monitor
        
        logger.info(f"🔧 Testing metrics logging...")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_metrics",
            enable_tracking=False,
            log_metrics=True
        )
        
        # Test metrics logging
        test_metrics = {
            'loss': 0.5,
            'learning_rate': 1e-4,
            'step': 100
        }
        
        monitor.log_metrics(test_metrics, step=100)
        
        logger.info(f"✅ Metrics logged successfully")
        logger.info(f"   Metrics history length: {len(monitor.metrics_history)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log metrics: {e}")
        return False

def test_configuration_logging():
    """Test configuration logging functionality"""
    try:
        from monitoring import SmolLM3Monitor
        
        logger.info(f"🔧 Testing configuration logging...")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_config",
            enable_tracking=False,
            log_config=True
        )
        
        # Test configuration logging
        test_config = {
            'model_name': 'test-model',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'max_steps': 1000
        }
        
        monitor.log_configuration(test_config)
        
        logger.info(f"✅ Configuration logged successfully")
        logger.info(f"   Artifacts count: {len(monitor.artifacts)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log configuration: {e}")
        return False

def test_system_metrics():
    """Test system metrics logging"""
    try:
        from monitoring import SmolLM3Monitor
        
        logger.info(f"🔧 Testing system metrics logging...")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_system",
            enable_tracking=False,
            log_metrics=True
        )
        
        # Test system metrics
        monitor.log_system_metrics(step=1)
        
        logger.info(f"✅ System metrics logged successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log system metrics: {e}")
        return False

def test_training_summary():
    """Test training summary logging"""
    try:
        from monitoring import SmolLM3Monitor
        
        logger.info(f"🔧 Testing training summary logging...")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_summary",
            enable_tracking=False,
            log_artifacts=True
        )
        
        # Test training summary
        test_summary = {
            'final_loss': 0.1,
            'total_steps': 1000,
            'training_duration': 3600,
            'model_path': '/output/model',
            'status': 'completed'
        }
        
        monitor.log_training_summary(test_summary)
        
        logger.info(f"✅ Training summary logged successfully")
        logger.info(f"   Artifacts count: {len(monitor.artifacts)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log training summary: {e}")
        return False

def test_callback_creation():
    """Test callback creation for trainer integration"""
    try:
        from monitoring import SmolLM3Monitor
        
        logger.info(f"🔧 Testing callback creation...")
        
        monitor = SmolLM3Monitor(
            experiment_name="test_callback",
            enable_tracking=False
        )
        
        # Test callback creation
        callback = monitor.create_monitoring_callback()
        
        logger.info(f"✅ Callback created successfully")
        logger.info(f"   Callback type: {type(callback).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create callback: {e}")
        return False

def main():
    """Run all monitoring integration tests"""
    
    print("🧪 Testing Monitoring Integration with HF Datasets")
    print("=" * 60)
    
    tests = [
        ("Module Import", test_monitoring_import),
        ("Monitor Creation", test_monitor_creation),
        ("Config Creation", test_config_creation),
        ("Metrics Logging", test_metrics_logging),
        ("Configuration Logging", test_configuration_logging),
        ("System Metrics", test_system_metrics),
        ("Training Summary", test_training_summary),
        ("Callback Creation", test_callback_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results")
    print("=" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Monitoring integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    print(f"\n📋 Environment Check:")
    print(f"   HF_TOKEN: {'Set' if os.environ.get('HF_TOKEN') else 'Not set'}")
    print(f"   TRACKIO_DATASET_REPO: {os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')}")
    
    if passed == total:
        print(f"\n✅ Monitoring integration is ready for use!")
        print(f"   Next step: Run a training experiment to verify full functionality")
    else:
        print(f"\n⚠️  Please fix the failed tests before using monitoring")

if __name__ == "__main__":
    main() 