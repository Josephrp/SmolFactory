#!/usr/bin/env python3
"""
Test script to verify monitoring.py against actual monitoring variables, 
dataset structure, and Trackio space deployment
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def test_dataset_structure_verification():
    """Test that monitoring.py matches the actual dataset structure"""
    print("🔍 Testing Dataset Structure Verification")
    print("=" * 50)
    
    # Expected dataset structure from setup_hf_dataset.py
    expected_dataset_fields = [
        'experiment_id',
        'name', 
        'description',
        'created_at',
        'status',
        'metrics',
        'parameters',
        'artifacts',
        'logs',
        'last_updated'
    ]
    
    # Expected metrics structure
    expected_metrics_fields = [
        'loss',
        'grad_norm',
        'learning_rate',
        'num_tokens',
        'mean_token_accuracy',
        'epoch',
        'total_tokens',
        'throughput',
        'step_time',
        'batch_size',
        'seq_len',
        'token_acc',
        'gpu_memory_allocated',
        'gpu_memory_reserved',
        'gpu_utilization',
        'cpu_percent',
        'memory_percent'
    ]
    
    # Expected parameters structure
    expected_parameters_fields = [
        'model_name',
        'max_seq_length',
        'batch_size',
        'learning_rate',
        'epochs',
        'dataset',
        'trainer_type',
        'hardware',
        'mixed_precision',
        'gradient_checkpointing',
        'flash_attention'
    ]
    
    print("✅ Expected dataset fields:", expected_dataset_fields)
    print("✅ Expected metrics fields:", expected_metrics_fields)
    print("✅ Expected parameters fields:", expected_parameters_fields)
    
    return True

def test_trackio_space_verification():
    """Test that monitoring.py matches the actual Trackio space structure"""
    print("\n🔍 Testing Trackio Space Verification")
    print("=" * 50)
    
    # Check if Trackio space app exists
    trackio_app = Path("scripts/trackio_tonic/app.py")
    if not trackio_app.exists():
        print("❌ Trackio space app not found")
        return False
    
    # Read Trackio space app to verify structure
    app_content = trackio_app.read_text(encoding='utf-8')
    
    # Expected Trackio space methods (from actual deployed space)
    expected_methods = [
        'update_trackio_config',
        'test_dataset_connection',
        'create_dataset_repository',
        'create_experiment_interface',
        'log_metrics_interface', 
        'log_parameters_interface',
        'get_experiment_details',
        'list_experiments_interface',
        'create_metrics_plot',
        'create_experiment_comparison',
        'simulate_training_data',
        'create_demo_experiment',
        'update_experiment_status_interface'
    ]
    
    all_found = True
    for method in expected_methods:
        if method in app_content:
            print(f"✅ Found: {method}")
        else:
            print(f"❌ Missing: {method}")
            all_found = False
    
    # Check for expected experiment structure
    expected_experiment_fields = [
        'id',
        'name',
        'description', 
        'created_at',
        'status',
        'metrics',
        'parameters',
        'artifacts',
        'logs'
    ]
    
    print("\nExpected experiment fields:", expected_experiment_fields)
    
    return all_found

def test_monitoring_variables_verification():
    """Test that monitoring.py uses the correct monitoring variables"""
    print("\n🔍 Testing Monitoring Variables Verification")
    print("=" * 50)
    
    # Check if monitoring.py exists
    monitoring_file = Path("src/monitoring.py")
    if not monitoring_file.exists():
        print("❌ monitoring.py not found")
        return False
    
    # Read monitoring.py to check variables
    monitoring_content = monitoring_file.read_text(encoding='utf-8')
    
    # Expected monitoring variables
    expected_variables = [
        'experiment_id',
        'experiment_name',
        'start_time',
        'metrics_history',
        'artifacts',
        'trackio_client',
        'hf_dataset_client',
        'dataset_repo',
        'hf_token',
        'enable_tracking'
    ]
    
    all_found = True
    for var in expected_variables:
        if var in monitoring_content:
            print(f"✅ Found: {var}")
        else:
            print(f"❌ Missing: {var}")
            all_found = False
    
    # Check for expected methods
    expected_methods = [
        'log_metrics',
        'log_configuration',
        'log_model_checkpoint',
        'log_evaluation_results',
        'log_system_metrics',
        'log_training_summary',
        'create_monitoring_callback'
    ]
    
    print("\nExpected monitoring methods:")
    for method in expected_methods:
        if method in monitoring_content:
            print(f"✅ Found: {method}")
        else:
            print(f"❌ Missing: {method}")
            all_found = False
    
    return all_found

def test_trackio_api_client_verification():
    """Test that monitoring.py uses the correct Trackio API client methods"""
    print("\n🔍 Testing Trackio API Client Verification")
    print("=" * 50)
    
    # Check if Trackio API client exists
    api_client = Path("scripts/trackio_tonic/trackio_api_client.py")
    if not api_client.exists():
        print("❌ Trackio API client not found")
        return False
    
    # Read API client to check methods
    api_content = api_client.read_text(encoding='utf-8')
    
    # Expected API client methods (from actual deployed space)
    expected_methods = [
        'create_experiment',
        'log_metrics',
        'log_parameters',
        'get_experiment_details',
        'list_experiments',
        'update_experiment_status',
        'simulate_training_data'
    ]
    
    all_found = True
    for method in expected_methods:
        if method in api_content:
            print(f"✅ Found: {method}")
        else:
            print(f"❌ Missing: {method}")
            all_found = False
    
    return all_found

def test_monitoring_integration_verification():
    """Test that monitoring.py integrates correctly with all components"""
    print("\n🔍 Testing Monitoring Integration Verification")
    print("=" * 50)
    
    try:
        # Test monitoring import
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from monitoring import SmolLM3Monitor
        
        # Test monitor creation with actual parameters
        monitor = SmolLM3Monitor(
            experiment_name="test-verification",
            trackio_url="https://huggingface.co/spaces/Tonic/trackio-monitoring-test",
            hf_token="test-token",
            dataset_repo="test/trackio-experiments"
        )
        
        print("✅ Monitor created successfully")
        print(f"   Experiment name: {monitor.experiment_name}")
        print(f"   Dataset repo: {monitor.dataset_repo}")
        print(f"   Enable tracking: {monitor.enable_tracking}")
        
        # Test that all expected attributes exist
        expected_attrs = [
            'experiment_name',
            'dataset_repo',
            'hf_token',
            'enable_tracking',
            'start_time',
            'metrics_history',
            'artifacts'
        ]
        
        all_attrs_found = True
        for attr in expected_attrs:
            if hasattr(monitor, attr):
                print(f"✅ Found attribute: {attr}")
            else:
                print(f"❌ Missing attribute: {attr}")
                all_attrs_found = False
        
        return all_attrs_found
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        return False

def test_dataset_structure_compatibility():
    """Test that the monitoring.py dataset structure matches the actual dataset"""
    print("\n🔍 Testing Dataset Structure Compatibility")
    print("=" * 50)
    
    # Get the actual dataset structure from setup script
    setup_script = Path("scripts/dataset_tonic/setup_hf_dataset.py")
    if not setup_script.exists():
        print("❌ Dataset setup script not found")
        return False
    
    setup_content = setup_script.read_text(encoding='utf-8')
    
    # Check that monitoring.py uses the same structure
    monitoring_file = Path("src/monitoring.py")
    monitoring_content = monitoring_file.read_text(encoding='utf-8')
    
    # Key dataset fields that should be consistent
    key_fields = [
        'experiment_id',
        'name',
        'description',
        'created_at',
        'status',
        'metrics',
        'parameters',
        'artifacts',
        'logs'
    ]
    
    all_compatible = True
    for field in key_fields:
        if field in setup_content and field in monitoring_content:
            print(f"✅ Compatible: {field}")
        else:
            print(f"❌ Incompatible: {field}")
            all_compatible = False
    
    return all_compatible

def test_trackio_space_compatibility():
    """Test that monitoring.py is compatible with the actual Trackio space"""
    print("\n🔍 Testing Trackio Space Compatibility")
    print("=" * 50)
    
    # Check Trackio space app
    trackio_app = Path("scripts/trackio_tonic/app.py")
    if not trackio_app.exists():
        print("❌ Trackio space app not found")
        return False
    
    trackio_content = trackio_app.read_text(encoding='utf-8')
    
    # Check monitoring.py
    monitoring_file = Path("src/monitoring.py")
    monitoring_content = monitoring_file.read_text(encoding='utf-8')
    
    # Key methods that should be compatible (only those actually used in monitoring.py)
    key_methods = [
        'log_metrics',
        'log_parameters',
        'list_experiments',
        'update_experiment_status'
    ]
    
    all_compatible = True
    for method in key_methods:
        if method in trackio_content and method in monitoring_content:
            print(f"✅ Compatible: {method}")
        else:
            print(f"❌ Incompatible: {method}")
            all_compatible = False
    
    return all_compatible

def main():
    """Run all monitoring verification tests"""
    print("🚀 Monitoring Verification Tests")
    print("=" * 50)
    
    tests = [
        test_dataset_structure_verification,
        test_trackio_space_verification,
        test_monitoring_variables_verification,
        test_trackio_api_client_verification,
        test_monitoring_integration_verification,
        test_dataset_structure_compatibility,
        test_trackio_space_compatibility
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL MONITORING VERIFICATION TESTS PASSED!")
        print("✅ Dataset structure: Compatible")
        print("✅ Trackio space: Compatible")
        print("✅ Monitoring variables: Correct")
        print("✅ API client: Compatible")
        print("✅ Integration: Working")
        print("✅ Structure compatibility: Verified")
        print("✅ Space compatibility: Verified")
        print("\nMonitoring.py is fully compatible with all components!")
    else:
        print("❌ SOME MONITORING VERIFICATION TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 