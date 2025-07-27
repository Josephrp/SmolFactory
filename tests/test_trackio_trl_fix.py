#!/usr/bin/env python3
"""
Test script to verify Trackio TRL compatibility fix
Tests that our trackio module provides the interface expected by TRL library
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_trackio_interface():
    """Test that trackio module provides the expected interface"""
    print("🔍 Testing Trackio TRL Interface")
    
    try:
        # Test importing trackio
        import trackio
        print("✅ Successfully imported trackio module")
        
        # Test that required functions exist
        required_functions = ['init', 'log', 'finish']
        for func_name in required_functions:
            if hasattr(trackio, func_name):
                print(f"✅ Found required function: {func_name}")
            else:
                print(f"❌ Missing required function: {func_name}")
                return False
        
        # Test initialization with arguments
        experiment_id = trackio.init(
            project_name="test_project",
            experiment_name="test_experiment",
            trackio_url="https://test.hf.space",
            dataset_repo="test/trackio-experiments"
        )
        print(f"✅ Trackio initialization with args successful: {experiment_id}")
        
        # Test initialization without arguments (TRL compatibility)
        experiment_id2 = trackio.init()
        print(f"✅ Trackio initialization without args successful: {experiment_id2}")
        
        # Test logging
        metrics = {'loss': 0.5, 'learning_rate': 1e-4}
        trackio.log(metrics, step=1)
        print("✅ Trackio logging successful")
        
        # Test finishing
        trackio.finish()
        print("✅ Trackio finish successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Trackio interface test failed: {e}")
        return False

def test_trl_compatibility():
    """Test that our trackio module is compatible with TRL expectations"""
    print("\n🔍 Testing TRL Compatibility")
    
    try:
        # Simulate what TRL would do
        import trackio
        
        # TRL expects these functions to be available
        assert hasattr(trackio, 'init'), "trackio.init not found"
        assert hasattr(trackio, 'log'), "trackio.log not found"
        assert hasattr(trackio, 'finish'), "trackio.finish not found"
        
        # Test function signatures
        import inspect
        
        # Check init signature
        init_sig = inspect.signature(trackio.init)
        print(f"✅ init signature: {init_sig}")
        
        # Test that init can be called without arguments (TRL compatibility)
        try:
            # This simulates what TRL might do
            trackio.init()
            print("✅ init() can be called without arguments")
        except Exception as e:
            print(f"❌ init() failed when called without arguments: {e}")
            return False
        
        # Test that config attribute is available (TRL compatibility)
        try:
            config = trackio.config
            print(f"✅ trackio.config is available: {type(config)}")
            print(f"✅ config.project_name: {config.project_name}")
            print(f"✅ config.experiment_name: {config.experiment_name}")
        except Exception as e:
            print(f"❌ trackio.config failed: {e}")
            return False

        # Check log signature
        log_sig = inspect.signature(trackio.log)
        print(f"✅ log signature: {log_sig}")
        
        # Check finish signature
        finish_sig = inspect.signature(trackio.finish)
        print(f"✅ finish signature: {finish_sig}")
        
        print("✅ TRL compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ TRL compatibility test failed: {e}")
        return False

def test_monitoring_integration():
    """Test that our trackio module integrates with our monitoring system"""
    print("\n🔍 Testing Monitoring Integration")
    
    try:
        import trackio
        
        # Test that we can get the monitor
        monitor = trackio.get_monitor()
        if monitor is not None:
            print("✅ Monitor integration working")
        else:
            print("⚠️ Monitor not available (this is normal if not initialized)")
        
        # Test availability check
        is_avail = trackio.is_available()
        print(f"✅ Trackio availability check: {is_avail}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Trackio TRL Fix")
    print("=" * 50)
    
    tests = [
        test_trackio_interface,
        test_trl_compatibility,
        test_monitoring_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Trackio TRL fix is working correctly.")
        print("\nThe trackio module now provides the interface expected by TRL library:")
        print("- init(): Initialize experiment")
        print("- log(): Log metrics")
        print("- finish(): Finish experiment")
        print("\nThis should resolve the 'module trackio has no attribute init' error.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 