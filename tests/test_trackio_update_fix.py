#!/usr/bin/env python3
"""
Test script to verify TrackioConfig update method fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_trackio_config_update():
    """Test that TrackioConfig update method works correctly"""
    print("🧪 Testing TrackioConfig update method...")
    
    try:
        # Import trackio module
        import trackio
        
        # Test that config attribute exists
        assert hasattr(trackio, 'config'), "trackio.config not found"
        print("✅ trackio.config exists")
        
        # Test that config has update method
        config = trackio.config
        assert hasattr(config, 'update'), "TrackioConfig.update method not found"
        print("✅ TrackioConfig.update method exists")
        
        # Test update method functionality with dictionary
        test_config = {
            'project_name': 'test_project',
            'experiment_name': 'test_experiment',
            'new_attribute': 'test_value'
        }
        
        # Call update method with dictionary
        config.update(test_config)
        
        # Verify updates
        assert config.project_name == 'test_project', f"Expected 'test_project', got '{config.project_name}'"
        assert config.experiment_name == 'test_experiment', f"Expected 'test_experiment', got '{config.experiment_name}'"
        assert config.new_attribute == 'test_value', f"Expected 'test_value', got '{config.new_attribute}'"
        
        print("✅ TrackioConfig.update method works correctly with dictionary")
        
        # Test update method with keyword arguments (TRL style)
        config.update(allow_val_change=True, trl_setting='test_value')
        
        # Verify keyword argument updates
        assert config.allow_val_change == True, f"Expected True, got '{config.allow_val_change}'"
        assert config.trl_setting == 'test_value', f"Expected 'test_value', got '{config.trl_setting}'"
        
        print("✅ TrackioConfig.update method works correctly with keyword arguments")
        print("✅ All attributes updated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_trackio_trl_compatibility():
    """Test that trackio is fully compatible with TRL expectations"""
    print("\n🔍 Testing TRL Compatibility...")
    
    try:
        import trackio
        
        # Test all required functions exist
        required_functions = ['init', 'log', 'finish']
        for func_name in required_functions:
            assert hasattr(trackio, func_name), f"trackio.{func_name} not found"
            print(f"✅ trackio.{func_name} exists")
        
        # Test config attribute exists and has update method
        assert hasattr(trackio, 'config'), "trackio.config not found"
        assert hasattr(trackio.config, 'update'), "trackio.config.update not found"
        print("✅ trackio.config.update exists")
        
        # Test that init can be called without arguments (TRL compatibility)
        try:
            experiment_id = trackio.init()
            print(f"✅ trackio.init() called successfully: {experiment_id}")
        except Exception as e:
            print(f"❌ trackio.init() failed: {e}")
            return False
        
        # Test that log can be called
        try:
            trackio.log({'test_metric': 1.0})
            print("✅ trackio.log() called successfully")
        except Exception as e:
            print(f"❌ trackio.log() failed: {e}")
            return False
        
        # Test that finish can be called
        try:
            trackio.finish()
            print("✅ trackio.finish() called successfully")
        except Exception as e:
            print(f"❌ trackio.finish() failed: {e}")
            return False
        
        print("✅ All TRL compatibility tests passed")
        return True
        
    except Exception as e:
        print(f"❌ TRL compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TrackioConfig Update Fix Test")
    print("=" * 40)
    
    # Test 1: Update method functionality
    test1_passed = test_trackio_config_update()
    
    # Test 2: TRL compatibility
    test2_passed = test_trackio_trl_compatibility()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    print(f"✅ Update Method Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ TRL Compatibility Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! TrackioConfig update fix is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 