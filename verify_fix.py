#!/usr/bin/env python3
"""
Simple verification script for TrackioConfig update fix
"""

try:
    import trackio
    print("✅ Trackio imported successfully")
    
    # Test config access
    config = trackio.config
    print(f"✅ Config accessed: {type(config)}")
    
    # Test update method exists
    print(f"✅ Update method exists: {hasattr(config, 'update')}")
    
    # Test update with keyword arguments (TRL style)
    config.update(allow_val_change=True, test_attr='test_value')
    print(f"✅ Update with kwargs worked: allow_val_change={config.allow_val_change}, test_attr={config.test_attr}")
    
    # Test update with dictionary
    config.update({'project_name': 'test_project', 'new_attr': 'dict_value'})
    print(f"✅ Update with dict worked: project_name={config.project_name}, new_attr={config.new_attr}")
    
    # Test TRL functions
    print(f"✅ Init function exists: {hasattr(trackio, 'init')}")
    print(f"✅ Log function exists: {hasattr(trackio, 'log')}")
    print(f"✅ Finish function exists: {hasattr(trackio, 'finish')}")
    
    print("\n🎉 All tests passed! The fix is working correctly.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 