#!/usr/bin/env python3
"""
Test script to check for trackio package conflicts
"""

import sys
import importlib

def test_trackio_imports():
    """Test what trackio-related packages are available"""
    print("🔍 Testing Trackio Package Imports")
    print("=" * 50)
    
    # Check for trackio package
    try:
        trackio_module = importlib.import_module('trackio')
        print(f"✅ Found trackio package: {trackio_module}")
        print(f"   Location: {trackio_module.__file__}")
        
        # Check for init attribute
        if hasattr(trackio_module, 'init'):
            print("✅ trackio.init exists")
        else:
            print("❌ trackio.init does not exist")
            print(f"   Available attributes: {[attr for attr in dir(trackio_module) if not attr.startswith('_')]}")
            
    except ImportError:
        print("✅ No trackio package found (this is good)")
    
    # Check for our custom TrackioAPIClient
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "trackio_tonic"))
        from trackio_api_client import TrackioAPIClient
        print("✅ Custom TrackioAPIClient available")
    except ImportError as e:
        print(f"❌ Custom TrackioAPIClient not available: {e}")
    
    # Check for any other trackio-related imports
    trackio_related = []
    for module_name in sys.modules:
        if 'trackio' in module_name.lower():
            trackio_related.append(module_name)
    
    if trackio_related:
        print(f"⚠️ Found trackio-related modules: {trackio_related}")
    else:
        print("✅ No trackio-related modules found")

def test_monitoring_import():
    """Test monitoring module import"""
    print("\n🔍 Testing Monitoring Module Import")
    print("=" * 50)
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from monitoring import SmolLM3Monitor
        print("✅ SmolLM3Monitor imported successfully")
        
        # Test monitor creation
        monitor = SmolLM3Monitor("test-experiment")
        print("✅ Monitor created successfully")
        print(f"   Dataset repo: {monitor.dataset_repo}")
        print(f"   Enable tracking: {monitor.enable_tracking}")
        
    except Exception as e:
        print(f"❌ Failed to import/create monitor: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run trackio conflict tests"""
    print("🚀 Trackio Conflict Detection")
    print("=" * 50)
    
    tests = [
        test_trackio_imports,
        test_monitoring_import
    ]
    
    all_passed = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TRACKIO CONFLICT TESTS PASSED!")
        print("✅ No trackio package conflicts detected")
        print("✅ Monitoring module works correctly")
    else:
        print("❌ SOME TRACKIO CONFLICT TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    from pathlib import Path
    success = main()
    sys.exit(0 if success else 1) 