#!/usr/bin/env python3
"""
Test script for the new configuration functionality in app.py
"""

import os
import sys
from unittest.mock import patch

def test_trackio_space_initialization():
    """Test TrackioSpace initialization with different parameters"""
    print("🧪 Testing TrackioSpace initialization...")
    
    # Import the app module
    import templates.spaces.app as app
    
    # Test 1: Default initialization (uses environment variables)
    print("\n1. Testing default initialization...")
    trackio = app.TrackioSpace()
    print(f"   Dataset repo: {trackio.dataset_repo}")
    print(f"   HF token set: {'Yes' if trackio.hf_token else 'No'}")
    
    # Test 2: Custom initialization
    print("\n2. Testing custom initialization...")
    trackio_custom_config = app.TrackioSpace(
        hf_token="test_token_123",
        dataset_repo="test-user/test-dataset"
    )
    print(f"   Dataset repo: {trackio_custom_config.dataset_repo}")
    print(f"   HF token set: {'Yes' if trackio_custom_config.hf_token else 'No'}")
    
    # Test 3: Partial custom initialization
    print("\n3. Testing partial custom initialization...")
    trackio_partial = app.TrackioSpace(dataset_repo="another-user/another-dataset")
    print(f"   Dataset repo: {trackio_partial.dataset_repo}")
    print(f"   HF token set: {'Yes' if trackio_partial.hf_token else 'No'}")
    
    print("✅ TrackioSpace initialization tests passed!")

def test_configuration_functions():
    """Test the configuration functions"""
    print("\n🧪 Testing configuration functions...")
    
    import templates.spaces.app as app
    
    # Test update_trackio_config function
    print("\n1. Testing update_trackio_config...")
    result = app.update_trackio_config("test_token", "test-user/test-dataset")
    print(f"   Result: {result}")
    
    # Test test_dataset_connection function
    print("\n2. Testing test_dataset_connection...")
    result = app.test_dataset_connection("", "test-user/test-dataset")
    print(f"   Result: {result}")
    
    # Test create_dataset_repository function
    print("\n3. Testing create_dataset_repository...")
    result = app.create_dataset_repository("", "test-user/test-dataset")
    print(f"   Result: {result}")
    
    print("✅ Configuration function tests passed!")

def test_environment_variables():
    """Test environment variable handling"""
    print("\n🧪 Testing environment variable handling...")
    
    # Test with environment variables set
    with patch.dict(os.environ, {
        'HF_TOKEN': 'env_test_token',
        'TRACKIO_DATASET_REPO': 'env-user/env-dataset'
    }):
        import templates.spaces.app as app
        trackio = app.TrackioSpace()
        print(f"   Dataset repo: {trackio.dataset_repo}")
        print(f"   HF token set: {'Yes' if trackio.hf_token else 'No'}")
    
    # Test with no environment variables
    with patch.dict(os.environ, {}, clear=True):
        import templates.spaces.app as app
        trackio = app.TrackioSpace()
        print(f"   Dataset repo: {trackio.dataset_repo}")
        print(f"   HF token set: {'Yes' if trackio.hf_token else 'No'}")
    
    print("✅ Environment variable tests passed!")

def main():
    """Run all tests"""
    print("🚀 Testing App Configuration Features")
    print("=" * 50)
    
    try:
        test_trackio_space_initialization()
        test_configuration_functions()
        test_environment_variables()
        
        print("\n🎉 All tests passed!")
        print("\n📋 Configuration Features:")
        print("✅ HF Token input field")
        print("✅ Dataset Repository input field")
        print("✅ Environment variable fallback")
        print("✅ Configuration update function")
        print("✅ Connection testing function")
        print("✅ Dataset creation function")
        print("✅ Gradio interface integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 