#!/usr/bin/env python3
"""
Test script to verify token validation works with the provided token
"""

import os
import sys
import json
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

def test_token_validation():
    """Test token validation with the provided token"""
    print("🔍 Testing Token Validation")
    print("=" * 50)
    
    # Test token from user
    test_token = ""
    
    print(f"Testing token: {'*' * 10}...{test_token[-4:]}")
    
    # Import the validation function
    try:
        from validate_hf_token import validate_hf_token
        print("✅ Token validation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import token validation module: {e}")
        return False
    
    # Test token validation
    try:
        success, username, error = validate_hf_token(test_token)
        
        if success:
            print(f"✅ Token validation successful!")
            print(f"✅ Username: {username}")
            return True
        else:
            print(f"❌ Token validation failed: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Token validation error: {e}")
        return False

def test_dataset_setup():
    """Test dataset setup with the provided token"""
    print("\n🔍 Testing Dataset Setup")
    print("=" * 50)
    
    # Test token from user
    test_token = "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"
    
    print(f"Testing dataset setup with token: {'*' * 10}...{test_token[-4:]}")
    
    # Set environment variable
    os.environ['HUGGING_FACE_HUB_TOKEN'] = test_token
    os.environ['HF_TOKEN'] = test_token
    
    # Import the dataset setup function
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "dataset_tonic"))
        from setup_hf_dataset import get_username_from_token
        print("✅ Dataset setup module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataset setup module: {e}")
        return False
    
    # Test username extraction
    try:
        username = get_username_from_token(test_token)
        
        if username:
            print(f"✅ Username extraction successful: {username}")
            return True
        else:
            print(f"❌ Username extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Username extraction error: {e}")
        return False

def test_space_deployment():
    """Test space deployment with the provided token"""
    print("\n🔍 Testing Space Deployment")
    print("=" * 50)
    
    # Test token from user
    test_token = ""
    
    print(f"Testing space deployment with token: {'*' * 10}...{test_token[-4:]}")
    
    # Import the space deployment class
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        print("✅ Space deployment module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import space deployment module: {e}")
        return False
    
    # Test deployer initialization
    try:
        deployer = TrackioSpaceDeployer("test-space", test_token)
        
        if deployer.username:
            print(f"✅ Space deployer initialization successful")
            print(f"✅ Username: {deployer.username}")
            return True
        else:
            print(f"❌ Space deployer initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Space deployer initialization error: {e}")
        return False

def main():
    """Run all token tests"""
    print("🚀 Token Validation and Deployment Tests")
    print("=" * 50)
    
    tests = [
        test_token_validation,
        test_dataset_setup,
        test_space_deployment
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
        print("🎉 ALL TOKEN TESTS PASSED!")
        print("✅ Token validation: Working")
        print("✅ Dataset setup: Working")
        print("✅ Space deployment: Working")
        print("\nThe token is working correctly with all components!")
    else:
        print("❌ SOME TOKEN TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 