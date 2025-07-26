#!/usr/bin/env python3
"""
Simple test to verify token works directly
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

def test_token_direct():
    """Test token validation directly"""
    print("🔍 Testing Token Directly")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxxx"
    
    print(f"Testing token directly: {'*' * 10}...{test_token[-4:]}")
    
    # Clear any existing environment variables
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']
    if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
        del os.environ['HUGGING_FACE_HUB_TOKEN']
    
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

def test_username_extraction_direct():
    """Test username extraction directly"""
    print("\n🔍 Testing Username Extraction Directly")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxx"
    
    print(f"Testing username extraction directly: {'*' * 10}...{test_token[-4:]}")
    
    # Clear any existing environment variables
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']
    if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
        del os.environ['HUGGING_FACE_HUB_TOKEN']
    
    # Import the username extraction function
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "dataset_tonic"))
        from setup_hf_dataset import get_username_from_token
        print("✅ Username extraction module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import username extraction module: {e}")
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

def main():
    """Run all direct token tests"""
    print("🚀 Direct Token Testing")
    print("=" * 50)
    
    tests = [
        test_token_direct,
        test_username_extraction_direct
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
        print("🎉 ALL DIRECT TOKEN TESTS PASSED!")
        print("✅ Token validation: Working")
        print("✅ Username extraction: Working")
        print("\nThe token works correctly when used directly!")
    else:
        print("❌ SOME DIRECT TOKEN TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 