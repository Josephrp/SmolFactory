#!/usr/bin/env python3
"""
Test script for Hugging Face token validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from validate_hf_token import validate_hf_token

def test_token_validation():
    """Test the token validation function."""
    
    # Test with a valid token (you can replace this with your own token for testing)
    test_token = "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"
    
    print("Testing token validation...")
    print(f"Token: {test_token[:10]}...")
    
    success, username, error = validate_hf_token(test_token)
    
    if success:
        print(f"✅ Token validation successful!")
        print(f"Username: {username}")
    else:
        print(f"❌ Token validation failed: {error}")
    
    return success

def test_invalid_token():
    """Test with an invalid token."""
    
    invalid_token = "hf_invalid_token_for_testing"
    
    print("\nTesting invalid token...")
    success, username, error = validate_hf_token(invalid_token)
    
    if not success:
        print(f"✅ Correctly rejected invalid token: {error}")
    else:
        print(f"❌ Unexpectedly accepted invalid token")
    
    return not success

if __name__ == "__main__":
    print("🧪 Testing Hugging Face Token Validation")
    print("=" * 50)
    
    # Test valid token
    valid_result = test_token_validation()
    
    # Test invalid token
    invalid_result = test_invalid_token()
    
    print("\n" + "=" * 50)
    if valid_result and invalid_result:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 