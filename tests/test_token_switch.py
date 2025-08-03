#!/usr/bin/env python3
"""
Test script for token switching functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def test_token_validation():
    """Test token validation script"""
    print("🧪 Testing token validation...")
    
    # Test with invalid token
    result = subprocess.run([
        "python3", "scripts/validate_hf_token.py", "invalid_token"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✅ Invalid token correctly rejected")
    else:
        print("❌ Invalid token should have been rejected")
        return False
    
    # Test with environment variable
    os.environ['HF_TOKEN'] = 'test_token'
    result = subprocess.run([
        "python3", "scripts/validate_hf_token.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✅ Environment token validation works")
    else:
        print("❌ Environment token validation failed")
        return False
    
    return True

def test_token_switch_script():
    """Test token switch script"""
    print("🧪 Testing token switch script...")
    
    # Test with invalid arguments
    result = subprocess.run([
        "python3", "scripts/trackio_tonic/switch_to_read_token.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✅ Script correctly handles missing arguments")
    else:
        print("❌ Script should have failed with missing arguments")
        return False
    
    # Test with invalid space_id format
    result = subprocess.run([
        "python3", "scripts/trackio_tonic/switch_to_read_token.py", 
        "invalid_space", "token1", "token2"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("✅ Script correctly validates space_id format")
    else:
        print("❌ Script should have failed with invalid space_id")
        return False
    
    return True

def test_secure_input_function():
    """Test the secure input function in launch.sh"""
    print("🧪 Testing secure input function...")
    
    # This would require interactive testing, so we'll just check if the function exists
    launch_script = Path("launch.sh")
    if launch_script.exists():
        try:
            with open(launch_script, 'r', encoding='utf-8') as f:
                content = f.read()
                if "get_secure_token_input" in content:
                    print("✅ Secure input function found in launch.sh")
                    return True
                else:
                    print("❌ Secure input function not found in launch.sh")
                    return False
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(launch_script, 'r', encoding='latin-1') as f:
                    content = f.read()
                    if "get_secure_token_input" in content:
                        print("✅ Secure input function found in launch.sh")
                        return True
                    else:
                        print("❌ Secure input function not found in launch.sh")
                        return False
            except Exception as e:
                print(f"❌ Error reading launch.sh: {e}")
                return False
    else:
        print("❌ launch.sh not found")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Token Security Features")
    print("=" * 40)
    
    tests = [
        test_token_validation,
        test_token_switch_script,
        test_secure_input_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error: {test.__name__} - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 