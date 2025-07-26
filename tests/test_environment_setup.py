#!/usr/bin/env python3
"""
Test script to verify environment variables are properly set in virtual environment
"""

import os
import sys
import subprocess
from pathlib import Path

def test_environment_variables():
    """Test that environment variables are properly set"""
    print("🔍 Testing Environment Variables")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxxxx"
    
    # Set environment variables
    os.environ['HF_TOKEN'] = test_token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = test_token
    os.environ['HF_USERNAME'] = 'Tonic'
    os.environ['TRACKIO_DATASET_REPO'] = 'Tonic/trackio-experiments'
    
    print(f"Testing environment setup with token: {'*' * 10}...{test_token[-4:]}")
    
    # Check if environment variables are set
    required_vars = ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_USERNAME', 'TRACKIO_DATASET_REPO']
    
    all_set = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"[OK] {var}: {value[:10] if len(value) > 10 else value}...{value[-4:] if len(value) > 4 else ''}")
        else:
            print(f"[ERROR] {var}: Not set")
            all_set = False
    
    return all_set

def test_virtual_environment():
    """Test that virtual environment can access environment variables"""
    print("\n🔍 Testing Virtual Environment Access")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxxx"
    
    # Create a simple Python script to test environment variables
    test_script = """
import os
import sys

# Check environment variables
required_vars = ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_USERNAME', 'TRACKIO_DATASET_REPO']

print("Environment variables in virtual environment:")
all_set = True
for var in required_vars:
    value = os.environ.get(var)
    if value:
        print(f"[OK] {var}: {value[:10] if len(value) > 10 else value}...{value[-4:] if len(value) > 4 else ''}")
    else:
        print(f"[ERROR] {var}: Not set")
        all_set = False

if all_set:
    print("\\n[OK] All environment variables are properly set in virtual environment")
    sys.exit(0)
else:
    print("\\n[ERROR] Some environment variables are missing in virtual environment")
    sys.exit(1)
"""
    
    # Write test script to temporary file
    test_file = Path("tests/temp_env_test.py")
    test_file.write_text(test_script)
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['HF_TOKEN'] = test_token
        env['HUGGING_FACE_HUB_TOKEN'] = test_token
        env['HF_USERNAME'] = 'Tonic'
        env['TRACKIO_DATASET_REPO'] = 'Tonic/trackio-experiments'
        
        # Run the test script
        result = subprocess.run([sys.executable, str(test_file)], 
                              env=env, 
                              capture_output=True, 
                              text=True)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

def test_launch_script_environment():
    """Test that launch script properly sets environment variables"""
    print("\n🔍 Testing Launch Script Environment Setup")
    print("=" * 50)
    
    # Check if launch.sh exists
    launch_script = Path("launch.sh")
    if not launch_script.exists():
        print("❌ launch.sh not found")
        return False
    
    # Read launch script and check for environment variable exports
    script_content = launch_script.read_text()
    
    required_exports = [
        'export HF_TOKEN=',
        'export HUGGING_FACE_HUB_TOKEN=',
        'export HF_USERNAME=',
        'export TRACKIO_DATASET_REPO='
    ]
    
    all_found = True
    for export in required_exports:
        if export in script_content:
            print(f"[OK] Found: {export}")
        else:
            print(f"[ERROR] Missing: {export}")
            all_found = False
    
    # Check for virtual environment activation
    if 'source smollm3_env/bin/activate' in script_content:
        print("[OK] Found virtual environment activation")
    else:
        print("[ERROR] Missing virtual environment activation")
        all_found = False
    
    # Check for environment variable re-export after activation
    if 'export HF_TOKEN="$HF_TOKEN"' in script_content:
        print("[OK] Found environment variable re-export after activation")
    else:
        print("[ERROR] Missing environment variable re-export after activation")
        all_found = False
    
    return all_found

def main():
    """Run all environment tests"""
    print("🚀 Environment Setup Verification")
    print("=" * 50)
    
    tests = [
        test_environment_variables,
        test_virtual_environment,
        test_launch_script_environment
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
        print("[SUCCESS] ALL ENVIRONMENT TESTS PASSED!")
        print("[OK] Environment variables: Properly set")
        print("[OK] Virtual environment: Can access variables")
        print("[OK] Launch script: Properly configured")
        print("\nThe environment setup is working correctly!")
    else:
        print("[ERROR] SOME ENVIRONMENT TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 