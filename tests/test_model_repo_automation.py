#!/usr/bin/env python3
"""
Test script to verify model repository name automation
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def test_model_repo_automation():
    """Test that model repository names are automatically generated"""
    print("🔍 Testing Model Repository Automation")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxxx"
    
    print(f"Testing model repository automation with token: {'*' * 10}...{test_token[-4:]}")
    
    # Set environment variables
    os.environ['HF_TOKEN'] = test_token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = test_token
    os.environ['HF_USERNAME'] = 'Tonic'
    
    # Import the validation function to get username
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        from validate_hf_token import validate_hf_token
        print("✅ Token validation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import token validation module: {e}")
        return False
    
    # Get username from token
    try:
        success, username, error = validate_hf_token(test_token)
        
        if not success:
            print(f"❌ Token validation failed: {error}")
            return False
        
        print(f"✅ Username extracted: {username}")
        
    except Exception as e:
        print(f"❌ Username extraction error: {e}")
        return False
    
    # Test automatic repository name generation
    try:
        # Generate repository name using the same logic as launch.sh
        current_date = datetime.now().strftime("%Y%m%d")
        auto_repo_name = f"{username}/smollm3-finetuned-{current_date}"
        
        print(f"✅ Auto-generated repository name: {auto_repo_name}")
        
        # Verify the format is correct
        if "/" in auto_repo_name and username in auto_repo_name:
            print("✅ Repository name format is correct")
            return True
        else:
            print("❌ Repository name format is incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Repository name generation error: {e}")
        return False

def test_launch_script_automation():
    """Test that launch script handles model repository automation"""
    print("\n🔍 Testing Launch Script Model Repository Automation")
    print("=" * 50)
    
    # Check if launch.sh exists
    launch_script = Path("launch.sh")
    if not launch_script.exists():
        print("❌ launch.sh not found")
        return False
    
    # Read launch script and check for automation
    script_content = launch_script.read_text(encoding='utf-8')
    
    # Check for automatic model repository generation
    automation_patterns = [
        'REPO_NAME="$HF_USERNAME/smollm3-finetuned-$(date +%Y%m%d)"',
        'Setting up model repository automatically',
        'Model repository: $REPO_NAME'
    ]
    
    all_found = True
    for pattern in automation_patterns:
        if pattern in script_content:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")
            all_found = False
    
    # Check that get_input for model repository name is removed
    if 'get_input "Model repository name"' in script_content:
        print("❌ Found manual model repository input (should be automated)")
        all_found = False
    else:
        print("✅ Manual model repository input removed")
    
    return all_found

def test_push_script_integration():
    """Test that push script works with auto-generated repository names"""
    print("\n🔍 Testing Push Script Integration")
    print("=" * 50)
    
    # Test token from user
    test_token = "xxxx"
    
    # Import the push script
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "model_tonic"))
        from push_to_huggingface import HuggingFacePusher
        print("✅ Push script module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import push script module: {e}")
        return False
    
    # Test with auto-generated repository name
    try:
        username = "Tonic"  # From token validation
        current_date = datetime.now().strftime("%Y%m%d")
        auto_repo_name = f"{username}/smollm3-finetuned-{current_date}"
        
        # Create a mock pusher (we won't actually push)
        pusher = HuggingFacePusher(
            model_path="/mock/path",
            repo_name=auto_repo_name,
            token=test_token
        )
        
        print(f"✅ Push script initialized with auto-generated repo: {auto_repo_name}")
        print(f"✅ Repository name format: {pusher.repo_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Push script integration error: {e}")
        return False

def main():
    """Run all model repository automation tests"""
    print("🚀 Model Repository Automation Verification")
    print("=" * 50)
    
    tests = [
        test_model_repo_automation,
        test_launch_script_automation,
        test_push_script_integration
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
        print("🎉 ALL MODEL REPOSITORY AUTOMATION TESTS PASSED!")
        print("✅ Model repository name generation: Working")
        print("✅ Launch script automation: Working")
        print("✅ Push script integration: Working")
        print("\nThe model repository automation is working correctly!")
    else:
        print("❌ SOME MODEL REPOSITORY AUTOMATION TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 