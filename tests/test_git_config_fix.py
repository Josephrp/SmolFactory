#!/usr/bin/env python3
"""
Test script to verify the git configuration fix for Trackio Space deployment
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_git_config_in_temp_dir():
    """Test that git configuration works in temporary directory"""
    print("🔍 Testing git configuration in temporary directory...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"✅ Created temp directory: {temp_dir}")
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        print("✅ Initialized git repository")
        
        # Test git configuration
        test_email = "test@example.com"
        test_name = "Test User"
        
        # Set git config
        subprocess.run(["git", "config", "user.email", test_email], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", test_name], check=True, capture_output=True)
        
        # Verify git config
        result = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == test_email:
            print("✅ Git email configured correctly")
        else:
            print(f"❌ Git email not configured correctly: {result.stdout}")
            return False
        
        result = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == test_name:
            print("✅ Git name configured correctly")
        else:
            print(f"❌ Git name not configured correctly: {result.stdout}")
            return False
        
        # Test git commit
        # Create a test file
        with open("test.txt", "w") as f:
            f.write("Test file for git commit")
        
        subprocess.run(["git", "add", "test.txt"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Test commit"], check=True, capture_output=True)
        print("✅ Git commit successful")
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing git config: {e}")
        # Return to original directory
        os.chdir(original_dir)
        return False

def test_deployment_script_git_config():
    """Test that the deployment script handles git configuration correctly"""
    print("\n🔍 Testing deployment script git configuration...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        # Test with git configuration
        deployer = TrackioSpaceDeployer(
            "test-space", 
            "test-user", 
            "test-token",
            git_email="test@example.com",
            git_name="Test User"
        )
        
        # Check that git config is set
        if deployer.git_email == "test@example.com":
            print("✅ Git email set correctly")
        else:
            print(f"❌ Git email not set correctly: {deployer.git_email}")
            return False
        
        if deployer.git_name == "Test User":
            print("✅ Git name set correctly")
        else:
            print(f"❌ Git name not set correctly: {deployer.git_name}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing deployment script: {e}")
        return False

def test_git_config_fallback():
    """Test git configuration fallback behavior"""
    print("\n🔍 Testing git configuration fallback...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        # Test without git configuration (should use defaults)
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        
        # Check default values
        expected_email = "test-user@huggingface.co"
        expected_name = "test-user"
        
        if deployer.git_email == expected_email:
            print("✅ Default git email set correctly")
        else:
            print(f"❌ Default git email not set correctly: {deployer.git_email}")
            return False
        
        if deployer.git_name == expected_name:
            print("✅ Default git name set correctly")
        else:
            print(f"❌ Default git name not set correctly: {deployer.git_name}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing git config fallback: {e}")
        return False

def test_git_commit_with_config():
    """Test that git commit works with proper configuration"""
    print("\n🔍 Testing git commit with configuration...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"✅ Created temp directory: {temp_dir}")
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Initialize git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        
        # Set git configuration
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        
        # Create test file
        with open("test.txt", "w") as f:
            f.write("Test content")
        
        # Add and commit
        subprocess.run(["git", "add", "test.txt"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Test commit"], check=True, capture_output=True)
        print("✅ Git commit successful with configuration")
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing git commit: {e}")
        # Return to original directory
        os.chdir(original_dir)
        return False

def main():
    """Run all git configuration tests"""
    print("🚀 Testing Git Configuration Fix")
    print("=" * 40)
    
    tests = [
        test_git_config_in_temp_dir,
        test_deployment_script_git_config,
        test_git_config_fallback,
        test_git_commit_with_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All git configuration tests passed! The deployment should work correctly.")
        print("\n🎯 Next steps:")
        print("1. Run the deployment script: python scripts/trackio_tonic/deploy_trackio_space.py")
        print("2. Provide your HF username, space name, token, and git config")
        print("3. The git commit should now work correctly")
        return True
    else:
        print("❌ Some git configuration tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 