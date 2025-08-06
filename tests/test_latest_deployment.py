#!/usr/bin/env python3
"""
Test script to verify the latest Trackio Space deployment using HF Hub API
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_hf_hub_import():
    """Test that huggingface_hub can be imported"""
    print("🔍 Testing huggingface_hub import...")
    
    try:
        from huggingface_hub import HfApi, create_repo
        print("✅ huggingface_hub imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import huggingface_hub: {e}")
        print("💡 Install with: pip install huggingface_hub>=0.19.0")
        return False

def test_deployment_script_import():
    """Test that the deployment script can be imported"""
    print("\n🔍 Testing deployment script import...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        # Test class instantiation
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        print("✅ TrackioSpaceDeployer class imported successfully")
        
        # Test API availability
        if hasattr(deployer, 'api'):
            print("✅ HF API initialized")
        else:
            print("⚠️  HF API not available (fallback to CLI)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing deployment script: {e}")
        return False

def test_api_methods():
    """Test that the deployment script has the new API methods"""
    print("\n🔍 Testing API methods...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        
        # Test required methods exist
        required_methods = [
            "create_space",
            "_create_space_cli",
            "prepare_space_files", 
            "upload_files_to_space",
            "test_space",
            "deploy"
        ]
        
        for method in required_methods:
            if hasattr(deployer, method):
                print(f"✅ Method exists: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API methods: {e}")
        return False

def test_create_repo_api():
    """Test the create_repo API function"""
    print("\n🔍 Testing create_repo API...")
    
    try:
        from huggingface_hub import create_repo
        
        # Test that the function exists and has the right parameters
        import inspect
        sig = inspect.signature(create_repo)
        
        # Check for required parameters
        required_params = ['repo_id', 'token']
        optional_params = ['repo_type', 'space_sdk', 'space_hardware']
        
        param_names = list(sig.parameters.keys())
        
        for param in required_params:
            if param in param_names:
                print(f"✅ Required parameter: {param}")
            else:
                print(f"❌ Missing required parameter: {param}")
                return False
        
        for param in optional_params:
            if param in param_names:
                print(f"✅ Optional parameter: {param}")
            else:
                print(f"⚠️  Optional parameter not found: {param}")
        
        print("✅ create_repo API signature looks correct")
        return True
        
    except Exception as e:
        print(f"❌ Error testing create_repo API: {e}")
        return False

def test_space_creation_logic():
    """Test the space creation logic"""
    print("\n🔍 Testing space creation logic...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        # Test with mock data
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        
        # Test that the space URL is correctly formatted
        expected_url = "https://huggingface.co/spaces/test-user/test-space"
        if deployer.space_url == expected_url:
            print("✅ Space URL formatted correctly")
        else:
            print(f"❌ Space URL incorrect: {deployer.space_url}")
            return False
        
        # Test that the repo_id is correctly formatted
        repo_id = f"{deployer.username}/{deployer.space_name}"
        expected_repo_id = "test-user/test-space"
        if repo_id == expected_repo_id:
            print("✅ Repo ID formatted correctly")
        else:
            print(f"❌ Repo ID incorrect: {repo_id}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing space creation logic: {e}")
        return False

def test_template_files():
    """Test that all required template files exist"""
    print("\n🔍 Testing template files...")
    
    spaces_dir = project_root / "templates" / "spaces"
    demo_types = ["demo_smol", "demo_gpt", "trackio"]
    required_files = ["app.py", "requirements.txt", "README.md"]
    
    for demo_type in demo_types:
        demo_dir = spaces_dir / demo_type
        print(f"Checking {demo_type} templates...")
        for file_name in required_files:
            file_path = demo_dir / file_name
            if file_path.exists():
                print(f"✅ {demo_type}/{file_name} exists")
            else:
                print(f"❌ {demo_type}/{file_name} missing")
                return False
    
    return True

def test_temp_directory_handling():
    """Test temporary directory handling"""
    print("\n🔍 Testing temporary directory handling...")
    
    try:
        import tempfile
        
        # Test temp directory creation
        temp_dir = tempfile.mkdtemp()
        print(f"✅ Created temp directory: {temp_dir}")
        
        # Test file copying
        templates_dir = project_root / "templates" / "spaces"
        test_file = templates_dir / "app.py"
        
        if test_file.exists():
            dest_file = Path(temp_dir) / "app.py"
            shutil.copy2(test_file, dest_file)
            print("✅ File copying works")
        else:
            print("❌ Source file not found")
            return False
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing temp directory handling: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("🚀 Testing Latest Trackio Space Deployment")
    print("=" * 55)
    
    tests = [
        test_hf_hub_import,
        test_deployment_script_import,
        test_api_methods,
        test_create_repo_api,
        test_space_creation_logic,
        test_template_files,
        test_temp_directory_handling
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
        print("✅ All deployment tests passed! The latest deployment should work correctly.")
        print("\n🎯 Next steps:")
        print("1. Install latest huggingface_hub: pip install huggingface_hub>=0.19.0")
        print("2. Run the deployment script: python scripts/trackio_tonic/deploy_trackio_space.py")
        print("3. Provide your HF username, space name, and token")
        print("4. Wait for the Space to build (2-5 minutes)")
        print("5. Test the Space URL")
        return True
    else:
        print("❌ Some deployment tests failed. Please check the errors above.")
        print("\n💡 Troubleshooting:")
        print("1. Install huggingface_hub: pip install huggingface_hub>=0.19.0")
        print("2. Check that all template files exist")
        print("3. Verify the deployment script structure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 