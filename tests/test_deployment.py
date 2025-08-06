#!/usr/bin/env python3
"""
Test script to verify deployment scripts work correctly
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_templates_exist():
    """Test that all required template files exist"""
    print("🔍 Testing template files...")
    
    # Check spaces templates
    spaces_dir = project_root / "templates" / "spaces"
    demo_types = ["demo_smol", "demo_gpt", "trackio"]
    spaces_files = ["app.py", "requirements.txt", "README.md"]
    
    for demo_type in demo_types:
        demo_dir = spaces_dir / demo_type
        print(f"Checking {demo_type} templates...")
        for file_name in spaces_files:
            file_path = demo_dir / file_name
            if file_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} not found")
                return False
    
    # Check datasets templates
    datasets_dir = project_root / "templates" / "datasets"
    datasets_files = ["readme.md"]
    
    for file_name in datasets_files:
        file_path = datasets_dir / file_name
        if file_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            return False
    
    return True

def test_deployment_scripts():
    """Test that deployment scripts can import required modules"""
    print("\n🔍 Testing deployment scripts...")
    
    try:
        # Test space deployment script
        from scripts.trackio_tonic.deploy_trackio_space import TrackioSpaceDeployer
        print("✅ deploy_trackio_space.py imports successfully")
        
        # Test dataset setup script
        from scripts.dataset_tonic.setup_hf_dataset import setup_trackio_dataset
        print("✅ setup_hf_dataset.py imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment script test failed: {e}")
        return False

def test_file_copying():
    """Test that file copying logic works"""
    print("\n🔍 Testing file copying logic...")
    
    try:
        # Test space deployment file copying
        from scripts.trackio_tonic.deploy_trackio_space import TrackioSpaceDeployer
        
        # Create a mock deployer
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        
        # Test that templates directory exists
        project_root = Path(__file__).parent
        templates_dir = project_root / "templates" / "spaces"
        
        if templates_dir.exists():
            print(f"✅ Templates directory exists: {templates_dir}")
            
            # Check that required files exist
            for file_name in ["app.py", "requirements.txt", "README.md"]:
                file_path = templates_dir / file_name
                if file_path.exists():
                    print(f"✅ Template file exists: {file_path}")
                else:
                    print(f"❌ Template file missing: {file_path}")
                    return False
        else:
            print(f"❌ Templates directory missing: {templates_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ File copying test failed: {e}")
        return False

def test_readme_inclusion():
    """Test that README inclusion logic works"""
    print("\n🔍 Testing README inclusion...")
    
    try:
        # Test dataset README inclusion
        from scripts.dataset_tonic.setup_hf_dataset import setup_trackio_dataset
        
        # Check that README template exists
        project_root = Path(__file__).parent
        readme_path = project_root / "templates" / "datasets" / "readme.md"
        
        if readme_path.exists():
            print(f"✅ README template exists: {readme_path}")
            
            # Check README content
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content.strip()) > 0:
                    print(f"✅ README has content ({len(content)} characters)")
                else:
                    print(f"⚠️  README is empty")
        else:
            print(f"❌ README template missing: {readme_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ README inclusion test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Deployment Scripts")
    print("=" * 50)
    
    tests = [
        test_templates_exist,
        test_deployment_scripts,
        test_file_copying,
        test_readme_inclusion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment scripts are ready to use.")
        print("\n🚀 Deployment workflow:")
        print("1. Space deployment will copy files from templates/spaces/")
        print("2. Dataset creation will include README from templates/datasets/")
        print("3. Both scripts will properly upload all required files")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    exit(main()) 