#!/usr/bin/env python3
"""
Test script to verify Trackio Space deployment
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_templates_structure():
    """Test that the templates structure is correct"""
    print("🔍 Testing templates structure...")
    
    templates_dir = project_root / "templates" / "spaces"
    
    required_files = ["app.py", "requirements.txt", "README.md"]
    
    for file_name in required_files:
        file_path = templates_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def test_app_py_content():
    """Test that app.py has the required structure"""
    print("\n🔍 Testing app.py content...")
    
    app_path = project_root / "templates" / "spaces" / "app.py"
    
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            "import gradio as gr",
            "class TrackioSpace",
            "def create_experiment_interface",
            "def log_metrics_interface",
            "def log_parameters_interface",
            "demo.launch()"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✅ Found: {component}")
            else:
                print(f"❌ Missing: {component}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def test_requirements_content():
    """Test that requirements.txt has the required dependencies"""
    print("\n🔍 Testing requirements.txt content...")
    
    req_path = project_root / "templates" / "spaces" / "requirements.txt"
    
    try:
        with open(req_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required dependencies
        required_deps = [
            "gradio>=",
            "pandas>=",
            "numpy>=",
            "plotly>=",
            "requests>=",
            "datasets>=",
            "huggingface-hub>="
        ]
        
        for dep in required_deps:
            if dep in content:
                print(f"✅ Found: {dep}")
            else:
                print(f"❌ Missing: {dep}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def test_readme_structure():
    """Test that README.md has the correct structure"""
    print("\n🔍 Testing README.md structure...")
    
    readme_path = project_root / "templates" / "spaces" / "README.md"
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "---",
            "title: Trackio Experiment Tracking",
            "sdk: gradio",
            "app_file: app.py",
            "# Trackio Experiment Tracking",
            "## Features",
            "## Usage",
            "Visit: {SPACE_URL}"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"✅ Found: {section}")
            else:
                print(f"❌ Missing: {section}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        return False

def test_deployment_script():
    """Test that the deployment script can be imported and has required methods"""
    print("\n🔍 Testing deployment script...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        
        # Test class instantiation
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        print("✅ TrackioSpaceDeployer class imported successfully")
        
        # Test required methods exist
        required_methods = [
            "create_space",
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
        print(f"❌ Error testing deployment script: {e}")
        return False

def test_temp_directory_creation():
    """Test that the deployment script can create temporary directories"""
    print("\n🔍 Testing temporary directory creation...")
    
    try:
        import tempfile
        import shutil
        
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
        print(f"❌ Error testing temp directory creation: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("🚀 Testing Trackio Space Deployment")
    print("=" * 50)
    
    tests = [
        test_templates_structure,
        test_app_py_content,
        test_requirements_content,
        test_readme_structure,
        test_deployment_script,
        test_temp_directory_creation
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
        print("✅ All deployment tests passed! The Trackio Space should deploy correctly.")
        print("\n🎯 Next steps:")
        print("1. Run the deployment script: python scripts/trackio_tonic/deploy_trackio_space.py")
        print("2. Provide your HF username, space name, and token")
        print("3. Wait for the Space to build (2-5 minutes)")
        print("4. Test the Space URL")
        return True
    else:
        print("❌ Some deployment tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 