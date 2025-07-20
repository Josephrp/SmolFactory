#!/usr/bin/env python3
"""
Test script to verify Trackio deployment fixes
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test that required packages are available"""
    print("🔍 Testing imports...")
    
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
        print("✅ huggingface_hub imports successful")
    except ImportError as e:
        print(f"❌ huggingface_hub import failed: {e}")
        return False
    
    try:
        from datasets import Dataset
        print("✅ datasets import successful")
    except ImportError as e:
        print(f"❌ datasets import failed: {e}")
        return False
    
    return True

def test_script_exists(script_path):
    """Test that a script exists and is executable"""
    path = Path(script_path)
    if not path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    if not path.is_file():
        print(f"❌ Not a file: {script_path}")
        return False
    
    print(f"✅ Script exists: {script_path}")
    return True

def test_script_syntax(script_path):
    """Test that a script has valid Python syntax"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            compile(f.read(), script_path, 'exec')
        print(f"✅ Syntax valid: {script_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {script_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {script_path}: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    print("🔍 Testing environment variables...")
    
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("✅ HF_TOKEN is set")
    else:
        print("⚠️  HF_TOKEN is not set (this is normal for testing)")
    
    dataset_repo = os.environ.get('TRACKIO_DATASET_REPO', 'tonic/trackio-experiments')
    print(f"📊 TRACKIO_DATASET_REPO: {dataset_repo}")
    
    return True

def test_api_connection():
    """Test HF API connection if token is available"""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠️  Skipping API connection test - no HF_TOKEN")
        return True
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        
        # Test basic API call
        user_info = api.whoami()
        print(f"✅ API connection successful - User: {user_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_script_functions():
    """Test that scripts can be imported and have required functions"""
    print("🔍 Testing script functions...")
    
    # Test deploy script
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "trackio_tonic"))
        from deploy_trackio_space import TrackioSpaceDeployer
        print("✅ TrackioSpaceDeployer class imported successfully")
    except Exception as e:
        print(f"❌ Failed to import TrackioSpaceDeployer: {e}")
        return False
    
    # Test dataset script
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "dataset_tonic"))
        import setup_hf_dataset
        print("✅ setup_hf_dataset module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import setup_hf_dataset: {e}")
        return False
    
    # Test configure script
    try:
        sys.path.append(str(Path(__file__).parent.parent / "scripts" / "trackio_tonic"))
        import configure_trackio
        print("✅ configure_trackio module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import configure_trackio: {e}")
        return False
    
    return True

def test_template_files():
    """Test that template files exist"""
    print("🔍 Testing template files...")
    
    project_root = Path(__file__).parent.parent
    templates_dir = project_root / "templates"
    
    required_files = [
        "spaces/app.py",
        "spaces/requirements.txt", 
        "spaces/README.md",
        "datasets/readme.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = templates_dir / file_path
        if full_path.exists():
            print(f"✅ Template exists: {file_path}")
        else:
            print(f"❌ Template missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Testing Trackio Deployment Fixes")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("Script Existence", lambda: all([
            test_script_exists("scripts/trackio_tonic/deploy_trackio_space.py"),
            test_script_exists("scripts/dataset_tonic/setup_hf_dataset.py"),
            test_script_exists("scripts/trackio_tonic/configure_trackio.py"),
            test_script_exists("scripts/model_tonic/push_to_huggingface.py")
        ])),
        ("Script Syntax", lambda: all([
            test_script_syntax("scripts/trackio_tonic/deploy_trackio_space.py"),
            test_script_syntax("scripts/dataset_tonic/setup_hf_dataset.py"),
            test_script_syntax("scripts/trackio_tonic/configure_trackio.py"),
            test_script_syntax("scripts/model_tonic/push_to_huggingface.py")
        ])),
        ("Environment Variables", test_environment_variables),
        ("API Connection", test_api_connection),
        ("Script Functions", test_script_functions),
        ("Template Files", test_template_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 