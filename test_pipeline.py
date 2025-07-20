#!/usr/bin/env python3
"""
Test script for the SmolLM3 end-to-end pipeline
Verifies all components are working correctly
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'trl',
        'huggingface_hub',
        'requests'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        return False
    
    print("✅ All imports successful")
    return True

def test_local_modules():
    """Test local module imports"""
    print("\n🔍 Testing local modules...")
    
    # Add src to path
    sys.path.append('src')
    
    local_modules = [
        'config',
        'model',
        'data',
        'trainer',
        'monitoring'
    ]
    
    failed_imports = []
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed local imports: {failed_imports}")
        return False
    
    print("✅ All local modules imported successfully")
    return True

def test_scripts():
    """Test script availability"""
    print("\n🔍 Testing scripts...")
    
    required_scripts = [
        'scripts/trackio_tonic/deploy_trackio_space.py',
        'scripts/trackio_tonic/configure_trackio.py',
        'scripts/dataset_tonic/setup_hf_dataset.py',
        'scripts/model_tonic/push_to_huggingface.py',
        'src/train.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script}")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n❌ Missing scripts: {missing_scripts}")
        return False
    
    print("✅ All scripts found")
    return True

def test_configs():
    """Test configuration files"""
    print("\n🔍 Testing configurations...")
    
    config_dir = Path('config')
    if not config_dir.exists():
        print("❌ config directory not found")
        return False
    
    config_files = list(config_dir.glob('*.py'))
    if not config_files:
        print("❌ No configuration files found")
        return False
    
    print(f"✅ Found {len(config_files)} configuration files:")
    for config in config_files:
        print(f"  - {config.name}")
    
    return True

def test_requirements():
    """Test requirements files"""
    print("\n🔍 Testing requirements...")
    
    requirements_dir = Path('requirements')
    if not requirements_dir.exists():
        print("❌ requirements directory not found")
        return False
    
    req_files = list(requirements_dir.glob('*.txt'))
    if not req_files:
        print("❌ No requirements files found")
        return False
    
    print(f"✅ Found {len(req_files)} requirements files:")
    for req in req_files:
        print(f"  - {req.name}")
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"  - Device 0: {device_name}")
        else:
            print("⚠️  CUDA not available (training will be slower)")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False
    
    return True

def test_hf_token():
    """Test Hugging Face token"""
    print("\n🔍 Testing HF token...")
    
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("⚠️  HF_TOKEN not set (will be prompted during setup)")
        return True
    
    try:
        result = subprocess.run(
            ['huggingface-cli', 'whoami'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ HF token valid: {username}")
            return True
        else:
            print(f"❌ HF token invalid: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ HF token test failed: {e}")
        return False

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🔍 Testing pipeline components...")
    
    # Test setup script
    if Path('setup_launch.py').exists():
        print("✅ setup_launch.py found")
    else:
        print("❌ setup_launch.py not found")
        return False
    
    # Test launch script
    if Path('launch.sh').exists():
        print("✅ launch.sh found")
    else:
        print("❌ launch.sh not found")
        return False
    
    # Test README
    if Path('README_END_TO_END.md').exists():
        print("✅ README_END_TO_END.md found")
    else:
        print("❌ README_END_TO_END.md not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 SmolLM3 End-to-End Pipeline Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_local_modules,
        test_scripts,
        test_configs,
        test_requirements,
        test_cuda,
        test_hf_token,
        test_pipeline_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: python setup_launch.py")
        print("2. Run: chmod +x launch.sh")
        print("3. Run: ./launch.sh")
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        print("\n🔧 Common fixes:")
        print("1. Install missing packages: pip install -r requirements/requirements_core.txt")
        print("2. Set HF_TOKEN environment variable")
        print("3. Check CUDA installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 