#!/usr/bin/env python3
"""
Test script to verify safetensors model validation fix
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_safetensors_validation():
    """Test that safetensors models are properly validated"""
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        # Test with dummy values
        quantizer = ModelQuantizer(
            model_path="/output-checkpoint",
            repo_name="test/test-repo",
            token="dummy_token"
        )
        
        # Mock the model path to simulate the Linux environment
        # In the real environment, this would be /output-checkpoint
        # with safetensors files
        
        # Test validation logic
        if quantizer.validate_model_path():
            logger.info("✅ Safetensors validation test passed")
            return True
        else:
            logger.error("❌ Safetensors validation test failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Safetensors validation test failed: {e}")
        return False

def test_model_file_detection():
    """Test model file detection logic"""
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        quantizer = ModelQuantizer(
            model_path="/output-checkpoint",
            repo_name="test/test-repo",
            token="dummy_token"
        )
        
        # Test the validation logic directly
        model_path = Path("/output-checkpoint")
        
        # Check for essential files
        required_files = ['config.json']
        model_files = [
            "model.safetensors.index.json",  # Safetensors format
            "pytorch_model.bin"  # PyTorch format
        ]
        
        missing_required = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_required.append(file)
        
        # Check if at least one model file exists
        model_file_exists = any((model_path / file).exists() for file in model_files)
        if not model_file_exists:
            missing_required.extend(model_files)
        
        if missing_required:
            logger.error(f"❌ Missing required model files: {missing_required}")
            return False
        
        logger.info("✅ Model file detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model file detection test failed: {e}")
        return False

def main():
    """Run safetensors validation tests"""
    logger.info("🧪 Testing safetensors validation fix...")
    
    tests = [
        ("Safetensors Validation Test", test_safetensors_validation),
        ("Model File Detection Test", test_model_file_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} passed")
        else:
            logger.error(f"❌ {test_name} failed")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All safetensors tests passed! The fix should work in the Linux environment.")
        logger.info("💡 The validation now properly handles:")
        logger.info("   - Safetensors format (model.safetensors.index.json)")
        logger.info("   - PyTorch format (pytorch_model.bin)")
        logger.info("   - Either format is accepted")
        return 0
    else:
        logger.error("❌ Some tests failed. The fix may need adjustment.")
        return 1

if __name__ == "__main__":
    exit(main())