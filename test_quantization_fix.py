#!/usr/bin/env python3
"""
Test script to verify quantization fixes
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

def test_quantization_imports():
    """Test that all required imports work"""
    try:
        # Test torchao imports
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
        from torchao.quantization import (
            Int8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig
        )
        from torchao.dtypes import Int4CPULayout
        logger.info("✅ torchao imports successful")
        
        # Test bitsandbytes imports
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            logger.info("✅ bitsandbytes imports successful")
        except ImportError:
            logger.warning("⚠️ bitsandbytes not available - alternative quantization disabled")
        
        # Test HF imports
        from huggingface_hub import HfApi
        logger.info("✅ huggingface_hub imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_model_quantizer():
    """Test ModelQuantizer initialization"""
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        # Test with dummy values
        quantizer = ModelQuantizer(
            model_path="/output-checkpoint",
            repo_name="test/test-repo",
            token="dummy_token"
        )
        
        logger.info("✅ ModelQuantizer initialization successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelQuantizer test failed: {e}")
        return False

def test_quantization_configs():
    """Test quantization config creation"""
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        quantizer = ModelQuantizer(
            model_path="/output-checkpoint",
            repo_name="test/test-repo",
            token="dummy_token"
        )
        
        # Test int8 config
        config = quantizer.create_quantization_config("int8_weight_only", 128)
        logger.info("✅ int8_weight_only config creation successful")
        
        # Test int4 config
        config = quantizer.create_quantization_config("int4_weight_only", 128)
        logger.info("✅ int4_weight_only config creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantization config test failed: {e}")
        return False

def test_device_selection():
    """Test optimal device selection"""
    try:
        from scripts.model_tonic.quantize_model import ModelQuantizer
        
        quantizer = ModelQuantizer(
            model_path="/output-checkpoint",
            repo_name="test/test-repo",
            token="dummy_token"
        )
        
        # Test device selection
        device = quantizer.get_optimal_device("int8_weight_only")
        logger.info(f"✅ int8 device selection: {device}")
        
        device = quantizer.get_optimal_device("int4_weight_only")
        logger.info(f"✅ int4 device selection: {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Device selection test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Testing quantization fixes...")
    
    tests = [
        ("Import Test", test_quantization_imports),
        ("ModelQuantizer Test", test_model_quantizer),
        ("Config Creation Test", test_quantization_configs),
        ("Device Selection Test", test_device_selection),
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
        logger.info("🎉 All tests passed! Quantization fixes are working.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())