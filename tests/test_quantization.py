#!/usr/bin/env python3
"""
Test script for quantization functionality
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.model_tonic.quantize_model import ModelQuantizer

def test_quantization_imports():
    """Test that all required imports are available"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
        from torchao.quantization import (
            Int8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig
        )
        from torchao.dtypes import Int4CPULayout
        print("✅ All quantization imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_quantizer_initialization():
    """Test quantizer initialization"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy model directory
            model_dir = Path(temp_dir) / "dummy_model"
            model_dir.mkdir()
            
            # Create minimal model files
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "pytorch_model.bin").write_text('dummy')
            
            quantizer = ModelQuantizer(
                model_path=str(model_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            print("✅ Quantizer initialization successful")
            return True
    except Exception as e:
        print(f"❌ Quantizer initialization failed: {e}")
        return False

def test_quantization_config_creation():
    """Test quantization configuration creation"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "dummy_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "pytorch_model.bin").write_text('dummy')
            
            quantizer = ModelQuantizer(
                model_path=str(model_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            # Test int8 config
            config_int8 = quantizer.create_quantization_config("int8_weight_only", 128)
            print("✅ int8 config creation successful")
            
            # Test int4 config
            config_int4 = quantizer.create_quantization_config("int4_weight_only", 128)
            print("✅ int4 config creation successful")
            
            return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_model_validation():
    """Test model path validation"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with valid model
            model_dir = Path(temp_dir) / "valid_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "pytorch_model.bin").write_text('dummy')
            
            quantizer = ModelQuantizer(
                model_path=str(model_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            if quantizer.validate_model_path():
                print("✅ Valid model validation successful")
            else:
                print("❌ Valid model validation failed")
                return False
            
            # Test with invalid model
            invalid_dir = Path(temp_dir) / "invalid_model"
            invalid_dir.mkdir()
            # Missing required files
            
            quantizer_invalid = ModelQuantizer(
                model_path=str(invalid_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            if not quantizer_invalid.validate_model_path():
                print("✅ Invalid model validation successful")
            else:
                print("❌ Invalid model validation failed")
                return False
            
            return True
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False

def test_quantized_model_card_creation():
    """Test quantized model card creation"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "dummy_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "pytorch_model.bin").write_text('dummy')
            
            quantizer = ModelQuantizer(
                model_path=str(model_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            # Test int8 model card
            card_int8 = quantizer.create_quantized_model_card("int8_weight_only", "test/model")
            if "int8_weight_only" in card_int8 and "GPU" in card_int8:
                print("✅ int8 model card creation successful")
            else:
                print("❌ int8 model card creation failed")
                return False
            
            # Test int4 model card
            card_int4 = quantizer.create_quantized_model_card("int4_weight_only", "test/model")
            if "int4_weight_only" in card_int4 and "CPU" in card_int4:
                print("✅ int4 model card creation successful")
            else:
                print("❌ int4 model card creation failed")
                return False
            
            return True
    except Exception as e:
        print(f"❌ Model card creation test failed: {e}")
        return False

def test_quantized_readme_creation():
    """Test quantized README creation"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "dummy_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "pytorch_model.bin").write_text('dummy')
            
            quantizer = ModelQuantizer(
                model_path=str(model_dir),
                repo_name="test/test-quantized",
                token="dummy_token"
            )
            
            # Test int8 README
            readme_int8 = quantizer.create_quantized_readme("int8_weight_only", "test/model")
            if "int8_weight_only" in readme_int8 and "GPU optimized" in readme_int8:
                print("✅ int8 README creation successful")
            else:
                print("❌ int8 README creation failed")
                return False
            
            # Test int4 README
            readme_int4 = quantizer.create_quantized_readme("int4_weight_only", "test/model")
            if "int4_weight_only" in readme_int4 and "CPU optimized" in readme_int4:
                print("✅ int4 README creation successful")
            else:
                print("❌ int4 README creation failed")
                return False
            
            return True
    except Exception as e:
        print(f"❌ README creation test failed: {e}")
        return False

def main():
    """Run all quantization tests"""
    print("🧪 Running Quantization Tests")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_quantization_imports),
        ("Initialization Test", test_quantizer_initialization),
        ("Config Creation Test", test_quantization_config_creation),
        ("Model Validation Test", test_model_validation),
        ("Model Card Test", test_quantized_model_card_creation),
        ("README Test", test_quantized_readme_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quantization tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit(main()) 