#!/usr/bin/env python3
"""
Test Setup Script
Verifies that all components are working correctly
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import transformers
        logger.info(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ transformers: {e}")
        return False
    
    try:
        import datasets
        logger.info(f"✓ datasets {datasets.__version__}")
    except ImportError as e:
        logger.error(f"✗ datasets: {e}")
        return False
    
    try:
        import trl
        logger.info(f"✓ trl {trl.__version__}")
    except ImportError as e:
        logger.error(f"✗ trl: {e}")
        return False
    
    try:
        import accelerate
        logger.info(f"✓ accelerate {accelerate.__version__}")
    except ImportError as e:
        logger.error(f"✗ accelerate: {e}")
        return False
    
    return True

def test_local_imports():
    """Test that local modules can be imported"""
    logger.info("Testing local imports...")
    
    try:
        from config import get_config
        logger.info("✓ config module")
    except ImportError as e:
        logger.error(f"✗ config module: {e}")
        return False
    
    try:
        from model import SmolLM3Model
        logger.info("✓ model module")
    except ImportError as e:
        logger.error(f"✗ model module: {e}")
        return False
    
    try:
        from data import SmolLM3Dataset
        logger.info("✓ data module")
    except ImportError as e:
        logger.error(f"✗ data module: {e}")
        return False
    
    try:
        from trainer import SmolLM3Trainer
        logger.info("✓ trainer module")
    except ImportError as e:
        logger.error(f"✗ trainer module: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        from config import get_config
        config = get_config("config/train_smollm3.py")
        logger.info(f"✓ Configuration loaded: {config.model_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation"""
    logger.info("Testing dataset creation...")
    
    try:
        from data import create_sample_dataset
        output_path = create_sample_dataset("test_dataset")
        
        # Check if files were created
        train_file = os.path.join(output_path, "train.json")
        val_file = os.path.join(output_path, "validation.json")
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            logger.info("✓ Sample dataset created successfully")
            
            # Clean up
            import shutil
            shutil.rmtree(output_path)
            return True
        else:
            logger.error("✗ Dataset files not created")
            return False
    except Exception as e:
        logger.error(f"✗ Dataset creation failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    logger.info("Testing GPU availability...")
    
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA version: {torch.version.cuda}")
        logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        logger.warning("⚠ No GPU available, will use CPU")
        return True

def test_model_loading():
    """Test model loading (without downloading)"""
    logger.info("Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM3-3B",
            trust_remote_code=True,
            use_fast=True
        )
        logger.info(f"✓ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
        
        # Test config loading
        config = AutoConfig.from_pretrained(
            "HuggingFaceTB/SmolLM3-3B",
            trust_remote_code=True
        )
        logger.info(f"✓ Config loaded, model type: {config.model_type}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting SmolLM3 setup tests...")
    
    tests = [
        ("Import Tests", test_imports),
        ("Local Import Tests", test_local_imports),
        ("Configuration Tests", test_config),
        ("Dataset Creation Tests", test_dataset_creation),
        ("GPU Availability Tests", test_gpu_availability),
        ("Model Loading Tests", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info('='*50)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info('='*50)
    
    if passed == total:
        logger.info("🎉 All tests passed! Setup is ready for SmolLM3 fine-tuning.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 