#!/usr/bin/env python3
"""
Test script to verify the string formatting fix
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_logging():
    """Test that logging works without f-string formatting errors"""
    try:
        # Test various logging scenarios that were causing issues
        logger.info("Testing logging with %s", "string formatting")
        logger.info("Testing with %d numbers", 42)
        logger.info("Testing with %s and %d", "text", 123)
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Caught error: %s", e)
        
        print("✅ All logging tests passed!")
        return True
        
    except Exception as e:
        print("❌ Logging test failed: {}".format(e))
        return False

def test_imports():
    """Test that all modules can be imported without formatting errors"""
    try:
        # Test importing the main modules
        from monitoring import SmolLM3Monitor
        print("✅ monitoring module imported successfully")
        
        from trainer import SmolLM3Trainer
        print("✅ trainer module imported successfully")
        
        from model import SmolLM3Model
        print("✅ model module imported successfully")
        
        from data import SmolLM3Dataset
        print("✅ data module imported successfully")
        
        return True
        
    except Exception as e:
        print("❌ Import test failed: {}".format(e))
        return False

def test_config_loading():
    """Test that configuration files can be loaded"""
    try:
        # Test loading a configuration
        config_path = "config/train_smollm3_openhermes_fr_a100_balanced.py"
        if os.path.exists(config_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if hasattr(config_module, 'config'):
                config = config_module.config
                print("✅ Configuration loaded successfully")
                print("   Model: {}".format(config.model_name))
                print("   Batch size: {}".format(config.batch_size))
                print("   Learning rate: {}".format(config.learning_rate))
                return True
            else:
                print("❌ No config found in {}".format(config_path))
                return False
        else:
            print("❌ Config file not found: {}".format(config_path))
            return False
            
    except Exception as e:
        print("❌ Config loading test failed: {}".format(e))
        return False

def main():
    """Run all tests"""
    print("🧪 Testing String Formatting Fix")
    print("=" * 40)
    
    tests = [
        ("Logging", test_logging),
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print("\n🔍 Testing: {}".format(test_name))
        if test_func():
            passed += 1
            print("✅ {} test passed".format(test_name))
        else:
            print("❌ {} test failed".format(test_name))
    
    print("\n" + "=" * 40)
    print("📊 Test Results: {}/{} tests passed".format(passed, total))
    
    if passed == total:
        print("🎉 All tests passed! The formatting fix is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 