#!/usr/bin/env python3
"""
Test script to verify trainer selection logic
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "config"))

def test_config_trainer_type():
    """Test that config files have the correct trainer_type"""
    print("Testing config trainer_type...")
    
    # Test base config
    from train_smollm3 import SmolLM3Config
    base_config = SmolLM3Config()
    assert base_config.trainer_type == "sft", f"Base config should have trainer_type='sft', got {base_config.trainer_type}"
    print("✅ Base config trainer_type: sft")
    
    # Test DPO config
    from train_smollm3_dpo import SmolLM3DPOConfig
    dpo_config = SmolLM3DPOConfig()
    assert dpo_config.trainer_type == "dpo", f"DPO config should have trainer_type='dpo', got {dpo_config.trainer_type}"
    print("✅ DPO config trainer_type: dpo")
    
    return True

def test_trainer_classes_exist():
    """Test that trainer classes exist in the trainer module"""
    print("Testing trainer class existence...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(project_root / "src"))
        
        # Import trainer module
        import trainer
        print("✅ Trainer module imported successfully")
        
        # Check if classes exist
        assert hasattr(trainer, 'SmolLM3Trainer'), "SmolLM3Trainer class not found"
        assert hasattr(trainer, 'SmolLM3DPOTrainer'), "SmolLM3DPOTrainer class not found"
        print("✅ Both trainer classes exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to check trainer classes: {e}")
        return False

def test_config_inheritance():
    """Test that DPO config properly inherits from base config"""
    print("Testing config inheritance...")
    
    try:
        from train_smollm3 import SmolLM3Config
        from train_smollm3_dpo import SmolLM3DPOConfig
        
        # Test that DPO config inherits from base config
        base_config = SmolLM3Config()
        dpo_config = SmolLM3DPOConfig()
        
        # Check that DPO config has all base config fields
        base_fields = set(base_config.__dict__.keys())
        dpo_fields = set(dpo_config.__dict__.keys())
        
        # DPO config should have all base fields plus DPO-specific ones
        assert base_fields.issubset(dpo_fields), "DPO config missing base config fields"
        print("✅ DPO config properly inherits from base config")
        
        # Check that trainer_type is overridden correctly
        assert dpo_config.trainer_type == "dpo", "DPO config should have trainer_type='dpo'"
        assert base_config.trainer_type == "sft", "Base config should have trainer_type='sft'"
        print("✅ Trainer type inheritance works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test config inheritance: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Trainer Selection Implementation")
    print("=" * 50)
    
    tests = [
        test_config_trainer_type,
        test_trainer_classes_exist,
        test_config_inheritance,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 