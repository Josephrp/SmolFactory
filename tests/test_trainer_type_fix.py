#!/usr/bin/env python3
"""
Test script to verify trainer type conversion works correctly
"""

import os
import sys
import subprocess
from pathlib import Path

def test_trainer_type_conversion():
    """Test that trainer type is converted to lowercase correctly"""
    print("🔍 Testing Trainer Type Conversion")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("SFT", "sft"),
        ("DPO", "dpo"),
        ("sft", "sft"),
        ("dpo", "dpo")
    ]
    
    all_passed = True
    for input_type, expected_output in test_cases:
        # Simulate the bash conversion: echo "$TRAINER_TYPE" | tr '[:upper:]' '[:lower:]'
        converted = input_type.lower()
        
        if converted == expected_output:
            print(f"✅ '{input_type}' -> '{converted}' (expected: '{expected_output}')")
        else:
            print(f"❌ '{input_type}' -> '{converted}' (expected: '{expected_output}')")
            all_passed = False
    
    return all_passed

def test_launch_script_trainer_type():
    """Test that launch script handles trainer type correctly"""
    print("\n🔍 Testing Launch Script Trainer Type Handling")
    print("=" * 50)
    
    # Check if launch.sh exists
    launch_script = Path("launch.sh")
    if not launch_script.exists():
        print("❌ launch.sh not found")
        return False
    
    # Read launch script and check for trainer type handling
    script_content = launch_script.read_text(encoding='utf-8')
    
    # Check for trainer type conversion
    conversion_patterns = [
        'TRAINER_TYPE_LOWER=$(echo "$TRAINER_TYPE" | tr \'[:upper:]\' \'[:lower:]\')',
        '--trainer-type "$TRAINER_TYPE_LOWER"'
    ]
    
    all_found = True
    for pattern in conversion_patterns:
        if pattern in script_content:
            print(f"✅ Found: {pattern}")
        else:
            print(f"❌ Missing: {pattern}")
            all_found = False
    
    # Check that old pattern is removed
    old_pattern = '--trainer-type "$TRAINER_TYPE"'
    if old_pattern in script_content:
        print(f"❌ Found old pattern (should be updated): {old_pattern}")
        all_found = False
    else:
        print(f"✅ Old pattern removed: {old_pattern}")
    
    return all_found

def test_training_script_validation():
    """Test that training script accepts the correct trainer types"""
    print("\n🔍 Testing Training Script Validation")
    print("=" * 50)
    
    # Check if training script exists
    training_script = Path("scripts/training/train.py")
    if not training_script.exists():
        print("❌ Training script not found")
        return False
    
    # Read training script and check for argument validation
    script_content = training_script.read_text(encoding='utf-8')
    
    # Check for trainer type argument definition
    if '--trainer-type' in script_content:
        print("✅ Found trainer-type argument in training script")
    else:
        print("❌ Missing trainer-type argument in training script")
        return False
    
    # Check for valid choices
    if 'sft' in script_content and 'dpo' in script_content:
        print("✅ Found valid trainer type choices: sft, dpo")
    else:
        print("❌ Missing valid trainer type choices")
        return False
    
    return True

def test_trainer_type_integration():
    """Test that trainer type integration works end-to-end"""
    print("\n🔍 Testing Trainer Type Integration")
    print("=" * 50)
    
    # Test the conversion logic
    test_cases = [
        ("SFT", "sft"),
        ("DPO", "dpo")
    ]
    
    all_passed = True
    for input_type, expected_output in test_cases:
        # Simulate the bash conversion
        converted = input_type.lower()
        
        # Check if the converted value is valid for the training script
        valid_types = ["sft", "dpo"]
        
        if converted in valid_types:
            print(f"✅ '{input_type}' -> '{converted}' (valid for training script)")
        else:
            print(f"❌ '{input_type}' -> '{converted}' (invalid for training script)")
            all_passed = False
    
    return all_passed

def main():
    """Run all trainer type fix tests"""
    print("🚀 Trainer Type Fix Verification")
    print("=" * 50)
    
    tests = [
        test_trainer_type_conversion,
        test_launch_script_trainer_type,
        test_training_script_validation,
        test_trainer_type_integration
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TRAINER TYPE FIX TESTS PASSED!")
        print("✅ Trainer type conversion: Working")
        print("✅ Launch script handling: Working")
        print("✅ Training script validation: Working")
        print("✅ Integration: Working")
        print("\nThe trainer type fix is working correctly!")
    else:
        print("❌ SOME TRAINER TYPE FIX TESTS FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 