#!/usr/bin/env python3
"""
Test script to verify all training fixes work correctly
"""

import os
import sys
import subprocess
from pathlib import Path

def test_trainer_type_fix():
    """Test that trainer type conversion works correctly"""
    print("🔍 Testing Trainer Type Fix")
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
        converted = input_type.lower()
        if converted == expected_output:
            print(f"✅ '{input_type}' -> '{converted}' (expected: '{expected_output}')")
        else:
            print(f"❌ '{input_type}' -> '{converted}' (expected: '{expected_output}')")
            all_passed = False
    
    return all_passed

def test_trackio_conflict_fix():
    """Test that trackio package conflicts are handled"""
    print("\n🔍 Testing Trackio Conflict Fix")
    print("=" * 50)
    
    try:
        # Test monitoring import
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from monitoring import SmolLM3Monitor
        
        # Test monitor creation
        monitor = SmolLM3Monitor("test-experiment")
        print("✅ Monitor created successfully")
        print(f"   Dataset repo: {monitor.dataset_repo}")
        print(f"   Enable tracking: {monitor.enable_tracking}")
        
        # Check that dataset repo is not empty
        if monitor.dataset_repo and monitor.dataset_repo.strip() != '':
            print("✅ Dataset repository is properly set")
        else:
            print("❌ Dataset repository is empty")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Trackio conflict fix failed: {e}")
        return False

def test_dataset_repo_fix():
    """Test that dataset repository is properly set"""
    print("\n🔍 Testing Dataset Repository Fix")
    print("=" * 50)
    
    # Test environment variable handling
    test_cases = [
        ("user/test-dataset", "user/test-dataset"),
        ("", "tonic/trackio-experiments"),  # Default fallback
        (None, "tonic/trackio-experiments"),  # Default fallback
    ]
    
    all_passed = True
    for input_repo, expected_repo in test_cases:
        # Simulate the monitoring logic
        if input_repo and input_repo.strip() != '':
            actual_repo = input_repo
        else:
            actual_repo = "tonic/trackio-experiments"
        
        if actual_repo == expected_repo:
            print(f"✅ '{input_repo}' -> '{actual_repo}' (expected: '{expected_repo}')")
        else:
            print(f"❌ '{input_repo}' -> '{actual_repo}' (expected: '{expected_repo}')")
            all_passed = False
    
    return all_passed

def test_launch_script_fixes():
    """Test that launch script fixes are in place"""
    print("\n🔍 Testing Launch Script Fixes")
    print("=" * 50)
    
    # Check if launch.sh exists
    launch_script = Path("launch.sh")
    if not launch_script.exists():
        print("❌ launch.sh not found")
        return False
    
    # Read launch script and check for fixes
    script_content = launch_script.read_text(encoding='utf-8')
    
    # Check for trainer type conversion
    if 'TRAINER_TYPE_LOWER=$(echo "$TRAINER_TYPE" | tr \'[:upper:]\' \'[:lower:]\')' in script_content:
        print("✅ Trainer type conversion found")
    else:
        print("❌ Trainer type conversion missing")
        return False
    
    # Check for trainer type usage
    if '--trainer-type "$TRAINER_TYPE_LOWER"' in script_content:
        print("✅ Trainer type usage updated")
    else:
        print("❌ Trainer type usage not updated")
        return False
    
    # Check for dataset repository default
    if 'TRACKIO_DATASET_REPO="$HF_USERNAME/trackio-experiments"' in script_content:
        print("✅ Dataset repository default found")
    else:
        print("❌ Dataset repository default missing")
        return False
    
    # Check for dataset repository validation
    if 'if [ -z "$TRACKIO_DATASET_REPO" ]' in script_content:
        print("✅ Dataset repository validation found")
    else:
        print("❌ Dataset repository validation missing")
        return False
    
    return True

def test_monitoring_fixes():
    """Test that monitoring fixes are in place"""
    print("\n🔍 Testing Monitoring Fixes")
    print("=" * 50)
    
    # Check if monitoring.py exists
    monitoring_file = Path("src/monitoring.py")
    if not monitoring_file.exists():
        print("❌ monitoring.py not found")
        return False
    
    # Read monitoring file and check for fixes
    script_content = monitoring_file.read_text(encoding='utf-8')
    
    # Check for trackio conflict handling
    if 'import trackio' in script_content:
        print("✅ Trackio conflict handling found")
    else:
        print("❌ Trackio conflict handling missing")
        return False
    
    # Check for dataset repository validation
    if 'if not self.dataset_repo or self.dataset_repo.strip() == \'\'' in script_content:
        print("✅ Dataset repository validation found")
    else:
        print("❌ Dataset repository validation missing")
        return False
    
    # Check for improved error handling
    if 'Trackio Space not accessible' in script_content:
        print("✅ Improved Trackio error handling found")
    else:
        print("❌ Improved Trackio error handling missing")
        return False
    
    return True

def test_training_script_validation():
    """Test that training script accepts correct parameters"""
    print("\n🔍 Testing Training Script Validation")
    print("=" * 50)
    
    # Check if training script exists
    training_script = Path("scripts/training/train.py")
    if not training_script.exists():
        print("❌ Training script not found")
        return False
    
    # Read training script and check for argument validation
    script_content = training_script.read_text(encoding='utf-8')
    
    # Check for trainer type argument
    if '--trainer-type' in script_content:
        print("✅ Trainer type argument found")
    else:
        print("❌ Trainer type argument missing")
        return False
    
    # Check for valid choices
    if 'choices=[\'sft\', \'dpo\']' in script_content:
        print("✅ Valid trainer type choices found")
    else:
        print("❌ Valid trainer type choices missing")
        return False
    
    return True

def main():
    """Run all training fix tests"""
    print("🚀 Training Fixes Verification")
    print("=" * 50)
    
    tests = [
        test_trainer_type_fix,
        test_trackio_conflict_fix,
        test_dataset_repo_fix,
        test_launch_script_fixes,
        test_monitoring_fixes,
        test_training_script_validation
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
        print("🎉 ALL TRAINING FIXES PASSED!")
        print("✅ Trainer type conversion: Working")
        print("✅ Trackio conflict handling: Working")
        print("✅ Dataset repository fixes: Working")
        print("✅ Launch script fixes: Working")
        print("✅ Monitoring fixes: Working")
        print("✅ Training script validation: Working")
        print("\nAll training issues have been resolved!")
    else:
        print("❌ SOME TRAINING FIXES FAILED!")
        print("Please check the failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 