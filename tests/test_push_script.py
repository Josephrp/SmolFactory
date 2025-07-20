#!/usr/bin/env python3
"""
Test script for the improved push_to_huggingface.py script
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_huggingface_pusher_initialization():
    """Test HuggingFacePusher initialization with new parameters"""
    print("🧪 Testing HuggingFacePusher initialization...")
    
    try:
        from scripts.model_tonic.push_to_huggingface import HuggingFacePusher
        
        # Test 1: Default initialization
        print("\n1. Testing default initialization...")
        with patch('push_to_huggingface.HfApi'):
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model"
            )
            print(f"   Dataset repo: {pusher.dataset_repo}")
            print(f"   HF token set: {'Yes' if pusher.hf_token else 'No'}")
        
        # Test 2: Custom initialization
        print("\n2. Testing custom initialization...")
        with patch('push_to_huggingface.HfApi'):
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model",
                dataset_repo="test-user/test-experiments",
                hf_token="test_token_123"
            )
            print(f"   Dataset repo: {pusher.dataset_repo}")
            print(f"   HF token set: {'Yes' if pusher.hf_token else 'No'}")
        
        # Test 3: Environment variable initialization
        print("\n3. Testing environment variable initialization...")
        with patch.dict(os.environ, {
            'HF_TOKEN': 'env_test_token',
            'TRACKIO_DATASET_REPO': 'env-user/env-dataset'
        }), patch('push_to_huggingface.HfApi'):
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model"
            )
            print(f"   Dataset repo: {pusher.dataset_repo}")
            print(f"   HF token set: {'Yes' if pusher.hf_token else 'No'}")
        
        print("✅ HuggingFacePusher initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test HuggingFacePusher initialization: {e}")
        return False

def test_model_card_creation():
    """Test model card creation with HF Datasets integration"""
    print("\n🧪 Testing model card creation...")
    
    try:
        from scripts.model_tonic.push_to_huggingface import HuggingFacePusher
        
        with patch('push_to_huggingface.HfApi'):
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model",
                dataset_repo="test-user/test-experiments"
            )
            
            training_config = {
                "model_name": "HuggingFaceTB/SmolLM3-3B",
                "batch_size": 8,
                "learning_rate": 1e-5
            }
            
            results = {
                "final_loss": 0.5,
                "total_steps": 1000,
                "training_time_hours": 2.5
            }
            
            model_card = pusher.create_model_card(training_config, results)
            
            # Check that dataset repository is included
            if "test-user/test-experiments" in model_card:
                print("✅ Dataset repository included in model card")
            else:
                print("❌ Dataset repository not found in model card")
                return False
            
            # Check that experiment tracking section is included
            if "Experiment Tracking" in model_card:
                print("✅ Experiment tracking section included")
            else:
                print("❌ Experiment tracking section not found")
                return False
            
            print("✅ Model card creation tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to test model card creation: {e}")
        return False

def test_logging_integration():
    """Test logging integration with HF Datasets"""
    print("\n🧪 Testing logging integration...")
    
    try:
        from scripts.model_tonic.push_to_huggingface import HuggingFacePusher
        
        with patch('push_to_huggingface.HfApi'), patch('push_to_huggingface.SmolLM3Monitor') as mock_monitor:
            # Create mock monitor
            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance
            
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model",
                dataset_repo="test-user/test-experiments",
                hf_token="test_token_123"
            )
            
            # Test logging
            details = {
                "model_path": "/tmp/test_model",
                "repo_name": "test-user/test-model"
            }
            
            pusher.log_to_trackio("model_push", details)
            
            # Check that monitor methods were called
            if mock_monitor_instance.log_metrics.called:
                print("✅ Log metrics called")
            else:
                print("❌ Log metrics not called")
                return False
            
            if mock_monitor_instance.log_training_summary.called:
                print("✅ Log training summary called")
            else:
                print("❌ Log training summary not called")
                return False
            
            print("✅ Logging integration tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to test logging integration: {e}")
        return False

def test_argument_parsing():
    """Test command line argument parsing"""
    print("\n🧪 Testing argument parsing...")
    
    try:
        from scripts.model_tonic.push_to_huggingface import parse_args
        
        # Test with new arguments
        test_args = [
            "push_to_huggingface.py",
            "/tmp/test_model",
            "test-user/test-model",
            "--dataset-repo", "test-user/test-experiments",
            "--hf-token", "test_token_123",
            "--private"
        ]
        
        with patch('sys.argv', test_args):
            args = parse_args()
            
            print(f"   Model path: {args.model_path}")
            print(f"   Repo name: {args.repo_name}")
            print(f"   Dataset repo: {args.dataset_repo}")
            print(f"   HF token: {'Set' if args.hf_token else 'Not set'}")
            print(f"   Private: {args.private}")
            
            if args.dataset_repo == "test-user/test-experiments":
                print("✅ Dataset repo argument parsed correctly")
            else:
                print("❌ Dataset repo argument not parsed correctly")
                return False
            
            if args.hf_token == "test_token_123":
                print("✅ HF token argument parsed correctly")
            else:
                print("❌ HF token argument not parsed correctly")
                return False
            
            print("✅ Argument parsing tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Failed to test argument parsing: {e}")
        return False

def test_environment_variable_handling():
    """Test environment variable handling"""
    print("\n🧪 Testing environment variable handling...")
    
    try:
        from scripts.model_tonic.push_to_huggingface import HuggingFacePusher
        
        # Test with environment variables set
        with patch.dict(os.environ, {
            'HF_TOKEN': 'env_test_token',
            'TRACKIO_DATASET_REPO': 'env-user/env-dataset'
        }), patch('push_to_huggingface.HfApi'):
            pusher = HuggingFacePusher(
                model_path="/tmp/test_model",
                repo_name="test-user/test-model"
            )
            
            print(f"   Dataset repo: {pusher.dataset_repo}")
            print(f"   HF token: {'Set' if pusher.hf_token else 'Not set'}")
            
            if pusher.dataset_repo == "env-user/env-dataset":
                print("✅ Environment variable for dataset repo used")
            else:
                print("❌ Environment variable for dataset repo not used")
                return False
            
            if pusher.hf_token == "env_test_token":
                print("✅ Environment variable for HF token used")
            else:
                print("❌ Environment variable for HF token not used")
                return False
        
        print("✅ Environment variable tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to test environment variables: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Improved Push Script")
    print("=" * 50)
    
    tests = [
        ("HuggingFacePusher Initialization", test_huggingface_pusher_initialization),
        ("Model Card Creation", test_model_card_creation),
        ("Logging Integration", test_logging_integration),
        ("Argument Parsing", test_argument_parsing),
        ("Environment Variables", test_environment_variable_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results")
    print("=" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Push script is working correctly.")
        print("\n📋 New Features:")
        print("✅ HF Datasets integration")
        print("✅ Environment variable support")
        print("✅ Enhanced model card creation")
        print("✅ Improved logging to HF Datasets")
        print("✅ Better argument parsing")
        print("✅ Dataset repository tracking")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    print(f"\n📋 Usage Examples:")
    print("Basic usage:")
    print("  python push_to_huggingface.py /path/to/model username/repo-name")
    print("\nWith HF Datasets:")
    print("  python push_to_huggingface.py /path/to/model username/repo-name --dataset-repo username/experiments")
    print("\nWith custom token:")
    print("  python push_to_huggingface.py /path/to/model username/repo-name --hf-token your_token_here")
    print("\nWith all options:")
    print("  python push_to_huggingface.py /path/to/model username/repo-name --dataset-repo username/experiments --hf-token your_token_here --private")

if __name__ == "__main__":
    main() 