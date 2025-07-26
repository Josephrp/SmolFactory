#!/usr/bin/env python3
"""
Test script for the unified model card system
Verifies template processing, variable substitution, and conditional sections
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.model_tonic.generate_model_card import ModelCardGenerator

def test_basic_model_card():
    """Test basic model card generation without quantized models"""
    print("🧪 Testing basic model card generation...")
    
    # Create test variables
    variables = {
        "model_name": "Test SmolLM3 Model",
        "model_description": "A test fine-tuned SmolLM3 model",
        "repo_name": "test-user/test-model",
        "base_model": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "OpenHermes-FR",
        "training_config_type": "H100 Lightweight",
        "trainer_type": "SFTTrainer",
        "batch_size": "8",
        "gradient_accumulation_steps": "16",
        "learning_rate": "5e-6",
        "max_epochs": "3",
        "max_seq_length": "2048",
        "hardware_info": "GPU (H100)",
        "experiment_name": "test-experiment",
        "trackio_url": "https://trackio.space/test",
        "dataset_repo": "test/trackio-experiments",
        "dataset_size": "~80K samples",
        "dataset_format": "Chat format",
        "author_name": "Test User",
        "model_name_slug": "test_smollm3_model",
        "quantized_models": False,
        "dataset_sample_size": "80000"
    }
    
    try:
        # Create generator
        generator = ModelCardGenerator()
        
        # Generate model card
        content = generator.generate_model_card(variables)
        
        # Check that content was generated
        assert content is not None
        assert len(content) > 0
        
        # Check that basic sections are present
        assert "Test SmolLM3 Model" in content
        assert "test-user/test-model" in content
        assert "HuggingFaceTB/SmolLM3-3B" in content
        
        # Check that quantized sections are NOT present
        assert "Quantized Models" not in content
        assert "int8" not in content
        assert "int4" not in content
        
        print("✅ Basic model card generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic model card generation test failed: {e}")
        return False

def test_quantized_model_card():
    """Test model card generation with quantized models"""
    print("🧪 Testing quantized model card generation...")
    
    # Create test variables with quantized models
    variables = {
        "model_name": "Test SmolLM3 Model with Quantization",
        "model_description": "A test fine-tuned SmolLM3 model with quantized versions",
        "repo_name": "test-user/test-model",
        "base_model": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "OpenHermes-FR",
        "training_config_type": "H100 Lightweight",
        "trainer_type": "SFTTrainer",
        "batch_size": "8",
        "gradient_accumulation_steps": "16",
        "learning_rate": "5e-6",
        "max_epochs": "3",
        "max_seq_length": "2048",
        "hardware_info": "GPU (H100)",
        "experiment_name": "test-experiment",
        "trackio_url": "https://trackio.space/test",
        "dataset_repo": "test/trackio-experiments",
        "dataset_size": "~80K samples",
        "dataset_format": "Chat format",
        "author_name": "Test User",
        "model_name_slug": "test_smollm3_model",
        "quantized_models": True,
        "dataset_sample_size": "80000"
    }
    
    try:
        # Create generator
        generator = ModelCardGenerator()
        
        # Generate model card
        content = generator.generate_model_card(variables)
        
        # Check that content was generated
        assert content is not None
        assert len(content) > 0
        
        # Check that basic sections are present
        assert "Test SmolLM3 Model with Quantization" in content
        assert "test-user/test-model" in content
        
        # Check that quantized sections ARE present
        assert "Quantized Models" in content
        assert "int8" in content
        assert "int4" in content
        assert "test-user/test-model/int8" in content
        assert "test-user/test-model/int4" in content
        
        print("✅ Quantized model card generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantized model card generation test failed: {e}")
        return False

def test_template_processing():
    """Test template processing and variable substitution"""
    print("🧪 Testing template processing...")
    
    try:
        # Create generator
        generator = ModelCardGenerator()
        
        # Test variable substitution
        test_variables = {
            "model_name": "Test Model",
            "repo_name": "test/repo",
            "quantized_models": True
        }
        
        # Generate content
        content = generator.generate_model_card(test_variables)
        
        # Check variable substitution
        assert "Test Model" in content
        assert "test/repo" in content
        
        # Check conditional processing
        assert "Quantized Models" in content
        
        print("✅ Template processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Template processing test failed: {e}")
        return False

def test_file_saving():
    """Test saving generated model cards to files"""
    print("🧪 Testing file saving...")
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_readme.md")
            
            # Create generator
            generator = ModelCardGenerator()
            
            # Test variables
            variables = {
                "model_name": "Test Model",
                "model_description": "Test description",
                "repo_name": "test/repo",
                "base_model": "HuggingFaceTB/SmolLM3-3B",
                "dataset_name": "Test Dataset",
                "training_config_type": "Test Config",
                "trainer_type": "SFTTrainer",
                "batch_size": "8",
                "gradient_accumulation_steps": "16",
                "learning_rate": "5e-6",
                "max_epochs": "3",
                "max_seq_length": "2048",
                "hardware_info": "GPU",
                "experiment_name": "test-exp",
                "trackio_url": "https://trackio.space/test",
                "dataset_repo": "test/dataset",
                "dataset_size": "1K samples",
                "dataset_format": "Chat format",
                "author_name": "Test User",
                "model_name_slug": "test_model",
                "quantized_models": False,
                "dataset_sample_size": None
            }
            
            # Generate and save
            content = generator.generate_model_card(variables)
            success = generator.save_model_card(content, output_path)
            
            # Check that file was created
            assert success
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            assert "Test Model" in saved_content
            assert "test/repo" in saved_content
            
            print("✅ File saving test passed")
            return True
            
    except Exception as e:
        print(f"❌ File saving test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for missing template and invalid variables"""
    print("🧪 Testing error handling...")
    
    try:
        # Test with non-existent template
        try:
            generator = ModelCardGenerator("non_existent_template.md")
            content = generator.generate_model_card({})
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("✅ Correctly handled missing template")
        
        # Test with minimal variables
        generator = ModelCardGenerator()
        content = generator.generate_model_card({})
        
        # Should still generate some content
        assert content is not None
        assert len(content) > 0
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting unified model card system tests...")
    print("=" * 50)
    
    tests = [
        test_basic_model_card,
        test_quantized_model_card,
        test_template_processing,
        test_file_saving,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified model card system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 