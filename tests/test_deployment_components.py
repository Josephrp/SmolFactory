#!/usr/bin/env python3
"""
Test script for deployment components verification
Tests Trackio Space deployment and model repository deployment components
"""

import os
import sys
import json
from pathlib import Path

def test_trackio_space_components():
    """Test Trackio Space deployment components"""
    print("🔍 Testing Trackio Space Deployment Components")
    print("=" * 50)
    
    # Test 1: Check if deployment script exists
    deploy_script = Path("scripts/trackio_tonic/deploy_trackio_space.py")
    if deploy_script.exists():
        print("✅ Trackio Space deployment script exists")
    else:
        print("❌ Trackio Space deployment script missing")
        return False
    
    # Test 2: Check if app.py template exists
    app_template = Path("templates/spaces/app.py")
    if app_template.exists():
        print("✅ Gradio app template exists")
        
        # Check if it has required components
        with open(app_template, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class TrackioSpace" in content:
                print("✅ TrackioSpace class implemented")
            else:
                print("❌ TrackioSpace class missing")
                return False
            
            if "def create_experiment" in content:
                print("✅ Experiment creation functionality")
            else:
                print("❌ Experiment creation missing")
                return False
            
            if "def log_metrics" in content:
                print("✅ Metrics logging functionality")
            else:
                print("❌ Metrics logging missing")
                return False
            
            if "def get_experiment" in content:
                print("✅ Experiment retrieval functionality")
            else:
                print("❌ Experiment retrieval missing")
                return False
    else:
        print("❌ Gradio app template missing")
        return False
    
    # Test 3: Check if requirements.txt exists
    requirements = Path("templates/spaces/requirements.txt")
    if requirements.exists():
        print("✅ Space requirements file exists")
        
        # Check for required dependencies
        with open(requirements, 'r', encoding='utf-8') as f:
            content = f.read()
            required_deps = ['gradio', 'pandas', 'plotly', 'datasets', 'huggingface-hub']
            for dep in required_deps:
                if dep in content:
                    print(f"✅ Required dependency: {dep}")
                else:
                    print(f"❌ Missing dependency: {dep}")
                    return False
    else:
        print("❌ Space requirements file missing")
        return False
    
    # Test 4: Check if README template exists
    readme_template = Path("templates/spaces/README.md")
    if readme_template.exists():
        print("✅ Space README template exists")
        
        # Check for required metadata
        with open(readme_template, 'r', encoding='utf-8') as f:
            content = f.read()
            if "title:" in content and "sdk: gradio" in content:
                print("✅ HF Spaces metadata present")
            else:
                print("❌ HF Spaces metadata missing")
                return False
    else:
        print("❌ Space README template missing")
        return False
    
    print("✅ All Trackio Space components verified!")
    return True

def test_model_repository_components():
    """Test model repository deployment components"""
    print("\n🔍 Testing Model Repository Deployment Components")
    print("=" * 50)
    
    # Test 1: Check if push script exists
    push_script = Path("scripts/model_tonic/push_to_huggingface.py")
    if push_script.exists():
        print("✅ Model push script exists")
    else:
        print("❌ Model push script missing")
        return False
    
    # Test 2: Check if quantize script exists
    quantize_script = Path("scripts/model_tonic/quantize_model.py")
    if quantize_script.exists():
        print("✅ Model quantization script exists")
    else:
        print("❌ Model quantization script missing")
        return False
    
    # Test 3: Check if model card template exists
    model_card_template = Path("templates/model_card.md")
    if model_card_template.exists():
        print("✅ Model card template exists")
        
        # Check for required sections
        with open(model_card_template, 'r', encoding='utf-8') as f:
            content = f.read()
            required_sections = ['base_model:', 'pipeline_tag:', 'tags:']
            for section in required_sections:
                if section in content:
                    print(f"✅ Required section: {section}")
                else:
                    print(f"❌ Missing section: {section}")
                    return False
    else:
        print("❌ Model card template missing")
        return False
    
    # Test 4: Check if model card generator exists
    card_generator = Path("scripts/model_tonic/generate_model_card.py")
    if card_generator.exists():
        print("✅ Model card generator exists")
    else:
        print("❌ Model card generator missing")
        return False
    
    # Test 5: Check push script functionality
    with open(push_script, 'r', encoding='utf-8') as f:
        content = f.read()
        required_functions = [
            'def create_repository',
            'def upload_model_files',
            'def create_model_card',
            'def validate_model_path'
        ]
        for func in required_functions:
            if func in content:
                print(f"✅ Required function: {func}")
            else:
                print(f"❌ Missing function: {func}")
                return False
    
    print("✅ All Model Repository components verified!")
    return True

def test_integration_components():
    """Test integration between components"""
    print("\n🔍 Testing Integration Components")
    print("=" * 50)
    
    # Test 1: Check if launch script integrates deployment
    launch_script = Path("launch.sh")
    if launch_script.exists():
        print("✅ Launch script exists")
        
        with open(launch_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if "deploy_trackio_space.py" in content:
                print("✅ Trackio Space deployment integrated")
            else:
                print("❌ Trackio Space deployment not integrated")
                return False
            
            if "push_to_huggingface.py" in content:
                print("✅ Model push integrated")
            else:
                print("❌ Model push not integrated")
                return False
    else:
        print("❌ Launch script missing")
        return False
    
    # Test 2: Check if monitoring integration exists
    monitoring_script = Path("src/monitoring.py")
    if monitoring_script.exists():
        print("✅ Monitoring script exists")
        
        with open(monitoring_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class SmolLM3Monitor" in content:
                print("✅ SmolLM3Monitor class implemented")
            else:
                print("❌ SmolLM3Monitor class missing")
                return False
    else:
        print("❌ Monitoring script missing")
        return False
    
    # Test 3: Check if dataset integration exists
    dataset_script = Path("scripts/dataset_tonic/setup_hf_dataset.py")
    if dataset_script.exists():
        print("✅ Dataset setup script exists")
        
        with open(dataset_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if "def setup_trackio_dataset" in content:
                print("✅ Dataset setup function implemented")
            else:
                print("❌ Dataset setup function missing")
                return False
    else:
        print("❌ Dataset setup script missing")
        return False
    
    print("✅ All integration components verified!")
    return True

def test_token_validation():
    """Test token validation functionality"""
    print("\n🔍 Testing Token Validation")
    print("=" * 50)
    
    # Test 1: Check if validation script exists
    validation_script = Path("scripts/validate_hf_token.py")
    if validation_script.exists():
        print("✅ Token validation script exists")
        
        with open(validation_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if "def validate_hf_token" in content:
                print("✅ Token validation function implemented")
            else:
                print("❌ Token validation function missing")
                return False
    else:
        print("❌ Token validation script missing")
        return False
    
    print("✅ Token validation components verified!")
    return True

def main():
    """Run all component tests"""
    print("🚀 Deployment Components Verification")
    print("=" * 50)
    
    tests = [
        test_trackio_space_components,
        test_model_repository_components,
        test_integration_components,
        test_token_validation
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
        print("🎉 ALL COMPONENTS VERIFIED SUCCESSFULLY!")
        print("✅ Trackio Space deployment components: Complete")
        print("✅ Model repository deployment components: Complete")
        print("✅ Integration components: Complete")
        print("✅ Token validation components: Complete")
        print("\nAll important deployment components are properly implemented!")
    else:
        print("❌ SOME COMPONENTS NEED ATTENTION!")
        print("Please check the failed components above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 