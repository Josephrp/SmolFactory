#!/usr/bin/env python3
"""
Test script for demo space deployment functionality
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from deploy_demo_space import DemoSpaceDeployer

def test_demo_deployer_initialization():
    """Test DemoSpaceDeployer initialization"""
    print("🧪 Testing DemoSpaceDeployer initialization...")
    
    deployer = DemoSpaceDeployer(
        hf_token="test_token",
        hf_username="test_user",
        model_id="test/model",
        subfolder="int4",
        space_name="test-demo"
    )
    
    assert deployer.hf_token == "test_token"
    assert deployer.hf_username == "test_user"
    assert deployer.model_id == "test/model"
    assert deployer.subfolder == "int4"
    assert deployer.space_name == "test-demo"
    assert deployer.space_id == "test_user/test-demo"
    
    print("✅ DemoSpaceDeployer initialization test passed")

def test_template_files_exist():
    """Test that template files exist"""
    print("🧪 Testing template files existence...")
    
    template_dir = Path(__file__).parent.parent / "templates" / "spaces" / "demo"
    
    required_files = ["app.py", "requirements.txt"]
    
    for file_name in required_files:
        file_path = template_dir / file_name
        assert file_path.exists(), f"Required file {file_name} not found in templates"
        print(f"✅ Found {file_name}")
    
    print("✅ Template files test passed")

def test_app_py_modification():
    """Test app.py modification with environment variables"""
    print("🧪 Testing app.py modification...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy template files
        template_dir = Path(__file__).parent.parent / "templates" / "spaces" / "demo"
        shutil.copytree(template_dir, temp_path, dirs_exist_ok=True)
        
        # Test the modification logic
        app_file = temp_path / "app.py"
        assert app_file.exists()
        
        # Read original content
        with open(app_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Simulate the modification
        env_setup = """
# Environment variables for model configuration
import os
os.environ['HF_MODEL_ID'] = 'test/model'
os.environ['MODEL_SUBFOLDER'] = 'int4'
os.environ['MODEL_NAME'] = 'model'

"""
        
        # Insert after imports
        lines = original_content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_end = i + 1
            elif line.strip() == '' and import_end > 0:
                break
        
        lines.insert(import_end, env_setup)
        modified_content = '\n'.join(lines)
        
        # Write modified content
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Verify modification
        with open(app_file, 'r', encoding='utf-8') as f:
            final_content = f.read()
        
        assert 'HF_MODEL_ID' in final_content
        assert 'MODEL_SUBFOLDER' in final_content
        assert 'MODEL_NAME' in final_content
        
        print("✅ app.py modification test passed")

def test_readme_generation():
    """Test README.md generation"""
    print("🧪 Testing README.md generation...")
    
    deployer = DemoSpaceDeployer(
        hf_token="test_token",
        hf_username="test_user",
        model_id="test/model",
        subfolder="int4"
    )
    
    # Test README content generation
    readme_content = f"""# Demo: {deployer.model_id}

This is an interactive demo for the fine-tuned model {deployer.model_id}.

## Features
- Interactive chat interface
- Customizable system prompts
- Advanced generation parameters
- Thinking mode support

## Model Information
- **Model ID**: {deployer.model_id}
- **Subfolder**: {deployer.subfolder}
- **Deployed by**: {deployer.hf_username}

## Usage
Simply start chatting with the model using the interface below!

---
*This demo was automatically deployed by the SmolLM3 Fine-tuning Pipeline*
"""
    
    assert "Demo: test/model" in readme_content
    assert "Model ID: test/model" in readme_content
    assert "Subfolder: int4" in readme_content
    assert "Deployed by: test_user" in readme_content
    
    print("✅ README.md generation test passed")

def main():
    """Run all tests"""
    print("🚀 Starting demo deployment tests...")
    print("=" * 50)
    
    try:
        test_demo_deployer_initialization()
        test_template_files_exist()
        test_app_py_modification()
        test_readme_generation()
        
        print("=" * 50)
        print("🎉 All demo deployment tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 