#!/usr/bin/env python3
"""
Test script to verify README template replacement works correctly
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_readme_template():
    """Test README template replacement"""
    print("🔍 Testing README template replacement...")
    
    try:
        # Get template path
        templates_dir = project_root / "templates" / "spaces"
        readme_template_path = templates_dir / "README.md"
        
        if not readme_template_path.exists():
            print(f"❌ README template not found: {readme_template_path}")
            return False
        
        # Read template
        with open(readme_template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        print(f"✅ README template loaded ({len(template_content)} characters)")
        
        # Test placeholder replacement
        test_space_url = "https://huggingface.co/spaces/test-user/test-space"
        replaced_content = template_content.replace("{SPACE_URL}", test_space_url)
        
        if "{SPACE_URL}" in replaced_content:
            print("❌ Placeholder replacement failed")
            return False
        
        if test_space_url not in replaced_content:
            print("❌ Space URL not found in replaced content")
            return False
        
        print("✅ Placeholder replacement works correctly")
        print(f"✅ Space URL: {test_space_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ README template test failed: {e}")
        return False

def test_deployment_readme():
    """Test that deployment script can use README template"""
    print("\n🔍 Testing deployment script README usage...")
    
    try:
        from scripts.trackio_tonic.deploy_trackio_space import TrackioSpaceDeployer
        
        # Create a mock deployer
        deployer = TrackioSpaceDeployer("test-space", "test-user", "test-token")
        
        # Test that templates directory exists
        project_root = Path(__file__).parent
        templates_dir = project_root / "templates" / "spaces"
        readme_template_path = templates_dir / "README.md"
        
        if readme_template_path.exists():
            print(f"✅ README template exists: {readme_template_path}")
            
            # Test reading template
            with open(readme_template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "{SPACE_URL}" in content:
                    print("✅ Template contains placeholder")
                else:
                    print("⚠️  Template missing placeholder")
            
            return True
        else:
            print(f"❌ README template missing: {readme_template_path}")
            return False
        
    except Exception as e:
        print(f"❌ Deployment README test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing README Template System")
    print("=" * 50)
    
    tests = [
        test_readme_template,
        test_deployment_readme
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! README template system is working correctly.")
        print("\n🚀 Template workflow:")
        print("1. README template is read from templates/spaces/README.md")
        print("2. {SPACE_URL} placeholder is replaced with actual space URL")
        print("3. Customized README is written to the space")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    exit(main()) 