#!/usr/bin/env python3
"""
Dependency Check Script

This script checks if all required dependencies are installed for the
SmolLM3 fine-tuning pipeline.
"""

import sys
import importlib

def check_dependency(module_name: str, package_name: str = None) -> bool:
    """
    Check if a Python module is available.
    
    Args:
        module_name (str): The module name to check
        package_name (str): The package name for pip installation (if different)
        
    Returns:
        bool: True if module is available, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Check all required dependencies."""
    
    print("🔍 Checking dependencies for SmolLM3 Fine-tuning Pipeline")
    print("=" * 60)
    
    # Required dependencies
    dependencies = [
        ("huggingface_hub", "huggingface_hub"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("bitsandbytes", "bitsandbytes"),
    ]
    
    missing_deps = []
    all_good = True
    
    for module_name, package_name in dependencies:
        if check_dependency(module_name):
            print(f"✅ {module_name}")
        else:
            print(f"❌ {module_name} (install with: pip install {package_name})")
            missing_deps.append(package_name)
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ All dependencies are installed!")
        print("🚀 You're ready to run the fine-tuning pipeline!")
    else:
        print("❌ Missing dependencies detected!")
        print("\nTo install missing dependencies, run:")
        print(f"pip install {' '.join(missing_deps)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements/requirements.txt")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 