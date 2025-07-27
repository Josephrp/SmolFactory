#!/usr/bin/env python3
"""
Setup script for the interactive SmolLM3 end-to-end fine-tuning pipeline
Helps users prepare for the interactive launch script
"""

import os
import re
from pathlib import Path

def setup_launch_script():
    """Setup the launch.sh script with user configuration"""
    
    print("🚀 SmolLM3 Interactive End-to-End Fine-tuning Setup")
    print("=" * 60)
    
    print("\n📋 This setup will help you prepare for the interactive pipeline.")
    print("The launch script will now prompt you for all necessary information.")
    
    # Check if launch.sh exists
    launch_path = Path("launch.sh")
    if not launch_path.exists():
        print("❌ launch.sh not found")
        return False
    
    print("\n✅ launch.sh found - no configuration needed!")
    print("The script is now interactive and will prompt you for all settings.")
    
    return True

def create_requirements_check():
    """Create a requirements check script"""
    
    check_script = """#!/usr/bin/env python3
\"\"\"
Requirements check for SmolLM3 fine-tuning
\"\"\"

import sys
import subprocess

def check_requirements():
    \"\"\"Check if all requirements are met\"\"\"
    
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'trl',
        'huggingface_hub',
        'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (training will be slower)")
    except:
        print("⚠️  Could not check CUDA availability")
    
    print("\\n✅ All requirements met!")
    return True

if __name__ == "__main__":
    check_requirements()
"""
    
    with open("check_requirements.py", 'w') as f:
        f.write(check_script)
    
    print("✅ Created check_requirements.py")

def create_quick_start_guide():
    """Create a quick start guide"""
    
    guide = """# SmolLM3 Interactive Pipeline - Quick Start Guide

## 🚀 Quick Start

### 1. Check Requirements
```bash
python check_requirements.py
```

### 2. Run the Interactive Pipeline
```bash
chmod +x launch.sh
./launch.sh
```

## 📋 What the Interactive Pipeline Does

The pipeline will guide you through:

1. **Authentication** - Enter your HF username and token
2. **Configuration Selection** - Choose from predefined training configs:
   - Basic Training (SmolLM3 + SmolTalk)
   - H100 Lightweight (Rapid training on H100)
   - A100 Large Scale (SmolLM3 + OpenHermes-FR)
   - Multiple Passes (Extended training)
   - Custom Configuration (User-defined)
3. **Experiment Setup** - Configure experiment name and repositories
4. **Training Parameters** - Adjust batch size, learning rate, etc.
5. **Deployment** - Automatic Trackio Space and HF Dataset setup
6. **Training** - Monitored fine-tuning with real-time tracking
7. **Model Push** - Upload to HF Hub with documentation

## 🎯 Available Training Configurations

### 1. Basic Training (Default)
- **Model**: SmolLM3-3B
- **Dataset**: SmolTalk
- **Epochs**: 3
- **Batch Size**: 2
- **Learning Rate**: 5e-6
- **Best for**: Quick experiments, learning

### 2. H100 Lightweight (Rapid)
- **Model**: SmolLM3-3B
- **Dataset**: OpenHermes-FR (80K samples)
- **Epochs**: 1
- **Batch Size**: 16
- **Learning Rate**: 8e-6
- **Sequence Length**: 8192
- **Best for**: Rapid training on H100

### 3. A100 Large Scale
- **Model**: SmolLM3-3B
- **Dataset**: OpenHermes-FR
- **Epochs**: 1.3 passes
- **Batch Size**: 8
- **Learning Rate**: 5e-6
- **Sequence Length**: 8192
- **Best for**: High-performance training

### 4. Multiple Passes
- **Model**: SmolLM3-3B
- **Dataset**: OpenHermes-FR
- **Epochs**: 4 passes
- **Batch Size**: 6
- **Learning Rate**: 3e-6
- **Sequence Length**: 8192
- **Best for**: Thorough training

### 5. Custom Configuration
- **User-defined parameters**
- **Flexible model and dataset selection**
- **Custom training parameters**

## 🔧 Prerequisites

1. **Hugging Face Account**
   - Create account at https://huggingface.co
   - Generate token at https://huggingface.co/settings/tokens

2. **System Requirements**
   - Python 3.8+
   - CUDA-compatible GPU (recommended)
   - 16GB+ RAM
   - 50GB+ storage

3. **Dependencies**
   - PyTorch with CUDA
   - Transformers
   - Datasets
   - Accelerate
   - TRL

## 📊 Expected Outputs

After running the pipeline, you'll have:

- **Model Repository**: `https://huggingface.co/your-username/smollm3-finetuned-YYYYMMDD`
- **Trackio Space**: `https://huggingface.co/spaces/your-username/trackio-monitoring-YYYYMMDD`
- **Experiment Dataset**: `https://huggingface.co/datasets/your-username/trackio-experiments`
- **Training Summary**: `training_summary.md`

## 🛠️ Troubleshooting

### Common Issues

1. **HF Token Issues**
   ```bash
   hf whoami
   ```

2. **CUDA Issues**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory Issues**
   - Reduce batch size in custom configuration
   - Increase gradient accumulation steps

4. **Network Issues**
   - Check internet connection
   - Verify HF token permissions

## 🎯 Tips for Success

1. **Start with Basic Training** for your first run
2. **Use H100 Lightweight** for rapid experiments on H100
3. **Use A100 Large Scale** for serious experiments
3. **Monitor in Trackio Space** for real-time progress
4. **Check logs** if something goes wrong
5. **Test the model** after training completes

## 📞 Support

- Check the troubleshooting section
- Review logs in `training.log`
- Monitor progress in Trackio Space
- Open an issue on GitHub

---

**Happy Fine-tuning! 🚀**
"""
    
    with open("QUICK_START_GUIDE.md", 'w') as f:
        f.write(guide)
    
    print("✅ Created QUICK_START_GUIDE.md")

def main():
    """Main setup function"""
    
    print("Welcome to SmolLM3 Interactive End-to-End Fine-tuning Setup!")
    print("This will help you prepare for the interactive pipeline.")
    
    if setup_launch_script():
        create_requirements_check()
        create_quick_start_guide()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Files created:")
        print("  - check_requirements.py (requirement checker)")
        print("  - QUICK_START_GUIDE.md (usage guide)")
        
        print("\n🚀 Ready to start training!")
        print("Next steps:")
        print("1. Run: python check_requirements.py")
        print("2. Run: chmod +x launch.sh")
        print("3. Run: ./launch.sh")
        print("4. Follow the interactive prompts")
        
        print("\n📚 For detailed information, see:")
        print("  - QUICK_START_GUIDE.md")
        print("  - README_END_TO_END.md")
    else:
        print("\n❌ Setup failed. Please check your input and try again.")

if __name__ == "__main__":
    main() 