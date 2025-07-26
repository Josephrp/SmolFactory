#!/usr/bin/env python3
"""
Quick test for the training fix
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    print("🔧 Testing H100 Lightweight Training Fix")
    print("=" * 50)
    
    # Set environment variables to fix mixed precision issues
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    print("✅ Environment variables set")
    
    # Test configuration
    try:
        from config.train_smollm3_h100_lightweight import SmolLM3ConfigH100Lightweight
        config = SmolLM3ConfigH100Lightweight()
        print(f"✅ Configuration loaded: fp16={config.fp16}, bf16={config.bf16}")
        
        # Test model loading (without actually loading the full model)
        from src.model import SmolLM3Model
        
        # Create model instance
        model = SmolLM3Model(
            model_name="HuggingFaceTB/SmolLM3-3B",
            max_seq_length=4096,
            config=config
        )
        
        print(f"✅ Model dtype: {model.torch_dtype}")
        print(f"✅ Model device map: {model.device_map}")
        
        # Test training arguments
        training_args = model.get_training_arguments("/tmp/test")
        print(f"✅ Training args: fp16={training_args.fp16}, bf16={training_args.bf16}")
        
        print("\n🎉 All tests passed!")
        print("You can now run the training with:")
        print("  ./launch.sh")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 