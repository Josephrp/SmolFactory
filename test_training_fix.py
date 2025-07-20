#!/usr/bin/env python3
"""
Quick test to verify the training configuration fix
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_configuration():
    """Test the H100 lightweight configuration"""
    print("Testing H100 Lightweight Configuration...")
    
    try:
        from config.train_smollm3_h100_lightweight import SmolLM3ConfigH100Lightweight
        
        config = SmolLM3ConfigH100Lightweight()
        
        print("✅ Configuration loaded successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  FP16: {config.fp16}")
        print(f"  BF16: {config.bf16}")
        print(f"  Mixed precision: {'fp16' if config.fp16 else 'bf16'}")
        print(f"  Sample size: {config.sample_size}")
        
        # Test training arguments creation
        from src.model import SmolLM3Model
        
        # Create a minimal model instance for testing
        model = SmolLM3Model(
            model_name="HuggingFaceTB/SmolLM3-3B",
            max_seq_length=4096,
            config=config
        )
        
        # Test training arguments
        training_args = model.get_training_arguments("/tmp/test")
        print(f"✅ Training arguments created successfully")
        print(f"  Training args FP16: {training_args.fp16}")
        print(f"  Training args BF16: {training_args.bf16}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration()
    if success:
        print("\n🎉 Configuration test passed!")
        print("You can now run the training with: ./launch.sh")
    else:
        print("\n❌ Configuration test failed!")
        sys.exit(1) 