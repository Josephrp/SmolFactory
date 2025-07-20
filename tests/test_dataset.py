#!/usr/bin/env python3
"""
Test script to verify OpenHermes-FR dataset loading
"""

from datasets import load_dataset
import json
import random

def test_openhermes_fr():
    """Test loading and processing OpenHermes-FR dataset"""
    
    print("Loading OpenHermes-FR dataset...")
    try:
        dataset = load_dataset('legmlai/openhermes-fr')
        print(f"✅ Dataset loaded successfully")
        print(f"   Train samples: {len(dataset['train'])}")
        if 'validation' in dataset:
            print(f"   Validation samples: {len(dataset['validation'])}")
        
        # Show sample structure
        sample = dataset['train'][0]
        print(f"\n📋 Sample structure:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        # Test conversion
        print(f"\n🔄 Testing conversion...")
        
        def convert_to_training_format(example):
            # Handle OpenHermes-FR format specifically
            if 'prompt' in example and 'accepted_completion' in example:
                return {
                    'prompt': example['prompt'],
                    'completion': example['accepted_completion']
                }
            elif 'prompt' in example and 'completion' in example:
                return {
                    'prompt': example['prompt'],
                    'completion': example['completion']
                }
            else:
                return None
        
        # Process first 10 examples
        train_data = []
        for i, example in enumerate(dataset['train'][:10]):
            training_example = convert_to_training_format(example)
            if training_example and training_example['prompt'] and training_example['completion']:
                # Filter out bad entries
                if 'bad_entry' in example and example['bad_entry']:
                    print(f"   Skipping bad entry {i}")
                    continue
                train_data.append(training_example)
                print(f"   ✅ Converted example {i}")
        
        print(f"\n📊 Conversion results:")
        print(f"   Converted: {len(train_data)} valid examples")
        
        if train_data:
            print(f"\n📝 Sample converted example:")
            sample = train_data[0]
            print(f"   Prompt: {sample['prompt'][:100]}...")
            print(f"   Completion: {sample['completion'][:100]}...")
        
        # Test sampling
        if len(dataset['train']) > 100:
            print(f"\n🎲 Testing sampling...")
            random.seed(42)
            sampled_indices = random.sample(range(len(dataset['train'])), 5)
            print(f"   Sampled indices: {sampled_indices}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_openhermes_fr()
    if success:
        print("\n✅ Dataset test completed successfully!")
    else:
        print("\n❌ Dataset test failed!")
        exit(1) 