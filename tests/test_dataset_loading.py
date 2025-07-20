#!/usr/bin/env python3
"""
Test script to verify dataset loading works correctly
"""

import os
import sys
import json
from datasets import load_dataset

def test_dataset_loading():
    """Test loading the OpenHermes-FR dataset"""
    print("Testing dataset loading...")
    
    try:
        # Load the dataset
        dataset = load_dataset("legmlai/openhermes-fr")
        print(f"✅ Dataset loaded successfully")
        print(f"  Train samples: {len(dataset['train'])}")
        
        # Check the first few examples
        print("\nSample examples:")
        for i in range(min(3, len(dataset['train'])):
            example = dataset['train'][i]
            print(f"\nExample {i+1}:")
            print(f"  Keys: {list(example.keys())}")
            print(f"  Prompt: {example.get('prompt', 'N/A')[:100]}...")
            print(f"  Accepted completion: {example.get('accepted_completion', 'N/A')[:100]}...")
            print(f"  Bad entry: {example.get('bad_entry', 'N/A')}")
        
        # Test filtering bad entries
        print(f"\nFiltering bad entries...")
        original_size = len(dataset['train'])
        filtered_dataset = dataset['train'].filter(lambda x: not x.get('bad_entry', False))
        filtered_size = len(filtered_dataset)
        print(f"  Original size: {original_size}")
        print(f"  Filtered size: {filtered_size}")
        print(f"  Removed: {original_size - filtered_size} bad entries")
        
        # Test conversion to training format
        print(f"\nTesting conversion to training format...")
        train_data = []
        for i, example in enumerate(filtered_dataset):
            if i >= 5:  # Just test first 5 examples
                break
            
            if 'prompt' in example and 'accepted_completion' in example:
                train_data.append({
                    'prompt': example['prompt'],
                    'completion': example['accepted_completion']
                })
        
        print(f"  Converted {len(train_data)} examples to training format")
        
        # Save a small sample
        os.makedirs('test_dataset', exist_ok=True)
        with open('test_dataset/train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        print(f"✅ Test completed successfully!")
        print(f"  Sample saved to: test_dataset/train.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1) 