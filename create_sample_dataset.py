#!/usr/bin/env python3
"""
Sample Dataset Creation Script
Creates sample datasets for testing SmolLM3 fine-tuning
"""

import os
import json
import argparse
from data import create_sample_dataset

def main():
    parser = argparse.ArgumentParser(description='Create sample dataset for SmolLM3 fine-tuning')
    parser.add_argument('--output_dir', type=str, default='my_dataset',
                       help='Output directory for the dataset')
    parser.add_argument('--format', type=str, default='chat',
                       choices=['chat', 'instruction', 'user_assistant'],
                       help='Dataset format')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to create')
    
    args = parser.parse_args()
    
    # Create sample dataset
    output_path = create_sample_dataset(args.output_dir)
    
    print(f"Sample dataset created in: {output_path}")
    print(f"Format: {args.format}")
    print(f"Samples: {args.num_samples}")
    print("\nFiles created:")
    print(f"- {os.path.join(output_path, 'train.json')}")
    print(f"- {os.path.join(output_path, 'validation.json')}")
    
    # Show sample data
    with open(os.path.join(output_path, 'train.json'), 'r') as f:
        data = json.load(f)
        print(f"\nSample data:")
        print(json.dumps(data[0], indent=2))

if __name__ == '__main__':
    main() 