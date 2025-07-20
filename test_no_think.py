#!/usr/bin/env python3
"""
Test script to verify /no_think tag handling in SmolLM3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from data import SmolLM3Dataset

def test_no_think_tag():
    """Test that /no_think tag is properly applied"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    
    # Test data
    test_data = [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI..."}
            ]
        }
    ]
    
    # Test with no_think_system_message=True
    print("=== Testing with no_think_system_message=True ===")
    dataset_with_no_think = SmolLM3Dataset(
        data_path="test_data",
        tokenizer=tokenizer,
        max_seq_length=4096,
        use_chat_template=True,
        chat_template_kwargs={
            "add_generation_prompt": True,
            "no_think_system_message": True
        }
    )
    
    # Test with no_think_system_message=False
    print("\n=== Testing with no_think_system_message=False ===")
    dataset_without_no_think = SmolLM3Dataset(
        data_path="test_data",
        tokenizer=tokenizer,
        max_seq_length=4096,
        use_chat_template=True,
        chat_template_kwargs={
            "add_generation_prompt": True,
            "no_think_system_message": False
        }
    )
    
    # Test manual chat template application
    print("\n=== Manual chat template test ===")
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
    
    # Without /no_think
    text_without = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Without /no_think:")
    print(text_without[:200] + "..." if len(text_without) > 200 else text_without)
    
    # With /no_think
    messages_with_system = [
        {"role": "system", "content": "You are a helpful assistant. /no_think"},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
    text_with = tokenizer.apply_chat_template(
        messages_with_system,
        tokenize=False,
        add_generation_prompt=True
    )
    print("\nWith /no_think:")
    print(text_with[:200] + "..." if len(text_with) > 200 else text_with)

if __name__ == "__main__":
    test_no_think_tag() 