"""
SmolLM3 Dataset Handler
Handles data loading, preprocessing, and tokenization for SmolLM3 fine-tuning
"""

import os
import json
import torch
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class SmolLM3Dataset:
    """Dataset handler for SmolLM3 fine-tuning"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 4096,
        use_chat_template: bool = True,
        chat_template_kwargs: Optional[Dict] = None,
        filter_bad_entries: bool = False,
        bad_entry_field: str = "bad_entry"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_chat_template = use_chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.filter_bad_entries = filter_bad_entries
        self.bad_entry_field = bad_entry_field
        
        # Load and process dataset
        self.dataset = self._load_dataset()
        self.processed_dataset = self._process_dataset()
    
    def _load_dataset(self) -> Dataset:
        """Load dataset from various formats"""
        logger.info(f"Loading dataset from {self.data_path}")
        
        # Check if it's a Hugging Face dataset
        if os.path.isdir(self.data_path):
            # Local directory
            try:
                dataset = load_dataset("json", data_files={
                    "train": os.path.join(self.data_path, "train.json"),
                    "validation": os.path.join(self.data_path, "validation.json") if os.path.exists(os.path.join(self.data_path, "validation.json")) else None,
                    "test": os.path.join(self.data_path, "test.json") if os.path.exists(os.path.join(self.data_path, "test.json")) else None
                })
                logger.info("Loaded dataset from local JSON files")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load as JSON dataset: {e}")
        
        # Try to load as a single JSON file
        if os.path.isfile(self.data_path) and self.data_path.endswith('.json'):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to dataset format
                if isinstance(data, list):
                    dataset = Dataset.from_list(data)
                else:
                    dataset = Dataset.from_dict(data)
                
                logger.info("Loaded dataset from single JSON file")
                return dataset
            except Exception as e:
                logger.error(f"Failed to load JSON file: {e}")
                raise
        
        # Try to load as a Hugging Face dataset name
        try:
            dataset = load_dataset(self.data_path)
            logger.info(f"Loaded Hugging Face dataset: {self.data_path}")
            
            # Filter bad entries if requested
            if self.filter_bad_entries and self.bad_entry_field in dataset["train"].column_names:
                logger.info(f"Filtering out bad entries using field: {self.bad_entry_field}")
                for split in dataset:
                    if self.bad_entry_field in dataset[split].column_names:
                        original_size = len(dataset[split])
                        dataset[split] = dataset[split].filter(lambda x: not x[self.bad_entry_field])
                        filtered_size = len(dataset[split])
                        logger.info(f"Filtered {split}: {original_size} -> {filtered_size} samples")
            
            # If only 'train' split exists, create validation and test splits
            if ("train" in dataset) and ("validation" not in dataset or "test" not in dataset):
                logger.info("Automatically splitting train into train/validation/test (98/1/1)")
                split_dataset = dataset["train"].train_test_split(test_size=0.02, seed=42)
                # Now split test into validation and test (1% each)
                val_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
                dataset = {
                    "train": split_dataset["train"],
                    "validation": val_test_split["train"],
                    "test": val_test_split["test"]
                }
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _process_dataset(self) -> Dataset:
        """Process the dataset for training"""
        logger.info("Processing dataset for training")
        
        def format_chat_template(example):
            """Format example using chat template"""
            if self.use_chat_template:
                try:
                    # Handle different input formats
                    if "messages" in example:
                        messages = example["messages"]
                    elif "conversations" in example:
                        messages = example["conversations"]
                    elif "user" in example and "assistant" in example:
                        messages = [
                            {"role": "user", "content": example["user"]},
                            {"role": "assistant", "content": example["assistant"]}
                        ]
                    elif "instruction" in example and "output" in example:
                        messages = [
                            {"role": "user", "content": example["instruction"]},
                            {"role": "assistant", "content": example["output"]}
                        ]
                    elif "prompt" in example and "completion" in example:
                        messages = [
                            {"role": "user", "content": example["prompt"]},
                            {"role": "assistant", "content": example["completion"]}
                        ]
                    elif "prompt" in example and "accepted_completion" in example:
                        messages = [
                            {"role": "user", "content": example["prompt"]},
                            {"role": "assistant", "content": example["accepted_completion"]}
                        ]
                    elif "prompt" in example and "completion" in example:
                        messages = [
                            {"role": "user", "content": example["prompt"]},
                            {"role": "assistant", "content": example["completion"]}
                        ]
                    else:
                        # Fallback: treat as plain text
                        return {"text": str(example)}
                    
                    # Add system message with /no_think tag if not present
                    if messages and messages[0]["role"] != "system":
                        # Check if we should add /no_think tag based on configuration
                        system_content = "You are a helpful assistant."
                        if hasattr(self, 'chat_template_kwargs') and self.chat_template_kwargs:
                            # If no_think_system_message is True, add /no_think tag
                            if self.chat_template_kwargs.get("no_think_system_message") == True:
                                system_content = "You are a helpful assistant. /no_think"
                        
                        messages.insert(0, {"role": "system", "content": system_content})
                    
                    # Apply chat template
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=self.chat_template_kwargs.get("add_generation_prompt", True)
                    )
                    return {"text": text}
                except Exception as e:
                    logger.warning(f"Failed to apply chat template: {e}")
                    # Fallback to plain text
                    return {"text": str(example)}
            else:
                # Use plain text
                if "text" in example:
                    return {"text": example["text"]}
                else:
                    return {"text": str(example)}
        
        def tokenize_function(examples):
            """Tokenize the examples"""
            # Tokenize the texts with fixed length
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,  # Enable padding during tokenization
                max_length=self.max_seq_length,
                return_overflowing_tokens=False,  # Don't return overflowing tokens
                return_length=True,
            )
            
            # Calculate input length
            input_length = [len(x) for x in tokenized["input_ids"]]
            
            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["labels"],
                "length": input_length,
            }
        
        # Process the dataset - handle both single dataset and dictionary of splits
        if isinstance(self.dataset, dict):
            # Process each split individually
            processed_dataset = {}
            for split_name, split_dataset in self.dataset.items():
                logger.info(f"Processing {split_name} split...")
                
                # Format the split
                processed_split = split_dataset.map(
                    format_chat_template,
                    remove_columns=split_dataset.column_names,
                    desc=f"Formatting {split_name} dataset"
                )
                
                # Tokenize the split
                tokenized_split = processed_split.map(
                    tokenize_function,
                    remove_columns=processed_split.column_names,
                    desc=f"Tokenizing {split_name} dataset",
                    batched=True,
                )
                
                processed_dataset[split_name] = tokenized_split
        else:
            # Single dataset
            processed_dataset = self.dataset.map(
                format_chat_template,
                remove_columns=self.dataset.column_names,
                desc="Formatting dataset"
            )
            
            # Tokenize the dataset
            processed_dataset = processed_dataset.map(
                tokenize_function,
                remove_columns=processed_dataset.column_names,
                desc="Tokenizing dataset",
                batched=True,
            )
        
        # Log processing results
        if isinstance(processed_dataset, dict):
            logger.info(f"Dataset processed. Train samples: {len(processed_dataset['train'])}")
            if "validation" in processed_dataset:
                logger.info(f"Validation samples: {len(processed_dataset['validation'])}")
            if "test" in processed_dataset:
                logger.info(f"Test samples: {len(processed_dataset['test'])}")
        else:
            logger.info(f"Dataset processed. Samples: {len(processed_dataset)}")
        
        return processed_dataset
    
    def get_train_dataset(self) -> Dataset:
        """Get training dataset"""
        return self.processed_dataset["train"]
    
    def get_eval_dataset(self) -> Optional[Dataset]:
        """Get evaluation dataset if available"""
        if "validation" in self.processed_dataset:
            return self.processed_dataset["validation"]
        elif "test" in self.processed_dataset:
            return self.processed_dataset["test"]
        else:
            return None
    
    def get_data_collator(self):
        """Get data collator for training"""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
            return_tensors="pt",  # Ensure we return PyTorch tensors
        )

def create_sample_dataset(output_path: str = "my_dataset"):
    """Create a sample dataset for testing"""
    os.makedirs(output_path, exist_ok=True)
    
    # Sample conversations
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain gravity in simple terms."},
                {"role": "assistant", "content": "Gravity is the force that pulls objects toward each other, like how the Earth pulls things down to the ground."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I make a cup of coffee?"},
                {"role": "assistant", "content": "To make a cup of coffee: 1) Boil water, 2) Add coffee grounds to a filter, 3) Pour hot water over the grounds, 4) Let it brew for a few minutes, 5) Enjoy!"}
            ]
        }
    ]
    
    # Split into train/validation
    train_data = conversations[:2]
    validation_data = conversations[2:]
    
    # Save to files
    with open(os.path.join(output_path, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_path, "validation.json"), 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample dataset created in {output_path}")
    return output_path 