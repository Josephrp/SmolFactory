#!/usr/bin/env python3
"""
GPT-OSS Training Script
Specialized training script for OpenAI's GPT-OSS models
Based on the GPT-OSS fine-tuning tutorial
"""

import os
import sys
import argparse
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from pathlib import Path

# Ensure project root and config package are importable for configs that do `from config...` imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
config_dir = project_root / "config"
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

def load_gpt_oss_model_and_tokenizer(config):
    """Load GPT-OSS model and tokenizer with proper configuration"""
    
    print("Loading GPT-OSS tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("Loading GPT-OSS model with quantization...")
    
    # Import quantization config
    from transformers import BitsAndBytesConfig
    
    # Set up quantization config based on config
    if config.quantization_config and config.quantization_config.get("load_in_4bit"):
        # Use BitsAndBytesConfig for 4-bit quantization (memory optimized)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif config.quantization_config and config.quantization_config.get("dequantize"):
        # Try to use Mxfp4Config if available (as per tutorial)
        try:
            from transformers import Mxfp4Config
            quantization_config = Mxfp4Config(dequantize=True)
        except ImportError:
            # Fallback to no quantization if Mxfp4Config not available
            print("Warning: Mxfp4Config not available, using no quantization")
            quantization_config = None
    else:
        # No quantization
        quantization_config = None
    
    # Model kwargs as per tutorial
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "device_map": "auto",
    }
    
    # Only add quantization_config if it's not None
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    return model, tokenizer

def setup_lora_for_gpt_oss(model, config):
    """Setup LoRA for GPT-OSS model"""
    
    print("Setting up LoRA for GPT-OSS...")
    
    # LoRA configuration as per tutorial
    lora_config = LoraConfig(
        r=config.lora_config.get("r", 8) if config.lora_config else 8,
        lora_alpha=config.lora_config.get("lora_alpha", 16) if config.lora_config else 16,
        target_modules=config.lora_config.get("target_modules", "all-linear") if config.lora_config else "all-linear",
        target_parameters=config.lora_config.get("target_parameters", [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ]) if config.lora_config else [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model

def load_dataset_from_config(config):
    """Load dataset based on configuration"""
    
    dataset_name = getattr(config, 'dataset_name', 'HuggingFaceH4/Multilingual-Thinking')
    dataset_split = getattr(config, 'dataset_split', 'train')
    dataset_config = getattr(config, 'dataset_config', None)
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Dataset split: {dataset_split}")
    if dataset_config:
        print(f"Dataset config: {dataset_config}")
    
    # Load the dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    print(f"Original dataset size: {len(dataset)} examples")
    
    # Apply filtering based on configuration
    dataset = apply_dataset_filtering(dataset, config)
    
    # Apply dataset processing based on format
    dataset = process_dataset_format(dataset, config)
    
    print(f"Final dataset size: {len(dataset)} examples")
    
    return dataset

def apply_dataset_filtering(dataset, config):
    """Apply filtering based on configuration"""
    
    # Filter bad entries if specified
    if getattr(config, 'filter_bad_entries', False):
        bad_entry_field = getattr(config, 'bad_entry_field', 'bad_entry')
        bad_prompt_field = getattr(config, 'bad_prompt_field', 'bad_prompt_detected')
        bad_response_field = getattr(config, 'bad_response_field', 'bad_response_detected')
        
        original_size = len(dataset)
        
        # Filter out bad entries
        if bad_entry_field in dataset.column_names:
            dataset = dataset.filter(lambda x: not x.get(bad_entry_field, False))
            print(f"Filtered {original_size - len(dataset)} bad entries")
        
        # Filter out bad prompts
        if bad_prompt_field in dataset.column_names:
            dataset = dataset.filter(lambda x: not x.get(bad_prompt_field, False))
            print(f"Filtered bad prompts, remaining: {len(dataset)} examples")
        
        # Filter out bad responses
        if bad_response_field in dataset.column_names:
            dataset = dataset.filter(lambda x: not x.get(bad_response_field, False))
            print(f"Filtered bad responses, remaining: {len(dataset)} examples")
    
    # Apply length filtering
    min_length = getattr(config, 'min_length', 10)
    max_length = getattr(config, 'max_length', None)
    
    input_field = getattr(config, 'input_field', 'prompt')
    target_field = getattr(config, 'target_field', 'accepted_completion')
    
    if min_length > 0 or max_length:
        def length_filter(example):
            input_len = len(example.get(input_field, ''))
            target_len = len(example.get(target_field, ''))
            total_len = input_len + target_len
            
            if total_len < min_length:
                return False
            if max_length and total_len > max_length:
                return False
            return True
        
        original_size = len(dataset)
        dataset = dataset.filter(length_filter)
        print(f"Length filtering: {original_size} -> {len(dataset)} examples")
    
    # Apply sampling if specified
    max_samples = getattr(config, 'max_samples', None)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled {max_samples} examples from dataset")
    
    return dataset

def format_gpt_oss_harmony(prompt, completion, add_eos_token=True):
    """
    Format data for GPT-OSS Harmony format following the exact template structure.
    Based on: https://huggingface.co/openai/gpt-oss-20b/raw/main/chat_template.jinja
    """
    # GPT-OSS Harmony format structure (exact template compliance)
    # User message: <|start|>user<|message|>content<|end|>
    # Assistant message: <|start|>assistant<|channel|>final<|message|>content<|end|> (inference)
    # Assistant message: <|start|>assistant<|channel|>final<|message|>content<|return|> (training)
    
    harmony_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}"
    
    if add_eos_token:
        # Use <|return|> for training as per template specification
        # This indicates the end of generation in training
        harmony_text += "<|return|>"
    else:
        # Use <|end|> for inference
        harmony_text += "<|end|>"
    
    return harmony_text

def process_dataset_format(dataset, config):
    """Process dataset based on format configuration with exact GPT-OSS Harmony compliance"""
    
    dataset_format = getattr(config, 'dataset_format', 'openhermes_fr')
    input_field = getattr(config, 'input_field', 'prompt')
    target_field = getattr(config, 'target_field', 'accepted_completion')
    concatenate_fields = getattr(config, 'concatenate_fields', True)
    field_separator = getattr(config, 'field_separator', '\n\n### Response:\n')
    add_eos_token = getattr(config, 'add_eos_token', True)
    use_harmony_format = getattr(config, 'use_harmony_format', True)
    
    print(f"Processing dataset format: {dataset_format}")
    print(f"Input field: {input_field}, Target field: {target_field}")
    print(f"GPT-OSS Harmony Format: {'Enabled' if use_harmony_format else 'Disabled'}")
    
    if dataset_format == "openhermes_fr":
        # Process OpenHermes-FR format: prompt + accepted_completion
        def format_openhermes_fr(example):
            prompt = example.get(input_field, '')
            completion = example.get(target_field, '')
            
            if concatenate_fields:
                if use_harmony_format:
                    # Use exact GPT-OSS Harmony format from template
                    text = format_gpt_oss_harmony(prompt, completion, add_eos_token)
                else:
                    # Fallback to standard format with separator
                    text = prompt + field_separator + completion
                    if add_eos_token:
                        text += "</s>"
                
                return {"text": text}
            else:
                # Keep separate for more advanced training setups
                return {
                    "input": prompt,
                    "output": completion
                }
        
        dataset = dataset.map(format_openhermes_fr, remove_columns=dataset.column_names)
        
    elif dataset_format == "messages":
        # Process messages format (like HuggingFaceH4/Multilingual-Thinking)
        def format_messages(example):
            messages = example.get(input_field, [])
            
            if use_harmony_format and len(messages) >= 2:
                # Extract user and assistant messages for harmony format
                user_message = ""
                assistant_message = ""
                
                for message in messages:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        user_message = content
                    elif role == "assistant":
                        assistant_message = content
                
                if user_message and assistant_message:
                    # Use GPT-OSS Harmony format
                    text = format_gpt_oss_harmony(user_message, assistant_message, add_eos_token)
                else:
                    # Fallback to simple concatenation
                    text = ""
                    for message in messages:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        text += f"{role}: {content}\n"
                    if add_eos_token:
                        text += "</s>"
            else:
                # Standard format - convert messages to simple text
                text = ""
                for message in messages:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    text += f"{role}: {content}\n"
                if add_eos_token:
                    text += "</s>"
            
            return {"text": text}
        
        dataset = dataset.map(format_messages, remove_columns=dataset.column_names)
        
    elif dataset_format == "text":
        # Process plain text format
        text_field = input_field
        def format_text(example):
            text = example.get(text_field, '')
            if add_eos_token:
                text += "</s>"
            return {"text": text}
        
        dataset = dataset.map(format_text, remove_columns=dataset.column_names)
    
    elif dataset_format == "custom":
        # Custom format - user handles this in their config
        print("Using custom dataset format - no automatic processing")
    
    return dataset

def setup_trackio_tracking(config):
    """Setup Trackio tracking if enabled"""
    
    if not config.enable_tracking or not config.trackio_url:
        print("Trackio tracking disabled or URL not provided")
        return None
    
    print(f"Setting up Trackio tracking: {config.trackio_url}")
    
    # Import the correct TrackioAPIClient
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trackio_tonic'))
    from trackio_api_client import TrackioAPIClient
    
    # Initialize Trackio client using the correct API
    trackio_client = TrackioAPIClient(
        space_id=config.trackio_url,
        hf_token=config.trackio_token
    )
    
    return trackio_client

def create_sft_config(config, output_dir):
    """Create enhanced SFTConfig for GPT-OSS training"""
    
    print("Creating enhanced SFT configuration...")
    
    # Helper coercion utilities to guarantee numeric types
    def _as_int(value, default):
        if value is None:
            return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value, default):
        if value is None:
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    # Extract training parameters from config with enhanced defaults and coercion
    num_train_epochs = _as_float(getattr(config, 'num_train_epochs', 1.0), 1.0)
    # Transformers expects max_steps default -1 (disabled). Some code compares > 0
    raw_max_steps = getattr(config, 'max_steps', None)
    max_steps = _as_int(raw_max_steps if raw_max_steps is not None else -1, -1)
    warmup_ratio = _as_float(getattr(config, 'warmup_ratio', 0.03), 0.03)
    # Ensure warmup_steps is an int; default 0 to avoid None comparisons in schedulers
    warmup_steps = _as_int(getattr(config, 'warmup_steps', 0), 0)
    
    # Learning rate configuration
    learning_rate = _as_float(getattr(config, 'learning_rate', 2e-4), 2e-4)
    lr_scheduler_type = getattr(config, 'scheduler', 'cosine_with_min_lr')
    
    # Batch configuration
    per_device_train_batch_size = _as_int(getattr(config, 'batch_size', 2), 2)
    per_device_eval_batch_size = _as_int(getattr(config, 'eval_batch_size', per_device_train_batch_size), per_device_train_batch_size)
    gradient_accumulation_steps = _as_int(getattr(config, 'gradient_accumulation_steps', 1), 1)
    
    # Evaluation and logging
    eval_strategy = getattr(config, 'eval_strategy', 'steps')
    eval_steps = _as_int(getattr(config, 'eval_steps', 100), 100)
    eval_accumulation_steps = _as_int(getattr(config, 'eval_accumulation_steps', 1), 1)
    logging_steps = _as_int(getattr(config, 'logging_steps', 10), 10)
    
    # Saving configuration
    save_strategy = getattr(config, 'save_strategy', 'steps')
    save_steps = _as_int(getattr(config, 'save_steps', 500), 500)
    save_total_limit = _as_int(getattr(config, 'save_total_limit', 3), 3)
    
    # Mixed precision
    fp16 = bool(getattr(config, 'fp16', False))
    bf16 = bool(getattr(config, 'bf16', True))
    tf32 = bool(getattr(config, 'tf32', False))
    
    # Regularization
    weight_decay = _as_float(getattr(config, 'weight_decay', 0.01), 0.01)
    max_grad_norm = _as_float(getattr(config, 'max_grad_norm', 1.0), 1.0)
    
    # HuggingFace Hub integration
    push_to_hub = getattr(config, 'push_to_hub', False)
    
    print(f"  • Epochs: {num_train_epochs}")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Batch size: {per_device_train_batch_size}")
    print(f"  • Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  • Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    
    # Build kwargs dynamically to be compatible across transformers versions
    ta_kwargs = {
        # Training duration
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        # Learning rate
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        # Batch configuration
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        # Model configuration
        "gradient_checkpointing": getattr(config, 'use_gradient_checkpointing', True),
        # Mixed precision
        "fp16": fp16,
        "bf16": bf16,
        # Some versions support tf32
        "tf32": tf32 if 'tf32' in TrainingArguments.__init__.__code__.co_varnames else None,
        # Regularization
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        # Evaluation (name may vary across versions)
        "evaluation_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "eval_accumulation_steps": eval_accumulation_steps,
        # Logging
        "logging_steps": logging_steps,
        # Saving
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        # Output
        "output_dir": output_dir,
        # Data loading
        "dataloader_num_workers": _as_int(getattr(config, 'dataloader_num_workers', 4), 4),
        "dataloader_pin_memory": getattr(config, 'dataloader_pin_memory', True),
        # Optional in some versions
        "dataloader_prefetch_factor": _as_int(getattr(config, 'dataloader_prefetch_factor', 2), 2),
        # Performance
        "group_by_length": getattr(config, 'group_by_length', True),
        "remove_unused_columns": getattr(config, 'remove_unused_columns', True),
        # HuggingFace Hub
        "push_to_hub": push_to_hub,
        # Monitoring
        "report_to": ("trackio" if getattr(config, 'enable_tracking', False) else None),
    }

    # Drop any None-valued kwargs
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if v is not None}

    # Adapt to transformers versions where 'evaluation_strategy' was renamed
    try:
        ta_sig = inspect.signature(TrainingArguments.__init__)
        param_names = set(ta_sig.parameters.keys())
    except Exception:
        param_names = set()

    if "evaluation_strategy" not in param_names and "eval_strategy" in param_names:
        # Move value to 'eval_strategy'
        ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" not in param_names:
        # If neither is supported, drop it
        ta_kwargs.pop("evaluation_strategy", None)

    # Remove any kwargs not supported by current transformers version
    if param_names:
        unsupported = [k for k in ta_kwargs.keys() if k not in param_names]
        for k in unsupported:
            ta_kwargs.pop(k, None)

    sft_config = TrainingArguments(**ta_kwargs)
    
    return sft_config

def train_gpt_oss(config_path, experiment_name, output_dir, trackio_url, trainer_type="sft"):
    """Main training function for GPT-OSS"""
    
    print("=== GPT-OSS Training Pipeline ===")
    print(f"Config: {config_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Trackio: {trackio_url}")
    print(f"Trainer: {trainer_type}")
    
    # Load configuration
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'config'):
            config = config_module.config
        else:
            # Try to find a config class
            for attr_name in dir(config_module):
                attr = getattr(config_module, attr_name)
                if hasattr(attr, 'model_name') and ('gpt_oss' in attr.model_name.lower() or 'GPTOSS' in attr_name):
                    config = attr
                    break
            else:
                raise ValueError(f"No GPT-OSS configuration found in {config_path}")
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Update config with runtime parameters
    config.experiment_name = experiment_name
    config.trackio_url = trackio_url
    config.trainer_type = trainer_type
    
    # Load model and tokenizer
    model, tokenizer = load_gpt_oss_model_and_tokenizer(config)
    
    # Setup LoRA
    peft_model = setup_lora_for_gpt_oss(model, config)
    
    # Load dataset
    dataset = load_dataset_from_config(config)
    
    # Setup Trackio tracking
    trackio_client = setup_trackio_tracking(config)
    
    # Create SFT configuration
    sft_config = create_sft_config(config, output_dir)
    
    # Create trainer
    print("Creating SFT trainer...")
    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=getattr(config, 'max_seq_length', 2048),
    )
    
    # Start training
    print("Starting GPT-OSS training...")
    trainer.train()
    
    # Save model
    print("Saving trained model...")
    trainer.save_model(output_dir)
    
    # Push to hub if enabled
    if sft_config.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")
    
    print("GPT-OSS training completed successfully!")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Training Script")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--trackio-url", help="Trackio URL for monitoring")
    parser.add_argument("--trainer-type", default="sft", choices=["sft", "dpo"], help="Trainer type")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        train_gpt_oss(
            config_path=args.config,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            trackio_url=args.trackio_url,
            trainer_type=args.trainer_type
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 