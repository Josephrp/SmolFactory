#!/usr/bin/env python3
"""
GPT-OSS Training Script
Specialized training script for OpenAI's GPT-OSS models
Based on the GPT-OSS fine-tuning tutorial
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def load_gpt_oss_model_and_tokenizer(config):
    """Load GPT-OSS model and tokenizer with proper configuration"""
    
    print("Loading GPT-OSS tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("Loading GPT-OSS model with quantization...")
    
    # Import quantization config
    from transformers import BitsAndBytesConfig
    
    # Set up quantization config based on config
    if config.quantization_config and config.quantization_config.get("load_in_4bit"):
        # Use BitsAndBytesConfig for 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        # Use BitsAndBytesConfig as default (no quantization)
        quantization_config = None
    
    # Model kwargs as per tutorial
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "use_cache": False,
        "device_map": "auto",
    }
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    return model, tokenizer

def setup_lora_for_gpt_oss(model, config):
    """Setup LoRA for GPT-OSS model"""
    
    print("Setting up LoRA for GPT-OSS...")
    
    # LoRA configuration as per tutorial
    lora_config = LoraConfig(
        r=config.lora_config.get("r", 8),
        lora_alpha=config.lora_config.get("lora_alpha", 16),
        target_modules=config.lora_config.get("target_modules", "all-linear"),
        target_parameters=config.lora_config.get("target_parameters", [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ]),
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model

def load_multilingual_thinking_dataset():
    """Load the Multilingual-Thinking dataset"""
    
    print("Loading Multilingual-Thinking dataset...")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    print(f"Dataset loaded: {len(dataset)} examples")
    
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

def create_sft_config(config):
    """Create SFTConfig for GPT-OSS training"""
    
    print("Creating SFT configuration...")
    
    sft_config = SFTConfig(
        learning_rate=config.learning_rate,
        gradient_checkpointing=True,
        num_train_epochs=1,  # Single epoch as per tutorial
        logging_steps=config.logging_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_seq_length,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir="gpt-oss-20b-multilingual-reasoner",
        report_to="trackio" if config.enable_tracking else None,
        push_to_hub=True,
    )
    
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
    dataset = load_multilingual_thinking_dataset()
    
    # Setup Trackio tracking
    trackio_client = setup_trackio_tracking(config)
    
    # Create SFT configuration
    sft_config = create_sft_config(config)
    
    # Create trainer
    print("Creating SFT trainer...")
    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
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