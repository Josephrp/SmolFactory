"""
SmolLM3 Model Wrapper
Handles model loading, tokenizer, and training setup
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SmolLM3Model:
    """Wrapper for SmolLM3 model and tokenizer"""
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        max_seq_length: int = 4096,
        config: Optional[Any] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.config = config
        
        # Set device and dtype
        if torch_dtype is None:
            if torch.cuda.is_available():
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
            
        if device_map is None:
            self.device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            self.device_map = device_map
        
        # Load tokenizer and model
        self._load_tokenizer()
        self._load_model()
        
    def _load_tokenizer(self):
        """Load the tokenizer"""
        logger.info(f"Loading tokenizer from {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self):
        """Load the model"""
        logger.info(f"Loading model from {self.model_name}")
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Update configuration if needed
            if hasattr(model_config, 'max_position_embeddings'):
                model_config.max_position_embeddings = self.max_seq_length
            
            # SmolLM3-specific optimizations for long context
            if hasattr(model_config, 'rope_scaling'):
                # Enable YaRN scaling for long context
                model_config.rope_scaling = {"type": "yarn", "factor": 2.0}
                logger.info("Enabled YaRN scaling for long context")
            
            # Load model
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
                "trust_remote_code": True
            }
            
            # Only add flash attention if the model supports it
            if hasattr(self.config, 'use_flash_attention') and self.config.use_flash_attention:
                try:
                    # Test if the model supports flash attention
                    test_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                    if hasattr(test_config, 'use_flash_attention_2'):
                        model_kwargs["use_flash_attention_2"] = True
                        logger.info("Enabled Flash Attention 2 for better long context performance")
                except:
                    # If flash attention is not supported, skip it
                    pass
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=model_config,
                **model_kwargs
            )
            
            # Enable gradient checkpointing if specified
            if self.config and self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            logger.info(f"Max sequence length: {self.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_training_arguments(self, output_dir: str, **kwargs) -> TrainingArguments:
        """Get training arguments for the Trainer"""
        if self.config is None:
            raise ValueError("Config is required to get training arguments")
        
        # Merge config with kwargs
        training_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self.config.max_iters,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "logging_steps": self.config.logging_steps,
            "save_total_limit": self.config.save_total_limit,
            "eval_strategy": self.config.eval_strategy,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            # Only enable DDP if multiple GPUs are available
            "ddp_backend": self.config.ddp_backend if torch.cuda.device_count() > 1 else None,
            "ddp_find_unused_parameters": self.config.ddp_find_unused_parameters if torch.cuda.device_count() > 1 else False,
            "report_to": None,  # Disable external logging - use None instead of "none"
            "remove_unused_columns": False,
            "dataloader_pin_memory": getattr(self.config, 'dataloader_pin_memory', False),
            "group_by_length": getattr(self.config, 'group_by_length', True),
            "length_column_name": "length",
            "ignore_data_skip": False,
            "seed": 42,
            "data_seed": 42,
            "dataloader_num_workers": getattr(self.config, 'dataloader_num_workers', 4),
            "max_grad_norm": getattr(self.config, 'max_grad_norm', 1.0),
            "optim": self.config.optimizer,
            "lr_scheduler_type": self.config.scheduler,
            "warmup_ratio": 0.1,
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "prediction_loss_only": True,
        }
        
        # Ensure boolean parameters are properly typed
        if "dataloader_pin_memory" in training_args:
            training_args["dataloader_pin_memory"] = bool(training_args["dataloader_pin_memory"])
        if "group_by_length" in training_args:
            training_args["group_by_length"] = bool(training_args["group_by_length"])
        if "prediction_loss_only" in training_args:
            training_args["prediction_loss_only"] = bool(training_args["prediction_loss_only"])
        if "ignore_data_skip" in training_args:
            training_args["ignore_data_skip"] = bool(training_args["ignore_data_skip"])
        if "remove_unused_columns" in training_args:
            training_args["remove_unused_columns"] = bool(training_args["remove_unused_columns"])
        if "ddp_find_unused_parameters" in training_args:
            training_args["ddp_find_unused_parameters"] = bool(training_args["ddp_find_unused_parameters"])
        if "fp16" in training_args:
            training_args["fp16"] = bool(training_args["fp16"])
        if "bf16" in training_args:
            training_args["bf16"] = bool(training_args["bf16"])
        if "load_best_model_at_end" in training_args:
            training_args["load_best_model_at_end"] = bool(training_args["load_best_model_at_end"])
        if "greater_is_better" in training_args:
            training_args["greater_is_better"] = bool(training_args["greater_is_better"])
        
        # Add dataloader_prefetch_factor if it exists in config
        if hasattr(self.config, 'dataloader_prefetch_factor'):
            try:
                # Test if the parameter is supported by creating a dummy TrainingArguments
                test_args = TrainingArguments(output_dir="/tmp/test", dataloader_prefetch_factor=2)
                training_args["dataloader_prefetch_factor"] = self.config.dataloader_prefetch_factor
                logger.info(f"Added dataloader_prefetch_factor: {self.config.dataloader_prefetch_factor}")
            except Exception as e:
                logger.warning(f"dataloader_prefetch_factor not supported in this transformers version: {e}")
                # Remove the parameter if it's not supported
                if "dataloader_prefetch_factor" in training_args:
                    del training_args["dataloader_prefetch_factor"]
        
        # Override with kwargs
        training_args.update(kwargs)
        
        # Clean up any None values that might cause issues
        training_args = {k: v for k, v in training_args.items() if v is not None}
        
        return TrainingArguments(**training_args)
    
    def save_pretrained(self, path: str):
        """Save model and tokenizer"""
        logger.info(f"Saving model and tokenizer to {path}")
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save configuration
        if self.config:
            import json
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                          if not k.startswith('_')}
            with open(os.path.join(path, 'training_config.json'), 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True
            )
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise 