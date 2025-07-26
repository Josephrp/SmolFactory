"""
SmolLM3 DPO Training Configuration
Based on nanoGPT structure but adapted for SmolLM3 DPO training
"""

import os
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3DPOConfig(SmolLM3Config):
    """Configuration for SmolLM3 DPO fine-tuning"""
    
    # Trainer type selection
    trainer_type: str = "dpo"  # Override default to use DPO trainer
    
    # DPO-specific configuration
    beta: float = 0.1
    max_prompt_length: int = 2048
    max_length: int = 4096
    
    # DPO training configuration
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"  # "sigmoid" or "hinge"
    dpo_alpha: float = 0.5
    
    # Reference model configuration
    ref_model_name: Optional[str] = None  # If None, will use the same as model_name
    ref_model_peft_config: Optional[dict] = None
    
    # Preference dataset configuration
    preference_dataset_format: str = "dpo"  # "dpo", "rlhf", "custom"
    preference_dataset_text_field: str = "text"
    preference_dataset_prompt_field: str = "prompt"
    preference_dataset_chosen_field: str = "chosen"
    preference_dataset_rejected_field: str = "rejected"
    
    # DPO training arguments
    dpo_gradient_checkpointing: bool = True
    dpo_gradient_checkpointing_kwargs: dict = None
    dpo_precompute_ref_log_probs: bool = False
    dpo_peft_config: Optional[dict] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set default values for DPO-specific settings
        if self.ref_model_name is None:
            self.ref_model_name = self.model_name
        
        if self.dpo_gradient_checkpointing_kwargs is None:
            self.dpo_gradient_checkpointing_kwargs = {
                "use_reentrant": False
            }
        
        if self.dpo_peft_config is None:
            self.dpo_peft_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        
        # Validate DPO configuration
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        
        if self.max_prompt_length > self.max_seq_length:
            raise ValueError("max_prompt_length cannot exceed max_seq_length")
        
        if self.max_length > self.max_seq_length:
            raise ValueError("max_length cannot exceed max_seq_length")

def get_dpo_config(config_path: str) -> SmolLM3DPOConfig:
    """Load DPO configuration from file or return default"""
    if os.path.exists(config_path):
        # Load from file if it exists
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'config'):
            return config_module.config
        else:
            # Try to find a config class
            for attr_name in dir(config_module):
                attr = getattr(config_module, attr_name)
                if isinstance(attr, SmolLM3DPOConfig):
                    return attr
    
    # Return default configuration
    return SmolLM3DPOConfig()

# Default DPO configuration instance
config = SmolLM3DPOConfig() 