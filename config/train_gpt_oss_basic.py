"""
GPT-OSS Basic Training Configuration
Based on OpenAI's GPT-OSS fine-tuning tutorial
Optimized for standard fine-tuning scenarios
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTOSSBasicConfig:
    """Basic configuration for GPT-OSS fine-tuning"""
    
    # Trainer type selection
    trainer_type: str = "sft"  # "sft" or "dpo"
    
    # Model configuration - GPT-OSS specific
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 2048  # GPT-OSS default
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Training configuration - optimized for GPT-OSS
    batch_size: int = 4  # Conservative for 20B model
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4  # Higher LR as per tutorial
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_iters: int = 1000
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 500
    
    # Optimizer configuration
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration
    scheduler: str = "cosine_with_min_lr"
    min_lr: float = 2e-5  # Higher min LR as per tutorial
    lr_scheduler_kwargs: dict = None
    
    # Mixed precision - GPT-OSS optimized
    fp16: bool = False  # Use bf16 for GPT-OSS
    bf16: bool = True
    
    # DDP configuration
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # Logging and saving
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: Optional[int] = 3
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Data configuration
    dataset_name: str = "HuggingFaceH4/Multilingual-Thinking"
    dataset_split: str = "train"
    input_field: str = "messages"  # GPT-OSS uses messages format
    target_field: str = None  # Not used for messages format
    filter_bad_entries: bool = False
    bad_entry_field: str = "bad_entry"
    
    # Chat template configuration - GPT-OSS specific
    use_chat_template: bool = True
    chat_template_kwargs: dict = None
    
    # Trackio monitoring configuration
    enable_tracking: bool = True
    trackio_url: Optional[str] = None
    trackio_token: Optional[str] = None
    log_artifacts: bool = True
    log_metrics: bool = True
    log_config: bool = True
    experiment_name: Optional[str] = None
    
    # HF Datasets configuration
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None
    
    # GPT-OSS specific configurations
    # LoRA configuration for GPT-OSS
    use_lora: bool = True
    lora_config: dict = None
    
    # Quantization for GPT-OSS (MXFP4)
    use_quantization: bool = True
    quantization_config: dict = None
    
    # GPT-OSS specific model kwargs
    model_kwargs: dict = None
    
    def __post_init__(self):
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False  # GPT-OSS specific
            }
        
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {
                "min_lr_rate": 0.1
            }
        
        if self.lora_config is None:
            self.lora_config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "target_parameters": [
                    "7.mlp.experts.gate_up_proj",
                    "7.mlp.experts.down_proj",
                    "15.mlp.experts.gate_up_proj",
                    "15.mlp.experts.down_proj",
                    "23.mlp.experts.gate_up_proj",
                    "23.mlp.experts.down_proj",
                ]
            }
        
        if self.quantization_config is None:
            self.quantization_config = {
                "dequantize": True
            }
        
        if self.model_kwargs is None:
            self.model_kwargs = {
                "attn_implementation": "eager",
                "torch_dtype": "auto",
                "use_cache": False,
                "device_map": "auto"
            }
        
        # Validate configuration
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        if self.max_seq_length > 131072:  # 128k limit
            raise ValueError("max_seq_length cannot exceed 131072")
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "gpt_oss_basic"

def get_config(config_path: str) -> GPTOSSBasicConfig:
    """Load configuration from file or return default"""
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
                if isinstance(attr, GPTOSSBasicConfig):
                    return attr
    
    # Return default configuration
    return GPTOSSBasicConfig()

# Default configuration instance
config = GPTOSSBasicConfig() 