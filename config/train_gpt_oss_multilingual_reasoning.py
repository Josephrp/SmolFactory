"""
GPT-OSS Multilingual Reasoning Training Configuration
Based on OpenAI's GPT-OSS fine-tuning tutorial
Specialized for multilingual reasoning tasks
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTOSSMultilingualReasoningConfig:
    """Multilingual reasoning configuration for GPT-OSS fine-tuning"""
    
    # Trainer type selection
    trainer_type: str = "sft"  # "sft" or "dpo"
    
    # Model configuration - GPT-OSS specific for multilingual reasoning
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 2048  # Standard for reasoning tasks
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Training configuration - optimized for multilingual reasoning
    batch_size: int = 4  # Conservative for reasoning tasks
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4  # As per tutorial
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_iters: int = 1000  # 1 epoch on Multilingual-Thinking
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 500
    
    # Optimizer configuration
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration - as per tutorial
    scheduler: str = "cosine_with_min_lr"
    min_lr: float = 2e-5  # As per tutorial
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
    
    # Data configuration - Multilingual-Thinking specific
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
    # LoRA configuration for GPT-OSS - as per tutorial
    use_lora: bool = True
    lora_config: dict = None
    
    # Quantization for GPT-OSS (MXFP4) - as per tutorial
    use_quantization: bool = True
    quantization_config: dict = None
    
    # GPT-OSS specific model kwargs - as per tutorial
    model_kwargs: dict = None
    
    # Multilingual reasoning specific configurations
    # Generation parameters for multilingual reasoning
    generation_config: dict = None
    
    # Multilingual reasoning evaluation languages
    reasoning_languages: list = None
    
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
        
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": None,
                "top_k": None
            }
        
        if self.reasoning_languages is None:
            self.reasoning_languages = [
                "English", "Spanish", "French", "Italian", "German",
                "Chinese", "Hindi", "Japanese", "Korean", "Arabic"
            ]
        
        # Validate configuration
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        if self.max_seq_length > 131072:  # 128k limit
            raise ValueError("max_seq_length cannot exceed 131072")
        
        # Calculate training statistics for Multilingual-Thinking
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = 1000 // effective_batch_size  # Multilingual-Thinking has 1000 examples
        epochs_for_max_iters = self.max_iters / steps_per_epoch
        
        print(f"=== GPT-OSS Multilingual Reasoning Configuration ===")
        print(f"Dataset: {self.dataset_name}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print(f"Training for ~{epochs_for_max_iters:.1f} epochs")
        print(f"Total training steps: {self.max_iters}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'bf16' if self.bf16 else 'fp16'}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"LoRA rank: {self.lora_config['r']}")
        print(f"Supported reasoning languages: {len(self.reasoning_languages)}")
        print("=" * 50)
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "gpt_oss_multilingual_reasoning"

def get_config(config_path: str) -> GPTOSSMultilingualReasoningConfig:
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
                if isinstance(attr, GPTOSSMultilingualReasoningConfig):
                    return attr
    
    # Return default configuration
    return GPTOSSMultilingualReasoningConfig()

# Default configuration instance
config = GPTOSSMultilingualReasoningConfig() 