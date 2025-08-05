"""
GPT-OSS H100 Optimized Training Configuration
Based on OpenAI's GPT-OSS fine-tuning tutorial
Optimized for H100 GPU with maximum performance
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTOSSH100OptimizedConfig:
    """H100-optimized configuration for GPT-OSS fine-tuning"""
    
    # Trainer type selection
    trainer_type: str = "sft"  # "sft" or "dpo"
    
    # Model configuration - GPT-OSS specific with H100 optimizations
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 4096  # Increased for H100
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Training configuration - H100 optimized
    batch_size: int = 8  # Larger batch size for H100
    gradient_accumulation_steps: int = 2  # Reduced for faster updates
    learning_rate: float = 3e-4  # Higher LR for H100
    weight_decay: float = 0.01
    warmup_steps: int = 50  # Reduced warmup for rapid training
    max_iters: int = 2000  # More iterations for H100
    eval_interval: int = 50  # More frequent evaluation
    log_interval: int = 5  # More frequent logging
    save_interval: int = 200  # More frequent saving
    
    # Optimizer configuration - H100 optimized
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration - faster learning
    scheduler: str = "cosine_with_min_lr"
    min_lr: float = 3e-5  # Higher min LR for H100
    lr_scheduler_kwargs: dict = None
    
    # Mixed precision - H100 optimized
    fp16: bool = False  # Use bf16 for H100
    bf16: bool = True
    
    # DDP configuration
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # Logging and saving - optimized for rapid training
    save_steps: int = 200
    eval_steps: int = 50
    logging_steps: int = 5
    save_total_limit: Optional[int] = 2  # Keep fewer checkpoints
    
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
    # LoRA configuration for GPT-OSS - H100 optimized
    use_lora: bool = True
    lora_config: dict = None
    
    # Quantization for GPT-OSS (MXFP4) - H100 optimized
    use_quantization: bool = True
    quantization_config: dict = None
    
    # GPT-OSS specific model kwargs - H100 optimized
    model_kwargs: dict = None
    
    # H100-specific optimizations
    dataloader_num_workers: int = 8  # More workers for H100
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4  # Increased prefetch
    
    # Memory optimizations for H100
    max_grad_norm: float = 1.0
    group_by_length: bool = True  # Group similar length sequences
    
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
                "r": 16,  # Increased for H100
                "lora_alpha": 32,  # Increased for H100
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
        
        # Calculate training statistics for H100
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = 1000 // effective_batch_size  # Approximate for Multilingual-Thinking
        epochs_for_max_iters = self.max_iters / steps_per_epoch
        
        print(f"=== GPT-OSS H100 Optimized Configuration ===")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print(f"Training for ~{epochs_for_max_iters:.1f} epochs")
        print(f"Total training steps: {self.max_iters}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'bf16' if self.bf16 else 'fp16'}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"LoRA rank: {self.lora_config['r']}")
        print(f"Data loader workers: {self.dataloader_num_workers}")
        print("=" * 50)
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "gpt_oss_h100_optimized"

def get_config(config_path: str) -> GPTOSSH100OptimizedConfig:
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
                if isinstance(attr, GPTOSSH100OptimizedConfig):
                    return attr
    
    # Return default configuration
    return GPTOSSH100OptimizedConfig()

# Default configuration instance
config = GPTOSSH100OptimizedConfig() 