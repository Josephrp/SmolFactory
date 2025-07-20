"""
SmolLM3 Training Configuration for OpenHermes-FR Dataset - A100 Large Scale
Optimized for A100 GPUs with large batch sizes and multiple passes on 800k+ datapoints
"""

import os
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3ConfigOpenHermesFRA100Large(SmolLM3Config):
    """Configuration for SmolLM3 fine-tuning on OpenHermes-FR dataset - A100 Large Scale"""
    
    # Model configuration - optimized for A100
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 8192  # Increased for better context understanding
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False  # Disabled for A100 efficiency
    
    # Training configuration - A100 optimized with large batch sizes
    batch_size: int = 8  # Large batch size for A100 (80GB VRAM)
    gradient_accumulation_steps: int = 16  # Effective batch size = 8 * 16 = 128
    learning_rate: float = 5e-6  # Lower LR for large effective batch size
    weight_decay: float = 0.01
    warmup_steps: int = 1000  # More warmup for large dataset
    max_iters: int = 8000  # Multiple passes on 800k dataset
    eval_interval: int = 500  # Less frequent evaluation
    log_interval: int = 25  # Less frequent logging
    save_interval: int = 1000  # Less frequent saving
    
    # Optimizer configuration - optimized for large batches
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.999  # Higher beta2 for stability with large batches
    eps: float = 1e-8
    
    # Scheduler configuration - longer training
    scheduler: str = "cosine"
    min_lr: float = 5e-7  # Lower min LR
    
    # Mixed precision - A100 optimized
    fp16: bool = False  # Use bf16 for A100
    bf16: bool = True  # Better for A100
    
    # DDP configuration
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # Logging and saving - optimized for long training
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: Optional[int] = 5  # Keep more checkpoints
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # OpenHermes-FR Dataset configuration
    dataset_name: str = "legmlai/openhermes-fr"
    dataset_split: str = "train"
    input_field: str = "prompt"
    target_field: str = "accepted_completion"
    filter_bad_entries: bool = True
    bad_entry_field: str = "bad_entry"
    
    # Data configuration (not used for HF datasets but kept for compatibility)
    data_dir: str = None
    train_file: str = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Chat template configuration
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
    
    # Additional A100 optimizations
    dataloader_num_workers: int = 8  # More workers for faster data loading
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Memory optimizations
    max_grad_norm: float = 1.0  # Gradient clipping
    group_by_length: bool = True  # Group similar length sequences
    
    # Training duration calculations
    # With 800k datapoints and effective batch size of 128:
    # Steps per epoch = 800,000 / 128 = 6,250 steps
    # For 3 passes: 6,250 * 3 = 18,750 steps
    # For 5 passes: 6,250 * 5 = 31,250 steps
    # Current max_iters = 8,000 (about 1.3 passes)
    
    def __post_init__(self):
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "add_generation_prompt": True,
                "no_think_system_message": True  # Set to True to add /no_think tag
            }
        
        # Validate configuration
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        if self.max_seq_length > 131072:  # 128k limit
            raise ValueError("max_seq_length cannot exceed 131072")
        
        # Calculate training statistics
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = 800000 // effective_batch_size  # Approximate for 800k dataset
        epochs_for_max_iters = self.max_iters / steps_per_epoch
        
        print(f"=== A100 Large Scale Training Configuration ===")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print(f"Training for ~{epochs_for_max_iters:.1f} epochs")
        print(f"Total training steps: {self.max_iters}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'bf16' if self.bf16 else 'fp16'}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print("=" * 50)
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "smollm3_openhermes_fr_a100_large"

def get_config(config_path: str) -> SmolLM3ConfigOpenHermesFRA100Large:
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
                if isinstance(attr, SmolLM3ConfigOpenHermesFRA100Large):
                    return attr
    
    # Return default configuration
    return SmolLM3ConfigOpenHermesFRA100Large()

# Default configuration instance
config = SmolLM3ConfigOpenHermesFRA100Large() 