"""
SmolLM3 Training Configuration for OpenHermes-FR Dataset - Multiple Passes
Optimized for A100 GPUs with multiple passes (3-5 epochs) on 800k+ datapoints
"""

import os
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3ConfigOpenHermesFRMultiplePasses(SmolLM3Config):
    """Configuration for SmolLM3 fine-tuning with multiple passes on OpenHermes-FR dataset"""
    
    # Model configuration - optimized for A100
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 8192  # Increased for better context understanding
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False  # Disabled for A100 efficiency
    
    # Training configuration - Multiple passes optimized
    batch_size: int = 6  # Slightly smaller for stability during long training
    gradient_accumulation_steps: int = 20  # Effective batch size = 6 * 20 = 120
    learning_rate: float = 3e-6  # Conservative LR for multiple passes
    weight_decay: float = 0.01
    warmup_steps: int = 2000  # Longer warmup for multiple passes
    max_iters: int = 25000  # 4 passes on 800k dataset (25k steps)
    eval_interval: int = 1000  # Less frequent evaluation
    log_interval: int = 50  # Less frequent logging
    save_interval: int = 2000  # Less frequent saving
    
    # Optimizer configuration - stability focused
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.999  # Higher beta2 for stability
    eps: float = 1e-8
    
    # Scheduler configuration - longer training with multiple passes
    scheduler: str = "cosine"
    min_lr: float = 3e-7  # Lower min LR
    
    # Mixed precision - A100 optimized
    fp16: bool = False  # Use bf16 for A100
    bf16: bool = True  # Better for A100
    
    # DDP configuration
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # Logging and saving - optimized for long training
    save_steps: int = 2000
    eval_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: Optional[int] = 8  # Keep more checkpoints for long training
    
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
    # With 800k datapoints and effective batch size of 120:
    # Steps per epoch = 800,000 / 120 = 6,667 steps
    # For 3 passes: 6,667 * 3 = 20,000 steps
    # For 4 passes: 6,667 * 4 = 26,667 steps
    # For 5 passes: 6,667 * 5 = 33,333 steps
    # Current max_iters = 25,000 (about 3.75 passes)
    
    def __post_init__(self):
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "enable_thinking": False,
                "add_generation_prompt": True
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
        
        print(f"=== Multiple Passes Training Configuration ===")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print(f"Training for ~{epochs_for_max_iters:.1f} epochs")
        print(f"Total training steps: {self.max_iters}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'bf16' if self.bf16 else 'fp16'}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Save interval: {self.save_interval}")
        print("=" * 50)
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "smollm3_openhermes_fr_multiple_passes"

def get_config(config_path: str) -> SmolLM3ConfigOpenHermesFRMultiplePasses:
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
                if isinstance(attr, SmolLM3ConfigOpenHermesFRMultiplePasses):
                    return attr
    
    # Return default configuration
    return SmolLM3ConfigOpenHermesFRMultiplePasses()

# Default configuration instance
config = SmolLM3ConfigOpenHermesFRMultiplePasses() 