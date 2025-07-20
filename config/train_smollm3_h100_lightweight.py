"""
SmolLM3 H100 Lightweight Training Configuration
Optimized for rapid training on H100 with 80K Hermes-FR samples
"""

import os
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3ConfigH100Lightweight(SmolLM3Config):
    """Configuration for SmolLM3 fine-tuning on OpenHermes-FR dataset - H100 Lightweight"""
    
    # Model configuration - optimized for H100
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 8192  # Increased for better context understanding
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True  # Enabled for memory efficiency
    
    # Training configuration - H100 optimized for rapid training
    batch_size: int = 16  # Larger batch size for H100
    gradient_accumulation_steps: int = 4  # Reduced for faster updates
    learning_rate: float = 8e-6  # Slightly higher for rapid convergence
    weight_decay: float = 0.01
    warmup_steps: int = 50  # Reduced warmup for rapid training
    max_iters: int = None  # Will be calculated based on epochs
    eval_interval: int = 50  # More frequent evaluation
    log_interval: int = 5  # More frequent logging
    save_interval: int = 200  # More frequent saving
    
    # Optimizer configuration - optimized for rapid training
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration - faster learning
    scheduler: str = "cosine"
    min_lr: float = 2e-6  # Higher minimum LR
    
    # Mixed precision - Using fp16 for better compatibility
    # Note: bf16 can cause issues on some GPU setups, fp16 is more universally supported
    fp16: bool = False
    bf16: bool = True
    
    # Logging and saving - more frequent for rapid training
    save_steps: int = 200
    eval_steps: int = 50
    logging_steps: int = 5
    save_total_limit: Optional[int] = 2  # Keep fewer checkpoints
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # OpenHermes-FR Dataset configuration with sampling
    dataset_name: str = "legmlai/openhermes-fr"
    dataset_split: str = "train"
    input_field: str = "prompt"
    target_field: str = "completion"
    filter_bad_entries: bool = False
    bad_entry_field: str = "bad_entry"
    sample_size: int = 80000  # 80K samples for lightweight training
    sample_seed: int = 42  # For reproducibility
    
    # Data configuration (not used for HF datasets but kept for compatibility)
    data_dir: str = "my_dataset"
    train_file: str = "train.json"
    validation_file: Optional[str] = "validation.json"
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
    
    # HF Datasets configuration
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None
    
    # H100-specific optimizations
    dataloader_num_workers: int = 4  # Optimized for H100
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Memory optimizations for rapid training
    max_grad_norm: float = 1.0
    group_by_length: bool = True  # Group similar length sequences
    
    # Training duration calculations
    # With 80k datapoints and effective batch size of 64:
    # Steps per epoch = 80,000 / 64 = 1,250 steps
    # For 1 epoch: 1,250 steps
    # For 2 epochs: 2,500 steps
    
    def __post_init__(self):
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "enable_thinking": False,
                "add_generation_prompt": True,
                "no_think_system_message": True
            }
        
        # Validate configuration
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        if self.max_seq_length > 131072:  # 128k limit
            raise ValueError("max_seq_length cannot exceed 131072")
        
        # Calculate training statistics
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        steps_per_epoch = self.sample_size // effective_batch_size  # For 80k dataset
        epochs_for_max_iters = self.max_iters / steps_per_epoch if self.max_iters else 1
        
        print(f"=== H100 Lightweight Training Configuration ===")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Steps per epoch: ~{steps_per_epoch}")
        print(f"Training for ~{epochs_for_max_iters:.1f} epochs")
        print(f"Total training steps: {self.max_iters or 'auto'}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Mixed precision: {'fp16' if self.fp16 else 'bf16'}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Gradient checkpointing: {self.use_gradient_checkpointing}")
        print(f"Dataset sample size: {self.sample_size}")
        print("=" * 50)
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "smollm3_h100_lightweight"

def get_config(config_path: str) -> SmolLM3ConfigH100Lightweight:
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
                if isinstance(attr, SmolLM3ConfigH100Lightweight):
                    return attr
    
    # Return default configuration
    return SmolLM3ConfigH100Lightweight()

# Default configuration instance
config = SmolLM3ConfigH100Lightweight() 