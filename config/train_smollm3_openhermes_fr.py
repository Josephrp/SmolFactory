"""
SmolLM3 Training Configuration for OpenHermes-FR Dataset
Optimized for French instruction tuning using legmlai/openhermes-fr
"""

import os
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3ConfigOpenHermesFR(SmolLM3Config):
    """Configuration for SmolLM3 fine-tuning on OpenHermes-FR dataset"""
    
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    max_seq_length: int = 4096
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Training configuration - optimized for French instruction tuning
    batch_size: int = 2  # Reduced for French text (longer sequences)
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    learning_rate: float = 1e-5  # Slightly lower for instruction tuning
    weight_decay: float = 0.01
    warmup_steps: int = 500  # More warmup for instruction tuning
    max_iters: int = 2000  # More iterations for large dataset
    eval_interval: int = 200
    log_interval: int = 10
    save_interval: int = 500
    
    # Optimizer configuration
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # DDP configuration
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # Logging and saving
    save_steps: int = 500
    eval_steps: int = 200
    logging_steps: int = 10
    save_total_limit: Optional[int] = 3
    
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
    # HF Datasets configuration
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None

    
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
        
        # Set default experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = "smollm3_openhermes_fr"

def get_config(config_path: str) -> SmolLM3ConfigOpenHermesFR:
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
                if isinstance(attr, SmolLM3ConfigOpenHermesFR):
                    return attr
    
    # Return default configuration
    return SmolLM3ConfigOpenHermesFR()

# Default configuration instance
config = SmolLM3ConfigOpenHermesFR() 