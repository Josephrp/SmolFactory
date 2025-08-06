"""
GPT-OSS Custom Training Configuration
Based on OpenAI's GPT-OSS fine-tuning tutorial
Fully customizable configuration for any dataset format

Supports specialized datasets like:
- legmlai/openhermes-fr (French instruction dataset)
- HuggingFaceH4/Multilingual-Thinking
- Custom prompt/completion formats
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Union

@dataclass
class GPTOSSEnhancedCustomConfig:
    """Enhanced custom configuration for GPT-OSS fine-tuning with maximum flexibility"""
    
    # ============================================================================
    # CORE MODEL CONFIGURATION
    # ============================================================================
    trainer_type: str = "sft"  # "sft" or "dpo"
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 2048  # Customizable: 512, 1024, 2048, 4096, 8192
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS - FULLY CUSTOMIZABLE
    # ============================================================================
    # Batch Configuration
    batch_size: int = 4  # Per-device batch size (1-32 depending on GPU memory)
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * accumulation * num_gpus
    eval_batch_size: Optional[int] = None  # If None, uses batch_size
    
    # Learning Rate Configuration
    learning_rate: float = 2e-4  # Main learning rate (1e-5 to 5e-4 typical range)
    min_lr: float = 2e-5  # Minimum learning rate for scheduler
    warmup_ratio: float = 0.03  # Fraction of steps for warmup (0.01-0.1)
    warmup_steps: Optional[int] = None  # If set, overrides warmup_ratio
    
    # Training Duration
    num_train_epochs: float = 1.0  # Number of epochs (0.5, 1.0, 2.0, 3.0)
    max_steps: Optional[int] = None  # If set, overrides num_train_epochs
    max_iters: Optional[int] = None  # Legacy compatibility
    
    # Regularization
    weight_decay: float = 0.01  # L2 regularization (0.0-0.1)
    max_grad_norm: float = 1.0  # Gradient clipping (0.5-2.0)
    
    # ============================================================================
    # OPTIMIZER CONFIGURATION
    # ============================================================================
    optimizer: str = "adamw_torch"  # "adamw_torch", "adamw_hf", "sgd"
    beta1: float = 0.9  # Adam beta1 parameter
    beta2: float = 0.95  # Adam beta2 parameter (0.95-0.999)
    eps: float = 1e-8  # Adam epsilon
    
    # ============================================================================
    # SCHEDULER CONFIGURATION
    # ============================================================================
    scheduler: str = "cosine_with_min_lr"  # "linear", "cosine", "cosine_with_min_lr", "constant"
    lr_scheduler_kwargs: Optional[Dict] = None
    
    # ============================================================================
    # MIXED PRECISION & DISTRIBUTED TRAINING
    # ============================================================================
    fp16: bool = False  # Use FP16 (not recommended for GPT-OSS)
    bf16: bool = True  # Use BF16 (recommended for GPT-OSS)
    tf32: Optional[bool] = None  # Use TF32 on A100/H100
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    # ============================================================================
    # LOGGING, EVALUATION & CHECKPOINTING
    # ============================================================================
    # Logging
    logging_steps: int = 10  # Log every N steps
    log_level: str = "info"  # "debug", "info", "warning", "error"
    
    # Evaluation
    eval_strategy: str = "steps"  # "no", "steps", "epoch"
    eval_steps: int = 100  # Evaluate every N steps
    eval_delay: float = 0  # Delay evaluation for N steps/epochs
    eval_accumulation_steps: Optional[int] = None  # Accumulate eval outputs
    
    # Checkpointing
    save_strategy: str = "steps"  # "no", "steps", "epoch"
    save_steps: int = 500  # Save checkpoint every N steps
    save_total_limit: Optional[int] = 3  # Keep only N best checkpoints
    save_only_model: bool = False  # Save only model weights
    
    # Model Selection
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # ============================================================================
    # DATASET CONFIGURATION - ENHANCED FOR CUSTOM FORMATS
    # ============================================================================
    # Dataset Source
    dataset_name: str = "legmlai/openhermes-fr"  # Default to French OpenHermes
    dataset_split: str = "train"  # Dataset split to use
    dataset_config: Optional[str] = None  # Dataset configuration name
    
    # Field Mapping - Customize for your dataset format
    input_field: str = "prompt"  # Field containing the input/prompt
    target_field: str = "accepted_completion"  # Field containing the target/completion
    
    # OpenHermes-FR specific fields
    filter_bad_entries: bool = True  # Filter entries marked as bad
    bad_entry_field: str = "bad_entry"  # Field indicating bad entries
    bad_prompt_field: str = "bad_prompt_detected"  # Field for bad prompts
    bad_response_field: str = "bad_response_detected"  # Field for bad responses
    
    # Data Processing Options
    concatenate_fields: bool = True  # Combine input and target fields for training
    field_separator: str = "\n\n### Response:\n"  # Separator between input and target
    add_eos_token: bool = True  # Add EOS token at the end
    
    # Dataset Filtering & Sampling
    max_samples: Optional[int] = None  # Limit dataset size (e.g., 100000 for testing)
    min_length: int = 10  # Minimum sequence length
    max_length: Optional[int] = None  # Maximum sequence length (None = use max_seq_length)
    
    # Custom Dataset Formats Support
    dataset_format: str = "openhermes_fr"  # "openhermes_fr", "messages", "text", "custom"
    
    # GPT-OSS Harmony Format Configuration
    use_harmony_format: bool = True  # Enable GPT-OSS harmony format
    use_chat_template: bool = False  # Set to True for messages format
    chat_template_kwargs: Optional[Dict] = None
    
    # ============================================================================
    # TRACKIO MONITORING CONFIGURATION
    # ============================================================================
    enable_tracking: bool = True
    trackio_url: Optional[str] = None
    trackio_token: Optional[str] = None
    log_artifacts: bool = True
    log_metrics: bool = True
    log_config: bool = True
    experiment_name: Optional[str] = None
    
    # ============================================================================
    # HUGGING FACE INTEGRATION
    # ============================================================================
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None
    push_to_hub: bool = False  # Push model to HF Hub after training
    hub_model_id: Optional[str] = None  # HF Hub model ID
    hub_private_repo: bool = False  # Make HF repo private
    
    # ============================================================================
    # GPT-OSS SPECIFIC CONFIGURATIONS
    # ============================================================================
    # LoRA Configuration
    use_lora: bool = True
    lora_config: Optional[Dict] = None
    
    # Quantization Configuration  
    use_quantization: bool = True
    quantization_config: Optional[Dict] = None
    
    # Model Loading Configuration
    model_kwargs: Optional[Dict] = None
    
    # Generation Configuration (for evaluation/testing)
    generation_config: Optional[Dict] = None
    
    # ============================================================================
    # MULTILINGUAL & DOMAIN SPECIFIC SETTINGS
    # ============================================================================
    # Language Support (for multilingual datasets)
    primary_language: str = "fr"  # Primary language code
    reasoning_languages: Optional[List[str]] = None  # Supported languages for reasoning
    
    # Domain-specific settings
    domain_focus: Optional[str] = None  # "reasoning", "conversation", "instruction", "general"
    
    # ============================================================================
    # PERFORMANCE & MEMORY OPTIMIZATION
    # ============================================================================
    # Data Loading
    dataloader_num_workers: int = 4  # Number of data loading workers
    dataloader_pin_memory: bool = True  # Pin memory for faster GPU transfer
    dataloader_prefetch_factor: int = 2  # Prefetch factor for data loading
    
    # Memory Management
    max_memory_per_gpu: Optional[str] = None  # e.g., "80GB", "40GB"
    low_cpu_mem_usage: bool = True  # Use low CPU memory loading
    
    # Performance Optimizations
    group_by_length: bool = True  # Group sequences by length
    length_column_name: str = "length"  # Column name for sequence lengths
    remove_unused_columns: bool = True  # Remove unused dataset columns
    
    def __post_init__(self):
        """Initialize default values and validate configuration"""
        
        # ============================================================================
        # LORA CONFIGURATION DEFAULTS
        # ============================================================================
        if self.lora_config is None:
            self.lora_config = {
                "r": 16,  # Rank (4, 8, 16, 32, 64) - higher = more parameters
                "lora_alpha": 32,  # Scaling factor (usually 2*r)
                "target_modules": "all-linear",  # Apply LoRA to all linear layers
                "target_parameters": [
                    "7.mlp.experts.gate_up_proj",
                    "7.mlp.experts.down_proj", 
                    "15.mlp.experts.gate_up_proj",
                    "15.mlp.experts.down_proj",
                    "23.mlp.experts.gate_up_proj", 
                    "23.mlp.experts.down_proj",
                ],
                "bias": "none",  # "none", "all", "lora_only"
                "task_type": "CAUSAL_LM",
                "lora_dropout": 0.05,  # LoRA dropout rate
            }
        
        # ============================================================================
        # QUANTIZATION CONFIGURATION DEFAULTS
        # ============================================================================
        if self.quantization_config is None:
            self.quantization_config = {
                "dequantize": True,  # Use Mxfp4Config as per GPT-OSS tutorial
                "load_in_4bit": False,  # Set to True for extreme memory optimization
                "bnb_4bit_compute_dtype": "bfloat16",  # For 4-bit quantization
                "bnb_4bit_use_double_quant": True,  # Double quantization
                "bnb_4bit_quant_type": "nf4"  # Quantization type
            }
        
        # ============================================================================
        # MODEL LOADING CONFIGURATION DEFAULTS
        # ============================================================================
        if self.model_kwargs is None:
            self.model_kwargs = {
                "attn_implementation": "eager",  # "eager", "flash_attention_2"
                "torch_dtype": "auto",  # "auto", "bfloat16", "float16"
                "use_cache": False,  # Disable KV cache for training
                "device_map": "auto",  # Automatic device mapping
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }
            
            # Add memory constraints if specified
            if self.max_memory_per_gpu:
                self.model_kwargs["max_memory"] = {0: self.max_memory_per_gpu}
        
        # ============================================================================
        # GENERATION CONFIGURATION DEFAULTS
        # ============================================================================
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 512,  # Maximum tokens to generate
                "do_sample": True,  # Use sampling
                "temperature": 0.7,  # Sampling temperature
                "top_p": 0.9,  # Nucleus sampling
                "top_k": 50,  # Top-k sampling
                "repetition_penalty": 1.1,  # Repetition penalty
                "pad_token_id": None,  # Will be set from tokenizer
                "eos_token_id": None,  # Will be set from tokenizer
            }
        
        # ============================================================================
        # LANGUAGE CONFIGURATION DEFAULTS
        # ============================================================================
        if self.reasoning_languages is None:
            if self.primary_language == "fr":
                self.reasoning_languages = [
                    "French", "English", "Spanish", "Italian", "German"
                ]
            else:
                self.reasoning_languages = [
                    "English", "Spanish", "French", "Italian", "German", 
                    "Chinese", "Hindi", "Japanese", "Korean", "Arabic"
                ]
        
        # ============================================================================
        # SCHEDULER CONFIGURATION DEFAULTS
        # ============================================================================
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {"min_lr_rate": 0.1}
        
        # ============================================================================
        # CHAT TEMPLATE CONFIGURATION DEFAULTS (GPT-OSS Harmony Format)
        # ============================================================================
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
                "auto_insert_role": True,
                # GPT-OSS Harmony Format specific settings
                "reasoning_effort": "medium",  # low, medium, high
                "model_identity": "You are GPT-Tonic, a large language model trained by TonicAI.",
                "builtin_tools": [],  # Can include "browser" and/or "python"
            }
        
        # ============================================================================
        # VALIDATION AND COMPUTED VALUES
        # ============================================================================
        # Compute effective batch size
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Set warmup steps if not provided
        if self.warmup_steps is None and self.max_steps:
            self.warmup_steps = int(self.max_steps * self.warmup_ratio)
        
        # Set max_length for dataset filtering
        if self.max_length is None:
            self.max_length = self.max_seq_length
        
        # Validate configuration
        self._validate_config()
        
        # Print comprehensive configuration summary
        self._print_config_summary(effective_batch_size)
    
    def _validate_config(self):
        """Validate configuration parameters"""
        
        # Validate batch configuration
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
            
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.min_lr >= self.learning_rate:
            raise ValueError("min_lr must be < learning_rate")
            
        # Validate sequence length
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")
            
        # Validate dataset format
        valid_formats = ["openhermes_fr", "messages", "text", "custom"]
        if self.dataset_format not in valid_formats:
            raise ValueError(f"dataset_format must be one of {valid_formats}")
    
    def _print_config_summary(self, effective_batch_size):
        """Print detailed configuration summary"""
        
        print("\n" + "="*80)
        print("🚀 GPT-OSS ENHANCED CUSTOM CONFIGURATION")
        print("="*80)
        
        print(f"📊 Model & Training:")
        print(f"   • Model: {self.model_name}")
        print(f"   • Dataset: {self.dataset_name} ({self.dataset_format})")
        print(f"   • Primary Language: {self.primary_language}")
        print(f"   • Sequence Length: {self.max_seq_length}")
        print(f"   • Epochs: {self.num_train_epochs}")
        
        print(f"\n🔄 Batch Configuration:")
        print(f"   • Per-device Batch Size: {self.batch_size}")
        print(f"   • Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   • Effective Batch Size: {effective_batch_size}")
        
        print(f"\n📈 Learning Configuration:")
        print(f"   • Learning Rate: {self.learning_rate}")
        print(f"   • Min Learning Rate: {self.min_lr}")
        print(f"   • Weight Decay: {self.weight_decay}")
        print(f"   • Warmup Ratio: {self.warmup_ratio}")
        
        print(f"\n🎛️ LoRA Configuration:")
        print(f"   • Rank: {self.lora_config['r']}")
        print(f"   • Alpha: {self.lora_config['lora_alpha']}")
        print(f"   • Target Modules: {self.lora_config['target_modules']}")
        
        print(f"\n📁 Dataset Configuration:")
        print(f"   • Input Field: {self.input_field}")
        print(f"   • Target Field: {self.target_field}")
        print(f"   • Filter Bad Entries: {self.filter_bad_entries}")
        print(f"   • Max Samples: {self.max_samples or 'All'}")
        
        print(f"\n💾 Memory & Performance:")
        print(f"   • Mixed Precision: {'BF16' if self.bf16 else 'FP32'}")
        print(f"   • Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"   • Data Workers: {self.dataloader_num_workers}")
        print(f"   • Group by Length: {self.group_by_length}")
        
        print("="*80 + "\n")

# Create the config instance with OpenHermes-FR optimized defaults
config = GPTOSSEnhancedCustomConfig()
