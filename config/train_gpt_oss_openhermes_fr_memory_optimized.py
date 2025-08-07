"""
GPT-OSS OpenHermes-FR Memory-Optimized Configuration
Combines memory optimization best practices with OpenHermes-FR dataset
Optimized for GPT-OSS harmony format and MXFP4 quantization
Based on OpenAI GPT-OSS specifications and memory optimization principles
"""

from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

# Memory-optimized OpenHermes-FR configuration for GPT-OSS
config = GPTOSSEnhancedCustomConfig(
    # ============================================================================
    # DATASET CONFIGURATION - OpenHermes-FR with Harmony Format
    # ============================================================================
    dataset_name="legmlai/openhermes-fr",
    dataset_split="train",
    dataset_format="openhermes_fr",
    
    # OpenHermes-FR field mapping optimized for harmony format
    input_field="prompt",                    # French prompts
    target_field="accepted_completion",      # GPT-4o generated completions
    
    # Enhanced quality filtering for memory-constrained training
    filter_bad_entries=True,                 # Critical for memory efficiency
    bad_entry_field="bad_entry",
    bad_prompt_field="bad_prompt_detected",
    bad_response_field="bad_response_detected",
    
    # Memory-optimized data processing with GPT-OSS Harmony Format
    concatenate_fields=True,
    field_separator="\n\n### Réponse:\n",   # Fallback separator (harmony format takes precedence)
    add_eos_token=True,                      # Required for proper training
    use_harmony_format=True,                 # Enable GPT-OSS harmony format
    
    # Dataset sampling optimized for memory constraints
    max_samples=800000,                      # Reduced from 800K for memory efficiency
    min_length=15,                          # Slightly higher minimum for quality
    max_length=2048,                        # Explicit max length for memory control
    
    # ============================================================================
    # MEMORY-OPTIMIZED TRAINING HYPERPARAMETERS
    # ============================================================================
    # Batch configuration following memory optimization principles
    num_train_epochs=1.0,                   # Single epoch to reduce memory pressure
    batch_size=8,                           # Reduced from 6 for memory efficiency
    gradient_accumulation_steps=8,         # Increased to maintain effective batch size 32
    
    # Learning rate optimized for single epoch + memory constraints
    learning_rate=2e-4,                     # Standard GPT-OSS learning rate
    min_lr=2e-5,                            # 10% of max learning rate
    warmup_ratio=0.03,                      # Reduced warmup for memory efficiency
    weight_decay=0.01,                      # Standard L2 regularization
    max_grad_norm=1.0,                      # Gradient clipping for stability
    
    # ============================================================================
    # MODEL CONFIGURATION - Memory Optimized for GPT-OSS
    # ============================================================================
    model_name="openai/gpt-oss-20b",
    max_seq_length=1024,                    # Reduced from 3072 for memory optimization
    use_flash_attention=True,               # Critical for memory efficiency
    use_gradient_checkpointing=True,        # Essential for memory optimization
    
    # Mixed precision optimized for GPT-OSS MXFP4
    fp16=False,                             # Not recommended for GPT-OSS
    bf16=True,                              # Required for GPT-OSS stability
    tf32=True,                              # Enable TF32 for A100/H100 efficiency
    
    # ============================================================================
    # LORA CONFIGURATION - Memory Optimized for GPT-OSS MoE
    # ============================================================================
    use_lora=True,
    lora_config={
        "r": 8,                             # Reduced rank for memory efficiency
        "lora_alpha": 16,                   # 2x rank scaling (memory optimized)
        "lora_dropout": 0.1,                # Higher dropout for better generalization
        "target_modules": "all-linear",     # Apply to all linear layers
        "target_parameters": [
            # GPT-OSS specific MoE expert targeting
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj", 
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        "bias": "none",                     # No bias adaptation for memory efficiency
        "task_type": "CAUSAL_LM",
        "modules_to_save": [],              # Don't save additional modules for memory
    },
    
    # ============================================================================
    # QUANTIZATION - GPT-OSS Native MXFP4 Optimization
    # ============================================================================
    use_quantization=True,
    quantization_config={
        "dequantize": True,                 # Use native MXFP4 as per GPT-OSS specs
        "load_in_4bit": False,              # Don't use BNB 4-bit with MXFP4
        "mxfp4_config": {                   # Native GPT-OSS MXFP4 settings
            "enabled": True,
            "block_size": 32,               # Optimized block size for MoE
        }
    },
    
    # ============================================================================
    # MEMORY OPTIMIZATION CONFIGURATION
    # ============================================================================
    # Model loading with memory constraints
    model_kwargs={
        "attn_implementation": "eager",     # Memory-safe attention
        "torch_dtype": "auto",              # Let model decide (MXFP4 compatible)
        "use_cache": False,                 # Disable KV cache for training
        "device_map": "auto",               # Automatic device mapping
        "low_cpu_mem_usage": True,          # Critical for memory optimization
        "max_memory": {0: "75GB"},          # Reserve memory for other processes
    },
    
    # Data loading optimized for memory efficiency
    dataloader_num_workers=2,               # Reduced workers to save memory
    dataloader_pin_memory=False,            # Disable to save memory
    dataloader_prefetch_factor=1,           # Minimal prefetch for memory
    
    # Memory management optimizations
    max_memory_per_gpu="75GB",              # Explicit memory limit
    low_cpu_mem_usage=True,                 # Essential for large models
    group_by_length=True,                   # Efficient batching for memory
    remove_unused_columns=True,             # Remove unnecessary data
    
    # ============================================================================
    # EVALUATION & LOGGING - Memory Efficient
    # ============================================================================
    eval_strategy="steps",
    eval_steps=500,                         # Less frequent evaluation for memory
    logging_steps=50,                       # Reduced logging frequency
    
    save_strategy="steps", 
    save_steps=1000,                        # Less frequent saves for memory/storage
    save_total_limit=2,                     # Keep only 2 checkpoints for memory
    save_only_model=True,                   # Save only model weights
    
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    
    # Evaluation memory optimization
    eval_accumulation_steps=4,              # Accumulate eval outputs to save memory
    eval_batch_size=1,                      # Smaller eval batch size
    # Split ratios for automatic validation/test creation
    eval_ratio=0.001,
    test_ratio=0.0005,
    
    # ============================================================================
    # GPT-OSS HARMONY FORMAT OPTIMIZATION
    # ============================================================================
    # Chat template for harmony format compatibility (following exact template)
    use_chat_template=False,                # Use custom harmony format instead
    chat_template_kwargs={
        "add_generation_prompt": True,
        "tokenize": False,
        # GPT-OSS Harmony Format specific settings (exact template format)
        "reasoning_effort": "medium",       # low, medium, high
        "model_identity": "You are GPT-Tonic, a large language model trained by TonicAI.",
        "builtin_tools": [],                # Can include "browser" and/or "python"
    },
    
    # Generation config optimized for GPT-OSS harmony format (exact template compliance)
    generation_config={
        "max_new_tokens": 256,              # Reduced for memory efficiency
        "do_sample": True,
        "temperature": 0.6,                 # Slightly lower for more focused training
        "top_p": 0.9,
        "top_k": 40,                        # Reduced for memory efficiency
        "repetition_penalty": 1.1,
        "pad_token_id": None,
        "eos_token_id": None,
        # GPT-OSS Harmony Format specific settings (exact template format)
        "reasoning_effort": "medium",       # Configurable reasoning level
        "use_harmony_format": True,         # Ensure harmony format in generation
    },
    
    # ============================================================================
    # MULTILINGUAL & REASONING OPTIMIZATION
    # ============================================================================
    primary_language="fr",                  # French as primary language
    reasoning_languages=["French", "English"],  # Bilingual reasoning capability
    domain_focus="reasoning",               # Align with GPT-OSS reasoning focus
    
    # ============================================================================
    # OPTIMIZER & SCHEDULER - Memory Optimized
    # ============================================================================
    optimizer="adamw_torch",                # Memory-efficient optimizer
    beta1=0.9,
    beta2=0.95,                             # GPT-OSS optimized beta2
    eps=1e-8,
    
    scheduler="cosine_with_min_lr",         # Stable scheduler for single epoch
    lr_scheduler_kwargs={
        "min_lr_rate": 0.1,
        "warmup_steps": None,               # Use warmup_ratio instead
    },
    
    # ============================================================================
    # MONITORING & HUB INTEGRATION
    # ============================================================================
    enable_tracking=True,                   # Trackio monitoring
    log_artifacts=False,                    # Disable to save memory/storage
    log_metrics=True,
    log_config=True,
    
    push_to_hub=False,                      # Set to True after successful training
    hub_model_id=None,
    hub_private_repo=False,
)

# Configuration validation and optimization tips
print("\n🔧 GPT-OSS Memory-Optimized OpenHermes-FR Configuration")
print("=" * 60)
print(f"📊 Dataset: {config.dataset_name} (200K samples)")
print(f"🗣️  Language: French with GPT-OSS Harmony Format")
print(f"📈 Training: {config.num_train_epochs} epoch (memory optimized)")
print(f"🔄 Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
print(f"🧠 LoRA Rank: {config.lora_config['r']} (memory optimized)")
print(f"📏 Sequence Length: {config.max_seq_length} (memory optimized)")
print(f"💾 Memory Limit: {config.max_memory_per_gpu}")
print(f"⚡ Quantization: MXFP4 (GPT-OSS native)")
print(f"🔍 Quality Filtering: Enabled")
print(f"🎵 GPT-OSS Harmony Format: {'Enabled' if config.use_harmony_format else 'Disabled'}")
print("=" * 60)
print("\n💡 Memory Optimization Features:")
print("  • Native MXFP4 quantization for GPT-OSS MoE layers")
print("  • Reduced batch size with increased gradient accumulation")
print("  • Limited sequence length for memory efficiency")
print("  • Reduced LoRA rank while maintaining effectiveness")
print("  • Dataset sampling (200K from 800K) for faster training")
print("  • Gradient checkpointing and efficient data loading")
print("  • Exact GPT-OSS Harmony format with <|return|> tokens")
print("=" * 60)
