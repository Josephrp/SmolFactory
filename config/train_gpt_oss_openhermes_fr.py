"""
GPT-OSS OpenHermes-FR Optimized Configuration
Specifically optimized for the legmlai/openhermes-fr dataset
800K French instruction-response pairs with quality filtering
"""

from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

# OpenHermes-FR optimized configuration
config = GPTOSSEnhancedCustomConfig(
    # ============================================================================
    # DATASET CONFIGURATION - OpenHermes-FR Specific
    # ============================================================================
    dataset_name="legmlai/openhermes-fr",
    dataset_split="train",
    dataset_format="openhermes_fr",
    
    # OpenHermes-FR field mapping
    input_field="prompt",                    # French prompts
    target_field="accepted_completion",      # GPT-4o generated completions
    
    # Quality filtering using OpenHermes-FR metadata
    filter_bad_entries=True,                 # Use built-in quality flags
    bad_entry_field="bad_entry",
    bad_prompt_field="bad_prompt_detected",
    bad_response_field="bad_response_detected",
    
    # Data processing optimized for French with GPT-OSS Harmony Format
    concatenate_fields=True,
    field_separator="\n\n### Réponse:\n",   # Fallback separator (harmony format takes precedence)
    add_eos_token=True,
    use_harmony_format=True,                 # Enable GPT-OSS harmony format
    
    # Dataset sampling (use all 800K examples by default)
    max_samples=None,                        # Use full dataset
    min_length=20,                          # Minimum for meaningful French text
    max_length=None,                        # Auto-set to max_seq_length
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS - French Language Optimized
    # ============================================================================
    num_train_epochs=1.5,                   # 1.5 epochs optimal for large dataset
    batch_size=6,                           # Balanced for most GPUs
    gradient_accumulation_steps=6,          # Effective batch size: 36
    
    # Learning rate schedule optimized for French fine-tuning
    learning_rate=2.5e-4,                   # Slightly higher for multilingual
    min_lr=2.5e-5,                          # 10% of max learning rate
    warmup_ratio=0.05,                      # 5% warmup for stability
    weight_decay=0.01,                      # Standard L2 regularization
    max_grad_norm=1.0,                      # Gradient clipping
    
    # ============================================================================
    # MODEL CONFIGURATION - Optimized for French
    # ============================================================================
    model_name="openai/gpt-oss-20b",
    max_seq_length=3072,                    # Balanced length for French
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Mixed precision for efficiency
    fp16=False,
    bf16=True,                              # Better for GPT-OSS
    
    # ============================================================================
    # LORA CONFIGURATION - Optimized for French Language Learning
    # ============================================================================
    use_lora=True,
    lora_config={
        "r": 24,                            # Higher rank for language adaptation
        "lora_alpha": 48,                   # 2x rank scaling
        "lora_dropout": 0.05,               # Light regularization
        "target_modules": "all-linear",
        "target_parameters": [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj", 
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    # ============================================================================
    # QUANTIZATION - Balanced Performance/Memory
    # ============================================================================
    use_quantization=True,
    quantization_config={
        "dequantize": True,                 # MXFP4 as per GPT-OSS tutorial
        "load_in_4bit": False,              # Standard precision for quality
    },
    
    # ============================================================================
    # PERFORMANCE OPTIMIZATION
    # ============================================================================
    # Data loading optimized for large dataset
    dataloader_num_workers=6,               # More workers for large dataset
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=3,           # Higher prefetch for efficiency
    
    # Memory management
    low_cpu_mem_usage=True,
    group_by_length=True,                   # Efficient batching
    remove_unused_columns=True,
    
    # ============================================================================
    # EVALUATION & LOGGING
    # ============================================================================
    eval_strategy="steps",
    eval_steps=200,                         # Evaluate every 200 steps
    logging_steps=20,                       # Log every 20 steps
    
    save_strategy="steps", 
    save_steps=500,                         # Save every 500 steps
    save_total_limit=3,                     # Keep 3 best checkpoints
    
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    
    # ============================================================================
    # MULTILINGUAL & FRENCH SPECIFIC SETTINGS
    # ============================================================================
    primary_language="fr",                  # French as primary language
    reasoning_languages=["French", "English"],  # Bilingual reasoning
    domain_focus="instruction",             # Instruction following
    
    # ============================================================================
    # GENERATION CONFIG FOR EVALUATION - GPT-OSS Harmony Format
    # ============================================================================
    generation_config={
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": None,
        "eos_token_id": None,
        # GPT-OSS Harmony Format specific settings
        "reasoning_effort": "medium",           # Configurable reasoning level
        "use_harmony_format": True,             # Ensure harmony format in generation
    },
    
    # ============================================================================
    # HF HUB INTEGRATION
    # ============================================================================
    push_to_hub=False,                      # Set to True to auto-push
    hub_model_id=None,                      # Will be set by launch script
    hub_private_repo=False,
    
    # ============================================================================
    # MONITORING
    # ============================================================================
    enable_tracking=True,                   # Trackio monitoring
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
)

# Print configuration summary on import
print("\n🇫🇷 OpenHermes-FR Configuration Loaded")
print("=" * 50)
print(f"📊 Dataset: {config.dataset_name}")
print(f"🗣️  Language: French (with {config.dataset_format} format)")
print(f"📈 Training: {config.num_train_epochs} epochs")
print(f"🔄 Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
print(f"🧠 LoRA Rank: {config.lora_config['r']}")
print(f"📏 Sequence Length: {config.max_seq_length}")
print(f"🔍 Quality Filtering: {'Enabled' if config.filter_bad_entries else 'Disabled'}")
print(f"🎵 GPT-OSS Harmony Format: {'Enabled' if config.use_harmony_format else 'Disabled'}")
print("=" * 50)
