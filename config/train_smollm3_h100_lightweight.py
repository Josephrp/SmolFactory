"""
SmolLM3 H100 Lightweight Training Configuration
Optimized for rapid training on H100 with 80K Hermes-FR samples
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B",
    max_seq_length=8192,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration - Optimized for H100
    batch_size=16,  # Larger batch size for H100
    gradient_accumulation_steps=4,  # Reduced for faster updates
    learning_rate=8e-6,  # Slightly higher for rapid convergence
    weight_decay=0.01,
    warmup_steps=50,  # Reduced warmup for rapid training
    max_iters=None,  # Will be calculated based on epochs
    eval_interval=50,  # More frequent evaluation
    log_interval=5,  # More frequent logging
    save_interval=200,  # More frequent saving
    
    # Optimizer configuration - Optimized for rapid training
    optimizer="adamw",
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    
    # Scheduler configuration - Faster learning
    scheduler="cosine",
    min_lr=2e-6,  # Higher minimum LR
    
    # Mixed precision - Full precision for H100
    fp16=True,
    bf16=False,
    
    # Logging and saving - More frequent for rapid training
    save_steps=200,
    eval_steps=50,
    logging_steps=5,
    save_total_limit=2,  # Keep fewer checkpoints
    
    # Evaluation
    eval_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    
    # Data configuration - Hermes-FR with sampling
    dataset_name="legmlai/openhermes-fr",
    dataset_split="train",
    input_field="prompt",
    target_field="completion",
    filter_bad_entries=False,
    bad_entry_field="bad_entry",
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": False,
        "add_generation_prompt": True,
        "no_think_system_message": True
    },
    
    # Trackio monitoring configuration
    enable_tracking=True,
    trackio_url=None,  # Will be set by launch script
    trackio_token=None,
    log_artifacts=True,
    log_metrics=True,
    log_config=True,
    experiment_name=None,  # Will be set by launch script
    
    # HF Datasets configuration
    dataset_repo=None,  # Will be set by launch script
    
    # H100-specific optimizations
    dataloader_num_workers=4,  # Optimized for H100
    dataloader_pin_memory=True,
    gradient_clipping=1.0,  # Prevent gradient explosion
    
    # Memory optimizations for rapid training
    max_grad_norm=1.0,
    warmup_ratio=0.1,  # 10% warmup
    lr_scheduler_type="cosine",
    
    # Early stopping for rapid training
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
    
    # H100-specific training optimizations
    remove_unused_columns=False,
    group_by_length=True,  # Group similar length sequences
    length_column_name="length",
    ignore_data_skip=False,
    
    # Reporting
    report_to=["tensorboard"],
    run_name="smollm3-h100-lightweight",
    
    # Seed for reproducibility
    seed=42,
    
    # Data collator settings
    data_collator_kwargs={
        "pad_to_multiple_of": 8,  # Optimized for H100
        "return_tensors": "pt"
    }
) 