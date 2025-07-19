"""
RunPod Optimized Configuration for SmolLM3 Fine-tuning
Optimized for cloud GPU training on RunPod
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B",
    max_seq_length=4096,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration - optimized for cloud GPUs
    batch_size=2,  # Conservative for cloud stability
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=1500,
    
    # Mixed precision for efficiency
    fp16=True,
    bf16=False,
    
    # Logging and saving - more frequent for cloud
    save_steps=200,
    eval_steps=100,
    logging_steps=10,
    save_total_limit=5,  # Keep more checkpoints
    
    # Cloud-specific optimizations
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    
    # Data loading optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": False,
        "add_generation_prompt": True
    }
) 