"""
SmolLM3 Long-Context Training Configuration
Optimized for long-context tasks (up to 128k tokens)
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B",
    max_seq_length=131072,  # 128k tokens
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration
    batch_size=1,  # Reduced for long sequences
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    learning_rate=1e-5,  # Lower learning rate for stability
    weight_decay=0.01,
    warmup_steps=200,
    max_iters=500,
    
    # Mixed precision
    fp16=True,
    bf16=False,
    
    # Logging and saving
    save_steps=100,
    eval_steps=50,
    logging_steps=10,
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": True,  # Enable reasoning mode
        "add_generation_prompt": True
    }
) 