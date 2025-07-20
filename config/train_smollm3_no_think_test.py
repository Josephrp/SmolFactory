"""
SmolLM3 Training Configuration with /no_think tag
Test configuration to verify /no_think tag functionality
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B",
    max_seq_length=4096,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=100,  # Short test run
    
    # Mixed precision
    fp16=True,
    bf16=False,
    
    # Logging and saving
    save_steps=50,
    eval_steps=25,
    logging_steps=10,
    
    # Chat template configuration with /no_think tag
    use_chat_template=True,
    chat_template_kwargs={
        "add_generation_prompt": True,
        "no_think_system_message": True  # Enable /no_think tag
    }
) 