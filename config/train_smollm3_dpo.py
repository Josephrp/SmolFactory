"""
SmolLM3 DPO Training Configuration
Optimized for Direct Preference Optimization
"""

from config.train_smollm3 import SmolLM3Config

config = SmolLM3Config(
    # Model configuration
    model_name="HuggingFaceTB/SmolLM3-3B-Instruct",  # Start from instruction-tuned model
    max_seq_length=4096,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    
    # Training configuration
    batch_size=2,  # Smaller batch size for DPO
    gradient_accumulation_steps=4,
    learning_rate=5e-6,  # Very low learning rate for DPO
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=1000,
    
    # Mixed precision
    fp16=True,
    bf16=False,
    
    # Logging and saving
    save_steps=200,
    eval_steps=100,
    logging_steps=20,
    
    # Chat template configuration
    use_chat_template=True,
    chat_template_kwargs={
        "enable_thinking": False,  # Disable reasoning for preference learning
        "add_generation_prompt": True
    }
) 