"""
GPT-OSS Medical o1 SFT Training Configuration
Dataset: FreedomIntelligence/medical-o1-reasoning-SFT
Format: Question | Complex_CoT | Response → GPT-OSS Harmony text

This configuration uses GPT-OSS Harmony formatting to combine the medical
dataset's question, chain-of-thought (Complex_CoT), and final response into a
single assistant turn, with optional system and developer messages.
"""

from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

# Medical-o1 SFT configuration for GPT-OSS
config = GPTOSSEnhancedCustomConfig(
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
    dataset_config="en",               # Use English split by default (can be changed to en_mix/zh/zh_mix)
    dataset_split="train",
    dataset_format="medical_o1_sft",   # Enable medical formatter in training script

    # Field mapping and prefixes
    input_field="Question",            # used for length filtering pre-format
    target_field="Response",           # used for length filtering pre-format
    question_field="Question",
    reasoning_field="Complex_CoT",
    response_field="Response",
    reason_prefix="Reasoning: ",
    answer_prefix="Final Answer: ",

    # GPT-OSS Harmony formatting
    use_harmony_format=True,
    use_chat_template=False,
    system_message=(
        "You are GPT-Tonic, a large language model trained by TonicAI."
    ),
    developer_message=(
        "You are are GPT-Tonic, an intelligent assistant that always answers health-related queries scientifically."
    ),
    chat_template_kwargs={
        "add_generation_prompt": True,
        "tokenize": False,
        "reasoning_effort": "low",
        "model_identity": "You are GPT-Tonic, a large language model trained by TonicAI.",
        "builtin_tools": [],
    },

    # Filtering & sampling
    filter_bad_entries=False,
    max_samples=None,
    min_length=10,
    max_length=2048,

    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    num_train_epochs=2.0,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    min_lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    warmup_steps=50,
    max_grad_norm=1.0,

    # Sequence length
    max_seq_length=2048,

    # ============================================================================
    # MIXED PRECISION / PERFORMANCE
    # ============================================================================
    fp16=False,
    bf16=True,
    tf32=True,

    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
    group_by_length=True,
    remove_unused_columns=True,

    # ============================================================================
    # LORA & QUANTIZATION
    # ============================================================================
    use_lora=True,
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
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

    use_quantization=True,
    quantization_config={
        "dequantize": True,
        "load_in_4bit": False,
        # Optional MXFP4 config is auto-applied by training script if available
    },

    # ============================================================================
    # LOGGING & EVAL
    # ============================================================================
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    save_only_model=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    eval_accumulation_steps=2,
    eval_batch_size=1,
    eval_ratio=0.001,
    test_ratio=0.0005,

    # ============================================================================
    # MONITORING & HUB
    # ============================================================================
    enable_tracking=True,
    log_artifacts=False,
    log_metrics=True,
    log_config=True,
    push_to_hub=False,
    hub_model_id=None,
    hub_private_repo=False,
)

# Quick summary for visibility when the config is imported
print("\n🩺 GPT-OSS Medical o1 SFT Configuration")
print("=" * 60)
print(f"📊 Dataset: {config.dataset_name} [{config.dataset_config}] (medical_o1_sft)")
print(f"📈 Training: {config.num_train_epochs} epoch | batch {config.batch_size} x acc {config.gradient_accumulation_steps}")
print(f"🧠 LoRA Rank: {config.lora_config['r']}")
print(f"📏 Sequence Length: {config.max_seq_length}")
print(f"🎵 Harmony Format: {'Enabled' if config.use_harmony_format else 'Disabled'}")
print("=" * 60)

