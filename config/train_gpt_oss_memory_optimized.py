"""
GPT-OSS Memory Optimized Training Configuration
Based on OpenAI's GPT-OSS fine-tuning tutorial
Optimized for limited GPU memory (40-80GB)
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTOSSMemoryOptimizedConfig:
    """Memory-optimized configuration for GPT-OSS fine-tuning"""
    trainer_type: str = "sft"
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 1024  # Reduced from 4096
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    batch_size: int = 1  # Reduced from 8
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_iters: int = 500  # Reduced for faster testing
    eval_interval: int = 50
    log_interval: int = 5
    save_interval: int = 100
    optimizer: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    scheduler: str = "cosine_with_min_lr"
    min_lr: float = 2e-5
    lr_scheduler_kwargs: dict = None
    fp16: bool = False
    bf16: bool = True
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 5
    save_total_limit: Optional[int] = 2
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    eval_accumulation_steps: Optional[int] = None
    eval_ratio: float = 0.01
    test_ratio: float = 0.01
    dataset_name: str = "HuggingFaceH4/Multilingual-Thinking"
    dataset_split: str = "train"
    input_field: str = "messages"
    target_field: str = None
    filter_bad_entries: bool = False
    bad_entry_field: str = "bad_entry"
    use_chat_template: bool = True
    chat_template_kwargs: dict = None
    enable_tracking: bool = True
    trackio_url: Optional[str] = None
    trackio_token: Optional[str] = None
    log_artifacts: bool = True
    log_metrics: bool = True
    log_config: bool = True
    experiment_name: Optional[str] = None
    hf_token: Optional[str] = None
    dataset_repo: Optional[str] = None
    use_lora: bool = True
    lora_config: dict = None
    use_quantization: bool = True
    quantization_config: dict = None
    model_kwargs: dict = None
    dataloader_prefetch_factor: int = 2
    tf32: Optional[bool] = None
    chosen_field: Optional[str] = None
    rejected_field: Optional[str] = None
    dpo_beta: float = 0.1
    generation_config: dict = None
    reasoning_languages: list = None
    
    def __post_init__(self):
        """Set default values for complex fields"""
        if self.lora_config is None:
            self.lora_config = {
                "r": 4,  # Reduced from 16
                "lora_alpha": 8,  # Reduced from 32
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
                "task_type": "CAUSAL_LM"
            }
        
        if self.quantization_config is None:
            self.quantization_config = {
                "dequantize": True,  # Use Mxfp4Config as per tutorial
                "load_in_4bit": False  # Only use 4-bit if explicitly needed
            }
        
        if self.model_kwargs is None:
            self.model_kwargs = {
                "attn_implementation": "eager",
                "torch_dtype": "auto",
                "use_cache": False,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "max_memory": {0: "75GB"},  # Reserve some memory
            }
        
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 256,  # Reduced from 512
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        
        if self.reasoning_languages is None:
            self.reasoning_languages = [
                "English", "Spanish", "French", "Italian", "German", 
                "Chinese", "Hindi", "Japanese", "Korean", "Arabic"
            ]
        
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {"min_lr_rate": 0.1}
        
        if self.chat_template_kwargs is None:
            self.chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
                "auto_insert_role": True
            }
        
        # Print memory optimization stats
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print("=== GPT-OSS Memory Optimized Configuration ===")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"LoRA rank: {self.lora_config['r']}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Memory optimization: Enabled")
        print(f"Quantization: {self.quantization_config}")
        print(f"Max memory per GPU: {self.model_kwargs.get('max_memory', 'Auto')}")
        print("==================================================") 