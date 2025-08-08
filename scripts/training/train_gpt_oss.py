#!/usr/bin/env python3
"""
GPT-OSS Training Script
Specialized training script for OpenAI's GPT-OSS models
Based on the GPT-OSS fine-tuning tutorial
"""

import os
import sys
import argparse
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
try:
    from trl import DPOTrainer
except Exception:  # pragma: no cover - optional import depending on TRL version
    DPOTrainer = None
from datasets import load_dataset
from pathlib import Path
# Import monitoring utilities from project src for persistent logging
try:
    from src.monitoring import create_monitor_from_config  # type: ignore
except Exception:
    create_monitor_from_config = None  # type: ignore

# Ensure project root and config package are importable for configs that do `from config...` imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    # Put project root early so top-level packages like `config` can be resolved
    sys.path.insert(0, str(project_root))
config_dir = project_root / "config"
if str(config_dir) not in sys.path:
    # Ensure the actual `config` package takes precedence over any `config.py` module elsewhere
    sys.path.insert(0, str(config_dir))
# Ensure 'src' is importable for modules like 'monitoring', 'model', etc., but do not shadow `config`
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    # Append to the end to avoid overshadowing the `config` package with `src/config.py`
    sys.path.append(str(src_dir))

# If a stray 'config' module (e.g., from src/config.py) is already imported, remove it so
# that the real package `config/` (with __init__.py) can be imported with submodules.
try:
    if 'config' in sys.modules and not hasattr(sys.modules['config'], '__path__'):
        del sys.modules['config']
except Exception:
    pass

# Reduce tokenizer thread contention and improve CUDA allocator behavior
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def load_gpt_oss_model_and_tokenizer(config):
    """Load GPT-OSS model and tokenizer with proper configuration"""
    
    print("Loading GPT-OSS tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("Loading GPT-OSS model with quantization...")
    
    # Import quantization config
    from transformers import BitsAndBytesConfig
    
    # Set up quantization config based on config
    if config.quantization_config and config.quantization_config.get("load_in_4bit"):
        # Use BitsAndBytesConfig for 4-bit quantization (memory optimized)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif config.quantization_config and (
        config.quantization_config.get("dequantize")
        or (
            isinstance(config.quantization_config.get("mxfp4_config"), dict)
            and config.quantization_config["mxfp4_config"].get("enabled", False)
        )
    ):
        # Try to use Mxfp4Config if available (as per tutorial)
        try:
            from transformers import Mxfp4Config
            quantization_config = Mxfp4Config(dequantize=True)
        except ImportError:
            # Fallback to no quantization if Mxfp4Config not available
            print("Warning: Mxfp4Config not available, using no quantization")
            quantization_config = None
    else:
        # No quantization
        quantization_config = None
    
    # Build model kwargs with sensible defaults and allow config overrides
    default_model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "device_map": "auto",
    }

    cfg_model_kwargs = getattr(config, "model_kwargs", None)
    if isinstance(cfg_model_kwargs, dict):
        # Config overrides defaults (e.g., attn_implementation="kernels-community/vllm-flash-attn3")
        model_kwargs = {**default_model_kwargs, **cfg_model_kwargs}
    else:
        model_kwargs = default_model_kwargs.copy()

    # Normalize torch_dtype if provided as a string in config
    if isinstance(model_kwargs.get("torch_dtype"), str):
        dtype_str = str(model_kwargs["torch_dtype"]).lower()
        if dtype_str in {"bf16", "bfloat16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype_str in {"fp16", "float16", "half"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype_str == "auto":
            # Leave as-is for HF to decide
            pass
        else:
            # Fallback to bfloat16 for safer memory footprint on A100/H100
            model_kwargs["torch_dtype"] = torch.bfloat16

    # Ensure we have an offload folder for tight-memory setups
    model_kwargs.setdefault("offload_folder", os.path.join(str(project_root), "offload"))
    
    # Only add quantization_config if it's not None
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    # If using MXFP4, follow tutorial exactly: eager attention + bf16
    try:
        from transformers import Mxfp4Config as _Mxfp4Config
        if isinstance(quantization_config, _Mxfp4Config):
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["use_cache"] = False
            model_kwargs["device_map"] = model_kwargs.get("device_map", "auto")
            model_kwargs["quantization_config"] = quantization_config
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    return model, tokenizer

def setup_lora_for_gpt_oss(model, config):
    """Setup LoRA for GPT-OSS model"""
    
    print("Setting up LoRA for GPT-OSS...")
    
    # LoRA configuration as per tutorial
    lora_config = LoraConfig(
        r=config.lora_config.get("r", 8) if config.lora_config else 8,
        lora_alpha=config.lora_config.get("lora_alpha", 16) if config.lora_config else 16,
        target_modules=config.lora_config.get("target_modules", "all-linear") if config.lora_config else "all-linear",
        target_parameters=config.lora_config.get("target_parameters", [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ]) if config.lora_config else [
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model

def load_dataset_from_config(config):
    """Load dataset based on configuration"""
    
    dataset_name = getattr(config, 'dataset_name', 'HuggingFaceH4/Multilingual-Thinking')
    dataset_split = getattr(config, 'dataset_split', 'train')
    dataset_config = getattr(config, 'dataset_config', None)
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Dataset split: {dataset_split}")
    if dataset_config:
        print(f"Dataset config: {dataset_config}")
    
    # Load the dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)
    
    print(f"Original dataset size: {len(dataset)} examples")
    
    # Apply filtering based on configuration
    dataset = apply_dataset_filtering(dataset, config)
    
    # Apply dataset processing based on format
    dataset = process_dataset_format(dataset, config)
    
    print(f"Final dataset size: {len(dataset)} examples")
    
    return dataset

def build_scheduler_kwargs(config):
    """Construct lr_scheduler_kwargs compatibly across TRL/Transformers versions.

    - For TRL's 'cosine_with_min_lr' scheduler, ensure a min_lr/min_lr_rate is set.
    - For all other schedulers, strip TRL-specific keys to avoid unexpected kwargs
      errors in Transformers' native schedulers.
    """
    skw = getattr(config, 'lr_scheduler_kwargs', {}) or {}
    if not isinstance(skw, dict):
        skw = {}

    scheduler_type = getattr(config, 'scheduler', None)

    # If we're NOT using TRL's special scheduler, drop incompatible keys early
    if scheduler_type != 'cosine_with_min_lr':
        for k in ('min_lr', 'min_lr_rate', 'warmup_steps', 'num_warmup_steps', 'warmup_ratio'):
            if k in skw:
                skw.pop(k, None)
        return skw

    # TRL cosine-with-min-lr: ensure one of min_lr or min_lr_rate is provided
    min_lr_cfg = getattr(config, 'min_lr', 1e-6)
    if 'min_lr' not in skw and 'min_lr_rate' not in skw:
        try:
            if min_lr_cfg is not None:
                skw['min_lr'] = float(min_lr_cfg)
            else:
                skw['min_lr_rate'] = 0.1
        except Exception:
            skw['min_lr_rate'] = 0.001

    # Remove warmup-related keys which conflict with some TRL schedulers
    for k in ('warmup_steps', 'num_warmup_steps', 'warmup_ratio'):
        if k in skw:
            skw.pop(k, None)
    return skw

def apply_dataset_filtering(dataset, config):
    """Apply filtering based on configuration"""
    
    # Parallel workers for datasets ops
    try:
        import os as _os
        num_proc = getattr(config, 'dataset_num_proc', None) or (_os.cpu_count() or 1)
    except Exception:
        num_proc = 1

    # Filter bad entries if specified
    if getattr(config, 'filter_bad_entries', False):
        bad_entry_field = getattr(config, 'bad_entry_field', 'bad_entry')
        bad_prompt_field = getattr(config, 'bad_prompt_field', 'bad_prompt_detected')
        bad_response_field = getattr(config, 'bad_response_field', 'bad_response_detected')
        
        original_size = len(dataset)
        
        # Filter out bad entries
        if bad_entry_field in dataset.column_names:
            def _keep_not_bad_entry(example, _field=bad_entry_field):
                return not example.get(_field, False)
            dataset = dataset.filter(_keep_not_bad_entry, num_proc=num_proc)
            print(f"Filtered {original_size - len(dataset)} bad entries")
        
        # Filter out bad prompts
        if bad_prompt_field in dataset.column_names:
            def _keep_not_bad_prompt(example, _field=bad_prompt_field):
                return not example.get(_field, False)
            dataset = dataset.filter(_keep_not_bad_prompt, num_proc=num_proc)
            print(f"Filtered bad prompts, remaining: {len(dataset)} examples")
        
        # Filter out bad responses
        if bad_response_field in dataset.column_names:
            def _keep_not_bad_response(example, _field=bad_response_field):
                return not example.get(_field, False)
            dataset = dataset.filter(_keep_not_bad_response, num_proc=num_proc)
            print(f"Filtered bad responses, remaining: {len(dataset)} examples")
    
    # Apply length filtering
    min_length = getattr(config, 'min_length', 10)
    max_length = getattr(config, 'max_length', None)
    
    input_field = getattr(config, 'input_field', 'prompt')
    target_field = getattr(config, 'target_field', 'accepted_completion')
    
    if min_length > 0 or max_length:
        def length_filter(example):
            input_len = len(example.get(input_field, ''))
            target_len = len(example.get(target_field, ''))
            total_len = input_len + target_len
            
            if total_len < min_length:
                return False
            if max_length and total_len > max_length:
                return False
            return True
        
        original_size = len(dataset)
        dataset = dataset.filter(length_filter, num_proc=num_proc)
        print(f"Length filtering: {original_size} -> {len(dataset)} examples")
    
    # Apply sampling if specified
    max_samples = getattr(config, 'max_samples', None)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled {max_samples} examples from dataset")
    
    return dataset

def _build_harmony_text(
    user_content: str,
    assistant_content: str,
    add_eos_token: bool = True,
    system_message: str | None = None,
    developer_message: str | None = None,
) -> str:
    """Compose a Harmony-formatted conversation with optional system/developer messages.

    Structure (training):
      <|start|>system<|message|>...<|end|> (optional)
      <|start|>developer<|message|>...<|end|> (optional)
      <|start|>user<|message|>...<|end|>
      <|start|>assistant<|channel|>final<|message|>...<|return|>
    """
    parts: list[str] = []
    if system_message:
        parts.append(f"<|start|>system<|message|>{system_message}<|end|>")
    if developer_message:
        parts.append(f"<|start|>developer<|message|>{developer_message}<|end|>")
    parts.append(f"<|start|>user<|message|>{user_content}<|end|>")
    parts.append(f"<|start|>assistant<|channel|>final<|message|>{assistant_content}")
    if add_eos_token:
        parts[-1] += "<|return|>"
    else:
        parts[-1] += "<|end|>"
    return "".join(parts)

def format_gpt_oss_harmony(
    prompt: str,
    completion: str,
    add_eos_token: bool = True,
    system_message: str | None = None,
    developer_message: str | None = None,
) -> str:
    """
    Format data for GPT-OSS Harmony format following the exact template structure.
    Spec: `https://huggingface.co/openai/gpt-oss-20b/raw/main/chat_template.jinja`.
    """
    return _build_harmony_text(
        user_content=prompt,
        assistant_content=completion,
        add_eos_token=add_eos_token,
        system_message=system_message,
        developer_message=developer_message,
    )

def format_gpt_oss_harmony_prompt(
    prompt: str,
    system_message: str | None = None,
    developer_message: str | None = None,
) -> str:
    """Prefix-only Harmony prompt up to assistant content marker for DPO, with optional context."""
    parts: list[str] = []
    if system_message:
        parts.append(f"<|start|>system<|message|>{system_message}<|end|>")
    if developer_message:
        parts.append(f"<|start|>developer<|message|>{developer_message}<|end|>")
    parts.append(f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>")
    return "".join(parts)

def process_dataset_format(dataset, config):
    """Process dataset based on format configuration with exact GPT-OSS Harmony compliance"""
    
    # Parallel workers for datasets ops
    try:
        import os as _os
        num_proc = getattr(config, 'dataset_num_proc', None) or (_os.cpu_count() or 1)
    except Exception:
        num_proc = 1

    dataset_format = getattr(config, 'dataset_format', 'openhermes_fr')
    input_field = getattr(config, 'input_field', 'prompt')
    target_field = getattr(config, 'target_field', 'accepted_completion')
    concatenate_fields = getattr(config, 'concatenate_fields', True)
    field_separator = getattr(config, 'field_separator', '\n\n### Response:\n')
    add_eos_token = getattr(config, 'add_eos_token', True)
    use_harmony_format = getattr(config, 'use_harmony_format', True)
    trainer_type = getattr(config, 'trainer_type', 'sft')
    system_message = getattr(config, 'system_message', None)
    developer_message = getattr(config, 'developer_message', None)
    
    print(f"Processing dataset format: {dataset_format}")
    print(f"Input field: {input_field}, Target field: {target_field}")
    print(f"GPT-OSS Harmony Format: {'Enabled' if use_harmony_format else 'Disabled'}")
    
    # Preference-format for DPO training (chosen/rejected pairs)
    if trainer_type == 'dpo':
        chosen_field = getattr(config, 'chosen_field', None)
        rejected_field = getattr(config, 'rejected_field', None)

        if dataset_format == 'preference':
            # Expect columns present; optionally reformat to ensure only necessary columns
            def id_map(example):
                prompt_val = example.get(input_field, '')
                chosen_val = example.get('chosen', example.get(chosen_field or 'chosen', ''))
                rejected_val = example.get('rejected', example.get(rejected_field or 'rejected', ''))
                if use_harmony_format:
                    prompt_text = format_gpt_oss_harmony_prompt(
                        prompt_val,
                        system_message=system_message,
                        developer_message=developer_message,
                    )
                    chosen_text = (chosen_val or '') + ("<|return|>" if add_eos_token else '')
                    rejected_text = (rejected_val or '') + ("<|return|>" if add_eos_token else '')
                    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}
                return {"prompt": prompt_val, "chosen": chosen_val, "rejected": rejected_val}

            keep_cols = [c for c in ['prompt', 'chosen', 'rejected'] if c in dataset.column_names]
            dataset = dataset.map(id_map, remove_columns=dataset.column_names if keep_cols else dataset.column_names, num_proc=num_proc)
            return dataset

        # Custom preference mapping via configured field names
        if chosen_field and rejected_field:
            def to_pref(example):
                prompt_val = example.get(input_field, '')
                chosen_val = example.get(chosen_field, '')
                rejected_val = example.get(rejected_field, '')
                if use_harmony_format:
                    prompt_text = format_gpt_oss_harmony_prompt(
                        prompt_val,
                        system_message=system_message,
                        developer_message=developer_message,
                    )
                    chosen_text = (chosen_val or '') + ("<|return|>" if add_eos_token else '')
                    rejected_text = (rejected_val or '') + ("<|return|>" if add_eos_token else '')
                    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}
                return {"prompt": prompt_val, "chosen": chosen_val, "rejected": rejected_val}

            dataset = dataset.map(to_pref, remove_columns=dataset.column_names, num_proc=num_proc)
            return dataset

        # If we reach here, we don't have required fields for DPO
        raise ValueError("DPO training requires preference data. Please set dataset_format='preference' with 'prompt', 'chosen', 'rejected' columns, or specify 'chosen_field' and 'rejected_field' in the config.")

    if dataset_format == "openhermes_fr":
        # Process OpenHermes-FR format: prompt + accepted_completion
        def format_openhermes_fr(example):
            prompt = example.get(input_field, '')
            completion = example.get(target_field, '')
            
            if concatenate_fields:
                if use_harmony_format:
                    # Use exact GPT-OSS Harmony format from template
                    text = format_gpt_oss_harmony(
                        prompt,
                        completion,
                        add_eos_token,
                        system_message=system_message,
                        developer_message=developer_message,
                    )
                else:
                    # Fallback to standard format with separator
                    text = prompt + field_separator + completion
                    if add_eos_token:
                        text += "</s>"
                
                return {"text": text}
            else:
                # Keep separate for more advanced training setups
                return {
                    "input": prompt,
                    "output": completion
                }
        
        dataset = dataset.map(format_openhermes_fr, remove_columns=dataset.column_names, num_proc=num_proc)
        
    elif dataset_format == "messages":
        # Process messages format (like HuggingFaceH4/Multilingual-Thinking)
        def format_messages(example):
            messages = example.get(input_field, [])
            
            if use_harmony_format and len(messages) >= 2:
                # Extract user and assistant messages for harmony format
                user_message = ""
                assistant_message = ""
                
                for message in messages:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    if role == "user":
                        user_message = content
                    elif role == "assistant":
                        assistant_message = content
                
                if user_message and assistant_message:
                    # Use GPT-OSS Harmony format
                    text = format_gpt_oss_harmony(
                        user_message,
                        assistant_message,
                        add_eos_token,
                        system_message=system_message,
                        developer_message=developer_message,
                    )
                else:
                    # Fallback to simple concatenation
                    text = ""
                    for message in messages:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        text += f"{role}: {content}\n"
                    if add_eos_token:
                        text += "</s>"
            else:
                # Standard format - convert messages to simple text
                text = ""
                for message in messages:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    text += f"{role}: {content}\n"
                if add_eos_token:
                    text += "</s>"
            
            return {"text": text}
        
        dataset = dataset.map(format_messages, remove_columns=dataset.column_names, num_proc=num_proc)
        
    elif dataset_format == "medical_o1_sft":
        # Process Medical-o1 SFT format: Question | Complex_CoT | Response
        # Defaults align with FreedomIntelligence/medical-o1-reasoning-SFT
        question_field = getattr(config, 'question_field', input_field or 'Question')
        reasoning_field = getattr(config, 'reasoning_field', 'Complex_CoT')
        response_field = getattr(config, 'response_field', target_field or 'Response')
        reason_prefix = getattr(config, 'reason_prefix', 'Reasoning: ')
        answer_prefix = getattr(config, 'answer_prefix', 'Final Answer: ')

        def format_medical(example):
            q = example.get(question_field, '') or ''
            cot = example.get(reasoning_field, '') or ''
            ans = example.get(response_field, '') or ''

            # Combine reasoning and final answer in a single assistant turn
            assistant_text = "\n\n".join(
                [s for s in [
                    f"{reason_prefix}{cot}".strip() if cot else '',
                    f"{answer_prefix}{ans}".strip() if ans else ''
                ] if s]
            ) or ans

            if use_harmony_format:
                text = format_gpt_oss_harmony(
                    q,
                    assistant_text,
                    add_eos_token,
                    system_message=system_message,
                    developer_message=developer_message,
                )
            else:
                text = f"Q: {q}\n\n{assistant_text}"
                if add_eos_token:
                    text += "</s>"
            return {"text": text}

        dataset = dataset.map(format_medical, remove_columns=dataset.column_names, num_proc=num_proc)

    elif dataset_format == "text":
        # Process plain text format
        text_field = input_field
        def format_text(example):
            text = example.get(text_field, '')
            if add_eos_token:
                text += "</s>"
            return {"text": text}
        
        dataset = dataset.map(format_text, remove_columns=dataset.column_names, num_proc=num_proc)
    
    elif dataset_format == "custom":
        # Custom format - user handles this in their config
        print("Using custom dataset format - no automatic processing")
    
    return dataset

def split_dataset(dataset, config):
    """Create train/validation/test splits from a single dataset.
    Defaults to 1% eval and 1% test if not specified.
    """
    from datasets import Dataset

    if not isinstance(dataset, Dataset):
        # If it's already a DatasetDict, try to use its splits
        try:
            train_split = dataset["train"]
            eval_split = dataset.get("validation") or dataset.get("eval")
            test_split = dataset.get("test")
            return train_split, eval_split, test_split
        except Exception:
            pass

    eval_ratio = getattr(config, 'eval_ratio', 0.01)
    test_ratio = getattr(config, 'test_ratio', 0.01)

    # Clamp ratios to sane bounds
    try:
        eval_ratio = max(0.0, float(eval_ratio))
        test_ratio = max(0.0, float(test_ratio))
        if eval_ratio + test_ratio >= 0.9:
            # Avoid extreme splits; cap combined at 0.2
            scale = 0.2 / max(1e-9, (eval_ratio + test_ratio))
            eval_ratio *= scale
            test_ratio *= scale
    except Exception:
        eval_ratio, test_ratio = 0.01, 0.01

    # No eval/test requested
    if eval_ratio <= 0 and test_ratio <= 0:
        return dataset, None, None

    ds_shuffled = dataset.shuffle(seed=42)

    # First carve out test split
    if test_ratio > 0:
        split1 = ds_shuffled.train_test_split(test_size=test_ratio, seed=42)
        train_part = split1["train"]
        test_split = split1["test"]
    else:
        train_part = ds_shuffled
        test_split = None

    # Then carve out eval from remaining train
    if eval_ratio > 0:
        remaining_fraction = 1.0 - test_ratio
        # Convert global eval fraction to fraction of remaining pool
        relative_eval = eval_ratio / remaining_fraction if remaining_fraction > 0 else eval_ratio
        split2 = train_part.train_test_split(test_size=relative_eval, seed=42)
        train_split = split2["train"]
        eval_split = split2["test"]
    else:
        train_split = train_part
        eval_split = None

    # Log sizes
    try:
        print(f"Created splits -> train: {len(train_split)}, eval: {len(eval_split) if eval_split else 0}, test: {len(test_split) if test_split else 0}")
    except Exception:
        pass

    return train_split, eval_split, test_split

def setup_trackio_tracking(config):
    """Setup Trackio tracking if enabled"""
    
    if not getattr(config, 'enable_tracking', False):
        print("Trackio tracking disabled or URL not provided")
        return None
    
    # Resolve Trackio URL from config or environment
    trackio_url = getattr(config, 'trackio_url', None) or os.environ.get('TRACKIO_URL') or os.environ.get('TRACKIO_SPACE_ID')
    if not trackio_url:
        print("Trackio tracking enabled but no TRACKIO_URL/TRACKIO_SPACE_ID provided; skipping Trackio setup")
        return None

    print(f"Setting up Trackio tracking: {trackio_url}")
    
    # Import the correct TrackioAPIClient
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trackio_tonic'))
    from trackio_api_client import TrackioAPIClient
    
    # Initialize Trackio client using the correct API
    trackio_client = TrackioAPIClient(
        space_id=trackio_url,
        hf_token=getattr(config, 'trackio_token', None) or os.environ.get('HF_TOKEN')
    )
    
    return trackio_client

def create_sft_config(config, output_dir):
    """Create enhanced SFTConfig for GPT-OSS training"""
    
    print("Creating enhanced SFT configuration...")
    
    # Helper coercion utilities to guarantee numeric types
    def _as_int(value, default):
        if value is None:
            return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value, default):
        if value is None:
            return float(default)
        try:
            return float(value)
        except Exception:
            return float(default)

    # Extract training parameters from config with enhanced defaults and coercion
    num_train_epochs = _as_float(getattr(config, 'num_train_epochs', 1.0), 1.0)
    # Transformers expects max_steps default -1 (disabled). Some code compares > 0
    raw_max_steps = getattr(config, 'max_steps', None)
    max_steps = _as_int(raw_max_steps if raw_max_steps is not None else -1, -1)
    warmup_ratio = _as_float(getattr(config, 'warmup_ratio', 0.03), 0.03)
    # Ensure warmup_steps is an int; default 0 to avoid None comparisons in schedulers
    warmup_steps = _as_int(getattr(config, 'warmup_steps', 0), 0)
    
    # Learning rate configuration
    learning_rate = _as_float(getattr(config, 'learning_rate', 2e-4), 2e-4)
    # Allow CLI/env override of scheduler
    lr_scheduler_type = os.environ.get('GPT_OSS_SCHEDULER', getattr(config, 'scheduler', 'cosine'))
    lr_scheduler_kwargs = build_scheduler_kwargs(config)

    # Detect TRL scheduler signature incompatibilities and fall back gracefully
    # Some TRL versions call get_cosine_with_min_lr_schedule_with_warmup with
    # 'warmup_steps' instead of 'num_warmup_steps', which raises:
    #   get_cosine_with_min_lr_schedule_with_warmup() got an unexpected keyword
    #   argument 'warmup_steps'
    # To avoid this, we fallback to the standard 'cosine' scheduler and strip
    # incompatible kwargs when the incompatible signature is detected.
    if lr_scheduler_type == 'cosine_with_min_lr':
        try:
            from trl.trainer import utils as trl_utils  # type: ignore
            import inspect as _inspect
            if hasattr(trl_utils, 'get_cosine_with_min_lr_schedule_with_warmup'):
                _sig = _inspect.signature(trl_utils.get_cosine_with_min_lr_schedule_with_warmup)
                # If the function does NOT accept 'warmup_steps' explicitly, some TRL versions
                # still pass it internally as a kwarg, causing a TypeError. Fallback to 'cosine'.
                if 'warmup_steps' not in _sig.parameters:
                    print("Warning: Incompatible TRL scheduler signature detected; falling back to 'cosine'.")
                    lr_scheduler_type = 'cosine'
                    lr_scheduler_kwargs = {}
            else:
                # Function missing; fallback
                print("Warning: TRL min-lr cosine scheduler not available; falling back to 'cosine'.")
                lr_scheduler_type = 'cosine'
                lr_scheduler_kwargs = {}
        except Exception:
            # Any import/signature issues -> safe fallback
            print("Warning: Unable to verify TRL scheduler; falling back to 'cosine'.")
            lr_scheduler_type = 'cosine'
            lr_scheduler_kwargs = {}
    
    # Batch configuration
    per_device_train_batch_size = _as_int(getattr(config, 'batch_size', 2), 2)
    per_device_eval_batch_size = _as_int(getattr(config, 'eval_batch_size', per_device_train_batch_size), per_device_train_batch_size)
    gradient_accumulation_steps = _as_int(getattr(config, 'gradient_accumulation_steps', 1), 1)
    
    # Evaluation and logging
    eval_strategy = getattr(config, 'eval_strategy', 'steps')
    eval_steps = _as_int(getattr(config, 'eval_steps', 100), 100)
    eval_accumulation_steps = _as_int(getattr(config, 'eval_accumulation_steps', 1), 1)
    logging_steps = _as_int(getattr(config, 'logging_steps', 10), 10)
    
    # Saving configuration
    save_strategy = getattr(config, 'save_strategy', 'steps')
    save_steps = _as_int(getattr(config, 'save_steps', 500), 500)
    save_total_limit = _as_int(getattr(config, 'save_total_limit', 3), 3)
    
    # Mixed precision
    fp16 = bool(getattr(config, 'fp16', False))
    bf16 = bool(getattr(config, 'bf16', True))
    tf32 = bool(getattr(config, 'tf32', False))
    
    # Regularization
    weight_decay = _as_float(getattr(config, 'weight_decay', 0.01), 0.01)
    max_grad_norm = _as_float(getattr(config, 'max_grad_norm', 1.0), 1.0)
    
    # HuggingFace Hub integration
    push_to_hub = getattr(config, 'push_to_hub', False)
    
    print(f"  • Epochs: {num_train_epochs}")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Batch size: {per_device_train_batch_size}")
    print(f"  • Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  • Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    
    # Build kwargs dynamically to be compatible across transformers versions
    ta_kwargs = {
        # Training duration
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        # Learning rate
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        # Batch configuration
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        # Model configuration
        "gradient_checkpointing": getattr(config, 'use_gradient_checkpointing', True),
        # Mixed precision
        "fp16": fp16,
        "bf16": bf16,
        # Some versions support tf32
        "tf32": tf32 if 'tf32' in TrainingArguments.__init__.__code__.co_varnames else None,
        # Optimizer (optionally use fused AdamW if available through config)
        "optim": getattr(config, 'optimizer', 'adamw_torch'),
        # Regularization
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        # Evaluation (name may vary across versions)
        "evaluation_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "eval_accumulation_steps": eval_accumulation_steps,
        # Logging
        "logging_steps": logging_steps,
        # Saving
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        # Output
        "output_dir": output_dir,
        # Data loading
        "dataloader_num_workers": _as_int(getattr(config, 'dataloader_num_workers', 4), 4),
        "dataloader_pin_memory": getattr(config, 'dataloader_pin_memory', True),
        # Optional in some versions
        "dataloader_prefetch_factor": _as_int(getattr(config, 'dataloader_prefetch_factor', 2), 2),
        # Performance
        "group_by_length": getattr(config, 'group_by_length', True),
        "remove_unused_columns": getattr(config, 'remove_unused_columns', True),
        # HuggingFace Hub
        "push_to_hub": push_to_hub,
        # Monitoring
        "report_to": ("trackio" if getattr(config, 'enable_tracking', False) else None),
    }

    # Drop any None-valued kwargs
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if v is not None}

    # Adapt to transformers versions where 'evaluation_strategy' was renamed
    try:
        ta_sig = inspect.signature(TrainingArguments.__init__)
        param_names = set(ta_sig.parameters.keys())
    except Exception:
        param_names = set()

    if "evaluation_strategy" not in param_names and "eval_strategy" in param_names:
        # Move value to 'eval_strategy'
        ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" not in param_names:
        # If neither is supported, drop it
        ta_kwargs.pop("evaluation_strategy", None)

    # Remove any kwargs not supported by current transformers version
    if param_names:
        unsupported = [k for k in ta_kwargs.keys() if k not in param_names]
        for k in unsupported:
            ta_kwargs.pop(k, None)

    sft_config = TrainingArguments(**ta_kwargs)
    
    return sft_config

def train_gpt_oss(config_path, experiment_name, output_dir, trackio_url, trainer_type="sft"):
    """Main training function for GPT-OSS"""
    
    print("=== GPT-OSS Training Pipeline ===")
    print(f"Config: {config_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Trackio: {trackio_url}")
    print(f"Trainer: {trainer_type}")
    
    # Load configuration
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'config'):
            config = config_module.config
        else:
            # Try to find a config class
            for attr_name in dir(config_module):
                attr = getattr(config_module, attr_name)
                if hasattr(attr, 'model_name') and ('gpt_oss' in attr.model_name.lower() or 'GPTOSS' in attr_name):
                    config = attr
                    break
            else:
                raise ValueError(f"No GPT-OSS configuration found in {config_path}")
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Update config with runtime parameters
    config.experiment_name = experiment_name
    config.trackio_url = trackio_url
    config.trainer_type = trainer_type

    # Optional: scheduler overrides via environment variables set by CLI
    try:
        env_scheduler = os.environ.get("GPT_OSS_SCHEDULER")
        if env_scheduler:
            # Apply scheduler override
            config.scheduler = env_scheduler
            # Prepare/normalize lr scheduler kwargs container
            if not hasattr(config, 'lr_scheduler_kwargs') or config.lr_scheduler_kwargs is None:
                config.lr_scheduler_kwargs = {}

            # Apply min lr overrides only when using TRL's special scheduler
            if env_scheduler == 'cosine_with_min_lr':
                env_min_lr = os.environ.get("GPT_OSS_MIN_LR")
                env_min_lr_rate = os.environ.get("GPT_OSS_MIN_LR_RATE")
                # Clear conflicting warmup keys to avoid signature issues
                for k in ('warmup_steps', 'num_warmup_steps', 'warmup_ratio'):
                    if k in config.lr_scheduler_kwargs:
                        config.lr_scheduler_kwargs.pop(k, None)
                # Prefer absolute min_lr if provided
                if env_min_lr is not None:
                    try:
                        config.min_lr = float(env_min_lr)
                        config.lr_scheduler_kwargs['min_lr'] = config.min_lr
                        # Remove relative rate if present
                        config.lr_scheduler_kwargs.pop('min_lr_rate', None)
                    except Exception:
                        pass
                elif env_min_lr_rate is not None:
                    try:
                        config.lr_scheduler_kwargs['min_lr_rate'] = float(env_min_lr_rate)
                        # Remove absolute min_lr if present in kwargs (leave config.min_lr untouched)
                        config.lr_scheduler_kwargs.pop('min_lr', None)
                    except Exception:
                        pass
                else:
                    # Ensure at least one constraint exists; prefer absolute from config if valid
                    try:
                        if hasattr(config, 'min_lr') and config.min_lr is not None:
                            config.lr_scheduler_kwargs['min_lr'] = float(config.min_lr)
                        else:
                            config.lr_scheduler_kwargs.setdefault('min_lr_rate', 0.1)
                    except Exception:
                        config.lr_scheduler_kwargs.setdefault('min_lr_rate', 0.1)
            else:
                # Non-TRL scheduler: strip TRL-specific keys to avoid unexpected kwargs
                if hasattr(config, 'lr_scheduler_kwargs') and isinstance(config.lr_scheduler_kwargs, dict):
                    for k in ('min_lr', 'min_lr_rate'):
                        config.lr_scheduler_kwargs.pop(k, None)
    except Exception:
        pass
    
    # Load model and tokenizer
    model, tokenizer = load_gpt_oss_model_and_tokenizer(config)
    
    # Setup LoRA
    peft_model = setup_lora_for_gpt_oss(model, config)
    
    # Load dataset
    dataset = load_dataset_from_config(config)

    # Split into train/eval/test
    train_dataset, eval_dataset, test_dataset = split_dataset(dataset, config)
    
    # Ensure TRACKIO_URL env is set so SmolLM3Monitor picks it up
    if trackio_url and not os.environ.get('TRACKIO_URL'):
        os.environ['TRACKIO_URL'] = trackio_url
        os.environ.setdefault('TRACKIO_SPACE_ID', trackio_url)

    # Setup Trackio tracking (Space API client) and monitoring (dataset + Space)
    trackio_client = setup_trackio_tracking(config)
    # Create unified monitor to ensure metrics get logged to dataset/Space
    monitor = None
    try:
        from monitoring import SmolLM3Monitor
        monitor = SmolLM3Monitor(
            experiment_name=experiment_name,
            trackio_url=trackio_url,
            trackio_token=getattr(config, 'trackio_token', None) or os.environ.get('HF_TOKEN'),
            enable_tracking=True,
            log_artifacts=True,
            log_metrics=True,
            log_config=True,
            hf_token=os.environ.get('HF_TOKEN'),
            dataset_repo=os.environ.get('TRACKIO_DATASET_REPO')
        )
        # Log configuration once
        try:
            cfg_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('_') and not callable(getattr(config, k))}
            monitor.log_configuration(cfg_dict)
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: failed to initialize monitor: {e}")
    
    # Initialize project monitor (HF Datasets + Trackio Space if configured)
    monitor_callback = None
    if create_monitor_from_config is not None:
        try:
            project_monitor = create_monitor_from_config(config, experiment_name=experiment_name)
            # Persist configuration immediately
            try:
                cfg_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
                project_monitor.log_config(cfg_dict)
            except Exception:
                pass
            # Create callback for SFTTrainer
            monitor_callback = project_monitor.create_monitoring_callback()
            # If we didn't initialize the explicit monitor above, use this one for summary/close
            if monitor is None:
                monitor = project_monitor
        except Exception:
            pass
    
    # Create SFT configuration
    sft_config = create_sft_config(config, output_dir)
    
    # Create trainer with version-robust kwargs
    if trainer_type == 'dpo':
        if DPOTrainer is None:
            raise RuntimeError("DPOTrainer is not available in this TRL version. Please upgrade 'trl'.")

        print("Creating DPO trainer...")
        try:
            dpo_sig = inspect.signature(DPOTrainer.__init__)
            dpo_params = set(dpo_sig.parameters.keys())
        except Exception:
            dpo_params = {"model", "args", "train_dataset", "tokenizer", "beta", "prompt_column", "chosen_column", "rejected_column"}

        dpo_kwargs = {
            "model": peft_model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "beta": getattr(config, 'dpo_beta', 0.1),
        }

        if "tokenizer" in dpo_params:
            dpo_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in dpo_params:
            dpo_kwargs["processing_class"] = tokenizer

        if "prompt_column" in dpo_params:
            dpo_kwargs["prompt_column"] = "prompt"
        if "chosen_column" in dpo_params:
            dpo_kwargs["chosen_column"] = "chosen"
        if "rejected_column" in dpo_params:
            dpo_kwargs["rejected_column"] = "rejected"

        # Remove Nones
        dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if v is not None}

        # Pass eval dataset if supported
        if "eval_dataset" in dpo_params and eval_dataset is not None:
            dpo_kwargs["eval_dataset"] = eval_dataset
        trainer = DPOTrainer(**dpo_kwargs)
    else:
        print("Creating SFT trainer...")
        try:
            sft_sig = inspect.signature(SFTTrainer.__init__)
            sft_params = set(sft_sig.parameters.keys())
        except Exception:
            sft_params = {"model", "args", "train_dataset", "tokenizer", "dataset_text_field", "max_seq_length"}

        sft_kwargs = {
            "model": peft_model,
            "args": sft_config,
            "train_dataset": train_dataset,
        }

        # Prefer passing tokenizer if supported; otherwise try processing_class
        if "tokenizer" in sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in sft_params:
            sft_kwargs["processing_class"] = tokenizer

        # Pass dataset text field if supported (we produced a 'text' column)
        if "dataset_text_field" in sft_params:
            sft_kwargs["dataset_text_field"] = "text"

        # Pass max sequence length if supported
        if "max_seq_length" in sft_params:
            sft_kwargs["max_seq_length"] = getattr(config, 'max_seq_length', 2048)

        # Enable sequence packing if supported by TRL (speeds up token utilization)
        if "packing" in sft_params:
            sft_kwargs["packing"] = getattr(config, 'packing', False)

        # Attach monitoring callback if supported
        if "callbacks" in sft_params:
            sft_kwargs["callbacks"] = ([monitor_callback] if monitor_callback is not None else [])

        # Attach monitoring callback if supported
        if monitor is not None:
            try:
                if "callbacks" in sft_params:
                    sft_kwargs["callbacks"] = [monitor.create_monitoring_callback()]
            except Exception:
                pass

        # Remove any None values
        sft_kwargs = {k: v for k, v in sft_kwargs.items() if v is not None}

        # Attach eval_dataset if supported
        if "eval_dataset" in sft_params and eval_dataset is not None:
            sft_kwargs["eval_dataset"] = eval_dataset
        trainer = SFTTrainer(**sft_kwargs)
    
    # Start training
    print("Starting GPT-OSS training...")
    try:
        trainer.train()
    finally:
        # Ensure periodic metrics are flushed at the end even if interrupted
        try:
            if monitor is not None:
                monitor._save_to_hf_dataset({'status': 'running'})
        except Exception:
            pass
    
    # Save model
    print("Saving trained model...")
    trainer.save_model(output_dir)
    
    # Push to hub if enabled
    if sft_config.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")
    
    # Log training summary and close monitor
    try:
        if monitor is not None:
            summary = {
                'output_dir': output_dir,
                'model_name': getattr(config, 'model_name', 'unknown'),
            }
            monitor.log_training_summary(summary)
            monitor.close()
    except Exception:
        pass

    # Close monitor cleanly
    try:
        if monitor is not None:
            monitor.close()
    except Exception:
        pass

    print("GPT-OSS training completed successfully!")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Training Script")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--trackio-url", help="Trackio URL for monitoring")
    parser.add_argument("--trainer-type", default="sft", choices=["sft", "dpo"], help="Trainer type")
    # Optional LR scheduler overrides (applied across any GPT-OSS config)
    parser.add_argument(
        "--scheduler",
        choices=["linear", "cosine", "cosine_with_min_lr", "constant"],
        help="Override LR scheduler for this run",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        dest="min_lr",
        help="Absolute floor for LR (used when scheduler is 'cosine_with_min_lr')",
    )
    parser.add_argument(
        "--min-lr-rate",
        type=float,
        dest="min_lr_rate",
        help="Relative LR floor rate in (0,1) for TRL scheduler (used when scheduler is 'cosine_with_min_lr')",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # If provided, expose scheduler overrides via environment so they can be picked up consistently
        # across helper functions if needed.
        if args.scheduler:
            os.environ["GPT_OSS_SCHEDULER"] = args.scheduler
        if args.min_lr is not None:
            os.environ["GPT_OSS_MIN_LR"] = str(args.min_lr)
        if args.min_lr_rate is not None:
            os.environ["GPT_OSS_MIN_LR_RATE"] = str(args.min_lr_rate)

        trainer = train_gpt_oss(
            config_path=args.config,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            trackio_url=args.trackio_url,
            trainer_type=args.trainer_type
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 