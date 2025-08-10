#!/usr/bin/env python3
"""
Gradio Interface for SmolLM3/GPT-OSS Fine-tuning Pipeline

This app mirrors the core flow of launch.sh with a click-and-run UI.
Tokens are read from environment variables:
  - HF_WRITE_TOKEN (required)
  - HF_READ_TOKEN (optional; used to switch the Trackio Space token after training)

Key steps (configurable via UI):
  1) Optional HF Dataset repo setup for Trackio
  2) Optional Trackio Space deployment
  3) Training (SmolLM3 or GPT-OSS)
  4) Push trained model to the HF Hub
  5) Optional switch Trackio HF_TOKEN to read token

This uses the existing scripts in scripts/ and config/ to avoid code duplication.
"""

from __future__ import annotations

import os
import sys
import time
import json
import shlex
import traceback
import importlib.util
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Generator, Optional, Tuple

# Third-party
try:
    import gradio as gr  # type: ignore
except Exception as _e:
    raise RuntimeError(
        "Gradio is required. Please install it first: pip install gradio"
    ) from _e


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


def mask_token(token: Optional[str]) -> str:
    if not token:
        return "<not set>"
    token = str(token)
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}****{token[-4:]}"


def get_python() -> str:
    return sys.executable or "python"


def get_username_from_token(token: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi  # type: ignore
        api = HfApi(token=token)
        info = api.whoami()
        if isinstance(info, dict):
            return info.get("name") or info.get("username")
        if isinstance(info, str):
            return info
    except Exception:
        return None
    return None


def detect_nvidia_driver() -> Tuple[bool, str]:
    """Detect NVIDIA driver/GPU presence with multiple strategies.

    Returns (available, human_message).
    """
    # 1) Try torch CUDA
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                num = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(num)]
                return True, f"NVIDIA GPU detected: {', '.join(names)}"
            except Exception:
                return True, "NVIDIA GPU detected (torch.cuda available)"
    except Exception:
        pass

    # 2) Try NVML via pynvml
    try:
        import pynvml  # type: ignore
        try:
            pynvml.nvmlInit()
            cnt = pynvml.nvmlDeviceGetCount()
            names = []
            for i in range(cnt):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                names.append(pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore"))
            drv = pynvml.nvmlSystemGetDriverVersion().decode("utf-8", errors="ignore")
            pynvml.nvmlShutdown()
            if cnt > 0:
                return True, f"NVIDIA driver {drv}; GPUs: {', '.join(names)}"
        except Exception:
            pass
    except Exception:
        pass

    # 3) Try nvidia-smi
    try:
        import subprocess
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return True, res.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return False, "No NVIDIA driver/GPU detected"


def duplicate_space_hint() -> str:
    space_id = os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID")
    if space_id:
        space_url = f"https://huggingface.co/spaces/{space_id}"
        dup_url = f"{space_url}?duplicate=true"
        return (
            f"ℹ️ No NVIDIA driver detected. If you're on Hugging Face Spaces, "
            f"please duplicate this Space to GPU hardware: [Duplicate this Space]({dup_url})."
        )
    return (
        "ℹ️ No NVIDIA driver detected. To enable training, run on a machine with an NVIDIA GPU/driver "
        "or duplicate this Space on Hugging Face with GPU hardware."
    )


def markdown_links_to_html(text: str) -> str:
    """Convert simple Markdown links [text](url) to HTML anchors for UI rendering."""
    try:
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text)
    except Exception:
        return text


def _write_generated_config(filename: str, content: str) -> Path:
    """Write a generated config under config/ and return the full path."""
    cfg_dir = PROJECT_ROOT / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def generate_medical_o1_config_file(
    dataset_config: str,
    system_message: Optional[str],
    developer_message: Optional[str],
    num_train_epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_length: int,
) -> Path:
    """Create a GPT-OSS Medical o1 SFT config file from user inputs."""
    # Sanitize quotes in messages
    def _q(s: Optional[str]) -> str:
        if s is None or s == "":
            return "None"
        return repr(s)

    py = f"""
from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

config = GPTOSSEnhancedCustomConfig(
    dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
    dataset_config={repr(dataset_config)},
    dataset_split="train",
    dataset_format="medical_o1_sft",

    # Field mapping and prefixes
    input_field="Question",
    target_field="Response",
    question_field="Question",
    reasoning_field="Complex_CoT",
    response_field="Response",
    reason_prefix="Reasoning: ",
    answer_prefix="Final Answer: ",

    # Optional context
    system_message={_q(system_message)},
    developer_message={_q(developer_message)},

    # Training hyperparameters
    num_train_epochs={num_train_epochs},
    batch_size={batch_size},
    gradient_accumulation_steps={gradient_accumulation_steps},
    learning_rate={learning_rate},
    min_lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,

    # Sequence length
    max_seq_length={max_seq_length},

    # Precision & performance
    fp16=False,
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
    group_by_length=True,
    remove_unused_columns=True,

    # LoRA & quantization
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
    },

    # Logging & evaluation
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
"""
    return _write_generated_config("_generated_gpt_oss_medical_o1_sft.py", py)


def generate_gpt_oss_custom_config_file(
    dataset_name: str,
    dataset_split: str,
    dataset_format: str,
    input_field: str,
    target_field: Optional[str],
    system_message: Optional[str],
    developer_message: Optional[str],
    model_identity: Optional[str],
    max_samples: Optional[int],
    min_length: int,
    max_length: Optional[int],
    num_train_epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    mixed_precision: str,  # "bf16"|"fp16"|"fp32"
    num_workers: int,
    quantization_type: str,  # "mxfp4"|"bnb4"|"none"
    max_grad_norm: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
) -> Path:
    # Precision flags
    if mixed_precision.lower() == "bf16":
        fp16_flag = False
        bf16_flag = True
    elif mixed_precision.lower() == "fp16":
        fp16_flag = True
        bf16_flag = False
    else:
        fp16_flag = False
        bf16_flag = False

    # Quantization flags/config
    if quantization_type == "mxfp4":
        use_quant = True
        quant_cfg = '{"dequantize": True, "load_in_4bit": False}'
    elif quantization_type == "bnb4":
        use_quant = True
        quant_cfg = '{"dequantize": False, "load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True, "bnb_4bit_quant_type": "nf4"}'
    else:
        use_quant = False
        quant_cfg = '{"dequantize": False, "load_in_4bit": False}'

    def _q(s: Optional[str]) -> str:
        if s is None or s == "":
            return "None"
        return repr(s)

    py = f"""
from config.train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

config = GPTOSSEnhancedCustomConfig(
    # Dataset
    dataset_name={repr(dataset_name)},
    dataset_split={repr(dataset_split)},
    dataset_format={repr(dataset_format)},
    input_field={repr(input_field)},
    target_field={repr(target_field)} if {repr(target_field)} != 'None' else None,
    system_message={_q(system_message)},
    developer_message={_q(developer_message)},
    max_samples={repr(max_samples)} if {repr(max_samples)} != 'None' else None,
    min_length={min_length},
    max_length={repr(max_length)} if {repr(max_length)} != 'None' else None,

    # Training hyperparameters
    num_train_epochs={num_train_epochs},
    batch_size={batch_size},
    gradient_accumulation_steps={gradient_accumulation_steps},
    learning_rate={learning_rate},
    min_lr={min_lr},
    weight_decay={weight_decay},
    warmup_ratio={warmup_ratio},
    max_grad_norm={max_grad_norm},

    # Model
    max_seq_length={max_seq_length},

    # Precision
    fp16={str(fp16_flag)},
    bf16={str(bf16_flag)},

    # LoRA
    lora_config={{
        "r": {lora_r},
        "lora_alpha": {lora_alpha},
        "lora_dropout": {lora_dropout},
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }},

    # Quantization
    use_quantization={str(use_quant)},
    quantization_config={quant_cfg},

    # Performance
    dataloader_num_workers={num_workers},
    dataloader_pin_memory=True,
    group_by_length=True,

    # Logging & eval
    logging_steps={logging_steps},
    eval_steps={eval_steps},
    save_steps={save_steps},
    
    # Chat template (Harmony)
    chat_template_kwargs={{
        "add_generation_prompt": True,
        "tokenize": False,
        "auto_insert_role": True,
        "reasoning_effort": "medium",
        "model_identity": {_q(model_identity) if _q(model_identity) != 'None' else repr('You are GPT-Tonic, a large language model trained by TonicAI.')},
        "builtin_tools": [],
    }},
)
"""
    return _write_generated_config("_generated_gpt_oss_custom.py", py)


def generate_smollm3_custom_config_file(
    model_name: str,
    dataset_name: Optional[str],
    max_seq_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    filter_bad_entries: bool,
    input_field: str,
    target_field: str,
    sample_size: Optional[int],
    sample_seed: int,
    trainer_type: str,
) -> Path:
    # Create subclass to include dataset fields similar to other configs
    def _bool(b: bool) -> str:
        return "True" if b else "False"

    ds_section = """
    # HF Dataset configuration
    dataset_name={}
    dataset_split="train"
    input_field={}
    target_field={}
    filter_bad_entries={}
    bad_entry_field="bad_entry"
    sample_size={}
    sample_seed={}
    """.format(
        repr(dataset_name) if dataset_name else "None",
        repr(input_field),
        repr(target_field),
        _bool(filter_bad_entries),
        repr(sample_size) if sample_size is not None else "None",
        sample_seed,
    )

    py = f"""
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3GeneratedConfig(SmolLM3Config):
{ds_section}

config = SmolLM3GeneratedConfig(
    trainer_type={repr(trainer_type.lower())},
    model_name={repr(model_name)},
    max_seq_length={max_seq_length},
    use_flash_attention=True,
    use_gradient_checkpointing=True,

    batch_size={batch_size},
    gradient_accumulation_steps={gradient_accumulation_steps},
    learning_rate={learning_rate},
    weight_decay=0.01,
    warmup_steps=100,
    max_iters=None,
    eval_interval={eval_steps},
    log_interval={logging_steps},
    save_interval={save_steps},

    optimizer="adamw",
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    scheduler="cosine",
    min_lr=1e-6,
    fp16=True,
    bf16=False,
    save_steps={save_steps},
    eval_steps={eval_steps},
    logging_steps={logging_steps},
    save_total_limit=3,
    eval_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
)
"""
    return _write_generated_config("_generated_smollm3_custom.py", py)


def generate_smollm3_long_context_config_file(
    model_name: str,
    dataset_name: Optional[str],
    input_field: str,
    target_field: str,
    filter_bad_entries: bool,
    sample_size: Optional[int],
    sample_seed: int,
    max_seq_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    max_iters: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
    use_chat_template: bool,
    no_think_system_message: bool,
    trainer_type: str,
) -> Path:
    """Create a SmolLM3 long-context config file with optional dataset fields."""
    def _bool(b: bool) -> str:
        return "True" if b else "False"

    ds_section = """
    # HF Dataset configuration
    dataset_name={}
    dataset_split="train"
    input_field={}
    target_field={}
    filter_bad_entries={}
    bad_entry_field="bad_entry"
    sample_size={}
    sample_seed={}
    """.format(
        repr(dataset_name) if dataset_name else "None",
        repr(input_field),
        repr(target_field),
        _bool(filter_bad_entries),
        repr(sample_size) if sample_size is not None else "None",
        sample_seed,
    )

    py = f"""
from dataclasses import dataclass
from typing import Optional
from config.train_smollm3 import SmolLM3Config

@dataclass
class SmolLM3LongContextGeneratedConfig(SmolLM3Config):
{ds_section}

config = SmolLM3LongContextGeneratedConfig(
    trainer_type={repr(trainer_type.lower())},
    model_name={repr(model_name)},
    max_seq_length={max_seq_length},
    use_flash_attention=True,
    use_gradient_checkpointing=True,

    batch_size={batch_size},
    gradient_accumulation_steps={gradient_accumulation_steps},
    learning_rate={learning_rate},
    weight_decay=0.01,
    warmup_steps={warmup_steps},
    max_iters={max_iters},

    fp16=True,
    bf16=False,
    save_steps={save_steps},
    eval_steps={eval_steps},
    logging_steps={logging_steps},
    save_total_limit=3,
    eval_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

    use_chat_template={_bool(use_chat_template)},
    chat_template_kwargs={{
        "add_generation_prompt": True,
        "no_think_system_message": {_bool(no_think_system_message)}
    }}
)
"""
    return _write_generated_config("_generated_smollm3_long_context.py", py)

def ensure_dataset_repo(username: str, dataset_name: str, token: str) -> Tuple[str, bool, str]:
    """Create or ensure dataset repo exists. Returns (repo_id, created_or_exists, message)."""
    from huggingface_hub import create_repo  # type: ignore
    repo_id = f"{username}/{dataset_name}"
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True, private=False)
        return repo_id, True, f"Dataset repo ready: {repo_id}"
    except Exception as e:
        return repo_id, False, f"Failed to create dataset repo {repo_id}: {e}"


def import_config_object(config_path: Path) -> Optional[Any]:
    """Import a config file and return its 'config' object if present, else None."""
    try:
        spec = importlib.util.spec_from_file_location("config_module", str(config_path))
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "config"):
            return getattr(module, "config")
        return None
    except Exception:
        return None


def run_command_stream(args: list[str], env: Dict[str, str], cwd: Optional[Path] = None) -> Generator[str, None, int]:
    """Run a command and yield stdout/stderr lines as they arrive. Returns exit code at the end."""
    import subprocess

    yield f"$ {' '.join(shlex.quote(a) for a in ([get_python()] + args))}"
    process = subprocess.Popen(
        [get_python()] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(cwd or PROJECT_ROOT),
        bufsize=1,
        universal_newlines=True,
    )
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        yield line.rstrip()
    process.stdout.close()
    code = process.wait()
    yield f"[exit_code={code}]"
    return code


# --------------------------------------------------------------------------------------
# Configuration Mappings (mirror launch.sh)
# --------------------------------------------------------------------------------------

SMOL_CONFIGS = {
    "Basic Training": {
        "config_file": "config/train_smollm3.py",
        "default_model": "HuggingFaceTB/SmolLM3-3B",
    },
    "H100 Lightweight (Rapid)": {
        "config_file": "config/train_smollm3_h100_lightweight.py",
        "default_model": "HuggingFaceTB/SmolLM3-3B",
    },
    "A100 Large Scale": {
        "config_file": "config/train_smollm3_openhermes_fr_a100_large.py",
        "default_model": "HuggingFaceTB/SmolLM3-3B",
    },
    "Multiple Passes": {
        "config_file": "config/train_smollm3_openhermes_fr_a100_multiple_passes.py",
        "default_model": "HuggingFaceTB/SmolLM3-3B",
    },
}

GPT_OSS_CONFIGS = {
    "GPT-OSS Basic Training": {
        "config_file": "config/train_gpt_oss_basic.py",
        "default_model": "openai/gpt-oss-20b",
    },
    "GPT-OSS H100 Optimized": {
        "config_file": "config/train_gpt_oss_h100_optimized.py",
        "default_model": "openai/gpt-oss-20b",
    },
    "GPT-OSS Multilingual Reasoning": {
        "config_file": "config/train_gpt_oss_multilingual_reasoning.py",
        "default_model": "openai/gpt-oss-20b",
    },
    "GPT-OSS Memory Optimized": {
        "config_file": "config/train_gpt_oss_memory_optimized.py",
        "default_model": "openai/gpt-oss-20b",
    },
    "GPT-OSS OpenHermes-FR (Recommended)": {
        "config_file": "config/train_gpt_oss_openhermes_fr.py",
        "default_model": "openai/gpt-oss-20b",
    },
    "GPT-OSS OpenHermes-FR Memory Optimized": {
        "config_file": "config/train_gpt_oss_openhermes_fr_memory_optimized.py",
        "default_model": "openai/gpt-oss-20b",
    },
    # Custom dataset and medical SFT can be added later as advanced UI panels
}


def get_config_map(family: str) -> Dict[str, Dict[str, str]]:
    return SMOL_CONFIGS if family == "SmolLM3" else GPT_OSS_CONFIGS


# --------------------------------------------------------------------------------------
# Pipeline Orchestration
# --------------------------------------------------------------------------------------

@dataclass
class PipelineInputs:
    model_family: str
    config_choice: str
    trainer_type: str  # "SFT" | "DPO"
    monitoring_mode: str  # "both" | "trackio" | "dataset" | "none"
    experiment_name: str
    repo_short: str
    author_name: str
    model_description: str
    trackio_space_name: Optional[str]
    deploy_trackio_space: bool
    create_dataset_repo: bool
    push_to_hub: bool
    switch_to_read_after: bool
    scheduler_override: Optional[str]
    min_lr: Optional[float]
    min_lr_rate: Optional[float]
    # Optional override config path generated from Advanced tab
    override_config_path: Optional[str] = None


def make_defaults(model_family: str) -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    family_slug = "gpt-oss" if model_family == "GPT-OSS" else "smollm3"
    exp = f"smolfactory-{family_slug}_{ts}"
    repo_short = f"smolfactory-{datetime.now().strftime('%Y%m%d')}"
    return exp, repo_short


def run_pipeline(params: PipelineInputs) -> Generator[str, None, None]:
    # Tokens from environment
    write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    read_token = os.environ.get("HF_READ_TOKEN")

    if not write_token:
        yield "❌ HF_WRITE_TOKEN (or HF_TOKEN) is not set in the environment."
        return

    # Resolve username
    username = get_username_from_token(write_token) or os.environ.get("HF_USERNAME")
    if not username:
        yield "❌ Could not resolve Hugging Face username from token."
        return
    yield f"✅ Authenticated as: {username}"

    # Compute Trackio URL if applicable
    trackio_url: Optional[str] = None
    if params.monitoring_mode != "none" and params.trackio_space_name:
        trackio_url = f"https://huggingface.co/spaces/{username}/{params.trackio_space_name}"
        yield f"Trackio Space URL: {trackio_url}"

    # Decide space deploy token per monitoring mode
    space_deploy_token = write_token if params.monitoring_mode in ("both", "trackio") else (read_token or write_token)

    # Dataset repo setup
    dataset_repo = f"{username}/trackio-experiments"
    if params.create_dataset_repo and params.monitoring_mode != "none":
        yield f"Creating/ensuring dataset repo exists: {dataset_repo}"
        rid, ok, msg = ensure_dataset_repo(username, "trackio-experiments", write_token)
        yield ("✅ " if ok else "⚠️ ") + msg
        dataset_repo = rid

    # Resolve config file and model name (allow override from Advanced tab)
    conf_map = get_config_map(params.model_family)
    if params.override_config_path:
        config_file = Path(params.override_config_path)
        if not config_file.exists():
            yield f"❌ Generated config file not found: {config_file}"
            return
        # Best-effort to infer base model from generated config
        cfg_obj = import_config_object(config_file)
        base_model_fallback = getattr(cfg_obj, "model_name", None) or (
            conf_map.get(params.config_choice, {}).get("default_model", "")
        )
    else:
        if params.config_choice not in conf_map:
            yield f"❌ Unknown config choice: {params.config_choice}"
            return
        config_file = PROJECT_ROOT / conf_map[params.config_choice]["config_file"]
        base_model_fallback = conf_map[params.config_choice]["default_model"]
        if not config_file.exists():
            yield f"❌ Config file not found: {config_file}"
            return
        cfg_obj = import_config_object(config_file)
    base_model = getattr(cfg_obj, "model_name", base_model_fallback) if cfg_obj else base_model_fallback
    dataset_name = getattr(cfg_obj, "dataset_name", None) if cfg_obj else None
    batch_size = getattr(cfg_obj, "batch_size", None) if cfg_obj else None
    learning_rate = getattr(cfg_obj, "learning_rate", None) if cfg_obj else None
    max_seq_length = getattr(cfg_obj, "max_seq_length", None) if cfg_obj else None

    # Prepare env for subprocesses
    env = os.environ.copy()
    env["HF_TOKEN"] = write_token
    env["HUGGING_FACE_HUB_TOKEN"] = write_token
    env["HF_USERNAME"] = username
    env["TRACKIO_DATASET_REPO"] = dataset_repo
    env["MONITORING_MODE"] = params.monitoring_mode

    # Optional Trackio Space deployment
    if params.deploy_trackio_space and params.monitoring_mode != "none" and params.trackio_space_name:
        yield f"\n=== Deploying Trackio Space: {params.trackio_space_name} ==="
        # deploy_trackio_space.py expects: space_name, token, git_email, git_name, dataset_repo
        args = [
            str(PROJECT_ROOT / "scripts/trackio_tonic/deploy_trackio_space.py"),
            params.trackio_space_name,
            space_deploy_token,
            f"{username}@users.noreply.hf.co",
            username,
            dataset_repo,
        ]
        for line in run_command_stream(args, env, cwd=PROJECT_ROOT / "scripts/trackio_tonic"):
            yield line

    # Dataset setup and Trackio configuration (mirror launch.sh) when monitoring is enabled
    if params.monitoring_mode != "none":
        # Ensure HF Dataset structure
        yield f"\n=== Setting up HF Dataset: {dataset_repo} ==="
        ds_args = [
            str(PROJECT_ROOT / "scripts/dataset_tonic/setup_hf_dataset.py"),
            write_token,
        ]
        for line in run_command_stream(ds_args, env, cwd=PROJECT_ROOT / "scripts/dataset_tonic"):
            yield line
        # Configure Trackio Space
        yield f"\n=== Configuring Trackio Space ({params.trackio_space_name or 'N/A'}) ==="
        conf_args = [str(PROJECT_ROOT / "scripts/trackio_tonic/configure_trackio.py")]
        # Use space deploy token (READ for dataset-only; WRITE otherwise)
        conf_env = env.copy()
        conf_env["HF_TOKEN"] = space_deploy_token
        conf_env["HUGGING_FACE_HUB_TOKEN"] = space_deploy_token
        for line in run_command_stream(conf_args, conf_env, cwd=PROJECT_ROOT / "scripts/trackio_tonic"):
            yield line

    # Training output directory
    out_dir = PROJECT_ROOT / "outputs" / f"{params.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    yield f"\nOutput directory: {out_dir}"

    # Scheduler overrides (GPT-OSS only)
    if params.model_family == "GPT-OSS" and params.scheduler_override:
        env["GPT_OSS_SCHEDULER"] = params.scheduler_override
        if params.min_lr is not None:
            env["GPT_OSS_MIN_LR"] = str(params.min_lr)
        if params.min_lr_rate is not None:
            env["GPT_OSS_MIN_LR_RATE"] = str(params.min_lr_rate)

    # Start training
    yield f"\n=== Starting Training ({params.model_family}) ==="
    if params.model_family == "GPT-OSS":
        args = [
            str(PROJECT_ROOT / "scripts/training/train_gpt_oss.py"),
            "--config", str(config_file),
            "--experiment-name", params.experiment_name,
            "--output-dir", str(out_dir),
            "--trackio-url", trackio_url or "",
            "--trainer-type", params.trainer_type.lower(),
        ]
    else:
        args = [
            str(PROJECT_ROOT / "scripts/training/train.py"),
            "--config", str(config_file),
            "--experiment-name", params.experiment_name,
            "--output-dir", str(out_dir),
            "--trackio-url", trackio_url or "",
            "--trainer-type", params.trainer_type.lower(),
        ]

    # Stream training logs
    train_failed = False
    for line in run_command_stream(args, env):
        yield line
        if line.strip().startswith("[exit_code=") and not line.strip().endswith("[exit_code=0]"):
            train_failed = True
    if train_failed:
        yield "❌ Training failed. Aborting remaining steps."
        return

    # Push to Hub
    if params.push_to_hub:
        yield "\n=== Pushing Model to Hugging Face Hub ==="
        repo_name = f"{username}/{params.repo_short}"
        if params.model_family == "GPT-OSS":
            push_args = [
                str(PROJECT_ROOT / "scripts/model_tonic/push_gpt_oss_to_huggingface.py"),
                str(out_dir),
                repo_name,
                "--token", write_token,
                "--trackio-url", trackio_url or "",
                "--experiment-name", params.experiment_name,
                "--dataset-repo", dataset_repo,
                "--author-name", params.author_name or username,
                "--model-description", params.model_description,
                "--training-config-type", params.config_choice,
                "--model-name", base_model,
            ]
            if dataset_name:
                push_args += ["--dataset-name", str(dataset_name)]
            if batch_size is not None:
                push_args += ["--batch-size", str(batch_size)]
            if learning_rate is not None:
                push_args += ["--learning-rate", str(learning_rate)]
            if max_seq_length is not None:
                push_args += ["--max-seq-length", str(max_seq_length)]
            push_args += ["--trainer-type", params.trainer_type]
        else:
            push_args = [
                str(PROJECT_ROOT / "scripts/model_tonic/push_to_huggingface.py"),
                str(out_dir),
                repo_name,
                "--token", write_token,
                "--trackio-url", trackio_url or "",
                "--experiment-name", params.experiment_name,
                "--dataset-repo", dataset_repo,
                "--author-name", params.author_name or username,
                "--model-description", params.model_description,
                "--training-config-type", params.config_choice,
                "--model-name", base_model,
            ]
            if dataset_name:
                push_args += ["--dataset-name", str(dataset_name)]
            if batch_size is not None:
                push_args += ["--batch-size", str(batch_size)]
            if learning_rate is not None:
                push_args += ["--learning-rate", str(learning_rate)]
            if max_seq_length is not None:
                push_args += ["--max-seq-length", str(max_seq_length)]
            push_args += ["--trainer-type", params.trainer_type]

        for line in run_command_stream(push_args, env):
            yield line

    # Switch Space token to read-only (security)
    if params.switch_to_read_after and params.monitoring_mode in ("both", "trackio") and params.trackio_space_name and read_token:
        yield "\n=== Switching Trackio Space HF_TOKEN to READ token ==="
        space_id = f"{username}/{params.trackio_space_name}"
        sw_args = [
            str(PROJECT_ROOT / "scripts/trackio_tonic/switch_to_read_token.py"),
            space_id,
            read_token,
            write_token,
        ]
        for line in run_command_stream(sw_args, env, cwd=PROJECT_ROOT / "scripts/trackio_tonic"):
            yield line
    elif params.switch_to_read_after and not read_token:
        yield "⚠️ HF_READ_TOKEN not set; skipping token switch."

    # Final summary
    yield "\n🎉 Pipeline completed."
    if params.monitoring_mode != "none" and trackio_url:
        yield f"Trackio: {trackio_url}"
    yield f"Model repo (if pushed): https://huggingface.co/{username}/{params.repo_short}"
    yield f"Outputs: {out_dir}"


# --------------------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------------------

MODEL_FAMILIES = ["SmolLM3", "GPT-OSS"]
TRAINER_CHOICES = ["SFT", "DPO"]
MONITORING_CHOICES = ["both", "trackio", "dataset", "none"]
SCHEDULER_CHOICES = [None, "linear", "cosine", "cosine_with_min_lr", "constant"]


def ui_defaults(family: str) -> Tuple[str, str, str, str]:
    exp, repo_short = make_defaults(family)
    default_desc = (
        "A fine-tuned GPT-OSS-20B model optimized for multilingual reasoning and instruction following."
        if family == "GPT-OSS"
        else "A fine-tuned SmolLM3-3B model optimized for instruction following and French language tasks."
    )
    trackio_space_name = f"trackio-monitoring-{datetime.now().strftime('%Y%m%d')}"
    return exp, repo_short, default_desc, trackio_space_name


joinus = """
## Join us :
🌟TeamTonic🌟 is always making cool demos! Join our active builder's 🛠️community 👻 [![Join us on Discord](https://img.shields.io/discord/1109943800132010065?label=Discord&logo=discord&style=flat-square)](https://discord.gg/qdfnvSPcqP) On 🤗Huggingface:[MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Tonic-AI](https://github.com/tonic-ai) & contribute to🌟 [Build Tonic](https://git.tonic-ai.com/contribute)🤗Big thanks to Yuvi Sharma and all the folks at huggingface for the community grant 🤗
"""


def on_family_change(family: str):
    """Update UI when the model family changes.

    - Refresh available prebuilt configuration choices
    - Reset defaults (experiment name, repo short, description, space name)
    - Reveal the next step (trainer type)
    """
    confs = list(get_config_map(family).keys())
    exp, repo_short, desc, space = ui_defaults(family)

    # Initial dataset information placeholder until a specific config is chosen
    training_md = (
        f"Select a training configuration for {family} to see details (dataset, batch size, etc.)."
    )

    # Update objects:
    return (
        gr.update(choices=confs, value=(confs[0] if confs else None)),
        exp,
        repo_short,
        desc,
        space,
        training_md,
        gr.update(choices=[], value=None),
        gr.update(visible=True),   # show step 2 (trainer)
        gr.update(visible=True),   # show step 3 immediately (default monitoring 'dataset')
        gr.update(visible=True),   # show step 4 immediately so users see configs
        gr.update(visible=False),  # GPT-OSS advanced group hidden until enabled
        gr.update(visible=False),  # SmolLM3 advanced group hidden until enabled
    )


def on_config_change(family: str, config_choice: str):
    """When a prebuilt configuration is selected, update dataset info and helpful details.

    Also auto-fill advanced fields with defaults from the selected config.
    """
    if not config_choice:
        return (
            "",
            gr.update(choices=[], value=None),
            # Advanced fields (GPT-OSS)
            "", "train", "openhermes_fr", "prompt", "accepted_completion", "", "", "",
            None, 10, None, 1.0, 4, 4, 2e-4, 2e-5, 0.01, 0.03,
            2048, 16, 32, 0.05, "bf16", 4, "mxfp4", 1.0, 10, 100, 500,
            # GPT-OSS Medical o1 SFT defaults
            "default", "", "", 1.0, 4, 4, 2e-4, 2048,
            # Advanced fields (SmolLM3)
            "HuggingFaceTB/SmolLM3-3B", None, "prompt", "completion", False, None, 42,
            4096, 2, 8, 5e-6, 500, 100, 10,
        )

    conf_map = get_config_map(family)
    cfg_path = PROJECT_ROOT / conf_map[config_choice]["config_file"]
    cfg_obj = import_config_object(cfg_path)

    dataset_name = getattr(cfg_obj, "dataset_name", None) if cfg_obj else None
    batch_size = getattr(cfg_obj, "batch_size", None) if cfg_obj else None
    learning_rate = getattr(cfg_obj, "learning_rate", None) if cfg_obj else None
    max_seq_length = getattr(cfg_obj, "max_seq_length", None) if cfg_obj else None
    base_model = conf_map[config_choice]["default_model"]

    md_lines = [
        f"**Configuration**: {config_choice}",
        f"**Base model**: {base_model}",
    ]
    if dataset_name:
        md_lines.append(f"**Dataset**: `{dataset_name}`")
    if batch_size is not None:
        md_lines.append(f"**Batch size**: {batch_size}")
    if learning_rate is not None:
        md_lines.append(f"**Learning rate**: {learning_rate}")
    if max_seq_length is not None:
        md_lines.append(f"**Max seq length**: {max_seq_length}")

    training_md = "\n".join(md_lines)

    # dataset selection (allow custom but prefill with the config's dataset if any)
    ds_choices = [dataset_name] if dataset_name else []

    # Defaults for Advanced (GPT-OSS)
    adv_dataset_name = dataset_name or ("HuggingFaceH4/Multilingual-Thinking" if family == "GPT-OSS" else (dataset_name or ""))
    adv_dataset_split = getattr(cfg_obj, "dataset_split", "train") if cfg_obj else "train"
    # Infer dataset_format heuristically
    if family == "GPT-OSS":
        adv_dataset_format = getattr(cfg_obj, "dataset_format", None) or (
            "messages" if getattr(cfg_obj, "input_field", "") == "messages" else "openhermes_fr"
        )
        adv_input_field = getattr(cfg_obj, "input_field", "prompt")
        adv_target_field = getattr(cfg_obj, "target_field", "accepted_completion") or ""
        adv_num_train_epochs = float(getattr(cfg_obj, "num_train_epochs", 1.0)) if cfg_obj and hasattr(cfg_obj, "num_train_epochs") else 1.0
        adv_batch_size = int(getattr(cfg_obj, "batch_size", 4) or 4)
        adv_gas = int(getattr(cfg_obj, "gradient_accumulation_steps", 4) or 4)
        adv_lr = float(getattr(cfg_obj, "learning_rate", 2e-4) or 2e-4)
        adv_min_lr = float(getattr(cfg_obj, "min_lr", 2e-5) or 2e-5)
        adv_wd = float(getattr(cfg_obj, "weight_decay", 0.01) or 0.01)
        adv_warmup = float(getattr(cfg_obj, "warmup_ratio", 0.03) or 0.03)
        adv_msl = int(getattr(cfg_obj, "max_seq_length", 2048) or 2048)
        lora_cfg = getattr(cfg_obj, "lora_config", {}) or {}
        adv_lora_r = int(lora_cfg.get("r", 16))
        adv_lora_alpha = int(lora_cfg.get("lora_alpha", 32))
        adv_lora_dropout = float(lora_cfg.get("lora_dropout", 0.05))
        adv_mixed_precision = "bf16" if getattr(cfg_obj, "bf16", True) else ("fp16" if getattr(cfg_obj, "fp16", False) else "fp32")
        adv_num_workers = int(getattr(cfg_obj, "dataloader_num_workers", 4) or 4)
        qcfg = getattr(cfg_obj, "quantization_config", {}) or {}
        if qcfg.get("load_in_4bit", False):
            adv_quantization_type = "bnb4"
        elif qcfg.get("dequantize", False):
            adv_quantization_type = "mxfp4"
        else:
            adv_quantization_type = "none"
        adv_mgn = float(getattr(cfg_obj, "max_grad_norm", 1.0) or 1.0)
        adv_log = int(getattr(cfg_obj, "logging_steps", 10) or 10)
        adv_eval = int(getattr(cfg_obj, "eval_steps", 100) or 100)
        adv_save = int(getattr(cfg_obj, "save_steps", 500) or 500)
    else:
        # SmolLM3 defaults for Advanced
        adv_dataset_format = "openhermes_fr"
        adv_input_field = getattr(cfg_obj, "input_field", "prompt") if cfg_obj else "prompt"
        adv_target_field = getattr(cfg_obj, "target_field", "completion") if cfg_obj else "completion"
        adv_num_train_epochs = 1.0
        adv_batch_size = int(getattr(cfg_obj, "batch_size", 2) or 2)
        adv_gas = int(getattr(cfg_obj, "gradient_accumulation_steps", 8) or 8)
        adv_lr = float(getattr(cfg_obj, "learning_rate", 5e-6) or 5e-6)
        adv_min_lr = float(getattr(cfg_obj, "min_lr", 1e-6) or 1e-6)
        adv_wd = float(getattr(cfg_obj, "weight_decay", 0.01) or 0.01)
        adv_warmup = float(getattr(cfg_obj, "warmup_steps", 100) or 100)  # Smol uses steps
        adv_msl = int(getattr(cfg_obj, "max_seq_length", 4096) or 4096)
        adv_lora_r = 16
        adv_lora_alpha = 32
        adv_lora_dropout = 0.05
        adv_mixed_precision = "fp16" if getattr(cfg_obj, "fp16", True) else ("bf16" if getattr(cfg_obj, "bf16", False) else "fp32")
        adv_num_workers = int(getattr(cfg_obj, "dataloader_num_workers", 4) or 4)
        adv_quantization_type = "none"
        adv_mgn = float(getattr(cfg_obj, "max_grad_norm", 1.0) or 1.0)
        adv_log = int(getattr(cfg_obj, "logging_steps", 10) or 10)
        adv_eval = int(getattr(cfg_obj, "eval_steps", 100) or 100)
        adv_save = int(getattr(cfg_obj, "save_steps", 500) or 500)

    # SmolLM3 advanced model/dataset
    adv_sm_model_name = getattr(cfg_obj, "model_name", "HuggingFaceTB/SmolLM3-3B") if cfg_obj else "HuggingFaceTB/SmolLM3-3B"
    adv_sm_dataset_name = dataset_name if family == "SmolLM3" else None
    adv_sm_input_field = adv_input_field
    adv_sm_target_field = adv_target_field
    adv_sm_filter_bad = bool(getattr(cfg_obj, "filter_bad_entries", False)) if cfg_obj else False
    adv_sm_sample_size = getattr(cfg_obj, "sample_size", None)
    adv_sm_sample_seed = getattr(cfg_obj, "sample_seed", 42)

    return (
        training_md,
        gr.update(choices=ds_choices, value=(dataset_name or None)),
        # Advanced (GPT-OSS)
        adv_dataset_name,
        adv_dataset_split,
        adv_dataset_format,
        adv_input_field,
        adv_target_field,
        getattr(cfg_obj, "system_message", None) if cfg_obj else "",
        getattr(cfg_obj, "developer_message", None) if cfg_obj else "",
        getattr(cfg_obj, "chat_template_kwargs", {}).get("model_identity") if cfg_obj and getattr(cfg_obj, "chat_template_kwargs", None) else "",
        getattr(cfg_obj, "max_samples", None) if cfg_obj else None,
        int(getattr(cfg_obj, "min_length", 10) or 10) if cfg_obj else 10,
        getattr(cfg_obj, "max_length", None) if cfg_obj else None,
        adv_num_train_epochs,
        adv_batch_size,
        adv_gas,
        adv_lr,
        adv_min_lr,
        adv_wd,
        adv_warmup,
        adv_msl,
        adv_lora_r,
        adv_lora_alpha,
        adv_lora_dropout,
        adv_mixed_precision,
        adv_num_workers,
        adv_quantization_type,
        adv_mgn,
        adv_log,
        adv_eval,
        adv_save,
        # GPT-OSS Medical o1 SFT defaults
        "default",
        "",
        "",
        1.0,
        4,
        4,
        2e-4,
        2048,
        # Advanced (SmolLM3)
        adv_sm_model_name,
        adv_sm_dataset_name,
        adv_sm_input_field,
        adv_sm_target_field,
        adv_sm_filter_bad,
        adv_sm_sample_size,
        adv_sm_sample_seed,
        # SmolLM3 training overrides
        int(getattr(cfg_obj, "max_seq_length", 4096) or 4096) if family == "SmolLM3" else 4096,
        int(getattr(cfg_obj, "batch_size", 2) or 2) if family == "SmolLM3" else 2,
        int(getattr(cfg_obj, "gradient_accumulation_steps", 8) or 8) if family == "SmolLM3" else 8,
        float(getattr(cfg_obj, "learning_rate", 5e-6) or 5e-6) if family == "SmolLM3" else 5e-6,
        int(getattr(cfg_obj, "save_steps", 500) or 500) if family == "SmolLM3" else 500,
        int(getattr(cfg_obj, "eval_steps", 100) or 100) if family == "SmolLM3" else 100,
        int(getattr(cfg_obj, "logging_steps", 10) or 10) if family == "SmolLM3" else 10,
    )


def on_trainer_selected(_: str):
    """Reveal monitoring step once trainer type is chosen."""
    return gr.update(visible=True)


def on_monitoring_change(mode: str):
    """Reveal configuration/details step and adjust Trackio-related visibility by mode."""
    show_trackio = mode in ("both", "trackio")
    show_dataset_repo = mode != "none"
    return (
        gr.update(visible=True),
        gr.update(visible=show_trackio),  # trackio space name
        gr.update(visible=show_trackio),  # deploy trackio space
        gr.update(visible=show_dataset_repo),  # create dataset repo
    )


def start_pipeline(
    model_family: str,
    config_choice: str,
    trainer_type: str,
    monitoring_mode: str,
    experiment_name: str,
    repo_short: str,
    author_name: str,
    model_description: str,
    trackio_space_name: str,
    deploy_trackio_space: bool,
    create_dataset_repo: bool,
    push_to_hub: bool,
    switch_to_read_after: bool,
    scheduler_override: Optional[str],
    min_lr: Optional[float],
    min_lr_rate: Optional[float],
) -> Generator[str, None, None]:
    try:
        params = PipelineInputs(
            model_family=model_family,
            config_choice=config_choice,
            trainer_type=trainer_type,
            monitoring_mode=monitoring_mode,
            experiment_name=experiment_name,
            repo_short=repo_short,
            author_name=author_name,
            model_description=model_description,
            trackio_space_name=trackio_space_name or None,
            deploy_trackio_space=deploy_trackio_space,
            create_dataset_repo=create_dataset_repo,
            push_to_hub=push_to_hub,
            switch_to_read_after=switch_to_read_after,
            scheduler_override=(scheduler_override or None),
            min_lr=min_lr,
            min_lr_rate=min_lr_rate,
        )

        # Show token presence
        write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
        read_token = os.environ.get("HF_READ_TOKEN")
        yield f"HF_WRITE_TOKEN: {mask_token(write_token)}"
        yield f"HF_READ_TOKEN:  {mask_token(read_token)}"

        # Run the orchestrated pipeline
        for line in run_pipeline(params):
            yield line
            # Small delay for smoother streaming
            time.sleep(0.01)
    except Exception as e:
        yield f"❌ Error: {e}"
        tb = traceback.format_exc(limit=2)
        yield tb


with gr.Blocks(title="SmolLM3 / GPT-OSS Fine-tuning Pipeline") as demo:
    # GPU/driver detection banner
    has_gpu, gpu_msg = detect_nvidia_driver()
    if has_gpu:
        gr.HTML(
            f"""
            <div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
                <p style="color: rgb(59, 130, 246); margin: 0; font-size: 14px; font-weight: 600;">
                    ✅ NVIDIA GPU ready — {gpu_msg}
                </p>
                <p style="color: rgb(59, 130, 246); margin: 6px 0 0; font-size: 12px;">
                    Reads tokens from environment: <code>HF_WRITE_TOKEN</code> (required), <code>HF_READ_TOKEN</code> (optional)
                </p>
                <p style="color: rgb(59, 130, 246); margin: 4px 0 0; font-size: 12px;">
                    Select a config and run training; optionally deploy Trackio and push to Hub
                </p>
            </div>
            """
        )
        gr.Markdown(joinus)
    else:
        hint_html = markdown_links_to_html(duplicate_space_hint())
        gr.HTML(
            f"""
            <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
                <p style="color: rgb(234, 88, 12); margin: 0; font-size: 14px; font-weight: 600;">
                    ⚠️ No NVIDIA GPU/driver detected — training requires a GPU runtime
                </p>
                <p style="color: rgb(234, 88, 12); margin: 6px 0 0; font-size: 12px;">
                    {hint_html}
                </p>
                <p style="color: rgb(234, 88, 12); margin: 4px 0 0; font-size: 12px;">
                    Reads tokens from environment: <code>HF_WRITE_TOKEN</code> (required), <code>HF_READ_TOKEN</code> (optional)
                </p>
                <p style="color: rgb(234, 88, 12); margin: 4px 0 0; font-size: 12px;">
                    You can still configure and push, but training requires a GPU runtime.
                </p>
            </div>
            """
        )
        gr.Markdown(joinus)

    # --- Progressive interface --------------------------------------------------------
    gr.Markdown("### Configure your run in simple steps")

    # Step 1: Model family
    with gr.Group():
        model_family = gr.Dropdown(choices=MODEL_FAMILIES, value="SmolLM3", label="1) Model family")

    # Step 2: Trainer (revealed after family)
    step2_group = gr.Group(visible=False)
    with step2_group:
        trainer_type = gr.Radio(choices=TRAINER_CHOICES, value="SFT", label="2) Trainer type")

    # Step 3: Monitoring (revealed after trainer)
    step3_group = gr.Group(visible=False)
    with step3_group:
        monitoring_mode = gr.Dropdown(choices=MONITORING_CHOICES, value="dataset", label="3) Monitoring mode")

    # Step 4: Config & details (revealed after monitoring)
    step4_group = gr.Group(visible=False)
    with step4_group:
        # Defaults based on initial family selection
        exp_default, repo_default, desc_default, trackio_space_default = ui_defaults("SmolLM3")

        config_choice = gr.Dropdown(
            choices=list(get_config_map("SmolLM3").keys()),
            value="Basic Training",
            label="4) Training configuration",
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                training_info = gr.Markdown("Select a training configuration to see details.")
                dataset_choice = gr.Dropdown(
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                    label="Dataset (from config; optional)",
                )
                with gr.Row():
                    experiment_name = gr.Textbox(value=exp_default, label="Experiment name")
                    repo_short = gr.Textbox(value=repo_default, label="Model repo (short name)")
                with gr.Row():
                    author_name = gr.Textbox(value=os.environ.get("HF_USERNAME", ""), label="Author name")
                    model_description = gr.Textbox(value=desc_default, label="Model description")
                trackio_space_name = gr.Textbox(
                    value=trackio_space_default,
                    label="Trackio Space name (used when monitoring != none)",
                    visible=False,
                )
                deploy_trackio_space = gr.Checkbox(value=True, label="Deploy Trackio Space", visible=False)
                create_dataset_repo = gr.Checkbox(value=True, label="Create/ensure HF Dataset repo", visible=True)
                with gr.Row():
                    push_to_hub = gr.Checkbox(value=True, label="Push model to Hugging Face Hub")
                    switch_to_read_after = gr.Checkbox(value=True, label="Switch Space token to READ after training")

            with gr.Tab("Advanced"):
                # GPT-OSS specific scheduler overrides
                advanced_enabled = gr.Checkbox(value=False, label="Use advanced overrides (generate config)")

                # Family-specific advanced groups
                gpt_oss_advanced_group = gr.Group(visible=False)
                with gpt_oss_advanced_group:
                    gr.Markdown("Advanced configuration for GPT-OSS")
                    adv_gpt_mode = gr.Radio(
                        choices=["custom", "medical_o1_sft"],
                        value="custom",
                        label="Advanced mode",
                    )

                    # --- GPT-OSS Custom advanced controls ---
                    gpt_oss_custom_group = gr.Group(visible=True)
                    with gpt_oss_custom_group:
                        with gr.Accordion("Dataset", open=True):
                            adv_dataset_name = gr.Textbox(value="", label="Dataset name")
                            with gr.Row():
                                adv_dataset_split = gr.Textbox(value="train", label="Dataset split")
                                adv_dataset_format = gr.Dropdown(
                                    choices=["openhermes_fr", "messages", "text"],
                                    value="openhermes_fr",
                                    label="Dataset format",
                                )
                            with gr.Row():
                                adv_input_field = gr.Textbox(value="prompt", label="Input field")
                                adv_target_field = gr.Textbox(value="accepted_completion", label="Target field (optional)")
                            with gr.Row():
                                adv_system_message = gr.Textbox(value="", label="System message (optional)")
                                adv_developer_message = gr.Textbox(value="", label="Developer message (optional)")
                            adv_model_identity = gr.Textbox(value="", label="Model identity (optional)")
                            with gr.Row():
                                adv_max_samples = gr.Number(value=None, precision=0, label="Max samples (optional)")
                                adv_min_length = gr.Number(value=10, precision=0, label="Min length")
                                adv_max_length = gr.Number(value=None, precision=0, label="Max length (optional)")

                        with gr.Accordion("Training", open=True):
                            with gr.Row():
                                adv_num_train_epochs = gr.Number(value=1.0, precision=2, label="Epochs")
                                adv_batch_size = gr.Number(value=4, precision=0, label="Batch size")
                                adv_gradient_accumulation_steps = gr.Number(value=4, precision=0, label="Grad accumulation")
                            with gr.Row():
                                adv_learning_rate = gr.Number(value=2e-4, precision=6, label="Learning rate")
                                adv_min_lr_num = gr.Number(value=2e-5, precision=6, label="Min LR")
                                adv_weight_decay = gr.Number(value=0.01, precision=6, label="Weight decay")
                                adv_warmup_ratio = gr.Number(value=0.03, precision=3, label="Warmup ratio")
                            adv_max_seq_length = gr.Number(value=2048, precision=0, label="Max seq length")

                        with gr.Accordion("LoRA & Quantization", open=False):
                            with gr.Row():
                                adv_lora_r = gr.Number(value=16, precision=0, label="LoRA r")
                                adv_lora_alpha = gr.Number(value=32, precision=0, label="LoRA alpha")
                                adv_lora_dropout = gr.Number(value=0.05, precision=3, label="LoRA dropout")
                            with gr.Row():
                                adv_mixed_precision = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value="bf16", label="Mixed precision")
                                adv_num_workers = gr.Number(value=4, precision=0, label="Data workers")
                                adv_quantization_type = gr.Dropdown(choices=["mxfp4", "bnb4", "none"], value="mxfp4", label="Quantization")
                            adv_max_grad_norm = gr.Number(value=1.0, precision=3, label="Max grad norm")

                        with gr.Accordion("Eval & Logging", open=False):
                            with gr.Row():
                                adv_logging_steps = gr.Number(value=10, precision=0, label="Logging steps")
                                adv_eval_steps = gr.Number(value=100, precision=0, label="Eval steps")
                                adv_save_steps = gr.Number(value=500, precision=0, label="Save steps")

                        with gr.Accordion("Scheduler (GPT-OSS only)", open=False):
                            scheduler_override = gr.Dropdown(
                                choices=[c for c in SCHEDULER_CHOICES if c is not None],
                                value=None,
                                allow_custom_value=True,
                                label="Scheduler override",
                            )
                            with gr.Row():
                                min_lr = gr.Number(value=None, precision=6, label="min_lr (cosine_with_min_lr)")
                                min_lr_rate = gr.Number(value=None, precision=6, label="min_lr_rate (cosine_with_min_lr)")

                    # --- GPT-OSS Medical o1 SFT controls ---
                    gpt_oss_medical_group = gr.Group(visible=False)
                    with gpt_oss_medical_group:
                        gr.Markdown("Build a Medical o1 SFT configuration (dataset fixed to FreedomIntelligence/medical-o1-reasoning-SFT)")
                        with gr.Accordion("Dataset", open=True):
                            adv_med_dataset_config = gr.Textbox(value="default", label="Dataset config (subset)")
                        with gr.Accordion("Context (optional)", open=False):
                            with gr.Row():
                                adv_med_system_message = gr.Textbox(value="", label="System message")
                                adv_med_developer_message = gr.Textbox(value="", label="Developer message")
                        with gr.Accordion("Training", open=True):
                            with gr.Row():
                                adv_med_num_train_epochs = gr.Number(value=1.0, precision=2, label="Epochs")
                                adv_med_batch_size = gr.Number(value=4, precision=0, label="Batch size")
                                adv_med_gradient_accumulation_steps = gr.Number(value=4, precision=0, label="Grad accumulation")
                            with gr.Row():
                                adv_med_learning_rate = gr.Number(value=2e-4, precision=6, label="Learning rate")
                                adv_med_max_seq_length = gr.Number(value=2048, precision=0, label="Max seq length")

                smollm3_advanced_group = gr.Group(visible=False)
                with smollm3_advanced_group:
                    gr.Markdown("Advanced configuration for SmolLM3")
                    adv_sm_mode = gr.Radio(
                        choices=["custom", "long_context"],
                        value="custom",
                        label="Advanced mode",
                    )
                    # --- SmolLM3 Custom ---
                    sm_custom_group = gr.Group(visible=True)
                    with sm_custom_group:
                        with gr.Accordion("Dataset", open=True):
                            adv_sm_model_name = gr.Textbox(value="HuggingFaceTB/SmolLM3-3B", label="Model name")
                            adv_sm_dataset_name = gr.Textbox(value="", label="Dataset name (optional)")
                            with gr.Row():
                                adv_sm_input_field = gr.Textbox(value="prompt", label="Input field")
                                adv_sm_target_field = gr.Textbox(value="completion", label="Target field")
                            with gr.Row():
                                adv_sm_filter_bad_entries = gr.Checkbox(value=False, label="Filter bad entries")
                                adv_sm_sample_size = gr.Number(value=None, precision=0, label="Sample size (optional)")
                                adv_sm_sample_seed = gr.Number(value=42, precision=0, label="Sample seed")
                        with gr.Accordion("Training", open=True):
                            with gr.Row():
                                adv_sm_max_seq_length = gr.Number(value=4096, precision=0, label="Max seq length")
                                adv_sm_batch_size = gr.Number(value=2, precision=0, label="Batch size")
                                adv_sm_gas = gr.Number(value=8, precision=0, label="Grad accumulation")
                                adv_sm_learning_rate = gr.Number(value=5e-6, precision=6, label="Learning rate")
                            with gr.Row():
                                adv_sm_save_steps = gr.Number(value=500, precision=0, label="Save steps")
                                adv_sm_eval_steps = gr.Number(value=100, precision=0, label="Eval steps")
                                adv_sm_logging_steps = gr.Number(value=10, precision=0, label="Logging steps")

                    # --- SmolLM3 Long-Context ---
                    sm_long_group = gr.Group(visible=False)
                    with sm_long_group:
                        gr.Markdown("Generate a Long-Context SmolLM3 config")
                        with gr.Accordion("Dataset", open=True):
                            adv_sm_lc_model_name = gr.Textbox(value="HuggingFaceTB/SmolLM3-3B", label="Model name")
                            adv_sm_lc_dataset_name = gr.Textbox(value="", label="Dataset name (optional)")
                            with gr.Row():
                                adv_sm_lc_input_field = gr.Textbox(value="prompt", label="Input field")
                                adv_sm_lc_target_field = gr.Textbox(value="completion", label="Target field")
                            with gr.Row():
                                adv_sm_lc_filter_bad_entries = gr.Checkbox(value=False, label="Filter bad entries")
                                adv_sm_lc_sample_size = gr.Number(value=None, precision=0, label="Sample size (optional)")
                                adv_sm_lc_sample_seed = gr.Number(value=42, precision=0, label="Sample seed")
                        with gr.Accordion("Training", open=True):
                            with gr.Row():
                                adv_sm_lc_max_seq_length = gr.Number(value=131072, precision=0, label="Max seq length (up to 131072)")
                                adv_sm_lc_batch_size = gr.Number(value=1, precision=0, label="Batch size")
                                adv_sm_lc_gas = gr.Number(value=8, precision=0, label="Grad accumulation")
                                adv_sm_lc_learning_rate = gr.Number(value=1e-5, precision=6, label="Learning rate")
                            with gr.Row():
                                adv_sm_lc_warmup_steps = gr.Number(value=200, precision=0, label="Warmup steps")
                                adv_sm_lc_max_iters = gr.Number(value=500, precision=0, label="Max iters")
                            with gr.Row():
                                adv_sm_lc_save_steps = gr.Number(value=100, precision=0, label="Save steps")
                                adv_sm_lc_eval_steps = gr.Number(value=50, precision=0, label="Eval steps")
                                adv_sm_lc_logging_steps = gr.Number(value=10, precision=0, label="Logging steps")
                        with gr.Accordion("Chat Template", open=False):
                            with gr.Row():
                                adv_sm_lc_use_chat_template = gr.Checkbox(value=True, label="Use chat template")
                                adv_sm_lc_no_think_system_message = gr.Checkbox(value=True, label="No-think system message")

                def _toggle_sm_mode(mode: str):
                    return (
                        gr.update(visible=mode == "custom"),
                        gr.update(visible=mode == "long_context"),
                    )

                adv_sm_mode.change(
                    _toggle_sm_mode,
                    inputs=[adv_sm_mode],
                    outputs=[sm_custom_group, sm_long_group],
                )

                def _toggle_advanced(enable: bool, family_val: str):
                    return (
                        gr.update(visible=enable and family_val == "GPT-OSS"),
                        gr.update(visible=enable and family_val == "SmolLM3"),
                    )

                advanced_enabled.change(
                    _toggle_advanced,
                    inputs=[advanced_enabled, model_family],
                    outputs=[gpt_oss_advanced_group, smollm3_advanced_group],
                )

                # Toggle between GPT-OSS Custom and Medical modes
                def _toggle_gpt_oss_mode(mode: str):
                    return (
                        gr.update(visible=mode == "custom"),
                        gr.update(visible=mode == "medical_o1_sft"),
                    )

                adv_gpt_mode.change(
                    _toggle_gpt_oss_mode,
                    inputs=[adv_gpt_mode],
                    outputs=[gpt_oss_custom_group, gpt_oss_medical_group],
                )

    # Final action & logs
    start_btn = gr.Button("Run Pipeline", variant="primary")
    logs = gr.Textbox(value="", label="Logs", lines=20)

    # --- Events ---------------------------------------------------------------------
    model_family.change(
        on_family_change,
        inputs=model_family,
        outputs=[
            config_choice,
            experiment_name,
            repo_short,
            model_description,
            trackio_space_name,
            training_info,
            dataset_choice,
            step2_group,
            step3_group,
            step4_group,
            gpt_oss_advanced_group,  # show advanced for GPT-OSS
            smollm3_advanced_group,  # show advanced for SmolLM3
        ],
    )

    trainer_type.change(on_trainer_selected, inputs=trainer_type, outputs=step3_group)

    monitoring_mode.change(
        on_monitoring_change,
        inputs=monitoring_mode,
        outputs=[step4_group, trackio_space_name, deploy_trackio_space, create_dataset_repo],
    )

    config_choice.change(
        on_config_change,
        inputs=[model_family, config_choice],
        outputs=[
            training_info,
            dataset_choice,
            # Advanced (GPT-OSS) outputs
            adv_dataset_name,
            adv_dataset_split,
            adv_dataset_format,
            adv_input_field,
            adv_target_field,
            adv_system_message,
            adv_developer_message,
            adv_model_identity,
            adv_max_samples,
            adv_min_length,
            adv_max_length,
            adv_num_train_epochs,
            adv_batch_size,
            adv_gradient_accumulation_steps,
            adv_learning_rate,
            adv_min_lr_num,
            adv_weight_decay,
            adv_warmup_ratio,
            adv_max_seq_length,
            adv_lora_r,
            adv_lora_alpha,
            adv_lora_dropout,
            adv_mixed_precision,
            adv_num_workers,
            adv_quantization_type,
            adv_max_grad_norm,
            adv_logging_steps,
            adv_eval_steps,
            adv_save_steps,
            # GPT-OSS Medical o1 SFT outputs (prefill defaults)
            adv_med_dataset_config,
            adv_med_system_message,
            adv_med_developer_message,
            adv_med_num_train_epochs,
            adv_med_batch_size,
            adv_med_gradient_accumulation_steps,
            adv_med_learning_rate,
            adv_med_max_seq_length,
            # Advanced (SmolLM3)
            adv_sm_model_name,
            adv_sm_dataset_name,
            adv_sm_input_field,
            adv_sm_target_field,
            adv_sm_filter_bad_entries,
            adv_sm_sample_size,
            adv_sm_sample_seed,
            adv_sm_max_seq_length,
            adv_sm_batch_size,
            adv_sm_gas,
            adv_sm_learning_rate,
            adv_sm_save_steps,
            adv_sm_eval_steps,
            adv_sm_logging_steps,
        ],
    )

    # Keep Advanced dataset fields in sync when user selects a different dataset
    def _sync_dataset_fields(ds_value: Optional[str]):
        ds_text = ds_value or ""
        return ds_text, ds_text

    dataset_choice.change(
        _sync_dataset_fields,
        inputs=[dataset_choice],
        outputs=[adv_dataset_name, adv_sm_dataset_name],
    )

    def _start_with_overrides(
        model_family_v,
        config_choice_v,
        trainer_type_v,
        monitoring_mode_v,
        experiment_name_v,
        repo_short_v,
        author_name_v,
        model_description_v,
        trackio_space_name_v,
        deploy_trackio_space_v,
        create_dataset_repo_v,
        push_to_hub_v,
        switch_to_read_after_v,
        scheduler_override_v,
        min_lr_v,
        min_lr_rate_v,
        advanced_enabled_v,
        adv_gpt_mode_v,
        # GPT-OSS advanced
        adv_dataset_name_v,
        adv_dataset_split_v,
        adv_dataset_format_v,
        adv_input_field_v,
        adv_target_field_v,
        adv_system_message_v,
        adv_developer_message_v,
        adv_model_identity_v,
        adv_max_samples_v,
        adv_min_length_v,
        adv_max_length_v,
        adv_num_train_epochs_v,
        adv_batch_size_v,
        adv_gas_v,
        adv_lr_v,
        adv_min_lr_num_v,
        adv_wd_v,
        adv_warmup_ratio_v,
        adv_max_seq_length_v,
        adv_lora_r_v,
        adv_lora_alpha_v,
        adv_lora_dropout_v,
        adv_mixed_precision_v,
        adv_num_workers_v,
        adv_quantization_type_v,
        adv_max_grad_norm_v,
        adv_logging_steps_v,
        adv_eval_steps_v,
        adv_save_steps_v,
        # GPT-OSS Medical o1 SFT
        adv_med_dataset_config_v,
        adv_med_system_message_v,
        adv_med_developer_message_v,
        adv_med_num_train_epochs_v,
        adv_med_batch_size_v,
        adv_med_gradient_accumulation_steps_v,
        adv_med_learning_rate_v,
        adv_med_max_seq_length_v,
        # SmolLM3 advanced
        adv_sm_mode_v,
        adv_sm_model_name_v,
        adv_sm_dataset_name_v,
        adv_sm_input_field_v,
        adv_sm_target_field_v,
        adv_sm_filter_bad_entries_v,
        adv_sm_sample_size_v,
        adv_sm_sample_seed_v,
        adv_sm_max_seq_length_v,
        adv_sm_batch_size_v,
        adv_sm_gas_v,
        adv_sm_learning_rate_v,
        adv_sm_save_steps_v,
        adv_sm_eval_steps_v,
        adv_sm_logging_steps_v,
        # SmolLM3 long context
        adv_sm_lc_model_name_v,
        adv_sm_lc_dataset_name_v,
        adv_sm_lc_input_field_v,
        adv_sm_lc_target_field_v,
        adv_sm_lc_filter_bad_entries_v,
        adv_sm_lc_sample_size_v,
        adv_sm_lc_sample_seed_v,
        adv_sm_lc_max_seq_length_v,
        adv_sm_lc_batch_size_v,
        adv_sm_lc_gas_v,
        adv_sm_lc_learning_rate_v,
        adv_sm_lc_warmup_steps_v,
        adv_sm_lc_max_iters_v,
        adv_sm_lc_save_steps_v,
        adv_sm_lc_eval_steps_v,
        adv_sm_lc_logging_steps_v,
        adv_sm_lc_use_chat_template_v,
        adv_sm_lc_no_think_system_message_v,
    ):
        # If advanced overrides enabled, generate a config file and pass its path
        override_path: Optional[str] = None
        if advanced_enabled_v:
            try:
                if model_family_v == "GPT-OSS":
                    if str(adv_gpt_mode_v) == "medical_o1_sft":
                        cfg_path = generate_medical_o1_config_file(
                            dataset_config=str(adv_med_dataset_config_v or "default"),
                            system_message=(str(adv_med_system_message_v) if adv_med_system_message_v else None),
                            developer_message=(str(adv_med_developer_message_v) if adv_med_developer_message_v else None),
                            num_train_epochs=float(adv_med_num_train_epochs_v or 1.0),
                            batch_size=int(adv_med_batch_size_v or 4),
                            gradient_accumulation_steps=int(adv_med_gradient_accumulation_steps_v or 4),
                            learning_rate=float(adv_med_learning_rate_v or 2e-4),
                            max_seq_length=int(adv_med_max_seq_length_v or 2048),
                        )
                    else:
                        cfg_path = generate_gpt_oss_custom_config_file(
                            dataset_name=str(adv_dataset_name_v or ""),
                            dataset_split=str(adv_dataset_split_v or "train"),
                            dataset_format=str(adv_dataset_format_v or "openhermes_fr"),
                            input_field=str(adv_input_field_v or "prompt"),
                            target_field=(str(adv_target_field_v) if adv_target_field_v else None),
                            system_message=(str(adv_system_message_v) if adv_system_message_v else None),
                            developer_message=(str(adv_developer_message_v) if adv_developer_message_v else None),
                            model_identity=(str(adv_model_identity_v) if adv_model_identity_v else None),
                            max_samples=(int(adv_max_samples_v) if adv_max_samples_v else None),
                            min_length=int(adv_min_length_v or 10),
                            max_length=(int(adv_max_length_v) if adv_max_length_v else None),
                            num_train_epochs=float(adv_num_train_epochs_v or 1.0),
                            batch_size=int(adv_batch_size_v or 4),
                            gradient_accumulation_steps=int(adv_gas_v or 4),
                            learning_rate=float(adv_lr_v or 2e-4),
                            min_lr=float(adv_min_lr_num_v or 2e-5),
                            weight_decay=float(adv_wd_v or 0.01),
                            warmup_ratio=float(adv_warmup_ratio_v or 0.03),
                            max_seq_length=int(adv_max_seq_length_v or 2048),
                            lora_r=int(adv_lora_r_v or 16),
                            lora_alpha=int(adv_lora_alpha_v or 32),
                            lora_dropout=float(adv_lora_dropout_v or 0.05),
                            mixed_precision=str(adv_mixed_precision_v or "bf16"),
                            num_workers=int(adv_num_workers_v or 4),
                            quantization_type=str(adv_quantization_type_v or "mxfp4"),
                            max_grad_norm=float(adv_max_grad_norm_v or 1.0),
                            logging_steps=int(adv_logging_steps_v or 10),
                            eval_steps=int(adv_eval_steps_v or 100),
                            save_steps=int(adv_save_steps_v or 500),
                        )
                else:
                    if str(adv_sm_mode_v) == "long_context":
                        cfg_path = generate_smollm3_long_context_config_file(
                            model_name=str(adv_sm_lc_model_name_v or "HuggingFaceTB/SmolLM3-3B"),
                            dataset_name=(str(adv_sm_lc_dataset_name_v) if adv_sm_lc_dataset_name_v else None),
                            input_field=str(adv_sm_lc_input_field_v or "prompt"),
                            target_field=str(adv_sm_lc_target_field_v or "completion"),
                            filter_bad_entries=bool(adv_sm_lc_filter_bad_entries_v),
                            sample_size=(int(adv_sm_lc_sample_size_v) if adv_sm_lc_sample_size_v else None),
                            sample_seed=int(adv_sm_lc_sample_seed_v or 42),
                            max_seq_length=int(adv_sm_lc_max_seq_length_v or 131072),
                            batch_size=int(adv_sm_lc_batch_size_v or 1),
                            gradient_accumulation_steps=int(adv_sm_lc_gas_v or 8),
                            learning_rate=float(adv_sm_lc_learning_rate_v or 1e-5),
                            warmup_steps=int(adv_sm_lc_warmup_steps_v or 200),
                            max_iters=int(adv_sm_lc_max_iters_v or 500),
                            save_steps=int(adv_sm_lc_save_steps_v or 100),
                            eval_steps=int(adv_sm_lc_eval_steps_v or 50),
                            logging_steps=int(adv_sm_lc_logging_steps_v or 10),
                            use_chat_template=bool(adv_sm_lc_use_chat_template_v),
                            no_think_system_message=bool(adv_sm_lc_no_think_system_message_v),
                            trainer_type=str(trainer_type_v).lower(),
                        )
                    else:
                        cfg_path = generate_smollm3_custom_config_file(
                            model_name=str(adv_sm_model_name_v or "HuggingFaceTB/SmolLM3-3B"),
                            dataset_name=(str(adv_sm_dataset_name_v) if adv_sm_dataset_name_v else None),
                            max_seq_length=int(adv_sm_max_seq_length_v or 4096),
                            batch_size=int(adv_sm_batch_size_v or 2),
                            gradient_accumulation_steps=int(adv_sm_gas_v or 8),
                            learning_rate=float(adv_sm_learning_rate_v or 5e-6),
                            save_steps=int(adv_sm_save_steps_v or 500),
                            eval_steps=int(adv_sm_eval_steps_v or 100),
                            logging_steps=int(adv_sm_logging_steps_v or 10),
                            filter_bad_entries=bool(adv_sm_filter_bad_entries_v),
                            input_field=str(adv_sm_input_field_v or "prompt"),
                            target_field=str(adv_sm_target_field_v or "completion"),
                            sample_size=(int(adv_sm_sample_size_v) if adv_sm_sample_size_v else None),
                            sample_seed=int(adv_sm_sample_seed_v or 42),
                            trainer_type=str(trainer_type_v).lower(),
                        )
                override_path = str(cfg_path)
            except Exception as e:
                # Surface error in logs via generator
                def _err_gen():
                    yield f"❌ Failed to generate advanced config: {e}"
                return _err_gen()

        def _gen():
            params = PipelineInputs(
                model_family=model_family_v,
                config_choice=config_choice_v,
                trainer_type=trainer_type_v,
                monitoring_mode=monitoring_mode_v,
                experiment_name=experiment_name_v,
                repo_short=repo_short_v,
                author_name=author_name_v,
                model_description=model_description_v,
                trackio_space_name=trackio_space_name_v or None,
                deploy_trackio_space=bool(deploy_trackio_space_v),
                create_dataset_repo=bool(create_dataset_repo_v),
                push_to_hub=bool(push_to_hub_v),
                switch_to_read_after=bool(switch_to_read_after_v),
                scheduler_override=(scheduler_override_v or None),
                min_lr=min_lr_v,
                min_lr_rate=min_lr_rate_v,
                override_config_path=override_path,
            )
            write_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
            read_token = os.environ.get("HF_READ_TOKEN")
            yield f"HF_WRITE_TOKEN: {mask_token(write_token)}"
            yield f"HF_READ_TOKEN:  {mask_token(read_token)}"
            for line in run_pipeline(params):
                yield line
                time.sleep(0.01)
        return _gen()

    start_btn.click(
        _start_with_overrides,
        inputs=[
            model_family,
            config_choice,
            trainer_type,
            monitoring_mode,
            experiment_name,
            repo_short,
            author_name,
            model_description,
            trackio_space_name,
            deploy_trackio_space,
            create_dataset_repo,
            push_to_hub,
            switch_to_read_after,
            scheduler_override,
            min_lr,
            min_lr_rate,
            advanced_enabled,
            adv_gpt_mode,
            # GPT-OSS advanced
            adv_dataset_name,
            adv_dataset_split,
            adv_dataset_format,
            adv_input_field,
            adv_target_field,
            adv_system_message,
            adv_developer_message,
            adv_model_identity,
            adv_max_samples,
            adv_min_length,
            adv_max_length,
            adv_num_train_epochs,
            adv_batch_size,
            adv_gradient_accumulation_steps,
            adv_learning_rate,
            adv_min_lr_num,
            adv_weight_decay,
            adv_warmup_ratio,
            adv_max_seq_length,
            adv_lora_r,
            adv_lora_alpha,
            adv_lora_dropout,
            adv_mixed_precision,
            adv_num_workers,
            adv_quantization_type,
            adv_max_grad_norm,
            adv_logging_steps,
            adv_eval_steps,
            adv_save_steps,
            # GPT-OSS Medical o1 SFT
            adv_med_dataset_config,
            adv_med_system_message,
            adv_med_developer_message,
            adv_med_num_train_epochs,
            adv_med_batch_size,
            adv_med_gradient_accumulation_steps,
            adv_med_learning_rate,
            adv_med_max_seq_length,
            # SmolLM3 advanced
            adv_sm_mode,
            adv_sm_model_name,
            adv_sm_dataset_name,
            adv_sm_input_field,
            adv_sm_target_field,
            adv_sm_filter_bad_entries,
            adv_sm_sample_size,
            adv_sm_sample_seed,
            adv_sm_max_seq_length,
            adv_sm_batch_size,
            adv_sm_gas,
            adv_sm_learning_rate,
            adv_sm_save_steps,
            adv_sm_eval_steps,
            adv_sm_logging_steps,
            # SmolLM3 long context
            adv_sm_lc_model_name,
            adv_sm_lc_dataset_name,
            adv_sm_lc_input_field,
            adv_sm_lc_target_field,
            adv_sm_lc_filter_bad_entries,
            adv_sm_lc_sample_size,
            adv_sm_lc_sample_seed,
            adv_sm_lc_max_seq_length,
            adv_sm_lc_batch_size,
            adv_sm_lc_gas,
            adv_sm_lc_learning_rate,
            adv_sm_lc_warmup_steps,
            adv_sm_lc_max_iters,
            adv_sm_lc_save_steps,
            adv_sm_lc_eval_steps,
            adv_sm_lc_logging_steps,
            adv_sm_lc_use_chat_template,
            adv_sm_lc_no_think_system_message,
        ],
        outputs=[logs],
    )


if __name__ == "__main__":
    # Optional: allow setting server parameters via env
    server_port = int(os.environ.get("INTERFACE_PORT", "7860"))
    server_name = os.environ.get("INTERFACE_HOST", "0.0.0.0")
    demo.queue().launch(server_name=server_name, server_port=server_port, mcp_server=True)


