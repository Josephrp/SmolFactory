"""
Configuration package for SmolLM3 and GPT-OSS training
"""

from .train_smollm3 import SmolLM3Config, get_config as get_base_config
from .train_smollm3_openhermes_fr import SmolLM3ConfigOpenHermesFR, get_config as get_openhermes_fr_config
from .train_smollm3_openhermes_fr_a100_large import SmolLM3ConfigOpenHermesFRA100Large, get_config as get_a100_large_config
from .train_smollm3_openhermes_fr_a100_multiple_passes import SmolLM3ConfigOpenHermesFRMultiplePasses, get_config as get_multiple_passes_config
from .train_smollm3_openhermes_fr_a100_max_performance import SmolLM3ConfigOpenHermesFRMaxPerformance, get_config as get_max_performance_config

# GPT-OSS configurations
from .train_gpt_oss_basic import GPTOSSBasicConfig, get_config as get_gpt_oss_basic_config
from .train_gpt_oss_multilingual_reasoning import (
    GPTOSSMultilingualReasoningConfig,
    get_config as get_gpt_oss_multilingual_reasoning_config,
)
from .train_gpt_oss_h100_optimized import (
    GPTOSSH100OptimizedConfig,
    get_config as get_gpt_oss_h100_optimized_config,
)
from .train_gpt_oss_memory_optimized import (
    GPTOSSMemoryOptimizedConfig,
    get_config as get_gpt_oss_memory_optimized_config,
)
from .train_gpt_oss_custom import GPTOSSEnhancedCustomConfig

# Pre-baked GPT-OSS configs exposing a `config` instance
from .train_gpt_oss_openhermes_fr import config as gpt_oss_openhermes_fr_config
from .train_gpt_oss_openhermes_fr_memory_optimized import (
    config as gpt_oss_openhermes_fr_memory_optimized_config,
)
from .train_gpt_oss_medical_o1_sft import config as gpt_oss_medical_o1_sft_config

# Generic get_config function that can handle different config types
def get_config(config_path: str):
    """Generic get_config function that tries different config types"""
    import os
    import importlib.util as _importlib
    
    if not os.path.exists(config_path):
        return get_base_config(config_path)
    
    # Try to determine config type based on filename
    if "a100_large" in config_path:
        return get_a100_large_config(config_path)
    elif "a100_multiple_passes" in config_path:
        return get_multiple_passes_config(config_path)
    elif "a100_max_performance" in config_path:
        return get_max_performance_config(config_path)
    elif "openhermes_fr" in config_path:
        return get_openhermes_fr_config(config_path)
    elif "gpt_oss" in config_path:
        # Load GPT-OSS style config module dynamically and return its `config` instance if present
        try:
            spec = _importlib.spec_from_file_location("config_module", config_path)
            module = _importlib.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)  # type: ignore
            if hasattr(module, "config"):
                return getattr(module, "config")
        except Exception:
            # Fallback to base config if dynamic load fails
            pass
        return get_base_config(config_path)
    else:
        return get_base_config(config_path)

__all__ = [
    'SmolLM3Config',
    'SmolLM3ConfigOpenHermesFR', 
    'SmolLM3ConfigOpenHermesFRA100Large',
    'SmolLM3ConfigOpenHermesFRMultiplePasses',
    'SmolLM3ConfigOpenHermesFRMaxPerformance',
    # GPT-OSS classes and accessors
    'GPTOSSBasicConfig',
    'GPTOSSMultilingualReasoningConfig',
    'GPTOSSH100OptimizedConfig',
    'GPTOSSMemoryOptimizedConfig',
    'GPTOSSEnhancedCustomConfig',
    'get_gpt_oss_basic_config',
    'get_gpt_oss_multilingual_reasoning_config',
    'get_gpt_oss_h100_optimized_config',
    'get_gpt_oss_memory_optimized_config',
    # Pre-baked GPT-OSS config instances
    'gpt_oss_openhermes_fr_config',
    'gpt_oss_openhermes_fr_memory_optimized_config',
    'gpt_oss_medical_o1_sft_config',
    'get_config',
    'get_base_config',
    'get_openhermes_fr_config',
    'get_a100_large_config',
    'get_multiple_passes_config',
    'get_max_performance_config',
] 