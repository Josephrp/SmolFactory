"""
Configuration package for SmolLM3 and GPT-OSS training
"""

try:
    from .train_smollm3 import SmolLM3Config, get_config as get_base_config
except Exception:
    SmolLM3Config = None  # type: ignore
    def get_base_config(config_path: str):  # type: ignore
        raise ImportError("train_smollm3 not available")
try:
    from .train_smollm3_openhermes_fr import SmolLM3ConfigOpenHermesFR, get_config as get_openhermes_fr_config
except Exception:
    SmolLM3ConfigOpenHermesFR = None  # type: ignore
    get_openhermes_fr_config = None  # type: ignore
try:
    from .train_smollm3_openhermes_fr_a100_large import SmolLM3ConfigOpenHermesFRA100Large, get_config as get_a100_large_config
except Exception:
    SmolLM3ConfigOpenHermesFRA100Large = None  # type: ignore
    get_a100_large_config = None  # type: ignore
try:
    from .train_smollm3_openhermes_fr_a100_multiple_passes import SmolLM3ConfigOpenHermesFRMultiplePasses, get_config as get_multiple_passes_config
except Exception:
    SmolLM3ConfigOpenHermesFRMultiplePasses = None  # type: ignore
    get_multiple_passes_config = None  # type: ignore
try:
    from .train_smollm3_openhermes_fr_a100_max_performance import SmolLM3ConfigOpenHermesFRMaxPerformance, get_config as get_max_performance_config
except Exception:
    SmolLM3ConfigOpenHermesFRMaxPerformance = None  # type: ignore
    get_max_performance_config = None  # type: ignore

# GPT-OSS configurations
try:
    from .train_gpt_oss_basic import GPTOSSBasicConfig, get_config as get_gpt_oss_basic_config
except Exception:
    GPTOSSBasicConfig = None  # type: ignore
    get_gpt_oss_basic_config = None  # type: ignore
try:
    from .train_gpt_oss_multilingual_reasoning import (
        GPTOSSMultilingualReasoningConfig,
        get_config as get_gpt_oss_multilingual_reasoning_config,
    )
except Exception:
    GPTOSSMultilingualReasoningConfig = None  # type: ignore
    get_gpt_oss_multilingual_reasoning_config = None  # type: ignore
try:
    from .train_gpt_oss_h100_optimized import (
        GPTOSSH100OptimizedConfig,
        get_config as get_gpt_oss_h100_optimized_config,
    )
except Exception:
    GPTOSSH100OptimizedConfig = None  # type: ignore
    get_gpt_oss_h100_optimized_config = None  # type: ignore
try:
    from .train_gpt_oss_memory_optimized import (
        GPTOSSMemoryOptimizedConfig,
        get_config as get_gpt_oss_memory_optimized_config,
    )
except Exception:
    GPTOSSMemoryOptimizedConfig = None  # type: ignore
    get_gpt_oss_memory_optimized_config = None  # type: ignore
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
        # Fall back to base config accessor if available
        return get_base_config(config_path) if get_base_config else None
    
    # Try to determine config type based on filename
    if "a100_large" in config_path:
        return get_a100_large_config(config_path) if get_a100_large_config else None
    elif "a100_multiple_passes" in config_path:
        return get_multiple_passes_config(config_path) if get_multiple_passes_config else None
    elif "a100_max_performance" in config_path:
        return get_max_performance_config(config_path) if get_max_performance_config else None
    elif "openhermes_fr" in config_path:
        return get_openhermes_fr_config(config_path) if get_openhermes_fr_config else None
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
        return get_base_config(config_path) if get_base_config else None
    else:
        return get_base_config(config_path) if get_base_config else None

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