"""
Configuration package for SmolLM3 training
"""

from .train_smollm3 import SmolLM3Config, get_config as get_base_config
from .train_smollm3_openhermes_fr import SmolLM3ConfigOpenHermesFR, get_config as get_openhermes_fr_config
from .train_smollm3_openhermes_fr_a100_large import SmolLM3ConfigOpenHermesFRA100Large, get_config as get_a100_large_config
from .train_smollm3_openhermes_fr_a100_multiple_passes import SmolLM3ConfigOpenHermesFRMultiplePasses, get_config as get_multiple_passes_config

# Generic get_config function that can handle different config types
def get_config(config_path: str):
    """Generic get_config function that tries different config types"""
    import os
    
    if not os.path.exists(config_path):
        return get_base_config(config_path)
    
    # Try to determine config type based on filename
    if "a100_large" in config_path:
        return get_a100_large_config(config_path)
    elif "a100_multiple_passes" in config_path:
        return get_multiple_passes_config(config_path)
    elif "openhermes_fr" in config_path:
        return get_openhermes_fr_config(config_path)
    else:
        return get_base_config(config_path)

__all__ = [
    'SmolLM3Config',
    'SmolLM3ConfigOpenHermesFR', 
    'SmolLM3ConfigOpenHermesFRA100Large',
    'SmolLM3ConfigOpenHermesFRMultiplePasses',
    'get_config',
    'get_base_config',
    'get_openhermes_fr_config',
    'get_a100_large_config',
    'get_multiple_passes_config',
] 