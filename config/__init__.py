"""
Configuration package for SmolLM3 training
"""

from .train_smollm3 import SmolLM3Config, get_config as get_base_config
from .train_smollm3_openhermes_fr import SmolLM3ConfigOpenHermesFR, get_config as get_openhermes_fr_config
from .train_smollm3_openhermes_fr_a100_large import SmolLM3ConfigOpenHermesFRA100Large, get_config as get_a100_large_config
from .train_smollm3_openhermes_fr_a100_multiple_passes import SmolLM3ConfigOpenHermesFRMultiplePasses, get_config as get_multiple_passes_config

__all__ = [
    'SmolLM3Config',
    'SmolLM3ConfigOpenHermesFR', 
    'SmolLM3ConfigOpenHermesFRA100Large',
    'SmolLM3ConfigOpenHermesFRMultiplePasses',
    'get_base_config',
    'get_openhermes_fr_config',
    'get_a100_large_config',
    'get_multiple_passes_config',
] 