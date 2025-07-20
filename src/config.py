"""
Configuration management for SmolLM3 fine-tuning
"""

import os
import importlib.util
from typing import Any
from config.train_smollm3 import SmolLM3Config, get_config as get_default_config

def get_config(config_path: str) -> SmolLM3Config:
    """Load configuration from file or return default"""
    if os.path.exists(config_path):
        # Load from file if it exists
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'config'):
            return config_module.config
        else:
            # Try to find a config class
            for attr_name in dir(config_module):
                attr = getattr(config_module, attr_name)
                if isinstance(attr, SmolLM3Config):
                    return attr
    
    # Return default configuration
    return get_default_config(config_path) 