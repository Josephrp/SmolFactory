"""
Configuration management for SmolLM3 fine-tuning
"""

import os
import sys
import importlib.util
from typing import Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add config directory to path
config_dir = os.path.join(project_root, 'config')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

try:
    from config.train_smollm3 import SmolLM3Config, get_config as get_default_config
except ImportError:
    # Fallback: try direct import
    import sys
    sys.path.insert(0, os.path.join(project_root, 'config'))
    from train_smollm3 import SmolLM3Config, get_config as get_default_config

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