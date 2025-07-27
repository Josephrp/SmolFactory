"""
SmolLM3 Fine-tuning Pipeline
Core training and monitoring modules
"""

from .config import SmolLM3Config
from .data import SmolLM3Dataset
from .model import SmolLM3Model
from .monitoring import SmolLM3Monitor, create_monitor_from_config
from .train import SmolLM3Trainer
from .trainer import SmolLM3Trainer as Trainer
from .trackio import init, log, finish, log_config, log_checkpoint, log_evaluation_results

__all__ = [
    'SmolLM3Config',
    'SmolLM3Dataset', 
    'SmolLM3Model',
    'SmolLM3Monitor',
    'create_monitor_from_config',
    'SmolLM3Trainer',
    'Trainer',
    # Trackio interface
    'init',
    'log', 
    'finish',
    'log_config',
    'log_checkpoint',
    'log_evaluation_results'
] 