"""
Trackio Module for TRL Library Compatibility
This module provides the interface expected by TRL library while using our custom monitoring system
"""

# Import all functions from our custom trackio implementation
from src.trackio import (
    init,
    log,
    finish,
    log_config,
    log_checkpoint,
    log_evaluation_results,
    get_experiment_url,
    is_available,
    get_monitor
)

# Make all functions available at module level
__all__ = [
    'init',
    'log', 
    'finish',
    'log_config',
    'log_checkpoint',
    'log_evaluation_results',
    'get_experiment_url',
    'is_available',
    'get_monitor'
] 