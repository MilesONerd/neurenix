"""
Neurenix CLI module for command-line interface operations.

This module provides a command-line interface for the Neurenix framework,
allowing users to easily create, train, evaluate, and deploy models.
"""

from .cli import main, parse_args
from .commands import (
    init_command,
    run_command,
    save_command,
    predict_command,
    eval_command,
    export_command,
    hardware_command,
    preprocess_command,
    monitor_command,
    optimize_command,
    dataset_command,
    serve_command,
    help_command
)

__all__ = [
    'main',
    'parse_args',
    'init_command',
    'run_command',
    'save_command',
    'predict_command',
    'eval_command',
    'export_command',
    'hardware_command',
    'preprocess_command',
    'monitor_command',
    'optimize_command',
    'dataset_command',
    'serve_command',
    'help_command'
]
