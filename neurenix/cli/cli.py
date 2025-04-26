"""
Main CLI module for the Neurenix framework.

This module provides the main entry point for the Neurenix CLI,
including argument parsing and command dispatching.
"""

import argparse
import sys
from typing import List, Optional, Dict, Any

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

COMMAND_DESCRIPTIONS = {
    "init": "Create a new Neurenix project with folder structure, config, and optional dataset",
    "run": "Train a model with the provided data",
    "save": "Save the current project state, including model, configurations, and data",
    "predict": "Make predictions using a trained model",
    "eval": "Evaluate a trained model with specific metrics",
    "export": "Export a trained model to a specific format",
    "hardware": "Manage hardware settings, including auto-selection",
    "preprocess": "Perform preprocessing on input data",
    "monitor": "Monitor model training in real-time",
    "optimize": "Optimize a model by adjusting hyperparameters or applying techniques like quantization",
    "dataset": "Manage datasets for training and evaluation",
    "serve": "Serve a trained model as a RESTful API",
    "help": "Display help information about Neurenix commands"
}

COMMAND_MAP = {
    "init": init_command,
    "run": run_command,
    "save": save_command,
    "predict": predict_command,
    "eval": eval_command,
    "export": export_command,
    "hardware": hardware_command,
    "preprocess": preprocess_command,
    "monitor": monitor_command,
    "optimize": optimize_command,
    "dataset": dataset_command,
    "serve": serve_command,
    "help": help_command
}

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Neurenix CLI - Command-line interface for the Neurenix framework",
        usage="neurenix <command> [<args>]"
    )
    
    parser.add_argument(
        "command",
        help=f"Command to run. Available commands: {', '.join(COMMAND_MAP.keys())}",
        choices=list(COMMAND_MAP.keys()),
        nargs="?",
        default="help"
    )
    
    if args is None:
        args = sys.argv[1:]
    
    if not args:
        return parser.parse_args(["help"])
    
    parsed_args, remaining = parser.parse_known_args(args)
    
    parsed_args.args = remaining
    
    return parsed_args

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Neurenix CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    command_func = COMMAND_MAP.get(parsed_args.command)
    
    if not command_func:
        print(f"Unknown command: {parsed_args.command}")
        help_command(argparse.Namespace(args=[]))
        return 1
    
    try:
        return command_func(parsed_args) or 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error executing command '{parsed_args.command}': {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
