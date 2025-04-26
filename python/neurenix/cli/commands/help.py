"""
Implementation of the 'help' command for the Neurenix CLI.

This module provides functionality to display help information about
available commands and their usage.
"""

import os
import argparse
from typing import Dict, Any, Optional, List, Union

def help_command(args: argparse.Namespace) -> int:
    """
    Display help information about available commands.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Display help information about available commands",
        usage="neurenix help [<command>]"
    )
    
    parser.add_argument(
        "command",
        help="Command to get help for",
        nargs="?",
        default=None
    )
    
    help_args = parser.parse_args(args.args)
    
    commands = {
        "init": "Initialize a new Neurenix project with a standard folder structure, configuration files, and optional dataset",
        "run": "Run a Neurenix model training script with specified configuration options",
        "save": "Save the current project state, including model, configurations, and data",
        "predict": "Make predictions using a trained model",
        "eval": "Evaluate a trained model with specific metrics",
        "export": "Export a trained model to a specific format (ONNX, TorchScript, etc.)",
        "hardware": "Manage hardware settings, including auto-selection, device information, and benchmarking",
        "preprocess": "Preprocess input data for model training",
        "monitor": "Monitor model training in real-time",
        "optimize": "Optimize a model by adjusting hyperparameters or applying techniques like quantization",
        "dataset": "Manage datasets for training and evaluation",
        "serve": "Serve a trained model as a RESTful API",
        "help": "Display help information about available commands"
    }
    
    if help_args.command:
        if help_args.command not in commands:
            print(f"Error: Unknown command '{help_args.command}'")
            print("\nAvailable commands:")
            for cmd, desc in commands.items():
                print(f"  {cmd:12} - {desc}")
            return 1
        
        try:
            module_name = f"neurenix.cli.commands.{help_args.command}"
            module = __import__(module_name, fromlist=[""])
            
            command_func = getattr(module, f"{help_args.command}_command")
            
            cmd_parser = argparse.ArgumentParser(
                description=commands[help_args.command],
                usage=f"neurenix {help_args.command} [<args>]"
            )
            
            command_func(argparse.Namespace(args=["--help"]))
            
            return 0
        except ImportError:
            print(f"Error: Command module for '{help_args.command}' not found.")
            return 1
        except AttributeError:
            print(f"Error: Command function for '{help_args.command}' not found.")
            return 1
    
    print("Neurenix CLI - Command Line Interface for the Neurenix Framework")
    print("\nUsage: neurenix <command> [<args>]")
    print("\nAvailable commands:")
    
    for cmd, desc in commands.items():
        print(f"  {cmd:12} - {desc}")
    
    print("\nFor more information about a specific command, use:")
    print("  neurenix help <command>")
    print("  neurenix <command> --help")
    
    return 0
