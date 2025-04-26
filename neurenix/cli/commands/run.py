"""
Implementation of the 'run' command for the Neurenix CLI.

This module provides functionality to run a Neurenix model training script
with specified configuration options.
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, Any, Optional

def run_command(args: argparse.Namespace) -> int:
    """
    Run a Neurenix model training script.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Run a Neurenix model training script",
        usage="neurenix run [<args>]"
    )
    
    parser.add_argument(
        "script",
        help="Script to run",
        nargs="?",
        default="train.py"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file",
        default="config.json"
    )
    
    parser.add_argument(
        "--device",
        help="Device to use (cpu, cuda, auto)",
        default=None
    )
    
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--learning-rate",
        help="Learning rate",
        type=float,
        default=None
    )
    
    run_args = parser.parse_args(args.args)
    
    if not os.path.exists(run_args.script):
        print(f"Error: Script '{run_args.script}' not found.")
        return 1
    
    if not os.path.exists(run_args.config):
        print(f"Error: Configuration file '{run_args.config}' not found.")
        return 1
    
    with open(run_args.config, "r") as f:
        config = json.load(f)
    
    if run_args.device:
        config["hardware"]["device"] = run_args.device
    
    if run_args.batch_size:
        config["training"]["batch_size"] = run_args.batch_size
    
    if run_args.epochs:
        config["training"]["epochs"] = run_args.epochs
    
    if run_args.learning_rate:
        config["training"]["learning_rate"] = run_args.learning_rate
    
    with open(run_args.config, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Running {run_args.script} with configuration from {run_args.config}...")
    
    try:
        env = os.environ.copy()
        env["NEURENIX_CONFIG"] = os.path.abspath(run_args.config)
        
        result = subprocess.run([sys.executable, run_args.script], env=env)
        
        if result.returncode != 0:
            print(f"Error: Script exited with code {result.returncode}")
            return result.returncode
        
        print(f"Script completed successfully.")
        return 0
    except Exception as e:
        print(f"Error running script: {str(e)}")
        return 1
