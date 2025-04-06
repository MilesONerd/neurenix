"""
Implementation of the 'save' command for the Neurenix CLI.

This module provides functionality to save the current project state,
including model, configurations, and data.
"""

import os
import json
import time
import shutil
import argparse
from typing import Dict, Any, Optional

import neurenix

def save_command(args: argparse.Namespace) -> int:
    """
    Save the current project state.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Save the current project state",
        usage="neurenix save [<args>]"
    )
    
    parser.add_argument(
        "--name",
        help="Checkpoint name",
        default=f"checkpoint_{int(time.time())}"
    )
    
    parser.add_argument(
        "--include-data",
        help="Include data in the checkpoint",
        action="store_true"
    )
    
    parser.add_argument(
        "--include-logs",
        help="Include logs in the checkpoint",
        action="store_true"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory",
        default="checkpoints"
    )
    
    save_args = parser.parse_args(args.args)
    
    os.makedirs(save_args.output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(save_args.output_dir, save_args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if os.path.exists("models"):
        print("Saving models...")
        shutil.copytree("models", os.path.join(checkpoint_dir, "models"), dirs_exist_ok=True)
    
    if os.path.exists("configs"):
        print("Saving configurations...")
        shutil.copytree("configs", os.path.join(checkpoint_dir, "configs"), dirs_exist_ok=True)
    elif os.path.exists("config.json"):
        print("Saving configuration...")
        os.makedirs(os.path.join(checkpoint_dir, "configs"), exist_ok=True)
        shutil.copy("config.json", os.path.join(checkpoint_dir, "configs", "config.json"))
    
    if save_args.include_data and os.path.exists("data"):
        print("Saving data...")
        shutil.copytree("data", os.path.join(checkpoint_dir, "data"), dirs_exist_ok=True)
    
    if save_args.include_logs and os.path.exists("logs"):
        print("Saving logs...")
        shutil.copytree("logs", os.path.join(checkpoint_dir, "logs"), dirs_exist_ok=True)
    
    metadata = {
        "timestamp": time.time(),
        "name": save_args.name,
        "neurenix_version": neurenix.__version__,
        "include_data": save_args.include_data,
        "include_logs": save_args.include_logs
    }
    
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Project state saved to {checkpoint_dir}")
    return 0
