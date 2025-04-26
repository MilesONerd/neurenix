"""
Implementation of the 'preprocess' command for the Neurenix CLI.

This module provides functionality to preprocess input data for model training.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List, Union

import neurenix

def preprocess_command(args: argparse.Namespace) -> int:
    """
    Preprocess input data.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Preprocess input data",
        usage="neurenix preprocess [<args>]"
    )
    
    parser.add_argument(
        "--input",
        help="Input data file or directory",
        required=True
    )
    
    parser.add_argument(
        "--output",
        help="Output directory",
        required=True
    )
    
    parser.add_argument(
        "--config",
        help="Preprocessing configuration file",
        default=None
    )
    
    parser.add_argument(
        "--normalize",
        help="Normalize data",
        action="store_true"
    )
    
    parser.add_argument(
        "--resize",
        help="Resize images to WxH (e.g., 224x224)",
        default=None
    )
    
    parser.add_argument(
        "--augment",
        help="Apply data augmentation",
        action="store_true"
    )
    
    parser.add_argument(
        "--split",
        help="Split data into train/val/test (e.g., 0.7,0.15,0.15)",
        default=None
    )
    
    preprocess_args = parser.parse_args(args.args)
    
    if not os.path.exists(preprocess_args.input):
        print(f"Error: Input '{preprocess_args.input}' not found.")
        return 1
    
    os.makedirs(preprocess_args.output, exist_ok=True)
    
    try:
        config = {}
        if preprocess_args.config and os.path.exists(preprocess_args.config):
            with open(preprocess_args.config, "r") as f:
                config = json.load(f)
        
        if preprocess_args.normalize:
            config["normalize"] = True
        
        if preprocess_args.resize:
            try:
                width, height = map(int, preprocess_args.resize.split("x"))
                config["resize"] = {"width": width, "height": height}
            except ValueError:
                print(f"Error: Invalid resize format. Use WxH (e.g., 224x224).")
                return 1
        
        if preprocess_args.augment:
            config["augment"] = True
        
        if preprocess_args.split:
            try:
                splits = list(map(float, preprocess_args.split.split(",")))
                if len(splits) not in [2, 3] or abs(sum(splits) - 1.0) > 1e-6:
                    raise ValueError("Split values must sum to 1.0")
                config["split"] = splits
            except ValueError as e:
                print(f"Error: Invalid split format. Use comma-separated values that sum to 1.0 (e.g., 0.7,0.15,0.15).")
                return 1
        
        print(f"Loading data from {preprocess_args.input}...")
        
        if os.path.isdir(preprocess_args.input):
            data = neurenix.load_dataset(preprocess_args.input)
        else:
            data = neurenix.load_dataset(preprocess_args.input)
        
        print("Preprocessing data...")
        processed_data = neurenix.preprocess(data, **config)
        
        if preprocess_args.split:
            splits = config.get("split", [0.7, 0.3])
            
            if len(splits) == 2:
                train_data, val_data = processed_data
                
                print(f"Saving training data ({len(train_data)} samples)...")
                train_path = os.path.join(preprocess_args.output, "train")
                os.makedirs(train_path, exist_ok=True)
                neurenix.save_dataset(train_data, train_path)
                
                print(f"Saving validation data ({len(val_data)} samples)...")
                val_path = os.path.join(preprocess_args.output, "val")
                os.makedirs(val_path, exist_ok=True)
                neurenix.save_dataset(val_data, val_path)
            else:
                train_data, val_data, test_data = processed_data
                
                print(f"Saving training data ({len(train_data)} samples)...")
                train_path = os.path.join(preprocess_args.output, "train")
                os.makedirs(train_path, exist_ok=True)
                neurenix.save_dataset(train_data, train_path)
                
                print(f"Saving validation data ({len(val_data)} samples)...")
                val_path = os.path.join(preprocess_args.output, "val")
                os.makedirs(val_path, exist_ok=True)
                neurenix.save_dataset(val_data, val_path)
                
                print(f"Saving test data ({len(test_data)} samples)...")
                test_path = os.path.join(preprocess_args.output, "test")
                os.makedirs(test_path, exist_ok=True)
                neurenix.save_dataset(test_data, test_path)
        else:
            print(f"Saving processed data ({len(processed_data)} samples)...")
            neurenix.save_dataset(processed_data, preprocess_args.output)
        
        config_path = os.path.join(preprocess_args.output, "preprocess_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Preprocessing completed successfully. Results saved to {preprocess_args.output}")
        return 0
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return 1
