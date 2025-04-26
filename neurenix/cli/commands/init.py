"""
Implementation of the 'init' command for the Neurenix CLI.

This module provides functionality to initialize a new Neurenix project
with a standard folder structure, configuration files, and optional dataset.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional

from neurenix.data import DatasetHub, DatasetFormat

PROJECT_TEMPLATES = {
    "basic": {
        "dirs": ["data", "models", "configs", "logs"],
        "files": {
            "config.json": {
                "model": {
                    "type": "mlp",
                    "layers": [128, 64, 10],
                    "activation": "relu"
                },
                "training": {
                    "batch_size": 32,
                    "epochs": 10,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                "hardware": {
                    "device": "auto",
                    "precision": "float32"
                }
            },
            "train.py": """
import neurenix
from neurenix.nn import Module
from neurenix.optim import Adam

config = neurenix.load_config("config.json")

train_data, val_data = neurenix.load_dataset("data/dataset.csv", split=0.8)

model = neurenix.create_model(config["model"])

optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
neurenix.train(
    model, 
    train_data, 
    val_data, 
    optimizer=optimizer,
    batch_size=config["training"]["batch_size"],
    epochs=config["training"]["epochs"]
)

neurenix.save_model(model, "models/model.nrx")
"""
        }
    },
    "advanced": {
        "dirs": ["data", "models", "configs", "logs", "scripts", "notebooks", "tests"],
        "files": {
            "config.json": {
                "model": {
                    "type": "resnet",
                    "depth": 18,
                    "pretrained": True
                },
                "training": {
                    "batch_size": 64,
                    "epochs": 20,
                    "learning_rate": 0.0001,
                    "optimizer": "adam",
                    "scheduler": {
                        "type": "cosine",
                        "warmup_epochs": 2
                    }
                },
                "hardware": {
                    "device": "auto",
                    "precision": "mixed",
                    "distributed": False
                },
                "data": {
                    "train": "data/train",
                    "val": "data/val",
                    "test": "data/test",
                    "augmentation": True
                }
            },
            "train.py": """
import neurenix
from neurenix.nn import Module
from neurenix.optim import Adam
from neurenix.data import DataLoader

config = neurenix.load_config("config.json")

neurenix.set_device(config["hardware"]["device"], precision=config["hardware"]["precision"])

train_loader = DataLoader(
    neurenix.load_dataset(config["data"]["train"]),
    batch_size=config["training"]["batch_size"],
    shuffle=True
)
val_loader = DataLoader(
    neurenix.load_dataset(config["data"]["val"]),
    batch_size=config["training"]["batch_size"]
)

model = neurenix.create_model(config["model"])

optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])

scheduler = neurenix.create_scheduler(optimizer, config["training"]["scheduler"])

neurenix.train(
    model, 
    train_loader, 
    val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=config["training"]["epochs"],
    log_dir="logs"
)

neurenix.save_model(model, "models/model.nrx")
"""
        }
    }
}

def init_command(args: argparse.Namespace) -> int:
    """
    Initialize a new Neurenix project.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Initialize a new Neurenix project",
        usage="neurenix init [<args>]"
    )
    
    parser.add_argument(
        "--name",
        help="Project name",
        default="neurenix-project"
    )
    
    parser.add_argument(
        "--template",
        help="Project template",
        choices=["basic", "advanced"],
        default="basic"
    )
    
    parser.add_argument(
        "--dataset",
        help="Dataset to download (URL or registered name)",
        default=None
    )
    
    parser.add_argument(
        "--force",
        help="Force overwrite if directory exists",
        action="store_true"
    )
    
    init_args = parser.parse_args(args.args)
    
    if os.path.exists(init_args.name) and not init_args.force:
        print(f"Error: Directory '{init_args.name}' already exists. Use --force to overwrite.")
        return 1
    
    os.makedirs(init_args.name, exist_ok=True)
    
    template = PROJECT_TEMPLATES[init_args.template]
    
    for dir_name in template["dirs"]:
        os.makedirs(os.path.join(init_args.name, dir_name), exist_ok=True)
    
    for file_name, content in template["files"].items():
        file_path = os.path.join(init_args.name, file_name)
        
        if isinstance(content, dict):
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2)
        else:
            with open(file_path, "w") as f:
                f.write(content.strip())
    
    if init_args.dataset:
        try:
            print(f"Downloading dataset from {init_args.dataset}...")
            hub = DatasetHub()
            dataset = hub.load_dataset(init_args.dataset)
            
            dataset_path = os.path.join(init_args.name, "data", "dataset.csv")
            print(f"Saving dataset to {dataset_path}...")
            
            with open(dataset_path, "w") as f:
                if hasattr(dataset, "columns") and dataset.columns:
                    f.write(",".join(dataset.columns) + "\n")
                
                for row in dataset:
                    f.write(",".join(str(item) for item in row) + "\n")
            
            print(f"Dataset saved successfully.")
        except Exception as e:
            print(f"Warning: Failed to download dataset: {str(e)}")
    
    print(f"Neurenix project '{init_args.name}' initialized successfully with {init_args.template} template.")
    return 0
