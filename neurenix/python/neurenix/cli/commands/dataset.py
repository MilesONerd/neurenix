"""
Implementation of the 'dataset' command for the Neurenix CLI.

This module provides functionality to manage datasets for training and evaluation.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List, Union

import neurenix
from neurenix.data import DatasetHub, DatasetFormat

def dataset_command(args: argparse.Namespace) -> int:
    """
    Manage datasets for training and evaluation.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Manage datasets for training and evaluation",
        usage="neurenix dataset [<args>]"
    )
    
    subparsers = parser.add_subparsers(dest="action", help="Dataset action")
    
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument(
        "--format",
        help="Output format (text, json)",
        choices=["text", "json"],
        default="text"
    )
    
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument(
        "source",
        help="Dataset source (URL or registered name)"
    )
    download_parser.add_argument(
        "--output",
        help="Output directory or file",
        default="data"
    )
    download_parser.add_argument(
        "--format",
        help="Dataset format",
        choices=[f.name.lower() for f in DatasetFormat],
        default=None
    )
    
    register_parser = subparsers.add_parser("register", help="Register a dataset")
    register_parser.add_argument(
        "name",
        help="Dataset name"
    )
    register_parser.add_argument(
        "url",
        help="Dataset URL or file path"
    )
    register_parser.add_argument(
        "--format",
        help="Dataset format",
        choices=[f.name.lower() for f in DatasetFormat],
        default=None
    )
    register_parser.add_argument(
        "--metadata",
        help="Dataset metadata (JSON string or file)",
        default=None
    )
    
    info_parser = subparsers.add_parser("info", help="Get dataset information")
    info_parser.add_argument(
        "name",
        help="Dataset name or path"
    )
    info_parser.add_argument(
        "--format",
        help="Output format (text, json)",
        choices=["text", "json"],
        default="text"
    )
    
    split_parser = subparsers.add_parser("split", help="Split a dataset")
    split_parser.add_argument(
        "input",
        help="Input dataset file or directory"
    )
    split_parser.add_argument(
        "--output",
        help="Output directory",
        default="data"
    )
    split_parser.add_argument(
        "--ratio",
        help="Split ratio (e.g., 0.7,0.15,0.15)",
        default="0.8,0.2"
    )
    split_parser.add_argument(
        "--shuffle",
        help="Shuffle data before splitting",
        action="store_true"
    )
    split_parser.add_argument(
        "--seed",
        help="Random seed for shuffling",
        type=int,
        default=None
    )
    
    convert_parser = subparsers.add_parser("convert", help="Convert a dataset to a different format")
    convert_parser.add_argument(
        "input",
        help="Input dataset file or directory"
    )
    convert_parser.add_argument(
        "output",
        help="Output file or directory"
    )
    convert_parser.add_argument(
        "--input-format",
        help="Input format",
        choices=[f.name.lower() for f in DatasetFormat],
        default=None
    )
    convert_parser.add_argument(
        "--output-format",
        help="Output format",
        choices=[f.name.lower() for f in DatasetFormat],
        required=True
    )
    
    dataset_args = parser.parse_args(args.args)
    
    hub = DatasetHub()
    
    try:
        if not dataset_args.action:
            parser.print_help()
            return 0
        
        elif dataset_args.action == "list":
            datasets = hub.list_datasets()
            
            if dataset_args.format == "json":
                print(json.dumps(datasets, indent=2))
            else:
                print("Available datasets:")
                for name, info in datasets.items():
                    print(f"\n{name}:")
                    print(f"  URL: {info['url']}")
                    print(f"  Format: {info.get('format', 'auto-detect')}")
                    if "metadata" in info and info["metadata"]:
                        print(f"  Metadata: {info['metadata']}")
            
            return 0
        
        elif dataset_args.action == "download":
            print(f"Downloading dataset from {dataset_args.source}...")
            
            format_enum = None
            if dataset_args.format:
                format_enum = getattr(DatasetFormat, dataset_args.format.upper())
            
            dataset = hub.load_dataset(
                dataset_args.source,
                format=format_enum,
                force_download=True
            )
            
            if os.path.isdir(dataset_args.output) or not os.path.exists(dataset_args.output):
                os.makedirs(dataset_args.output, exist_ok=True)
                
                if os.path.isdir(dataset_args.output):
                    if hasattr(dataset, "name") and dataset.name:
                        base_name = dataset.name
                    else:
                        base_name = os.path.basename(dataset_args.source).split("?")[0]
                        if not base_name:
                            base_name = "dataset"
                    
                    output_file = os.path.join(dataset_args.output, base_name)
                    
                    if not os.path.splitext(output_file)[1]:
                        output_file += ".csv"
                else:
                    output_file = dataset_args.output
            else:
                output_file = dataset_args.output
            
            print(f"Saving dataset to {output_file}...")
            neurenix.save_dataset(dataset, output_file)
            
            print(f"Dataset downloaded and saved to {output_file}")
            return 0
        
        elif dataset_args.action == "register":
            
            format_enum = None
            if dataset_args.format:
                format_enum = getattr(DatasetFormat, dataset_args.format.upper())
            
            metadata = {}
            if dataset_args.metadata:
                if os.path.exists(dataset_args.metadata):
                    with open(dataset_args.metadata, "r") as f:
                        metadata = json.load(f)
                else:
                    try:
                        metadata = json.loads(dataset_args.metadata)
                    except json.JSONDecodeError:
                        print(f"Error: Invalid metadata JSON: {dataset_args.metadata}")
                        return 1
            
            hub.register_dataset(
                dataset_args.name,
                dataset_args.url,
                format=format_enum,
                metadata=metadata
            )
            
            print(f"Dataset '{dataset_args.name}' registered successfully.")
            return 0
        
        elif dataset_args.action == "info":
            try:
                info = hub.get_dataset_info(dataset_args.name)
                
                if dataset_args.format == "json":
                    print(json.dumps(info, indent=2))
                else:
                    print(f"Dataset: {dataset_args.name}")
                    print(f"URL: {info['url']}")
                    print(f"Format: {info.get('format', 'auto-detect')}")
                    if "metadata" in info and info["metadata"]:
                        print(f"Metadata: {info['metadata']}")
            except KeyError:
                if os.path.exists(dataset_args.name):
                    print(f"Loading dataset from {dataset_args.name}...")
                    dataset = hub.load_dataset(dataset_args.name)
                    
                    info = {
                        "path": os.path.abspath(dataset_args.name),
                        "format": str(dataset.format) if hasattr(dataset, "format") else "unknown",
                        "size": len(dataset) if hasattr(dataset, "__len__") else "unknown",
                        "metadata": dataset.metadata if hasattr(dataset, "metadata") else {}
                    }
                    
                    if dataset_args.format == "json":
                        print(json.dumps(info, indent=2))
                    else:
                        print(f"Dataset: {os.path.basename(dataset_args.name)}")
                        print(f"Path: {info['path']}")
                        print(f"Format: {info['format']}")
                        print(f"Size: {info['size']}")
                        if info["metadata"]:
                            print(f"Metadata: {info['metadata']}")
                else:
                    print(f"Error: Dataset '{dataset_args.name}' not found.")
                    return 1
            
            return 0
        
        elif dataset_args.action == "split":
            if not os.path.exists(dataset_args.input):
                print(f"Error: Input dataset '{dataset_args.input}' not found.")
                return 1
            
            os.makedirs(dataset_args.output, exist_ok=True)
            
            try:
                ratios = list(map(float, dataset_args.ratio.split(",")))
                if abs(sum(ratios) - 1.0) > 1e-6:
                    raise ValueError("Split ratios must sum to 1.0")
            except ValueError as e:
                print(f"Error: Invalid split ratio: {str(e)}")
                return 1
            
            print(f"Loading dataset from {dataset_args.input}...")
            dataset = hub.load_dataset(dataset_args.input)
            
            print(f"Splitting dataset with ratio {dataset_args.ratio}...")
            splits = dataset.split(
                ratio=ratios,
                shuffle=dataset_args.shuffle,
                seed=dataset_args.seed
            )
            
            split_names = ["train", "val", "test"][:len(splits)]
            
            for i, (name, split) in enumerate(zip(split_names, splits)):
                split_path = os.path.join(dataset_args.output, name)
                os.makedirs(split_path, exist_ok=True)
                
                output_file = os.path.join(split_path, f"{name}_data.csv")
                print(f"Saving {name} split ({len(split)} samples) to {output_file}...")
                neurenix.save_dataset(split, output_file)
            
            print(f"Dataset split successfully. Results saved to {dataset_args.output}")
            return 0
        
        elif dataset_args.action == "convert":
            if not os.path.exists(dataset_args.input):
                print(f"Error: Input dataset '{dataset_args.input}' not found.")
                return 1
            
            input_format = None
            if dataset_args.input_format:
                input_format = getattr(DatasetFormat, dataset_args.input_format.upper())
            
            output_format = getattr(DatasetFormat, dataset_args.output_format.upper())
            
            print(f"Loading dataset from {dataset_args.input}...")
            dataset = hub.load_dataset(dataset_args.input, format=input_format)
            
            output_dir = os.path.dirname(dataset_args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            print(f"Converting dataset to {dataset_args.output_format} format...")
            neurenix.save_dataset(dataset, dataset_args.output, format=output_format)
            
            print(f"Dataset converted and saved to {dataset_args.output}")
            return 0
        
        else:
            print(f"Error: Unknown action '{dataset_args.action}'")
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error managing dataset: {str(e)}")
        return 1
