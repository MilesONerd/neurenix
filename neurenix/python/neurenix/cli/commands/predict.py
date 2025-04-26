"""
Implementation of the 'predict' command for the Neurenix CLI.

This module provides functionality to make predictions using a trained model.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List, Union

import neurenix

def predict_command(args: argparse.Namespace) -> int:
    """
    Make predictions using a trained model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model",
        usage="neurenix predict [<args>]"
    )
    
    parser.add_argument(
        "--model",
        help="Model file",
        required=True
    )
    
    parser.add_argument(
        "--input",
        help="Input data file or directory",
        required=True
    )
    
    parser.add_argument(
        "--output",
        help="Output file",
        default="predictions.csv"
    )
    
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=32
    )
    
    parser.add_argument(
        "--device",
        help="Device to use (cpu, cuda, auto)",
        default="auto"
    )
    
    parser.add_argument(
        "--format",
        help="Output format (csv, json, npy)",
        choices=["csv", "json", "npy"],
        default=None
    )
    
    predict_args = parser.parse_args(args.args)
    
    if not os.path.exists(predict_args.model):
        print(f"Error: Model file '{predict_args.model}' not found.")
        return 1
    
    if not os.path.exists(predict_args.input):
        print(f"Error: Input '{predict_args.input}' not found.")
        return 1
    
    try:
        print(f"Loading model from {predict_args.model}...")
        model = neurenix.load_model(predict_args.model)
        
        neurenix.set_device(predict_args.device)
        
        print(f"Loading input data from {predict_args.input}...")
        
        if os.path.isdir(predict_args.input):
            data = neurenix.load_dataset(predict_args.input)
        else:
            data = neurenix.load_dataset(predict_args.input)
        
        print(f"Making predictions with batch size {predict_args.batch_size}...")
        predictions = neurenix.predict(model, data, batch_size=predict_args.batch_size)
        
        print(f"Saving predictions to {predict_args.output}...")
        
        format_type = predict_args.format
        if not format_type:
            _, ext = os.path.splitext(predict_args.output)
            if ext.lower() == ".json":
                format_type = "json"
            elif ext.lower() == ".npy":
                format_type = "npy"
            else:
                format_type = "csv"
        
        if format_type == "json":
            with open(predict_args.output, "w") as f:
                json.dump(predictions, f, indent=2)
        elif format_type == "npy":
            try:
                import numpy as np
                np.save(predict_args.output, predictions)
            except ImportError:
                print("Warning: NumPy not available. Falling back to CSV format.")
                format_type = "csv"
        
        if format_type == "csv":
            with open(predict_args.output, "w") as f:
                if hasattr(predictions, "columns") and predictions.columns:
                    f.write(",".join(predictions.columns) + "\n")
                
                for row in predictions:
                    f.write(",".join(str(item) for item in row) + "\n")
        
        print(f"Predictions saved to {predict_args.output}")
        return 0
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return 1
