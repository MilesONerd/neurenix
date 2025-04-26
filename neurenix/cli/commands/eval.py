"""
Implementation of the 'eval' command for the Neurenix CLI.

This module provides functionality to evaluate a trained model
with specific metrics.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List, Union

import neurenix

def eval_command(args: argparse.Namespace) -> int:
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model",
        usage="neurenix eval [<args>]"
    )
    
    parser.add_argument(
        "--model",
        help="Model file",
        required=True
    )
    
    parser.add_argument(
        "--data",
        help="Evaluation data file or directory",
        required=True
    )
    
    parser.add_argument(
        "--metrics",
        help="Metrics to evaluate (comma-separated)",
        default="accuracy,precision,recall,f1"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for evaluation results",
        default="evaluation.json"
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
    
    eval_args = parser.parse_args(args.args)
    
    if not os.path.exists(eval_args.model):
        print(f"Error: Model file '{eval_args.model}' not found.")
        return 1
    
    if not os.path.exists(eval_args.data):
        print(f"Error: Data '{eval_args.data}' not found.")
        return 1
    
    try:
        print(f"Loading model from {eval_args.model}...")
        model = neurenix.load_model(eval_args.model)
        
        neurenix.set_device(eval_args.device)
        
        print(f"Loading evaluation data from {eval_args.data}...")
        
        if os.path.isdir(eval_args.data):
            data = neurenix.load_dataset(eval_args.data)
        else:
            data = neurenix.load_dataset(eval_args.data)
        
        metrics = [m.strip() for m in eval_args.metrics.split(",")]
        print(f"Evaluating model with metrics: {', '.join(metrics)}...")
        
        results = neurenix.evaluate(model, data, metrics=metrics, batch_size=eval_args.batch_size)
        
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
        
        if eval_args.output:
            print(f"\nSaving evaluation results to {eval_args.output}...")
            
            with open(eval_args.output, "w") as f:
                json.dump(results, f, indent=2)
        
        return 0
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return 1
