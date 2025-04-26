"""
Implementation of the 'hardware' command for the Neurenix CLI.

This module provides functionality to manage hardware settings,
including auto-selection, device information, and benchmarking.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List, Union

import neurenix

def hardware_command(args: argparse.Namespace) -> int:
    """
    Manage hardware settings.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Manage hardware settings",
        usage="neurenix hardware [<args>]"
    )
    
    parser.add_argument(
        "action",
        help="Hardware action",
        choices=["list", "info", "auto", "select", "benchmark"],
        default="list",
        nargs="?"
    )
    
    parser.add_argument(
        "--device",
        help="Device to use (required for 'select' action)",
        default=None
    )
    
    parser.add_argument(
        "--precision",
        help="Precision to use",
        choices=["float32", "float16", "mixed", "int8"],
        default="float32"
    )
    
    parser.add_argument(
        "--memory-limit",
        help="Memory limit in GB",
        type=float,
        default=None
    )
    
    hw_args = parser.parse_args(args.args)
    
    try:
        if hw_args.action == "list":
            print("Available hardware devices:")
            devices = neurenix.list_devices()
            
            for i, device in enumerate(devices):
                print(f"{i+1}. {device['name']} ({device['type']})")
                print(f"   - Memory: {device['memory']} GB")
                print(f"   - Compute capability: {device.get('compute_capability', 'N/A')}")
                print(f"   - Supported precisions: {', '.join(device['supported_precisions'])}")
                print()
            
            return 0
        
        elif hw_args.action == "info":
            device_info = neurenix.get_current_device()
            
            print("Current device information:")
            print(f"Name: {device_info['name']}")
            print(f"Type: {device_info['type']}")
            print(f"Memory: {device_info['memory']} GB")
            print(f"Compute capability: {device_info.get('compute_capability', 'N/A')}")
            print(f"Current precision: {device_info['precision']}")
            print(f"Memory usage: {device_info['memory_used']} / {device_info['memory']} GB")
            
            return 0
        
        elif hw_args.action == "auto":
            print("Auto-selecting optimal hardware...")
            
            device = neurenix.auto_select_device(precision=hw_args.precision)
            
            print(f"Selected device: {device['name']} ({device['type']})")
            print(f"Memory: {device['memory']} GB")
            print(f"Precision: {hw_args.precision}")
            
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                
                if "hardware" in config:
                    config["hardware"]["device"] = device["name"]
                    config["hardware"]["precision"] = hw_args.precision
                    
                    with open("config.json", "w") as f:
                        json.dump(config, f, indent=2)
                    
                    print("Updated configuration in config.json")
            
            return 0
        
        elif hw_args.action == "select":
            if not hw_args.device:
                print("Error: --device is required for 'select' action.")
                return 1
            
            print(f"Selecting hardware: {hw_args.device}")
            
            options = {
                "precision": hw_args.precision
            }
            
            if hw_args.memory_limit:
                options["memory_limit"] = hw_args.memory_limit
            
            device = neurenix.set_device(hw_args.device, **options)
            
            print(f"Selected device: {device['name']} ({device['type']})")
            print(f"Memory: {device['memory']} GB")
            print(f"Precision: {hw_args.precision}")
            
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                
                if "hardware" in config:
                    config["hardware"]["device"] = hw_args.device
                    config["hardware"]["precision"] = hw_args.precision
                    
                    with open("config.json", "w") as f:
                        json.dump(config, f, indent=2)
                    
                    print("Updated configuration in config.json")
            
            return 0
        
        elif hw_args.action == "benchmark":
            print("Benchmarking available hardware...")
            
            results = neurenix.benchmark_devices(precision=hw_args.precision)
            
            print("\nBenchmark Results:")
            for device, metrics in results.items():
                print(f"\n{device}:")
                print(f"  - Inference time: {metrics['inference_time']:.4f} ms")
                print(f"  - Training time: {metrics['training_time']:.4f} ms")
                print(f"  - Memory usage: {metrics['memory_usage']:.2f} GB")
                print(f"  - Throughput: {metrics['throughput']:.2f} samples/sec")
            
            best_device = max(results.items(), key=lambda x: x[1]['throughput'])[0]
            print(f"\nRecommended device: {best_device}")
            
            return 0
    
    except Exception as e:
        print(f"Error managing hardware: {str(e)}")
        return 1
