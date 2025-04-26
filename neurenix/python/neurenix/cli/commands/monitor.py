"""
Implementation of the 'monitor' command for the Neurenix CLI.

This module provides functionality to monitor model training in real-time.
"""

import os
import json
import time
import argparse
import threading
from typing import Dict, Any, Optional, List, Union

import neurenix

def monitor_command(args: argparse.Namespace) -> int:
    """
    Monitor model training in real-time.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Monitor model training in real-time",
        usage="neurenix monitor [<args>]"
    )
    
    parser.add_argument(
        "--log-dir",
        help="Log directory to monitor",
        default="logs"
    )
    
    parser.add_argument(
        "--refresh-rate",
        help="Refresh rate in seconds",
        type=float,
        default=1.0
    )
    
    parser.add_argument(
        "--metrics",
        help="Metrics to display (comma-separated)",
        default="loss,accuracy"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for monitoring data",
        default=None
    )
    
    parser.add_argument(
        "--plot",
        help="Generate plots for metrics",
        action="store_true"
    )
    
    monitor_args = parser.parse_args(args.args)
    
    if not os.path.exists(monitor_args.log_dir):
        print(f"Error: Log directory '{monitor_args.log_dir}' not found.")
        return 1
    
    try:
        metrics = [m.strip() for m in monitor_args.metrics.split(",")]
        
        print(f"Monitoring training logs in '{monitor_args.log_dir}'...")
        print(f"Metrics: {', '.join(metrics)}")
        print(f"Refresh rate: {monitor_args.refresh_rate} seconds")
        print("\nPress Ctrl+C to stop monitoring.\n")
        
        monitoring_data = {metric: [] for metric in metrics}
        epochs = []
        
        output_file = None
        if monitor_args.output:
            output_file = open(monitor_args.output, "w")
            output_file.write(f"epoch,{','.join(metrics)}\n")
        
        stop_event = threading.Event()
        
        def monitor_loop():
            try:
                while not stop_event.is_set():
                    log_files = sorted([
                        os.path.join(monitor_args.log_dir, f)
                        for f in os.listdir(monitor_args.log_dir)
                        if f.endswith(".json") or f.endswith(".log")
                    ], key=os.path.getmtime)
                    
                    if not log_files:
                        print("No log files found. Waiting for logs...")
                        time.sleep(monitor_args.refresh_rate)
                        continue
                    
                    latest_log = log_files[-1]
                    
                    if latest_log.endswith(".json"):
                        with open(latest_log, "r") as f:
                            log_data = json.load(f)
                        
                        current_epoch = log_data.get("epoch", len(epochs) + 1)
                        
                        if current_epoch not in epochs:
                            epochs.append(current_epoch)
                            
                            metric_values = []
                            for metric in metrics:
                                value = log_data.get(metric, None)
                                if value is not None:
                                    monitoring_data[metric].append(value)
                                    metric_values.append(str(value))
                                else:
                                    metric_values.append("N/A")
                            
                            print(f"Epoch {current_epoch}:")
                            for i, metric in enumerate(metrics):
                                print(f"  {metric}: {metric_values[i]}")
                            print()
                            
                            if output_file:
                                output_file.write(f"{current_epoch},{','.join(metric_values)}\n")
                                output_file.flush()
                    
                    time.sleep(monitor_args.refresh_rate)
            except Exception as e:
                print(f"Error monitoring logs: {str(e)}")
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            while monitor_thread.is_alive():
                monitor_thread.join(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            stop_event.set()
        
        if output_file:
            output_file.close()
        
        if monitor_args.plot and epochs:
            try:
                import matplotlib.pyplot as plt
                
                plot_dir = os.path.join(monitor_args.log_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                
                for metric in metrics:
                    if monitoring_data[metric]:
                        plt.figure(figsize=(10, 6))
                        plt.plot(epochs, monitoring_data[metric], marker='o')
                        plt.title(f"{metric.capitalize()} vs. Epoch")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric.capitalize())
                        plt.grid(True)
                        plt.savefig(os.path.join(plot_dir, f"{metric}_plot.png"))
                        plt.close()
                
                print(f"Plots saved to {plot_dir}")
            except ImportError:
                print("Warning: matplotlib not available. Plots not generated.")
        
        return 0
    except Exception as e:
        print(f"Error monitoring training: {str(e)}")
        return 1
