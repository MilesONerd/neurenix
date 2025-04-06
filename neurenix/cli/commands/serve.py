"""
Implementation of the 'serve' command for the Neurenix CLI.

This module provides functionality to serve a trained model as a RESTful API.
"""

import os
import json
import argparse
import threading
import time
from typing import Dict, Any, Optional, List, Union

import neurenix

def serve_command(args: argparse.Namespace) -> int:
    """
    Serve a trained model as a RESTful API.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Serve a trained model as a RESTful API",
        usage="neurenix serve [<args>]"
    )
    
    parser.add_argument(
        "--model",
        help="Model file",
        required=True
    )
    
    parser.add_argument(
        "--host",
        help="Host to bind the server to",
        default="0.0.0.0"
    )
    
    parser.add_argument(
        "--port",
        help="Port to bind the server to",
        type=int,
        default=8000
    )
    
    parser.add_argument(
        "--device",
        help="Device to use for inference",
        default="auto"
    )
    
    parser.add_argument(
        "--batch-size",
        help="Batch size for inference",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--workers",
        help="Number of worker processes",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--api-type",
        help="API type",
        choices=["rest", "websocket", "grpc"],
        default="rest"
    )
    
    parser.add_argument(
        "--config",
        help="API configuration file",
        default=None
    )
    
    parser.add_argument(
        "--auth",
        help="Enable authentication",
        action="store_true"
    )
    
    parser.add_argument(
        "--cors",
        help="Enable CORS",
        action="store_true"
    )
    
    serve_args = parser.parse_args(args.args)
    
    if not os.path.exists(serve_args.model):
        print(f"Error: Model file '{serve_args.model}' not found.")
        return 1
    
    try:
        print(f"Loading model from {serve_args.model}...")
        model = neurenix.load_model(serve_args.model)
        
        neurenix.set_device(serve_args.device)
        
        config = {}
        if serve_args.config and os.path.exists(serve_args.config):
            with open(serve_args.config, "r") as f:
                config = json.load(f)
        
        config["host"] = serve_args.host
        config["port"] = serve_args.port
        config["batch_size"] = serve_args.batch_size
        config["workers"] = serve_args.workers
        config["api_type"] = serve_args.api_type
        config["auth"] = serve_args.auth
        config["cors"] = serve_args.cors
        
        print(f"Creating {serve_args.api_type.upper()} API server...")
        
        if serve_args.api_type == "rest":
            from neurenix.api_support import RESTfulAPI
            api = RESTfulAPI(model, **config)
        elif serve_args.api_type == "websocket":
            from neurenix.api_support import WebSocketAPI
            api = WebSocketAPI(model, **config)
        elif serve_args.api_type == "grpc":
            from neurenix.api_support import gRPCAPI
            api = gRPCAPI(model, **config)
        
        print(f"Starting server on {serve_args.host}:{serve_args.port}...")
        
        stop_event = threading.Event()
        
        def signal_handler(sig, frame):
            print("\nStopping server...")
            stop_event.set()
        
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        server_thread = threading.Thread(target=api.start)
        server_thread.daemon = True
        server_thread.start()
        
        print("\nAPI Endpoints:")
        
        if serve_args.api_type == "rest":
            print(f"  POST http://{serve_args.host}:{serve_args.port}/predict")
            print(f"  GET  http://{serve_args.host}:{serve_args.port}/info")
            print(f"  GET  http://{serve_args.host}:{serve_args.port}/health")
        elif serve_args.api_type == "websocket":
            print(f"  WebSocket ws://{serve_args.host}:{serve_args.port}/ws")
        elif serve_args.api_type == "grpc":
            print(f"  gRPC {serve_args.host}:{serve_args.port}")
            print(f"  Use the generated client to connect to the server.")
        
        print("\nPress Ctrl+C to stop the server.\n")
        
        while not stop_event.is_set():
            time.sleep(0.1)
        
        api.stop()
        
        print("Server stopped.")
        return 0
    except Exception as e:
        print(f"Error serving model: {str(e)}")
        return 1
