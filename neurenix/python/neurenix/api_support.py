"""
API support module for the Neurenix framework.

This module provides built-in API endpoints with RESTful, WebSocket, and gRPC
support for serving Neurenix models and handling distributed computing tasks.
"""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Union, Any, Callable

from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.device_manager import DeviceManager

logger = logging.getLogger("neurenix")

class APIServer:
    """
    Base class for API servers in the Neurenix framework.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the API server.
        
        Args:
            host: Host to bind the server to.
            port: Port to bind the server to.
        """
        self.host = host
        self.port = port
        self.models: Dict[str, Module] = {}
        self.device_manager = DeviceManager()
        self.running = False
        self.server_thread = None
    
    def add_model(self, name: str, model: Module) -> None:
        """
        Add a model to the server.
        
        Args:
            name: Name of the model.
            model: The model to add.
        """
        self.models[name] = model
        logger.info(f"Added model '{name}' to API server")
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the server.
        
        Args:
            name: Name of the model to remove.
        """
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model '{name}' from API server")
    
    def start(self) -> None:
        """Start the API server."""
        if self.running:
            logger.warning("API server is already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logger.info(f"API server started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the API server."""
        if not self.running:
            logger.warning("API server is not running")
            return
        
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
            self.server_thread = None
        
        logger.info("API server stopped")
    
    def _run_server(self) -> None:
        """Run the server in a separate thread."""
        raise NotImplementedError("Subclasses must implement _run_server")
    
    def _preprocess_input(self, data: Any) -> Tensor:
        """
        Preprocess input data into a tensor.
        
        Args:
            data: Input data to preprocess.
            
        Returns:
            A tensor containing the preprocessed data.
        """
        if isinstance(data, list):
            import numpy as np
            return Tensor(np.array(data, dtype=np.float32))
        elif isinstance(data, dict):
            if "data" in data:
                return self._preprocess_input(data["data"])
            else:
                raise ValueError("Input dictionary must contain a 'data' key")
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
    
    def _postprocess_output(self, tensor: Tensor) -> Any:
        """
        Postprocess a tensor into a serializable format.
        
        Args:
            tensor: The tensor to postprocess.
            
        Returns:
            A serializable representation of the tensor.
        """
        return tensor.numpy().tolist()


class RESTfulServer(APIServer):
    """
    RESTful API server for the Neurenix framework.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the RESTful API server.
        
        Args:
            host: Host to bind the server to.
            port: Port to bind the server to.
        """
        super().__init__(host, port)
        self.app = None
    
    def _run_server(self) -> None:
        """Run the RESTful server."""
        try:
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.responses import JSONResponse
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
        except ImportError:
            logger.error("FastAPI and uvicorn are required for RESTful API server. "
                        "Install them with: pip install fastapi uvicorn")
            return
        
        self.app = FastAPI(title="Neurenix API", version="1.0.0")
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/")
        async def root():
            return {"message": "Neurenix API Server", "version": "1.0.0"}
        
        @self.app.get("/models")
        async def list_models():
            return {"models": list(self.models.keys())}
        
        @self.app.post("/models/{model_name}/predict")
        async def predict(model_name: str, request: Request):
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            
            try:
                data = await request.json()
                
                input_tensor = self._preprocess_input(data)
                
                model = self.models[model_name]
                output_tensor = model(input_tensor)
                
                result = self._postprocess_output(output_tensor)
                
                return {"result": result}
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        asyncio.run(server.serve())


class WebSocketServer(APIServer):
    """
    WebSocket API server for the Neurenix framework.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        """
        Initialize the WebSocket API server.
        
        Args:
            host: Host to bind the server to.
            port: Port to bind the server to.
        """
        super().__init__(host, port)
        self.clients = set()
        self.server = None
    
    def _run_server(self) -> None:
        """Run the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets is required for WebSocket API server. "
                        "Install it with: pip install websockets")
            return
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if "action" not in data:
                            await websocket.send(json.dumps({
                                "error": "Missing 'action' field"
                            }))
                            continue
                        
                        if data["action"] == "list_models":
                            await websocket.send(json.dumps({
                                "models": list(self.models.keys())
                            }))
                        elif data["action"] == "predict":
                            if "model" not in data:
                                await websocket.send(json.dumps({
                                    "error": "Missing 'model' field"
                                }))
                                continue
                            
                            if "data" not in data:
                                await websocket.send(json.dumps({
                                    "error": "Missing 'data' field"
                                }))
                                continue
                            
                            model_name = data["model"]
                            if model_name not in self.models:
                                await websocket.send(json.dumps({
                                    "error": f"Model '{model_name}' not found"
                                }))
                                continue
                            
                            input_tensor = self._preprocess_input(data["data"])
                            
                            model = self.models[model_name]
                            output_tensor = model(input_tensor)
                            
                            result = self._postprocess_output(output_tensor)
                            
                            await websocket.send(json.dumps({
                                "result": result
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "error": f"Unknown action: {data['action']}"
                            }))
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "error": "Invalid JSON"
                        }))
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        await websocket.send(json.dumps({
                            "error": str(e)
                        }))
            finally:
                self.clients.remove(websocket)
        
        async def serve():
            async with websockets.serve(handler, self.host, self.port):
                await asyncio.Future()  # Run forever
        
        asyncio.run(serve())


class GRPCServer(APIServer):
    """
    gRPC API server for the Neurenix framework.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002):
        """
        Initialize the gRPC API server.
        
        Args:
            host: Host to bind the server to.
            port: Port to bind the server to.
        """
        super().__init__(host, port)
        self.server = None
    
    def _run_server(self) -> None:
        """Run the gRPC server."""
        try:
            import grpc
            from concurrent import futures
        except ImportError:
            logger.error("grpc is required for gRPC API server. "
                        "Install it with: pip install grpcio grpcio-tools")
            return
        
        proto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto")
        proto_file = os.path.join(proto_dir, "neurenix_api.proto")
        
        if not os.path.exists(proto_file):
            os.makedirs(proto_dir, exist_ok=True)
            
            with open(proto_file, "w") as f:
                f.write("""
syntax = "proto3";

package neurenix;

service NeurenixService {
    // List available models
    rpc ListModels (ListModelsRequest) returns (ListModelsResponse);
    
    // Run inference on a model
    rpc Predict (PredictRequest) returns (PredictResponse);
}

message ListModelsRequest {
    // Empty request
}

message ListModelsResponse {
    repeated string models = 1;
}

message PredictRequest {
    string model_name = 1;
    repeated float data = 2;
    repeated int32 shape = 3;
}

message PredictResponse {
    repeated float result = 1;
    repeated int32 shape = 2;
}
                """)
        
        import subprocess
        try:
            subprocess.run([
                "python", "-m", "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={proto_dir}",
                f"--grpc_python_out={proto_dir}",
                proto_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile proto file: {e}")
            return
        
        import sys
        sys.path.append(proto_dir)
        
        try:
            import neurenix_api_pb2
            import neurenix_api_pb2_grpc
        except ImportError:
            logger.error("Failed to import generated proto modules")
            return
        
        class NeurenixServicer(neurenix_api_pb2_grpc.NeurenixServiceServicer):
            def __init__(self, server):
                self.server = server
            
            def ListModels(self, request, context):
                return neurenix_api_pb2.ListModelsResponse(
                    models=list(self.server.models.keys())
                )
            
            def Predict(self, request, context):
                try:
                    if request.model_name not in self.server.models:
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(f"Model '{request.model_name}' not found")
                        return neurenix_api_pb2.PredictResponse()
                    
                    import numpy as np
                    data = np.array(request.data, dtype=np.float32)
                    if request.shape:
                        data = data.reshape(request.shape)
                    
                    input_tensor = Tensor(data)
                    
                    model = self.server.models[request.model_name]
                    output_tensor = model(input_tensor)
                    
                    output_data = output_tensor.numpy()
                    result = output_data.flatten().tolist()
                    shape = list(output_data.shape)
                    
                    return neurenix_api_pb2.PredictResponse(
                        result=result,
                        shape=shape
                    )
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(str(e))
                    return neurenix_api_pb2.PredictResponse()
        
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        neurenix_api_pb2_grpc.add_NeurenixServiceServicer_to_server(
            NeurenixServicer(self), self.server
        )
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop(0)


class APIManager:
    """
    Manager for API servers in the Neurenix framework.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern for APIManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(APIManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the API manager."""
        if self._initialized:
            return
            
        self._initialized = True
        self.servers = {}
        
        logger.info("API manager initialized")
    
    def create_server(self, server_type: str, host: str = "0.0.0.0", port: int = None) -> APIServer:
        """
        Create a new API server.
        
        Args:
            server_type: Type of server to create ("rest", "websocket", or "grpc").
            host: Host to bind the server to.
            port: Port to bind the server to. If None, uses the default port for the server type.
            
        Returns:
            The created server.
        """
        if port is None:
            if server_type == "rest":
                port = 8000
            elif server_type == "websocket":
                port = 8001
            elif server_type == "grpc":
                port = 8002
            else:
                raise ValueError(f"Unsupported server type: {server_type}")
        
        if server_type == "rest":
            server = RESTfulServer(host, port)
        elif server_type == "websocket":
            server = WebSocketServer(host, port)
        elif server_type == "grpc":
            server = GRPCServer(host, port)
        else:
            raise ValueError(f"Unsupported server type: {server_type}")
        
        server_id = f"{server_type}_{host}_{port}"
        self.servers[server_id] = server
        
        logger.info(f"Created {server_type} server on {host}:{port}")
        return server
    
    def get_server(self, server_type: str, host: str = "0.0.0.0", port: int = None) -> Optional[APIServer]:
        """
        Get an existing API server.
        
        Args:
            server_type: Type of server to get ("rest", "websocket", or "grpc").
            host: Host of the server.
            port: Port of the server. If None, uses the default port for the server type.
            
        Returns:
            The server if it exists, None otherwise.
        """
        if port is None:
            if server_type == "rest":
                port = 8000
            elif server_type == "websocket":
                port = 8001
            elif server_type == "grpc":
                port = 8002
            else:
                raise ValueError(f"Unsupported server type: {server_type}")
        
        server_id = f"{server_type}_{host}_{port}"
        return self.servers.get(server_id)
    
    def start_server(self, server_type: str, host: str = "0.0.0.0", port: int = None) -> APIServer:
        """
        Start an API server.
        
        Args:
            server_type: Type of server to start ("rest", "websocket", or "grpc").
            host: Host of the server.
            port: Port of the server. If None, uses the default port for the server type.
            
        Returns:
            The started server.
        """
        server = self.get_server(server_type, host, port)
        if server is None:
            server = self.create_server(server_type, host, port)
        
        server.start()
        
        return server
    
    def stop_server(self, server_type: str, host: str = "0.0.0.0", port: int = None) -> None:
        """
        Stop an API server.
        
        Args:
            server_type: Type of server to stop ("rest", "websocket", or "grpc").
            host: Host of the server.
            port: Port of the server. If None, uses the default port for the server type.
        """
        server = self.get_server(server_type, host, port)
        if server is None:
            logger.warning(f"Server {server_type} on {host}:{port} not found")
            return
        
        server.stop()
    
    def stop_all_servers(self) -> None:
        """Stop all API servers."""
        for server in self.servers.values():
            server.stop()
        
        logger.info("All API servers stopped")



def create_rest_server(host: str = "0.0.0.0", port: int = 8000) -> RESTfulServer:
    """
    Create a RESTful API server.
    
    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
        
    Returns:
        The created server.
    """
    manager = APIManager()
    return manager.create_server("rest", host, port)


def create_websocket_server(host: str = "0.0.0.0", port: int = 8001) -> WebSocketServer:
    """
    Create a WebSocket API server.
    
    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
        
    Returns:
        The created server.
    """
    manager = APIManager()
    return manager.create_server("websocket", host, port)


def create_grpc_server(host: str = "0.0.0.0", port: int = 8002) -> GRPCServer:
    """
    Create a gRPC API server.
    
    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
        
    Returns:
        The created server.
    """
    manager = APIManager()
    return manager.create_server("grpc", host, port)


def serve_model(model: Module, name: str, server_types: List[str] = ["rest"],
               host: str = "0.0.0.0", ports: Dict[str, int] = None) -> Dict[str, APIServer]:
    """
    Serve a model using one or more API servers.
    
    Args:
        model: The model to serve.
        name: Name of the model.
        server_types: Types of servers to use.
        host: Host to bind the servers to.
        ports: Dictionary mapping server types to ports.
            If None, uses default ports.
            
    Returns:
        Dictionary mapping server types to servers.
    """
    manager = APIManager()
    servers = {}
    
    for server_type in server_types:
        port = None
        if ports and server_type in ports:
            port = ports[server_type]
        
        server = manager.start_server(server_type, host, port)
        server.add_model(name, model)
        servers[server_type] = server
    
    return servers
