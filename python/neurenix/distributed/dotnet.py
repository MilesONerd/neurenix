"""
.NET integration for distributed computing in Neurenix.

This module provides integration with the .NET distributed computing implementation
using ASP.NET for RESTful APIs and Orleans for distributed computing functionality.
"""

import os
import subprocess
import requests
import json
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin

from neurenix.device import Device, DeviceType
from neurenix.tensor import Tensor
from neurenix.distributed.distributed import DistributedContext


class DotNetDistributedClient:
    """
    Client for interacting with the .NET distributed computing services.
    
    This class provides a Python interface to the ASP.NET RESTful APIs and
    Orleans-based distributed computing functionality.
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:5000",
        silo_host_path: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the .NET distributed client.
        
        Args:
            api_base_url: Base URL for the ASP.NET API
            silo_host_path: Path to the Orleans silo host executable
            timeout: Timeout for API requests in seconds
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.silo_host_path = silo_host_path
        self.timeout = timeout
        self._silo_process = None
        
        if self.silo_host_path is None:
            self.silo_host_path = os.path.join(
                os.path.dirname(__file__),
                "../../src/phynexus/dotnet/Neurenix.Distributed.Orleans/SiloHost/bin/Debug/net8.0/Neurenix.Distributed.Orleans.SiloHost"
            )
    
    def start_silo(self) -> bool:
        """
        Start the Orleans silo host.
        
        Returns:
            True if the silo started successfully, False otherwise
        """
        if self._silo_process is not None:
            return True
        
        try:
            if self.silo_host_path is None:
                raise ValueError("Silo host path is not set")
            
            self._silo_process = subprocess.Popen(
                self.silo_host_path,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            import time
            time.sleep(5)
            
            if self._silo_process.poll() is None:
                return True
            else:
                stdout, stderr = self._silo_process.communicate()
                print(f"Silo failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                return False
        except Exception as e:
            print(f"Error starting silo: {e}")
            return False
    
    def stop_silo(self):
        """Stop the Orleans silo host."""
        if self._silo_process is not None:
            self._silo_process.terminate()
            self._silo_process.wait()
            self._silo_process = None
    
    def create_tensor(self, tensor_id: str, data: List[float], shape: List[int]) -> bool:
        """
        Create a tensor in the distributed system.
        
        Args:
            tensor_id: Unique identifier for the tensor
            data: Tensor data as a flat list
            shape: Shape of the tensor
            
        Returns:
            True if the tensor was created successfully, False otherwise
        """
        url = urljoin(self.api_base_url, f"/api/tensor/{tensor_id}")
        payload = {
            "data": data,
            "shape": shape
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error creating tensor: {e}")
            return False
    
    def get_tensor(self, tensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tensor from the distributed system.
        
        Args:
            tensor_id: Unique identifier for the tensor
            
        Returns:
            Dictionary containing tensor data and shape, or None if not found
        """
        url = urljoin(self.api_base_url, f"/api/tensor/{tensor_id}")
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting tensor: {e}")
            return None
    
    def add_tensors(self, tensor_id: str, other_data: List[float]) -> Optional[List[float]]:
        """
        Add two tensors element-wise.
        
        Args:
            tensor_id: ID of the first tensor
            other_data: Data of the second tensor
            
        Returns:
            Result of the addition, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/tensor/{tensor_id}/add")
        
        try:
            response = requests.post(url, json=other_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error adding tensors: {e}")
            return None
    
    def multiply_tensors(self, tensor_id: str, other_data: List[float]) -> Optional[List[float]]:
        """
        Multiply two tensors element-wise.
        
        Args:
            tensor_id: ID of the first tensor
            other_data: Data of the second tensor
            
        Returns:
            Result of the multiplication, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/tensor/{tensor_id}/multiply")
        
        try:
            response = requests.post(url, json=other_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error multiplying tensors: {e}")
            return None
    
    def matmul_tensors(self, tensor_id: str, other_data: List[float], other_shape: List[int]) -> Optional[List[float]]:
        """
        Perform matrix multiplication of two tensors.
        
        Args:
            tensor_id: ID of the first tensor
            other_data: Data of the second tensor
            other_shape: Shape of the second tensor
            
        Returns:
            Result of the matrix multiplication, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/tensor/{tensor_id}/matmul")
        payload = {
            "otherData": other_data,
            "otherShape": other_shape
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error performing matrix multiplication: {e}")
            return None
    
    def forward(self, compute_id: str, model_id: str, input_data: List[float]) -> Optional[List[float]]:
        """
        Perform a forward pass through a model.
        
        Args:
            compute_id: ID of the compute grain
            model_id: ID of the model
            input_data: Input data for the forward pass
            
        Returns:
            Output of the forward pass, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/compute/{compute_id}/forward")
        payload = {
            "modelId": model_id,
            "input": input_data
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error performing forward pass: {e}")
            return None
    
    def backward(self, compute_id: str, model_id: str, gradients: List[float]) -> Optional[List[float]]:
        """
        Perform a backward pass through a model.
        
        Args:
            compute_id: ID of the compute grain
            model_id: ID of the model
            gradients: Gradients for the backward pass
            
        Returns:
            Gradients from the backward pass, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/compute/{compute_id}/backward")
        payload = {
            "modelId": model_id,
            "gradients": gradients
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error performing backward pass: {e}")
            return None
    
    def update_model(self, compute_id: str, model_id: str, parameters: List[float]) -> bool:
        """
        Update model parameters.
        
        Args:
            compute_id: ID of the compute grain
            model_id: ID of the model
            parameters: New model parameters
            
        Returns:
            True if the update was successful, False otherwise
        """
        url = urljoin(self.api_base_url, f"/api/compute/{compute_id}/update")
        payload = {
            "modelId": model_id,
            "parameters": parameters
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error updating model: {e}")
            return False
    
    def get_model_parameters(self, compute_id: str, model_id: str) -> Optional[List[float]]:
        """
        Get model parameters.
        
        Args:
            compute_id: ID of the compute grain
            model_id: ID of the model
            
        Returns:
            Model parameters, or None if failed
        """
        url = urljoin(self.api_base_url, f"/api/compute/{compute_id}/parameters/{model_id}")
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting model parameters: {e}")
            return None


class DotNetDistributedContext(DistributedContext):
    """
    Distributed context that uses the .NET implementation.
    
    This class extends the base DistributedContext to use the .NET
    distributed computing services.
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:5000",
        silo_host_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the .NET distributed context.
        
        Args:
            api_base_url: Base URL for the ASP.NET API
            silo_host_path: Path to the Orleans silo host executable
            **kwargs: Additional arguments for the base DistributedContext
        """
        super().__init__(**kwargs)
        self.client = DotNetDistributedClient(api_base_url, silo_host_path)
    
    def initialize(self):
        """Initialize the distributed context."""
        if not self.client.start_silo():
            raise RuntimeError("Failed to start Orleans silo")
        
        super().initialize()
    
    def shutdown(self):
        """Shut down the distributed context."""
        self.client.stop_silo()
        
        super().shutdown()
    
    def create_distributed_tensor(self, tensor: Tensor, tensor_id: str) -> bool:
        """
        Create a distributed tensor from a local tensor.
        
        Args:
            tensor: Local tensor to distribute
            tensor_id: Unique identifier for the distributed tensor
            
        Returns:
            True if the tensor was created successfully, False otherwise
        """
        import numpy as np
        from typing import Any, List, cast
        
        numpy_data = tensor.numpy()
        
        if isinstance(numpy_data, np.ndarray):
            data = numpy_data.astype(np.float32).flatten().tolist()
        else:
            try:
                if numpy_data is None:
                    data = [0.0]
                elif hasattr(numpy_data, '__len__') and not isinstance(numpy_data, str):
                    data = []
                    iterable_data = cast(List[Any], numpy_data)
                    for x in iterable_data:
                        data.append(float(x))
                else:
                    data = [float(numpy_data)]
            except Exception as e:
                print(f"Error converting tensor data: {e}")
                data = [0.0]
                
        shape = list(tensor.shape)
        return self.client.create_tensor(tensor_id, data, shape)
    
    def get_distributed_tensor(self, tensor_id: str) -> Optional[Tensor]:
        """
        Get a distributed tensor as a local tensor.
        
        Args:
            tensor_id: Unique identifier for the distributed tensor
            
        Returns:
            Local tensor, or None if not found
        """
        result = self.client.get_tensor(tensor_id)
        if result is None:
            return None
        
        data = result["data"]
        shape = result["shape"]
        return Tensor(data, shape=shape)
