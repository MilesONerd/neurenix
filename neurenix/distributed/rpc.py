"""
Remote Procedure Call (RPC) module for distributed training.

This module provides RPC functionality for distributed training,
enabling communication between processes.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from neurenix.distributed.distributed import get_rank, get_world_size


class RpcContext:
    """
    RPC context for distributed training.
    
    This class manages the RPC context for distributed training,
    enabling communication between processes.
    """
    
    def __init__(
        self,
        backend: str = "tensorpipe",
        init_method: Optional[str] = None,
        world_size: int = -1,
        rank: int = -1,
        timeout: float = 1800.0,
    ):
        """
        Initialize RPC context.
        
        Args:
            backend: RPC backend ('tensorpipe' or 'gloo')
            init_method: URL specifying how to initialize the RPC
            world_size: Number of processes in the group
            rank: Rank of the current process
            timeout: Timeout for operations
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.timeout = timeout
        self._initialized = False
        
        # Function registry
        self._functions: Dict[str, Callable] = {}
        
        # Request queue
        self._request_queue: List[Dict[str, Any]] = []
        
        # Response registry
        self._responses: Dict[str, Any] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def __enter__(self):
        """Initialize the RPC context."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the RPC context."""
        self.shutdown()
    
    def initialize(self):
        """Initialize the RPC context."""
        if self._initialized:
            return
        
        # Initialize RPC (placeholder for actual implementation)
        print(f"Initializing RPC: rank={self.rank}, world_size={self.world_size}, backend={self.backend}")
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        
        # Mark as initialized
        self._initialized = True
    
    def shutdown(self):
        """Shut down the RPC context."""
        if not self._initialized:
            return
        
        # Clean up RPC (placeholder for actual implementation)
        print("Shutting down RPC")
        
        # Mark as not initialized
        self._initialized = False
    
    def register_function(self, name: str, func: Callable):
        """
        Register a function for RPC.
        
        Args:
            name: Function name
            func: Function to register
        """
        with self._lock:
            self._functions[name] = func
    
    def _worker(self):
        """Worker thread for processing RPC requests."""
        while self._initialized:
            # Process requests
            with self._lock:
                if self._request_queue:
                    request = self._request_queue.pop(0)
                    self._process_request(request)
            
            # Sleep to avoid busy waiting
            time.sleep(0.01)
    
    def _process_request(self, request: Dict[str, Any]):
        """
        Process an RPC request.
        
        Args:
            request: Request to process
        """
        # Get request details
        request_id = request["id"]
        function_name = request["function"]
        args = request["args"]
        kwargs = request["kwargs"]
        
        # Get function
        if function_name not in self._functions:
            # Function not found
            response = {
                "status": "error",
                "error": f"Function '{function_name}' not found",
            }
        else:
            # Call function
            try:
                result = self._functions[function_name](*args, **kwargs)
                response = {
                    "status": "success",
                    "result": result,
                }
            except Exception as e:
                # Function call failed
                response = {
                    "status": "error",
                    "error": str(e),
                }
        
        # Store response
        self._responses[request_id] = response
    
    def _send_request(
        self,
        dst_rank: int,
        function_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        sync: bool = True,
    ) -> Any:
        """
        Send an RPC request.
        
        Args:
            dst_rank: Destination rank
            function_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
            sync: Whether to wait for the response
            
        Returns:
            Response if sync=True, None otherwise
        """
        # Generate request ID
        request_id = f"{self.rank}_{dst_rank}_{function_name}_{time.time()}"
        
        # Create request
        request = {
            "id": request_id,
            "src_rank": self.rank,
            "dst_rank": dst_rank,
            "function": function_name,
            "args": args,
            "kwargs": kwargs,
        }
        
        # Send request (placeholder for actual implementation)
        # In a real implementation, this would send the request to the destination rank
        # For now, we'll simulate it by adding the request to our own queue
        with self._lock:
            self._request_queue.append(request)
        
        if not sync:
            # Asynchronous call
            return None
        
        # Synchronous call, wait for response
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # Check if response is available
            with self._lock:
                if request_id in self._responses:
                    # Get response
                    response = self._responses.pop(request_id)
                    
                    # Check status
                    if response["status"] == "success":
                        return response["result"]
                    else:
                        raise RuntimeError(f"RPC call failed: {response['error']}")
            
            # Sleep to avoid busy waiting
            time.sleep(0.01)
        
        # Timeout
        raise TimeoutError(f"RPC call timed out after {self.timeout} seconds")


# Global RPC context
_GLOBAL_RPC_CONTEXT: Optional[RpcContext] = None


def init_rpc(
    backend: str = "tensorpipe",
    init_method: Optional[str] = None,
    world_size: int = -1,
    rank: int = -1,
) -> RpcContext:
    """
    Initialize RPC.
    
    Args:
        backend: RPC backend ('tensorpipe' or 'gloo')
        init_method: URL specifying how to initialize the RPC
        world_size: Number of processes in the group
        rank: Rank of the current process
        
    Returns:
        RPC context
    """
    global _GLOBAL_RPC_CONTEXT
    
    if _GLOBAL_RPC_CONTEXT is not None and _GLOBAL_RPC_CONTEXT._initialized:
        return _GLOBAL_RPC_CONTEXT
    
    _GLOBAL_RPC_CONTEXT = RpcContext(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    _GLOBAL_RPC_CONTEXT.initialize()
    
    return _GLOBAL_RPC_CONTEXT


def rpc_sync(
    dst_rank: int,
    function_name: str,
    *args,
    **kwargs,
) -> Any:
    """
    Synchronous RPC call.
    
    Args:
        dst_rank: Destination rank
        function_name: Function name
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    global _GLOBAL_RPC_CONTEXT
    
    if _GLOBAL_RPC_CONTEXT is None or not _GLOBAL_RPC_CONTEXT._initialized:
        raise RuntimeError("RPC not initialized")
    
    return _GLOBAL_RPC_CONTEXT._send_request(
        dst_rank=dst_rank,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
        sync=True,
    )


def rpc_async(
    dst_rank: int,
    function_name: str,
    *args,
    **kwargs,
) -> None:
    """
    Asynchronous RPC call.
    
    Args:
        dst_rank: Destination rank
        function_name: Function name
        *args: Function arguments
        **kwargs: Function keyword arguments
    """
    global _GLOBAL_RPC_CONTEXT
    
    if _GLOBAL_RPC_CONTEXT is None or not _GLOBAL_RPC_CONTEXT._initialized:
        raise RuntimeError("RPC not initialized")
    
    _GLOBAL_RPC_CONTEXT._send_request(
        dst_rank=dst_rank,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
        sync=False,
    )
