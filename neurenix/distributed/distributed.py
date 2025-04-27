"""
Distributed computing core functionality for Neurenix.

This module provides the core functionality for distributed training and inference
across multiple GPUs and compute nodes.
"""

import datetime
import os
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from neurenix.device import Device, get_device
from neurenix.tensor import Tensor


class DistributedContext:
    """
    Context manager for distributed training.
    
    This class manages the distributed training context, including rank, world size,
    and communication between processes.
    """
    _master_socket = None
    _worker_sockets = {}
    _socket_lock = threading.Lock()
    _socket_port = 23457  # Different from init_method port
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: int = -1,
        rank: int = -1,
        device_id: Optional[int] = None,
        timeout: float = 1800.0,
    ):
        """
        Initialize distributed context.
        
        Args:
            backend: Communication backend ('nccl', 'gloo', or 'mpi')
            init_method: URL specifying how to initialize the process group
            world_size: Number of processes in the group
            rank: Rank of the current process
            device_id: Device ID for the current process
            timeout: Timeout for operations
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.device_id = device_id
        self.timeout = timeout
        self._initialized = False
        
        # Auto-detect world size and rank if not provided
        if world_size == -1 or rank == -1:
            if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.rank = int(os.environ["RANK"])
            else:
                # Single process mode
                self.world_size = 1
                self.rank = 0
        
        # Auto-detect device ID if not provided
        if device_id is None and "LOCAL_RANK" in os.environ:
            self.device_id = int(os.environ["LOCAL_RANK"])
        elif device_id is None:
            self.device_id = self.rank % Device.device_count()
    
    def __enter__(self):
        """Initialize the distributed context."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the distributed context."""
        self.shutdown()
    
    def initialize(self):
        """Initialize the distributed context."""
        if self._initialized:
            return
        
        # Set device
        if self.device_id is not None:
            device = get_device(f"cuda:{self.device_id}")
            from neurenix.device_manager import DeviceManager
            device_manager = DeviceManager()
            device_manager.active_device = device
        
        # Initialize process group
        if self.init_method is None:
            # Default initialization method
            if self.backend == "nccl":
                # Use shared file system for NCCL
                self.init_method = f"file:///tmp/neurenix_dist_init_{int(time.time())}"
            else:
                # Use TCP for other backends
                hostname = socket.gethostname()
                self.init_method = f"tcp://{hostname}:23456"
        
        # Initialize process group
        try:
            import torch
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=torch.distributed.default_pg_timeout if self.timeout is None else datetime.timedelta(seconds=self.timeout)
                )
                print(f"Initialized torch.distributed process group: rank={self.rank}, "
                      f"world_size={self.world_size}, backend={self.backend}")
        except (ImportError, AttributeError):
            # Fallback to custom implementation
            print(f"Initializing custom distributed process group: rank={self.rank}, "
                  f"world_size={self.world_size}, backend={self.backend}")
            # Set up socket-based communication for simple operations
            self._setup_socket_communication()
        
        # Mark as initialized
        self._initialized = True
    
    def _setup_socket_communication(self):
        """Set up socket-based communication for distributed operations."""
        if self.rank == 0:  # Master process
            self._master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._master_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            hostname = socket.gethostname()
            self._master_socket.bind((hostname, self._socket_port))
            self._master_socket.listen(self.world_size - 1)
            
            # Accept connections from worker processes
            for _ in range(self.world_size - 1):
                client_socket, client_address = self._master_socket.accept()
                worker_rank = int(client_socket.recv(1024).decode())
                self._worker_sockets[worker_rank] = client_socket
                print(f"Master accepted connection from worker {worker_rank}")
        else:  # Worker process
            if self.init_method and "//" in self.init_method:
                master_hostname = self.init_method.split("//")[1].split(":")[0]
            else:
                # Default to localhost if init_method is not available
                master_hostname = "localhost"
            
            worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            worker_socket.connect((master_hostname, self._socket_port))
            worker_socket.send(str(self.rank).encode())
            self._worker_sockets[0] = worker_socket
            print(f"Worker {self.rank} connected to master")
    
    def shutdown(self):
        """Shut down the distributed context."""
        if not self._initialized:
            return
        
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                print("Shut down torch.distributed process group")
        except (ImportError, AttributeError):
            # Clean up socket-based communication
            with self._socket_lock:
                if self.rank == 0 and self._master_socket:
                    for worker_socket in self._worker_sockets.values():
                        worker_socket.close()
                    self._master_socket.close()
                    self._master_socket = None
                    self._worker_sockets = {}
                elif self.rank != 0:
                    for socket_conn in self._worker_sockets.values():
                        socket_conn.close()
                    self._worker_sockets = {}
            print(f"Process {self.rank} shut down socket communication")
        
        # Mark as not initialized
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the distributed context is initialized."""
        return self._initialized
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0


# Global distributed context
_GLOBAL_CONTEXT: Optional[DistributedContext] = None


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: int = -1,
    rank: int = -1,
    device_id: Optional[int] = None,
) -> DistributedContext:
    """
    Initialize distributed training.
    
    Args:
        backend: Communication backend ('nccl', 'gloo', or 'mpi')
        init_method: URL specifying how to initialize the process group
        world_size: Number of processes in the group
        rank: Rank of the current process
        device_id: Device ID for the current process
        
    Returns:
        Distributed context
    """
    global _GLOBAL_CONTEXT
    
    if _GLOBAL_CONTEXT is not None and _GLOBAL_CONTEXT.is_initialized:
        return _GLOBAL_CONTEXT
    
    _GLOBAL_CONTEXT = DistributedContext(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        device_id=device_id,
    )
    _GLOBAL_CONTEXT.initialize()
    
    return _GLOBAL_CONTEXT


def get_rank() -> int:
    """
    Get the rank of the current process.
    
    Returns:
        Rank of the current process
    """
    global _GLOBAL_CONTEXT
    
    if _GLOBAL_CONTEXT is None or not _GLOBAL_CONTEXT.is_initialized:
        return 0
    
    return _GLOBAL_CONTEXT.rank


def get_world_size() -> int:
    """
    Get the world size (number of processes).
    
    Returns:
        Number of processes
    """
    global _GLOBAL_CONTEXT
    
    if _GLOBAL_CONTEXT is None or not _GLOBAL_CONTEXT.is_initialized:
        return 1
    
    return _GLOBAL_CONTEXT.world_size


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).
    
    Returns:
        True if this is the main process, False otherwise
    """
    return get_rank() == 0


def barrier():
    """
    Synchronize all processes.
    
    This function blocks until all processes reach this barrier.
    """
    global _GLOBAL_CONTEXT
    
    if _GLOBAL_CONTEXT is None or not _GLOBAL_CONTEXT.is_initialized:
        return
    
    try:
        # Try to use torch.distributed if available
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            return
    except (ImportError, AttributeError):
        pass
    
    # Fallback to socket-based synchronization
    rank = get_rank()
    world_size = get_world_size()
    
    if world_size <= 1:
        return
    
    with _GLOBAL_CONTEXT._socket_lock:
        if rank == 0:  # Master process
            barrier_counts = {r: False for r in range(1, world_size)}
            
            for worker_rank in range(1, world_size):
                if worker_rank in _GLOBAL_CONTEXT._worker_sockets:
                    try:
                        _GLOBAL_CONTEXT._worker_sockets[worker_rank].recv(1024)
                        barrier_counts[worker_rank] = True
                    except (socket.error, ConnectionError) as e:
                        print(f"Error receiving barrier signal from worker {worker_rank}: {e}")
            
            for worker_rank in range(1, world_size):
                if worker_rank in _GLOBAL_CONTEXT._worker_sockets and barrier_counts[worker_rank]:
                    try:
                        _GLOBAL_CONTEXT._worker_sockets[worker_rank].send(b"barrier_complete")
                    except (socket.error, ConnectionError) as e:
                        print(f"Error sending barrier completion to worker {worker_rank}: {e}")
        else:  # Worker process
            if 0 in _GLOBAL_CONTEXT._worker_sockets:
                try:
                    _GLOBAL_CONTEXT._worker_sockets[0].send(b"barrier_reached")
                    _GLOBAL_CONTEXT._worker_sockets[0].recv(1024)
                except (socket.error, ConnectionError) as e:
                    print(f"Error in barrier for worker {rank}: {e}")
    
    print(f"Process {rank} completed barrier")
