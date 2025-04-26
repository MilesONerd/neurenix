"""
Distributed computing core functionality for Neurenix.

This module provides the core functionality for distributed training and inference
across multiple GPUs and compute nodes.
"""

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
            device.set_as_default()
        
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
        
        # Initialize process group (placeholder for actual implementation)
        print(f"Initializing distributed process group: rank={self.rank}, "
              f"world_size={self.world_size}, backend={self.backend}")
        
        # Mark as initialized
        self._initialized = True
    
    def shutdown(self):
        """Shut down the distributed context."""
        if not self._initialized:
            return
        
        # Clean up process group (placeholder for actual implementation)
        print("Shutting down distributed process group")
        
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
    # Placeholder for actual implementation
    print(f"Process {get_rank()} reached barrier")
    time.sleep(0.1)  # Simulate synchronization
