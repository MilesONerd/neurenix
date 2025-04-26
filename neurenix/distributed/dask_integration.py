"""
Dask integration for distributed computing in Neurenix.

This module provides integration with Dask for distributed computing,
enabling scalable data processing and model training.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import dask
    import dask.array as da
    import dask.distributed as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


class DaskCluster:
    """
    Dask cluster for distributed computing.
    
    This class manages a Dask cluster for distributed computing,
    enabling scalable data processing and model training.
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB",
        scheduler_address: Optional[str] = None,
    ):
        """
        Initialize Dask cluster.
        
        Args:
            n_workers: Number of workers
            threads_per_worker: Number of threads per worker
            memory_limit: Memory limit per worker
            scheduler_address: Address of existing scheduler to connect to
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not available. Please install it with 'pip install dask[distributed]'."
            )
        
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.scheduler_address = scheduler_address
        self.client = None
        self.cluster = None
    
    def __enter__(self):
        """Initialize the Dask cluster."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the Dask cluster."""
        self.stop()
    
    def start(self):
        """Start the Dask cluster."""
        if self.client is not None:
            return
        
        if self.scheduler_address is not None:
            # Connect to existing scheduler
            self.client = dd.Client(self.scheduler_address)
        else:
            # Create local cluster
            self.cluster = dd.LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
            )
            self.client = dd.Client(self.cluster)
        
        print(f"Dask dashboard available at: {self.client.dashboard_link}")
    
    def stop(self):
        """Stop the Dask cluster."""
        if self.client is None:
            return
        
        self.client.close()
        
        if self.cluster is not None:
            self.cluster.close()
        
        self.client = None
        self.cluster = None
    
    @property
    def is_running(self) -> bool:
        """Check if the Dask cluster is running."""
        return self.client is not None
    
    def map(self, func: Callable, *iterables, **kwargs) -> List[Any]:
        """
        Map a function to iterables in parallel.
        
        Args:
            func: Function to map
            *iterables: Iterables to map over
            **kwargs: Keyword arguments for map
            
        Returns:
            List of results
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.gather(self.client.map(func, *iterables, **kwargs))
    
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Submit a function for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.submit(func, *args, **kwargs)
    
    def gather(self, futures: Union[Any, List[Any]]) -> Any:
        """
        Gather results from futures.
        
        Args:
            futures: Future or list of futures
            
        Returns:
            Result or list of results
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.gather(futures)
    
    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """
        Scatter data to workers.
        
        Args:
            data: Data to scatter
            broadcast: Whether to broadcast data to all workers
            
        Returns:
            Future object
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.scatter(data, broadcast=broadcast)
    
    def replicate(self, future: Any) -> Any:
        """
        Replicate data to all workers.
        
        Args:
            future: Future object
            
        Returns:
            Future object
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.replicate(future)
    
    def get_worker_info(self) -> Dict[str, Any]:
        """
        Get worker information.
        
        Returns:
            Dictionary of worker information
        """
        if self.client is None:
            raise RuntimeError("Dask cluster is not running")
        
        return self.client.scheduler_info()


def tensor_to_dask(tensor, chunks="auto"):
    """
    Convert a Neurenix tensor to a Dask array.
    
    Args:
        tensor: Neurenix tensor
        chunks: Chunk size for Dask array
        
    Returns:
        Dask array
    """
    if not DASK_AVAILABLE:
        raise ImportError(
            "Dask is not available. Please install it with 'pip install dask[distributed]'."
        )
    
    # Convert to numpy array
    array = tensor.numpy()
    
    # Convert to Dask array
    return da.from_array(array, chunks=chunks)


def dask_to_tensor(dask_array):
    """
    Convert a Dask array to a Neurenix tensor.
    
    Args:
        dask_array: Dask array
        
    Returns:
        Neurenix tensor
    """
    if not DASK_AVAILABLE:
        raise ImportError(
            "Dask is not available. Please install it with 'pip install dask[distributed]'."
        )
    
    from neurenix.tensor import Tensor
    
    # Compute Dask array
    array = dask_array.compute()
    
    # Convert to Neurenix tensor
    return Tensor(array)
