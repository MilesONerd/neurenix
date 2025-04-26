"""
Horovod support for distributed training in Neurenix.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..core import get_config
from ..device import get_device
from ..binding import get_binding

logger = logging.getLogger(__name__)

class HorovodManager:
    """
    Manager for Horovod-based distributed training.
    
    This class provides an interface to Horovod functionality for distributed training
    in Neurenix. It uses the Phynexus Rust/C++ backend for actual Horovod operations.
    """
    
    def __init__(self, 
                 backend: str = "auto",
                 init_method: str = "env",
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 local_rank: Optional[int] = None,
                 timeout: float = 1800.0):
        """
        Initialize the Horovod manager.
        
        Args:
            backend: Communication backend to use ('mpi', 'gloo', 'nccl', 'auto')
            init_method: Initialization method ('env', 'tcp', 'file')
            world_size: Total number of processes (auto-detected if None)
            rank: Global rank of this process (auto-detected if None)
            local_rank: Local rank of this process (auto-detected if None)
            timeout: Timeout for operations in seconds
        """
        self.backend = backend
        self.init_method = init_method
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self.timeout = timeout
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_HOROVOD_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the Horovod environment.
        """
        if self._initialized:
            logger.warning("Horovod environment already initialized")
            return
        
        try:
            result = self._binding.horovod_initialize(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self._world_size,
                rank=self._rank,
                local_rank=self._local_rank,
                timeout=self.timeout
            )
            
            self._world_size = result["world_size"]
            self._rank = result["rank"]
            self._local_rank = result["local_rank"]
            self._initialized = True
            
            logger.info(f"Horovod initialized: rank {self._rank}/{self._world_size-1}, local_rank {self._local_rank}")
            
            if get_config().get("auto_device_selection", True):
                device = get_device()
                if device.type == "cpu":
                    try:
                        device = get_device(f"cuda:{self._local_rank}")
                        logger.info(f"Auto-selected device: {device}")
                    except Exception as e:
                        logger.warning(f"Failed to auto-select GPU device: {e}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Horovod: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the Horovod environment.
        """
        if not self._initialized:
            logger.warning("Horovod environment not initialized")
            return
        
        try:
            self._binding.horovod_finalize()
            self._initialized = False
            logger.info("Horovod finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize Horovod: {e}")
            raise
    
    @property
    def world_size(self) -> int:
        """Get the total number of processes."""
        if not self._initialized:
            self.initialize()
        return self._world_size
    
    @property
    def rank(self) -> int:
        """Get the global rank of this process."""
        if not self._initialized:
            self.initialize()
        return self._rank
    
    @property
    def local_rank(self) -> int:
        """Get the local rank of this process."""
        if not self._initialized:
            self.initialize()
        return self._local_rank
    
    @property
    def is_master(self) -> bool:
        """Check if this process is the master (rank 0)."""
        return self.rank == 0
    
    def barrier(self) -> None:
        """
        Synchronize all processes.
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.horovod_barrier()
        except Exception as e:
            logger.error(f"Failed to synchronize at barrier: {e}")
            raise
    
    def broadcast(self, data: Any, src: int = 0) -> Any:
        """
        Broadcast data from the source rank to all processes.
        
        Args:
            data: Data to broadcast
            src: Source rank
            
        Returns:
            Broadcasted data
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.horovod_broadcast(data, src)
        except Exception as e:
            logger.error(f"Failed to broadcast data: {e}")
            raise
    
    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        """
        Perform an all-reduce operation.
        
        Args:
            data: Data to reduce
            op: Reduction operation ('sum', 'prod', 'min', 'max', 'avg')
            
        Returns:
            Reduced data
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.horovod_all_reduce(data, op)
        except Exception as e:
            logger.error(f"Failed to perform all-reduce: {e}")
            raise
    
    def all_gather(self, data: Any) -> List[Any]:
        """
        Gather data from all processes.
        
        Args:
            data: Data to gather
            
        Returns:
            List of gathered data from all processes
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.horovod_all_gather(data)
        except Exception as e:
            logger.error(f"Failed to perform all-gather: {e}")
            raise
    
    def broadcast_parameters(self, model: Any, root_rank: int = 0) -> None:
        """
        Broadcast model parameters from root rank to all other processes.
        
        Args:
            model: Model to broadcast parameters from
            root_rank: Source rank
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.horovod_broadcast_parameters(model, root_rank)
        except Exception as e:
            logger.error(f"Failed to broadcast parameters: {e}")
            raise
    
    def distributed_optimizer(self, optimizer: Any, named_parameters: Any = None,
                             compression: Optional[str] = None,
                             backward_passes_per_step: int = 1,
                             op: str = "sum") -> Any:
        """
        Wrap optimizer with Horovod distributed optimizer.
        
        Args:
            optimizer: Optimizer to wrap
            named_parameters: Named parameters (optional)
            compression: Compression algorithm ('fp16', 'none')
            backward_passes_per_step: Number of backward passes before parameter update
            op: Reduction operation ('sum', 'avg')
            
        Returns:
            Wrapped optimizer
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.horovod_distributed_optimizer(
                optimizer, 
                named_parameters,
                compression,
                backward_passes_per_step,
                op
            )
        except Exception as e:
            logger.error(f"Failed to create distributed optimizer: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


_horovod_manager = None

def get_horovod_manager() -> HorovodManager:
    """
    Get the global Horovod manager instance.
    
    Returns:
        HorovodManager instance
    """
    global _horovod_manager
    if _horovod_manager is None:
        _horovod_manager = HorovodManager()
    return _horovod_manager
