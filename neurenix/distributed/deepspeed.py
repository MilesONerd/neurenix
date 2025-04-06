"""
DeepSpeed support for distributed training in Neurenix.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..core import get_config
from ..device import get_device
from ..binding import get_binding

logger = logging.getLogger(__name__)

class DeepSpeedManager:
    """
    Manager for DeepSpeed-based distributed training.
    
    This class provides an interface to DeepSpeed functionality for distributed training
    in Neurenix. It uses the Phynexus Rust/C++ backend for actual DeepSpeed operations.
    """
    
    def __init__(self, 
                 backend: str = "nccl",
                 init_method: str = "env",
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 local_rank: Optional[int] = None,
                 timeout: float = 1800.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DeepSpeed manager.
        
        Args:
            backend: Communication backend to use ('nccl', 'gloo', 'auto')
            init_method: Initialization method ('env', 'tcp', 'file')
            world_size: Total number of processes (auto-detected if None)
            rank: Global rank of this process (auto-detected if None)
            local_rank: Local rank of this process (auto-detected if None)
            timeout: Timeout for operations in seconds
            config: DeepSpeed configuration dictionary
        """
        self.backend = backend
        self.init_method = init_method
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self.timeout = timeout
        self.config = config or {}
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_DEEPSPEED_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the DeepSpeed environment.
        """
        if self._initialized:
            logger.warning("DeepSpeed environment already initialized")
            return
        
        try:
            result = self._binding.deepspeed_initialize(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self._world_size,
                rank=self._rank,
                local_rank=self._local_rank,
                timeout=self.timeout,
                config=self.config
            )
            
            self._world_size = result["world_size"]
            self._rank = result["rank"]
            self._local_rank = result["local_rank"]
            self._initialized = True
            
            logger.info(f"DeepSpeed initialized: rank {self._rank}/{self._world_size-1}, local_rank {self._local_rank}")
            
            if get_config().get("auto_device_selection", True):
                device = get_device()
                if device.type == "cpu":
                    try:
                        device = get_device(f"cuda:{self._local_rank}")
                        logger.info(f"Auto-selected device: {device}")
                    except Exception as e:
                        logger.warning(f"Failed to auto-select GPU device: {e}")
        
        except Exception as e:
            logger.error(f"Failed to initialize DeepSpeed: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the DeepSpeed environment.
        """
        if not self._initialized:
            logger.warning("DeepSpeed environment not initialized")
            return
        
        try:
            self._binding.deepspeed_finalize()
            self._initialized = False
            logger.info("DeepSpeed finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize DeepSpeed: {e}")
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
            self._binding.deepspeed_barrier()
        except Exception as e:
            logger.error(f"Failed to synchronize at barrier: {e}")
            raise
    
    def initialize_model(self, model: Any, optimizer: Any = None, 
                        model_parameters: Optional[Any] = None,
                        training_batch_size: Optional[int] = None,
                        gradient_accumulation_steps: int = 1,
                        fp16: bool = False,
                        amp: bool = False,
                        zero_optimization: bool = False,
                        zero_stage: int = 1,
                        offload_optimizer: bool = False,
                        offload_parameters: bool = False) -> Any:
        """
        Initialize a model with DeepSpeed.
        
        Args:
            model: Model to initialize
            optimizer: Optimizer to use
            model_parameters: Model parameters
            training_batch_size: Training batch size
            gradient_accumulation_steps: Number of gradient accumulation steps
            fp16: Whether to use FP16 precision
            amp: Whether to use automatic mixed precision
            zero_optimization: Whether to use ZeRO optimization
            zero_stage: ZeRO stage (1, 2, or 3)
            offload_optimizer: Whether to offload optimizer states to CPU
            offload_parameters: Whether to offload parameters to CPU
            
        Returns:
            DeepSpeed engine
        """
        if not self._initialized:
            self.initialize()
        
        ds_config = self.config.copy()
        
        if training_batch_size is not None:
            ds_config["train_batch_size"] = training_batch_size
        
        if gradient_accumulation_steps != 1:
            ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        
        if fp16:
            ds_config["fp16"] = {"enabled": True}
        
        if amp:
            ds_config["amp"] = {"enabled": True}
        
        if zero_optimization:
            ds_config["zero_optimization"] = {
                "stage": zero_stage,
                "offload_optimizer": offload_optimizer,
                "offload_param": offload_parameters
            }
        
        try:
            return self._binding.deepspeed_initialize_model(
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                config=ds_config
            )
        except Exception as e:
            logger.error(f"Failed to initialize model with DeepSpeed: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


_deepspeed_manager = None

def get_deepspeed_manager() -> DeepSpeedManager:
    """
    Get the global DeepSpeed manager instance.
    
    Returns:
        DeepSpeedManager instance
    """
    global _deepspeed_manager
    if _deepspeed_manager is None:
        _deepspeed_manager = DeepSpeedManager()
    return _deepspeed_manager
