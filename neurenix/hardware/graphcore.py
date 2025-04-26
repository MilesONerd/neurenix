"""
GraphCore IPU support for Neurenix.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..core import get_config
from ..device import get_device
from ..binding import get_binding

logger = logging.getLogger(__name__)

class GraphCoreManager:
    """
    Manager for GraphCore IPU hardware acceleration.
    
    This class provides an interface to GraphCore IPU functionality for hardware
    acceleration in Neurenix. It uses the Phynexus Rust/C++ backend for actual
    IPU operations.
    """
    
    def __init__(self, 
                 num_ipus: int = 1,
                 precision: str = "float16",
                 memory_proportion: float = 0.6,
                 enable_half_partials: bool = True,
                 compile_only: bool = False,
                 device_id: Optional[int] = None):
        """
        Initialize the GraphCore IPU manager.
        
        Args:
            num_ipus: Number of IPUs to use
            precision: Precision to use ('float16', 'float32')
            memory_proportion: Proportion of IPU memory to use for model
            enable_half_partials: Whether to use half-precision for partial results
            compile_only: Whether to compile the model without executing
            device_id: Device ID to use (default: auto-select)
        """
        self.num_ipus = num_ipus
        self.precision = precision
        self.memory_proportion = memory_proportion
        self.enable_half_partials = enable_half_partials
        self.compile_only = compile_only
        self._device_id = device_id
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_GRAPHCORE_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the GraphCore IPU environment.
        """
        if self._initialized:
            logger.warning("GraphCore IPU environment already initialized")
            return
        
        try:
            result = self._binding.graphcore_initialize(
                num_ipus=self.num_ipus,
                precision=self.precision,
                memory_proportion=self.memory_proportion,
                enable_half_partials=self.enable_half_partials,
                compile_only=self.compile_only,
                device_id=self._device_id
            )
            
            if self._device_id is None:
                self._device_id = result.get("device_id", 0)
            
            self._initialized = True
            
            logger.info(f"GraphCore IPU initialized: {self.num_ipus} IPUs, device_id={self._device_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize GraphCore IPU: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the GraphCore IPU environment.
        """
        if not self._initialized:
            logger.warning("GraphCore IPU environment not initialized")
            return
        
        try:
            self._binding.graphcore_finalize()
            self._initialized = False
            logger.info("GraphCore IPU finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize GraphCore IPU: {e}")
            raise
    
    def get_ipu_count(self) -> int:
        """
        Get the number of available IPUs.
        
        Returns:
            Number of available IPUs
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.graphcore_get_ipu_count()
        except Exception as e:
            logger.error(f"Failed to get IPU count: {e}")
            raise
    
    def get_ipu_info(self) -> Dict[str, Any]:
        """
        Get information about available IPUs.
        
        Returns:
            Dictionary with IPU information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.graphcore_get_ipu_info()
        except Exception as e:
            logger.error(f"Failed to get IPU information: {e}")
            raise
    
    def compile_model(self, model: Any, inputs: Any) -> Any:
        """
        Compile a model for IPU execution.
        
        Args:
            model: Model to compile
            inputs: Example inputs for the model
            
        Returns:
            Compiled model
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.graphcore_compile_model(model, inputs)
        except Exception as e:
            logger.error(f"Failed to compile model for IPU: {e}")
            raise
    
    def execute_model(self, model: Any, inputs: Any) -> Any:
        """
        Execute a model on IPU.
        
        Args:
            model: Model to execute
            inputs: Inputs for the model
            
        Returns:
            Model outputs
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.graphcore_execute_model(model, inputs)
        except Exception as e:
            logger.error(f"Failed to execute model on IPU: {e}")
            raise
    
    def optimize_model(self, model: Any, inputs: Any) -> Any:
        """
        Optimize a model for IPU execution.
        
        Args:
            model: Model to optimize
            inputs: Example inputs for the model
            
        Returns:
            Optimized model
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.graphcore_optimize_model(model, inputs)
        except Exception as e:
            logger.error(f"Failed to optimize model for IPU: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


_graphcore_manager = None

def get_graphcore_manager() -> GraphCoreManager:
    """
    Get the global GraphCore IPU manager instance.
    
    Returns:
        GraphCoreManager instance
    """
    global _graphcore_manager
    if _graphcore_manager is None:
        _graphcore_manager = GraphCoreManager()
    return _graphcore_manager
