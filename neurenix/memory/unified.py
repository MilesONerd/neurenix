"""
Unified Memory (UM) and Heterogeneous Memory Management (HMM) support for Neurenix.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..core import get_config
from ..device import get_device
from ..binding import get_binding

logger = logging.getLogger(__name__)

class UnifiedMemoryManager:
    """
    Manager for Unified Memory (UM) and Heterogeneous Memory Management (HMM).
    
    This class provides an interface to Unified Memory functionality for efficient
    memory management in Neurenix. It uses the Phynexus Rust/C++ backend for actual
    memory operations.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 prefetch_policy: str = "adaptive",
                 migration_policy: str = "adaptive",
                 advise_policy: str = "preferred_location",
                 device: Optional[str] = None):
        """
        Initialize the Unified Memory manager.
        
        Args:
            mode: Memory management mode ('auto', 'manual', 'managed')
            prefetch_policy: Prefetch policy ('none', 'adaptive', 'aggressive')
            migration_policy: Migration policy ('none', 'adaptive', 'aggressive')
            advise_policy: Memory advise policy ('preferred_location', 'read_mostly', 'accessed_by')
            device: Device to use for memory operations (default: current device)
        """
        self.mode = mode
        self.prefetch_policy = prefetch_policy
        self.migration_policy = migration_policy
        self.advise_policy = advise_policy
        self._device = device
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_UM_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the Unified Memory environment.
        """
        if self._initialized:
            logger.warning("Unified Memory environment already initialized")
            return
        
        try:
            if self._device is None:
                self._device = get_device().name
            
            result = self._binding.um_initialize(
                mode=self.mode,
                prefetch_policy=self.prefetch_policy,
                migration_policy=self.migration_policy,
                advise_policy=self.advise_policy,
                device=self._device
            )
            
            self._initialized = True
            
            logger.info(f"Unified Memory initialized: mode={self.mode}, device={self._device}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Unified Memory: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the Unified Memory environment.
        """
        if not self._initialized:
            logger.warning("Unified Memory environment not initialized")
            return
        
        try:
            self._binding.um_finalize()
            self._initialized = False
            logger.info("Unified Memory finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize Unified Memory: {e}")
            raise
    
    def allocate(self, size: int, dtype: str = "float32") -> Any:
        """
        Allocate unified memory.
        
        Args:
            size: Size of memory to allocate in bytes
            dtype: Data type ('float32', 'float64', 'int32', 'int64', etc.)
            
        Returns:
            Memory handle
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.um_allocate(size, dtype)
        except Exception as e:
            logger.error(f"Failed to allocate unified memory: {e}")
            raise
    
    def free(self, handle: Any) -> None:
        """
        Free unified memory.
        
        Args:
            handle: Memory handle to free
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.um_free(handle)
        except Exception as e:
            logger.error(f"Failed to free unified memory: {e}")
            raise
    
    def prefetch(self, handle: Any, device: Optional[str] = None) -> None:
        """
        Prefetch unified memory to a device.
        
        Args:
            handle: Memory handle to prefetch
            device: Device to prefetch to (default: current device)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if device is None:
                device = self._device
            
            self._binding.um_prefetch(handle, device)
        except Exception as e:
            logger.error(f"Failed to prefetch unified memory: {e}")
            raise
    
    def advise(self, handle: Any, advice: str, device: Optional[str] = None) -> None:
        """
        Set memory usage advice for unified memory.
        
        Args:
            handle: Memory handle to advise
            advice: Memory usage advice ('preferred_location', 'read_mostly', 'accessed_by')
            device: Device for the advice (default: current device)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if device is None:
                device = self._device
            
            self._binding.um_advise(handle, advice, device)
        except Exception as e:
            logger.error(f"Failed to set memory advice: {e}")
            raise
    
    def is_managed(self, handle: Any) -> bool:
        """
        Check if memory is managed by unified memory.
        
        Args:
            handle: Memory handle to check
            
        Returns:
            True if memory is managed, False otherwise
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.um_is_managed(handle)
        except Exception as e:
            logger.error(f"Failed to check if memory is managed: {e}")
            raise
    
    def get_info(self, handle: Any) -> Dict[str, Any]:
        """
        Get information about unified memory.
        
        Args:
            handle: Memory handle to get information about
            
        Returns:
            Dictionary with memory information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.um_get_info(handle)
        except Exception as e:
            logger.error(f"Failed to get memory information: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class HeterogeneousMemoryManager:
    """
    Manager for Heterogeneous Memory Management (HMM).
    
    This class provides an interface to HMM functionality for efficient
    memory management in Neurenix. It uses the Phynexus Rust/C++ backend for actual
    memory operations.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 migration_policy: str = "adaptive",
                 device: Optional[str] = None):
        """
        Initialize the HMM manager.
        
        Args:
            mode: Memory management mode ('auto', 'manual', 'managed')
            migration_policy: Migration policy ('none', 'adaptive', 'aggressive')
            device: Device to use for memory operations (default: current device)
        """
        self.mode = mode
        self.migration_policy = migration_policy
        self._device = device
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_HMM_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the HMM environment.
        """
        if self._initialized:
            logger.warning("HMM environment already initialized")
            return
        
        try:
            if self._device is None:
                self._device = get_device().name
            
            result = self._binding.hmm_initialize(
                mode=self.mode,
                migration_policy=self.migration_policy,
                device=self._device
            )
            
            self._initialized = True
            
            logger.info(f"HMM initialized: mode={self.mode}, device={self._device}")
        
        except Exception as e:
            logger.error(f"Failed to initialize HMM: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the HMM environment.
        """
        if not self._initialized:
            logger.warning("HMM environment not initialized")
            return
        
        try:
            self._binding.hmm_finalize()
            self._initialized = False
            logger.info("HMM finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize HMM: {e}")
            raise
    
    def allocate(self, size: int, dtype: str = "float32") -> Any:
        """
        Allocate HMM memory.
        
        Args:
            size: Size of memory to allocate in bytes
            dtype: Data type ('float32', 'float64', 'int32', 'int64', etc.)
            
        Returns:
            Memory handle
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.hmm_allocate(size, dtype)
        except Exception as e:
            logger.error(f"Failed to allocate HMM memory: {e}")
            raise
    
    def free(self, handle: Any) -> None:
        """
        Free HMM memory.
        
        Args:
            handle: Memory handle to free
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.hmm_free(handle)
        except Exception as e:
            logger.error(f"Failed to free HMM memory: {e}")
            raise
    
    def migrate(self, handle: Any, device: Optional[str] = None) -> None:
        """
        Migrate HMM memory to a device.
        
        Args:
            handle: Memory handle to migrate
            device: Device to migrate to (default: current device)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if device is None:
                device = self._device
            
            self._binding.hmm_migrate(handle, device)
        except Exception as e:
            logger.error(f"Failed to migrate HMM memory: {e}")
            raise
    
    def get_info(self, handle: Any) -> Dict[str, Any]:
        """
        Get information about HMM memory.
        
        Args:
            handle: Memory handle to get information about
            
        Returns:
            Dictionary with memory information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.hmm_get_info(handle)
        except Exception as e:
            logger.error(f"Failed to get memory information: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


_um_manager = None
_hmm_manager = None

def get_um_manager() -> UnifiedMemoryManager:
    """
    Get the global Unified Memory manager instance.
    
    Returns:
        UnifiedMemoryManager instance
    """
    global _um_manager
    if _um_manager is None:
        _um_manager = UnifiedMemoryManager()
    return _um_manager

def get_hmm_manager() -> HeterogeneousMemoryManager:
    """
    Get the global HMM manager instance.
    
    Returns:
        HeterogeneousMemoryManager instance
    """
    global _hmm_manager
    if _hmm_manager is None:
        _hmm_manager = HeterogeneousMemoryManager()
    return _hmm_manager
