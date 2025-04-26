"""
Device manager for hot-swappable backends in the Neurenix framework.
"""

import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from neurenix.device import Device, DeviceType, get_available_devices

logger = logging.getLogger("neurenix")

class DeviceManager:
    """
    Manages device switching and hardware detection for the Neurenix framework.
    
    This class provides functionality for hot-swappable backends, allowing
    tensors to be moved between different hardware devices at runtime.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern for DeviceManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeviceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the device manager."""
        if self._initialized:
            return
            
        self._initialized = True
        self._active_device = Device(DeviceType.CPU)
        self._available_devices = get_available_devices()
        self._device_memory = {}  # Track memory usage per device
        self._registered_tensors = set()  # Track tensors for hot-swapping
        self._device_streams = {}  # Track CUDA/ROCm streams per device
        self._device_hooks = {}  # Hooks for device switching events
        
        for device in self._available_devices:
            self._device_memory[device] = 0
            self._device_streams[device] = []
        
        logger.info(f"DeviceManager initialized with {len(self._available_devices)} available devices")
    
    @property
    def active_device(self) -> Device:
        """Get the currently active device."""
        return self._active_device
    
    @active_device.setter
    def active_device(self, device: Device) -> None:
        """Set the active device."""
        if device not in self._available_devices:
            if not self._is_device_available(device):
                raise ValueError(f"Device {device} is not available")
            self._available_devices.append(device)
        
        self._active_device = device
        logger.debug(f"Active device set to {device}")
        
        if device in self._device_hooks:
            for hook in self._device_hooks[device]:
                hook(device)
    
    def register_tensor(self, tensor_id: int) -> None:
        """Register a tensor for hot-swapping."""
        self._registered_tensors.add(tensor_id)
    
    def unregister_tensor(self, tensor_id: int) -> None:
        """Unregister a tensor from hot-swapping."""
        if tensor_id in self._registered_tensors:
            self._registered_tensors.remove(tensor_id)
    
    def register_device_hook(self, device: Device, hook: callable) -> None:
        """Register a hook to be called when switching to a device."""
        if device not in self._device_hooks:
            self._device_hooks[device] = []
        self._device_hooks[device].append(hook)
    
    def get_available_devices(self) -> List[Device]:
        """Get a list of all available devices."""
        self._available_devices = get_available_devices()
        return self._available_devices
    
    def _is_device_available(self, device: Device) -> bool:
        """Check if a device is available."""
        try:
            from neurenix.tensor import Tensor
            test_tensor = Tensor([1.0], device=device)
            return True
        except Exception as e:
            logger.warning(f"Device {device} is not available: {e}")
            return False
    
    def synchronize(self, device: Optional[Device] = None) -> None:
        """
        Synchronize the specified device or all devices.
        
        Args:
            device: The device to synchronize. If None, synchronize all devices.
        """
        if device is None:
            for dev in self._available_devices:
                self._synchronize_device(dev)
        else:
            self._synchronize_device(device)
    
    def _synchronize_device(self, device: Device) -> None:
        """Synchronize a specific device."""
        if device.type == DeviceType.CPU:
            return
        
        try:
            from neurenix.binding import synchronize_device
            synchronize_device(device)
        except (ImportError, AttributeError):
            logger.warning(f"Cannot synchronize device {device}: bindings not available")
    
    def get_memory_stats(self, device: Optional[Device] = None) -> Dict[str, Any]:
        """
        Get memory statistics for the specified device or all devices.
        
        Args:
            device: The device to get memory stats for. If None, get stats for all devices.
            
        Returns:
            A dictionary with memory statistics.
        """
        if device is None:
            stats = {}
            for dev in self._available_devices:
                stats[str(dev)] = self._get_device_memory_stats(dev)
            return stats
        else:
            return {str(device): self._get_device_memory_stats(device)}
    
    def _get_device_memory_stats(self, device: Device) -> Dict[str, Any]:
        """Get memory statistics for a specific device."""
        if device.type == DeviceType.CPU:
            try:
                import psutil
                mem = psutil.virtual_memory()
                return {
                    "total": mem.total,
                    "available": mem.available,
                    "used": mem.used,
                    "percent": mem.percent
                }
            except ImportError:
                return {"error": "psutil not available"}
        else:
            try:
                from neurenix.binding import get_device_memory_stats
                return get_device_memory_stats(device)
            except (ImportError, AttributeError):
                return {"allocated": self._device_memory.get(device, 0)}


class Genesis:
    """
    Automatic hardware detection and selection system for Neurenix.
    
    Genesis automatically detects available hardware and selects the most
    appropriate device for the current workload.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern for Genesis."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Genesis, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the Genesis system."""
        if self._initialized:
            return
            
        self._initialized = True
        self._device_manager = DeviceManager()
        self._device_scores = {}  # Performance scores for each device
        self._workload_history = []  # History of workloads for better selection
        
        self._initialize_device_scores()
        
        logger.info("Genesis initialized for automatic hardware selection")
    
    def _initialize_device_scores(self) -> None:
        """Initialize performance scores for available devices."""
        devices = self._device_manager.get_available_devices()
        
        for device in devices:
            if device.type == DeviceType.CPU:
                self._device_scores[device] = 1.0
            elif device.type == DeviceType.CUDA:
                self._device_scores[device] = 10.0
            elif device.type == DeviceType.ROCM:
                self._device_scores[device] = 9.0
            elif device.type == DeviceType.TPU:
                self._device_scores[device] = 15.0
            elif device.type == DeviceType.WEBGPU:
                self._device_scores[device] = 5.0
            else:
                self._device_scores[device] = 1.0
    
    def select_device(self, workload_type: str = "general", tensor_shape: Optional[Tuple[int, ...]] = None) -> Device:
        """
        Select the most appropriate device for the given workload.
        
        Args:
            workload_type: Type of workload (e.g., "general", "training", "inference").
            tensor_shape: Shape of the tensor for the workload.
            
        Returns:
            The selected device.
        """
        devices = self._device_manager.get_available_devices()
        
        if not devices:
            return Device(DeviceType.CPU)
        
        if tensor_shape is not None:
            element_size = 4  # Assume float32 (4 bytes)
            memory_required = element_size * int(sum(tensor_shape))
            
            devices = [d for d in devices if self._has_enough_memory(d, memory_required)]
        
        if not devices:
            return Device(DeviceType.CPU)
        
        adjusted_scores = {}
        for device in devices:
            score = self._device_scores[device]
            
            if workload_type == "training":
                if device.type in [DeviceType.CUDA, DeviceType.ROCM, DeviceType.TPU]:
                    score *= 1.5
            elif workload_type == "inference":
                if device.type == DeviceType.TPU:
                    score *= 2.0
                elif device.type in [DeviceType.CUDA, DeviceType.ROCM]:
                    score *= 1.2
            
            adjusted_scores[device] = score
        
        selected_device = max(adjusted_scores, key=adjusted_scores.get)
        
        self._workload_history.append((workload_type, tensor_shape, selected_device))
        if len(self._workload_history) > 100:
            self._workload_history.pop(0)
        
        logger.debug(f"Selected device {selected_device} for workload {workload_type}")
        return selected_device
    
    def _has_enough_memory(self, device: Device, memory_required: int) -> bool:
        """Check if a device has enough memory for the workload."""
        if device.type == DeviceType.CPU:
            return True
        
        stats = self._device_manager._get_device_memory_stats(device)
        
        if "error" in stats:
            return True
        
        if "available" in stats:
            return stats["available"] >= memory_required
        
        return True
    
    def benchmark_devices(self) -> Dict[Device, float]:
        """
        Benchmark available devices to update performance scores.
        
        Returns:
            A dictionary mapping devices to their benchmark scores.
        """
        devices = self._device_manager.get_available_devices()
        benchmark_results = {}
        
        for device in devices:
            score = self._benchmark_device(device)
            benchmark_results[device] = score
            
            self._device_scores[device] = score
        
        return benchmark_results
    
    def _benchmark_device(self, device: Device) -> float:
        """Run a benchmark on a device and return a performance score."""
        try:
            from neurenix.tensor import Tensor
            import time
            
            size = 1000
            a = Tensor.randn((size, size), device=device)
            b = Tensor.randn((size, size), device=device)
            
            start_time = time.time()
            _ = a.matmul(b)
            self._device_manager.synchronize(device)
            end_time = time.time()
            
            ops = size * size * size * 2  # Approximate number of operations
            score = ops / (end_time - start_time)
            
            return score
        except Exception as e:
            logger.warning(f"Benchmark failed for device {device}: {e}")
            return self._device_scores.get(device, 1.0)  # Return existing score or default
