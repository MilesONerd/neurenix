"""
Device abstraction for the Neurenix framework.
"""

from enum import Enum
from typing import Optional, List, Dict, Any

class DeviceType(Enum):
    """Types of devices supported by Neurenix."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    WEBGPU = "webgpu"  # WebGPU for WebAssembly context (client-side execution)
    TPU = "tpu"  # Tensor Processing Unit for machine learning acceleration

class Device:
    """
    Represents a computational device (CPU, GPU, etc.).
    
    This class abstracts away the details of different hardware devices,
    allowing the framework to run on various platforms.
    """
    
    def __init__(self, device_type: DeviceType, index: int = 0):
        """
        Create a new device.
        
        Args:
            device_type: The type of the device.
            index: The index of the device (for multiple devices of the same type).
        """
        self._type = device_type
        self._index = index
        
        # Check if the device is available
        self._available = True  # Default to available, will be updated if needed
        
        if device_type == DeviceType.CUDA:
            # TODO: Check if CUDA is available when Phynexus bindings are available
            try:
                # Import binding module to check CUDA availability
                from neurenix.binding import is_cuda_available
                self._available = is_cuda_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.ROCM:
            # TODO: Check if ROCm is available when Phynexus bindings are available
            try:
                # Import binding module to check ROCm availability
                from neurenix.binding import is_rocm_available
                self._available = is_rocm_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.WEBGPU:
            # Check if running in a WebAssembly context
            import sys
            
            # Check for WebAssembly environment
            # Emscripten sets sys.platform to 'emscripten'
            # Pyodide is another Python implementation for WebAssembly
            self._available = sys.platform == "emscripten" or "pyodide" in sys.modules
            
            # If not in WebAssembly context, check if WebGPU is available through bindings
            if not self._available:
                try:
                    from neurenix.binding import is_webgpu_available
                    self._available = is_webgpu_available()
                except (ImportError, AttributeError):
                    self._available = False
        elif device_type == DeviceType.TPU:
            # Check if TPU is available through bindings
            try:
                from neurenix.binding import is_tpu_available
                self._available = is_tpu_available()
            except (ImportError, AttributeError):
                self._available = False
    
    @property
    def type(self) -> DeviceType:
        """Get the type of the device."""
        return self._type
    
    @property
    def index(self) -> int:
        """Get the index of the device."""
        return self._index
    
    @property
    def name(self) -> str:
        """Get the name of the device."""
        if self._type == DeviceType.CPU:
            return "CPU"
        elif self._type == DeviceType.CUDA:
            return f"CUDA:{self._index}"
        elif self._type == DeviceType.ROCM:
            return f"ROCm:{self._index}"
        elif self._type == DeviceType.WEBGPU:
            return f"WebGPU:{self._index}"
        else:
            return f"{self._type}:{self._index}"
    
    def __eq__(self, other: object) -> bool:
        """Check if two devices are equal."""
        if not isinstance(other, Device):
            return False
        return self._type == other._type and self._index == other._index
    
    def __hash__(self) -> int:
        """Get a hash of the device."""
        return hash((self._type, self._index))
    
    def __repr__(self) -> str:
        """Get a string representation of the device."""
        return f"Device({self.name})"
    
    @classmethod
    def device_count(cls) -> int:
        """
        Get the total number of devices available.
        
        Returns:
            The total number of devices available.
        """
        count = 1  # CPU is always available
        
        # Add CUDA devices
        count += get_device_count(DeviceType.CUDA)
        
        # Add ROCm devices
        count += get_device_count(DeviceType.ROCM)
        
        # Add WebGPU devices
        count += get_device_count(DeviceType.WEBGPU)
        
        # Add TPU devices
        count += get_device_count(DeviceType.TPU)
        
        return count

def get_device_count(device_type: DeviceType) -> int:
    """
    Get the number of devices of the given type.
    
    Args:
        device_type: The type of device to count.
        
    Returns:
        The number of devices of the given type.
    """
    if device_type == DeviceType.CPU:
        return 1
    elif device_type == DeviceType.CUDA:
        try:
            # Import binding module to get CUDA device count
            from neurenix.binding import get_cuda_device_count
            return get_cuda_device_count()
        except (ImportError, AttributeError):
            # For now, assume no CUDA devices if bindings are not available
            return 0
    elif device_type == DeviceType.ROCM:
        try:
            # Import binding module to get ROCm device count
            from neurenix.binding import get_rocm_device_count
            return get_rocm_device_count()
        except (ImportError, AttributeError):
            # For now, assume no ROCm devices if bindings are not available
            return 0
    elif device_type == DeviceType.WEBGPU:
        # Check if running in a WebAssembly context
        import sys
        
        # In WebAssembly context, there's at most one WebGPU device
        if sys.platform == "emscripten" or "pyodide" in sys.modules:
            # Return 1 if WebGPU is available in the browser
            try:
                # This is a simplified check, in a real implementation
                # we would check if the browser supports WebGPU
                return 1
            except:
                return 0
        else:
            # If not in WebAssembly context, check through bindings
            try:
                from neurenix.binding import get_webgpu_device_count
                return get_webgpu_device_count()
            except (ImportError, AttributeError):
                # For now, assume no WebGPU devices if bindings are not available
                return 0
    elif device_type == DeviceType.TPU:
        try:
            # Import binding module to get TPU device count
            from neurenix.binding import get_tpu_device_count
            return get_tpu_device_count()
        except (ImportError, AttributeError):
            # For now, assume no TPU devices if bindings are not available
            return 0
    else:
        return 0

def get_device(device_str: str) -> Device:
    """
    Get a device from a string representation.
    
    Args:
        device_str: String representation of the device (e.g., 'cpu', 'cuda:0').
        
    Returns:
        The corresponding device.
    """
    if device_str == "cpu":
        return Device(DeviceType.CPU)
    
    # Parse device type and index
    if ":" in device_str:
        device_type_str, index_str = device_str.split(":", 1)
        index = int(index_str)
    else:
        device_type_str = device_str
        index = 0
    
    # Map string to device type
    if device_type_str == "cuda":
        device_type = DeviceType.CUDA
    elif device_type_str == "rocm":
        device_type = DeviceType.ROCM
    elif device_type_str == "webgpu":
        device_type = DeviceType.WEBGPU
    elif device_type_str == "tpu":
        device_type = DeviceType.TPU
    else:
        raise ValueError(f"Unknown device type: {device_type_str}")
    
    return Device(device_type, index)

def get_available_devices() -> List[Device]:
    """
    Get a list of all available devices.
    
    Returns:
        A list of available devices.
    """
    devices = [Device(DeviceType.CPU)]
    
    # Add CUDA devices
    cuda_count = get_device_count(DeviceType.CUDA)
    for i in range(cuda_count):
        devices.append(Device(DeviceType.CUDA, i))
    
    # Add ROCm devices
    rocm_count = get_device_count(DeviceType.ROCM)
    for i in range(rocm_count):
        devices.append(Device(DeviceType.ROCM, i))
    
    # Add WebGPU devices
    webgpu_count = get_device_count(DeviceType.WEBGPU)
    for i in range(webgpu_count):
        devices.append(Device(DeviceType.WEBGPU, i))
    
    # Add TPU devices
    tpu_count = get_device_count(DeviceType.TPU)
    for i in range(tpu_count):
        devices.append(Device(DeviceType.TPU, i))
    
    return devices
