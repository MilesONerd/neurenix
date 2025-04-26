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
    NPU = "npu"  # Neural Processing Unit for machine learning acceleration
    VULKAN = "vulkan"  # Vulkan for cross-platform GPU acceleration
    OPENCL = "opencl"  # OpenCL for cross-platform GPU acceleration
    ONEAPI = "oneapi"  # oneAPI for cross-platform acceleration
    DIRECTML = "directml"  # DirectML for Windows-specific acceleration
    ONEDNN = "onednn"  # oneDNN for optimized deep learning primitives
    MKLDNN = "mkldnn"  # MKL-DNN for optimized deep learning primitives
    TENSORRT = "tensorrt"  # TensorRT for NVIDIA-specific optimizations
    QUANTUM = "quantum"  # Quantum computing device for quantum operations
    ARM = "arm"  # ARM architecture with NEON SIMD and SVE support

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
            try:
                # Import binding module to check CUDA availability
                from neurenix.binding import is_cuda_available
                self._available = is_cuda_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.ROCM:
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
        elif device_type == DeviceType.NPU:
            # Check if NPU is available through bindings
            try:
                from neurenix.binding import is_npu_available
                self._available = is_npu_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.VULKAN:
            try:
                from neurenix.binding import is_vulkan_available
                self._available = is_vulkan_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.OPENCL:
            try:
                from neurenix.binding import is_opencl_available
                self._available = is_opencl_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.ONEAPI:
            try:
                from neurenix.binding import is_oneapi_available
                self._available = is_oneapi_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.DIRECTML:
            try:
                from neurenix.binding import is_directml_available
                self._available = is_directml_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.ONEDNN:
            try:
                from neurenix.binding import is_onednn_available
                self._available = is_onednn_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.MKLDNN:
            try:
                from neurenix.binding import is_mkldnn_available
                self._available = is_mkldnn_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.TENSORRT:
            try:
                from neurenix.binding import is_tensorrt_available
                self._available = is_tensorrt_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.QUANTUM:
            try:
                from neurenix.binding import is_quantum_available
                self._available = is_quantum_available()
            except (ImportError, AttributeError):
                self._available = False
        elif device_type == DeviceType.ARM:
            try:
                from neurenix.binding import is_arm_available
                self._available = is_arm_available()
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
        elif self._type == DeviceType.TPU:
            return f"TPU:{self._index}"
        elif self._type == DeviceType.NPU:
            return f"NPU:{self._index}"
        elif self._type == DeviceType.VULKAN:
            return f"Vulkan:{self._index}"
        elif self._type == DeviceType.OPENCL:
            return f"OpenCL:{self._index}"
        elif self._type == DeviceType.ONEAPI:
            return f"oneAPI:{self._index}"
        elif self._type == DeviceType.DIRECTML:
            return f"DirectML:{self._index}"
        elif self._type == DeviceType.ONEDNN:
            return f"oneDNN:{self._index}"
        elif self._type == DeviceType.MKLDNN:
            return f"MKL-DNN:{self._index}"
        elif self._type == DeviceType.TENSORRT:
            return f"TensorRT:{self._index}"
        elif self._type == DeviceType.ARM:
            return f"ARM:{self._index}"
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
        
        # Add NPU devices
        count += get_device_count(DeviceType.NPU)
        
        count += get_device_count(DeviceType.VULKAN)
        
        count += get_device_count(DeviceType.OPENCL)
        
        count += get_device_count(DeviceType.ONEAPI)
        
        count += get_device_count(DeviceType.DIRECTML)
        
        count += get_device_count(DeviceType.ONEDNN)
        
        count += get_device_count(DeviceType.MKLDNN)
        
        count += get_device_count(DeviceType.TENSORRT)
        
        count += get_device_count(DeviceType.ARM)
        
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
            try:
                import js
                
                if hasattr(js.navigator, 'gpu') and hasattr(js.navigator.gpu, 'requestAdapter'):
                    adapter_promise = js.navigator.gpu.requestAdapter()
                    
                    def check_adapter(adapter):
                        return 1 if adapter else 0
                    
                    result = js.Promise.resolve(adapter_promise).then(check_adapter).catch(lambda _: 0)
                    
                    return result.valueOf()
                else:
                    return 0
            except ImportError:
                try:
                    import emscripten
                    
                    check_webgpu_js = """
                    function checkWebGPU() {
                        if (navigator.gpu && navigator.gpu.requestAdapter) {
                            return navigator.gpu.requestAdapter()
                                .then(adapter => adapter ? 1 : 0)
                                .catch(() => 0);
                        }
                        return Promise.resolve(0);
                    }
                    """
                    
                    emscripten.run_script(check_webgpu_js)
                    result = emscripten.run_script_int("checkWebGPU()")
                    
                    return result
                except ImportError:
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
    elif device_type == DeviceType.NPU:
        try:
            # Import binding module to get NPU device count
            from neurenix.binding import get_npu_device_count
            return get_npu_device_count()
        except (ImportError, AttributeError):
            # For now, assume no NPU devices if bindings are not available
            return 0
    elif device_type == DeviceType.VULKAN:
        try:
            from neurenix.binding import get_vulkan_device_count
            return get_vulkan_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.OPENCL:
        try:
            from neurenix.binding import get_opencl_device_count
            return get_opencl_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.ONEAPI:
        try:
            from neurenix.binding import get_oneapi_device_count
            return get_oneapi_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.DIRECTML:
        try:
            from neurenix.binding import get_directml_device_count
            return get_directml_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.ONEDNN:
        try:
            from neurenix.binding import get_onednn_device_count
            return get_onednn_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.MKLDNN:
        try:
            from neurenix.binding import get_mkldnn_device_count
            return get_mkldnn_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.TENSORRT:
        try:
            from neurenix.binding import get_tensorrt_device_count
            return get_tensorrt_device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == DeviceType.ARM:
        try:
            from neurenix.binding import get_arm_device_count
            return get_arm_device_count()
        except (ImportError, AttributeError):
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
    elif device_type_str == "npu":
        device_type = DeviceType.NPU
    elif device_type_str == "vulkan":
        device_type = DeviceType.VULKAN
    elif device_type_str == "opencl":
        device_type = DeviceType.OPENCL
    elif device_type_str == "oneapi":
        device_type = DeviceType.ONEAPI
    elif device_type_str == "directml":
        device_type = DeviceType.DIRECTML
    elif device_type_str == "onednn":
        device_type = DeviceType.ONEDNN
    elif device_type_str == "mkldnn":
        device_type = DeviceType.MKLDNN
    elif device_type_str == "tensorrt":
        device_type = DeviceType.TENSORRT
    elif device_type_str == "arm":
        device_type = DeviceType.ARM
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
    
    # Add NPU devices
    npu_count = get_device_count(DeviceType.NPU)
    for i in range(npu_count):
        devices.append(Device(DeviceType.NPU, i))
    
    vulkan_count = get_device_count(DeviceType.VULKAN)
    for i in range(vulkan_count):
        devices.append(Device(DeviceType.VULKAN, i))
    
    opencl_count = get_device_count(DeviceType.OPENCL)
    for i in range(opencl_count):
        devices.append(Device(DeviceType.OPENCL, i))
    
    oneapi_count = get_device_count(DeviceType.ONEAPI)
    for i in range(oneapi_count):
        devices.append(Device(DeviceType.ONEAPI, i))
    
    directml_count = get_device_count(DeviceType.DIRECTML)
    for i in range(directml_count):
        devices.append(Device(DeviceType.DIRECTML, i))
    
    onednn_count = get_device_count(DeviceType.ONEDNN)
    for i in range(onednn_count):
        devices.append(Device(DeviceType.ONEDNN, i))
    
    mkldnn_count = get_device_count(DeviceType.MKLDNN)
    for i in range(mkldnn_count):
        devices.append(Device(DeviceType.MKLDNN, i))
    
    tensorrt_count = get_device_count(DeviceType.TENSORRT)
    for i in range(tensorrt_count):
        devices.append(Device(DeviceType.TENSORRT, i))
    
    arm_count = get_device_count(DeviceType.ARM)
    for i in range(arm_count):
        devices.append(Device(DeviceType.ARM, i))
    
    return devices
