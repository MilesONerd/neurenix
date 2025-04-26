"""
FPGA support for Neurenix via OpenCL, Xilinx Vitis, and Intel OpenVINO.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ..core import get_config
from ..device import get_device
from ..binding import get_binding

logger = logging.getLogger(__name__)

class FPGAManager:
    """
    Manager for FPGA hardware acceleration.
    
    This class provides an interface to FPGA functionality for hardware
    acceleration in Neurenix. It supports multiple FPGA frameworks:
    - OpenCL for general FPGA programming
    - Xilinx Vitis for Xilinx FPGAs
    - Intel OpenVINO for Intel FPGAs
    
    It uses the Phynexus Rust/C++ backend for actual FPGA operations.
    """
    
    def __init__(self, 
                 framework: str = "opencl",
                 device_id: Optional[int] = None,
                 bitstream: Optional[str] = None,
                 precision: str = "float32",
                 optimize_for: str = "throughput",
                 memory_allocation: str = "dynamic"):
        """
        Initialize the FPGA manager.
        
        Args:
            framework: FPGA framework to use ('opencl', 'vitis', 'openvino')
            device_id: Device ID to use (default: auto-select)
            bitstream: Path to FPGA bitstream file (optional)
            precision: Precision to use ('float16', 'float32', 'int8')
            optimize_for: Optimization target ('throughput', 'latency', 'power')
            memory_allocation: Memory allocation strategy ('static', 'dynamic')
        """
        self.framework = framework
        self._device_id = device_id
        self.bitstream = bitstream
        self.precision = precision
        self.optimize_for = optimize_for
        self.memory_allocation = memory_allocation
        self._initialized = False
        
        self._binding = get_binding()
        
        if os.environ.get("NEURENIX_FPGA_AUTO_INIT", "1") == "1":
            self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the FPGA environment.
        """
        if self._initialized:
            logger.warning("FPGA environment already initialized")
            return
        
        try:
            result = self._binding.fpga_initialize(
                framework=self.framework,
                device_id=self._device_id,
                bitstream=self.bitstream,
                precision=self.precision,
                optimize_for=self.optimize_for,
                memory_allocation=self.memory_allocation
            )
            
            if self._device_id is None:
                self._device_id = result.get("device_id", 0)
            
            self._initialized = True
            
            logger.info(f"FPGA initialized: framework={self.framework}, device_id={self._device_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize FPGA: {e}")
            raise
    
    def finalize(self) -> None:
        """
        Finalize the FPGA environment.
        """
        if not self._initialized:
            logger.warning("FPGA environment not initialized")
            return
        
        try:
            self._binding.fpga_finalize()
            self._initialized = False
            logger.info("FPGA finalized")
        
        except Exception as e:
            logger.error(f"Failed to finalize FPGA: {e}")
            raise
    
    def get_fpga_count(self) -> int:
        """
        Get the number of available FPGAs.
        
        Returns:
            Number of available FPGAs
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.fpga_get_count()
        except Exception as e:
            logger.error(f"Failed to get FPGA count: {e}")
            raise
    
    def get_fpga_info(self) -> Dict[str, Any]:
        """
        Get information about available FPGAs.
        
        Returns:
            Dictionary with FPGA information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.fpga_get_info()
        except Exception as e:
            logger.error(f"Failed to get FPGA information: {e}")
            raise
    
    def compile_model(self, model: Any, inputs: Any) -> Any:
        """
        Compile a model for FPGA execution.
        
        Args:
            model: Model to compile
            inputs: Example inputs for the model
            
        Returns:
            Compiled model
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.fpga_compile_model(model, inputs)
        except Exception as e:
            logger.error(f"Failed to compile model for FPGA: {e}")
            raise
    
    def execute_model(self, model: Any, inputs: Any) -> Any:
        """
        Execute a model on FPGA.
        
        Args:
            model: Model to execute
            inputs: Inputs for the model
            
        Returns:
            Model outputs
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.fpga_execute_model(model, inputs)
        except Exception as e:
            logger.error(f"Failed to execute model on FPGA: {e}")
            raise
    
    def load_bitstream(self, bitstream_path: str) -> None:
        """
        Load a bitstream to the FPGA.
        
        Args:
            bitstream_path: Path to the bitstream file
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.fpga_load_bitstream(bitstream_path)
            self.bitstream = bitstream_path
            logger.info(f"Loaded bitstream: {bitstream_path}")
        except Exception as e:
            logger.error(f"Failed to load bitstream: {e}")
            raise
    
    def create_kernel(self, kernel_name: str, kernel_source: str) -> Any:
        """
        Create an OpenCL kernel for FPGA execution.
        
        Args:
            kernel_name: Name of the kernel function
            kernel_source: OpenCL kernel source code
            
        Returns:
            Kernel handle
        """
        if not self._initialized:
            self.initialize()
        
        if self.framework != "opencl":
            raise ValueError(f"Creating kernels is only supported with OpenCL, not {self.framework}")
        
        try:
            return self._binding.fpga_create_kernel(kernel_name, kernel_source)
        except Exception as e:
            logger.error(f"Failed to create kernel: {e}")
            raise
    
    def execute_kernel(self, kernel: Any, global_size: Tuple[int, ...], 
                      local_size: Optional[Tuple[int, ...]] = None,
                      args: Optional[List[Any]] = None) -> None:
        """
        Execute an OpenCL kernel on FPGA.
        
        Args:
            kernel: Kernel handle
            global_size: Global work size
            local_size: Local work size (optional)
            args: Kernel arguments (optional)
        """
        if not self._initialized:
            self.initialize()
        
        if self.framework != "opencl":
            raise ValueError(f"Executing kernels is only supported with OpenCL, not {self.framework}")
        
        try:
            self._binding.fpga_execute_kernel(kernel, global_size, local_size, args)
        except Exception as e:
            logger.error(f"Failed to execute kernel: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class OpenCLManager(FPGAManager):
    """
    Manager for OpenCL-based FPGA acceleration.
    """
    
    def __init__(self, 
                 device_id: Optional[int] = None,
                 platform_id: Optional[int] = None,
                 precision: str = "float32",
                 optimize_for: str = "throughput"):
        """
        Initialize the OpenCL FPGA manager.
        
        Args:
            device_id: Device ID to use (default: auto-select)
            platform_id: Platform ID to use (default: auto-select)
            precision: Precision to use ('float16', 'float32', 'int8')
            optimize_for: Optimization target ('throughput', 'latency', 'power')
        """
        super().__init__(
            framework="opencl",
            device_id=device_id,
            precision=precision,
            optimize_for=optimize_for
        )
        self.platform_id = platform_id
    
    def initialize(self) -> None:
        """
        Initialize the OpenCL FPGA environment.
        """
        if self._initialized:
            logger.warning("OpenCL FPGA environment already initialized")
            return
        
        try:
            result = self._binding.opencl_initialize(
                device_id=self._device_id,
                platform_id=self.platform_id,
                precision=self.precision,
                optimize_for=self.optimize_for
            )
            
            if self._device_id is None:
                self._device_id = result.get("device_id", 0)
            
            if self.platform_id is None:
                self.platform_id = result.get("platform_id", 0)
            
            self._initialized = True
            
            logger.info(f"OpenCL FPGA initialized: device_id={self._device_id}, platform_id={self.platform_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL FPGA: {e}")
            raise
    
    def get_platforms(self) -> List[Dict[str, Any]]:
        """
        Get information about available OpenCL platforms.
        
        Returns:
            List of dictionaries with platform information
        """
        try:
            return self._binding.opencl_get_platforms()
        except Exception as e:
            logger.error(f"Failed to get OpenCL platforms: {e}")
            raise
    
    def get_devices(self, platform_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get information about available OpenCL devices.
        
        Args:
            platform_id: Platform ID to get devices for (default: current platform)
            
        Returns:
            List of dictionaries with device information
        """
        if platform_id is None:
            platform_id = self.platform_id
        
        try:
            return self._binding.opencl_get_devices(platform_id)
        except Exception as e:
            logger.error(f"Failed to get OpenCL devices: {e}")
            raise


class VitisManager(FPGAManager):
    """
    Manager for Xilinx Vitis-based FPGA acceleration.
    """
    
    def __init__(self, 
                 device_id: Optional[int] = None,
                 xclbin_path: Optional[str] = None,
                 target_device: str = "u250",
                 precision: str = "float32",
                 optimize_for: str = "throughput"):
        """
        Initialize the Xilinx Vitis FPGA manager.
        
        Args:
            device_id: Device ID to use (default: auto-select)
            xclbin_path: Path to XCLBIN file (optional)
            target_device: Target FPGA device ('u200', 'u250', 'u280', etc.)
            precision: Precision to use ('float16', 'float32', 'int8')
            optimize_for: Optimization target ('throughput', 'latency', 'power')
        """
        super().__init__(
            framework="vitis",
            device_id=device_id,
            bitstream=xclbin_path,
            precision=precision,
            optimize_for=optimize_for
        )
        self.target_device = target_device
    
    def initialize(self) -> None:
        """
        Initialize the Xilinx Vitis FPGA environment.
        """
        if self._initialized:
            logger.warning("Xilinx Vitis FPGA environment already initialized")
            return
        
        try:
            result = self._binding.vitis_initialize(
                device_id=self._device_id,
                xclbin_path=self.bitstream,
                target_device=self.target_device,
                precision=self.precision,
                optimize_for=self.optimize_for
            )
            
            if self._device_id is None:
                self._device_id = result.get("device_id", 0)
            
            self._initialized = True
            
            logger.info(f"Xilinx Vitis FPGA initialized: device_id={self._device_id}, target_device={self.target_device}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Xilinx Vitis FPGA: {e}")
            raise
    
    def load_xclbin(self, xclbin_path: str) -> None:
        """
        Load an XCLBIN file to the FPGA.
        
        Args:
            xclbin_path: Path to the XCLBIN file
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.vitis_load_xclbin(xclbin_path)
            self.bitstream = xclbin_path
            logger.info(f"Loaded XCLBIN: {xclbin_path}")
        except Exception as e:
            logger.error(f"Failed to load XCLBIN: {e}")
            raise
    
    def get_kernels(self) -> List[str]:
        """
        Get the names of available kernels in the loaded XCLBIN.
        
        Returns:
            List of kernel names
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.vitis_get_kernels()
        except Exception as e:
            logger.error(f"Failed to get kernels: {e}")
            raise
    
    def execute_kernel(self, kernel_name: str, args: List[Any]) -> None:
        """
        Execute a kernel on the FPGA.
        
        Args:
            kernel_name: Name of the kernel to execute
            args: Kernel arguments
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self._binding.vitis_execute_kernel(kernel_name, args)
        except Exception as e:
            logger.error(f"Failed to execute kernel: {e}")
            raise


class OpenVINOManager(FPGAManager):
    """
    Manager for Intel OpenVINO-based FPGA acceleration.
    """
    
    def __init__(self, 
                 device_id: Optional[int] = None,
                 model_path: Optional[str] = None,
                 precision: str = "float32",
                 optimize_for: str = "throughput",
                 cache_dir: Optional[str] = None):
        """
        Initialize the Intel OpenVINO FPGA manager.
        
        Args:
            device_id: Device ID to use (default: auto-select)
            model_path: Path to OpenVINO IR model (optional)
            precision: Precision to use ('float16', 'float32', 'int8')
            optimize_for: Optimization target ('throughput', 'latency', 'power')
            cache_dir: Directory to cache compiled models (optional)
        """
        super().__init__(
            framework="openvino",
            device_id=device_id,
            precision=precision,
            optimize_for=optimize_for
        )
        self.model_path = model_path
        self.cache_dir = cache_dir
    
    def initialize(self) -> None:
        """
        Initialize the Intel OpenVINO FPGA environment.
        """
        if self._initialized:
            logger.warning("Intel OpenVINO FPGA environment already initialized")
            return
        
        try:
            result = self._binding.openvino_initialize(
                device_id=self._device_id,
                model_path=self.model_path,
                precision=self.precision,
                optimize_for=self.optimize_for,
                cache_dir=self.cache_dir
            )
            
            if self._device_id is None:
                self._device_id = result.get("device_id", 0)
            
            self._initialized = True
            
            logger.info(f"Intel OpenVINO FPGA initialized: device_id={self._device_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Intel OpenVINO FPGA: {e}")
            raise
    
    def load_model(self, model_path: str, weights_path: Optional[str] = None) -> Any:
        """
        Load an OpenVINO IR model.
        
        Args:
            model_path: Path to the model XML file
            weights_path: Path to the model weights file (optional)
            
        Returns:
            Model handle
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.openvino_load_model(model_path, weights_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def infer(self, model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference with a model on the FPGA.
        
        Args:
            model: Model handle
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.openvino_infer(model, inputs)
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise
    
    def get_input_info(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """
        Get information about model inputs.
        
        Args:
            model: Model handle
            
        Returns:
            Dictionary with input information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.openvino_get_input_info(model)
        except Exception as e:
            logger.error(f"Failed to get input information: {e}")
            raise
    
    def get_output_info(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """
        Get information about model outputs.
        
        Args:
            model: Model handle
            
        Returns:
            Dictionary with output information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            return self._binding.openvino_get_output_info(model)
        except Exception as e:
            logger.error(f"Failed to get output information: {e}")
            raise


_fpga_manager = None
_opencl_manager = None
_vitis_manager = None
_openvino_manager = None

def get_fpga_manager(framework: str = "opencl") -> FPGAManager:
    """
    Get a global FPGA manager instance.
    
    Args:
        framework: FPGA framework to use ('opencl', 'vitis', 'openvino')
        
    Returns:
        FPGAManager instance
    """
    global _fpga_manager, _opencl_manager, _vitis_manager, _openvino_manager
    
    if framework == "opencl":
        if _opencl_manager is None:
            _opencl_manager = OpenCLManager()
        return _opencl_manager
    
    elif framework == "vitis":
        if _vitis_manager is None:
            _vitis_manager = VitisManager()
        return _vitis_manager
    
    elif framework == "openvino":
        if _openvino_manager is None:
            _openvino_manager = OpenVINOManager()
        return _openvino_manager
    
    else:
        if _fpga_manager is None:
            _fpga_manager = FPGAManager(framework=framework)
        return _fpga_manager
