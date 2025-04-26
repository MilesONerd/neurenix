"""
WebAssembly Multithreaded support for the Neurenix framework.

This module provides multithreading capabilities for WebAssembly, enabling
parallel execution of computations in browser environments using Web Workers
and SharedArrayBuffer.
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import numpy as np
import threading
import queue

from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module

def is_multithreading_supported() -> bool:
    """
    Check if WebAssembly multithreading is supported in the current environment.
    
    Returns:
        bool: True if WebAssembly multithreading is supported, False otherwise.
    """
    try:
        import js
        return (hasattr(js.WebAssembly, 'Feature') and 
                js.WebAssembly.Feature.threads and 
                hasattr(js.window, 'SharedArrayBuffer'))
    except ImportError:
        try:
            import emscripten
            return bool(emscripten.run_script_int(
                "typeof WebAssembly.Feature !== 'undefined' && "
                "WebAssembly.Feature.threads && "
                "typeof SharedArrayBuffer !== 'undefined' ? 1 : 0"
            ))
        except ImportError:
            return False

def enable_multithreading() -> bool:
    """
    Enable multithreading for WebAssembly if available.
    
    Returns:
        bool: True if multithreading was enabled, False otherwise.
    """
    if not is_multithreading_supported():
        return False
    
    try:
        import js
        js.globalThis.NEURENIX_MULTITHREADING_ENABLED = True
        return True
    except ImportError:
        try:
            import emscripten
            emscripten.run_script("globalThis.NEURENIX_MULTITHREADING_ENABLED = true;")
            return True
        except ImportError:
            return False

def get_num_workers() -> int:
    """
    Get the number of available worker threads.
    
    Returns:
        int: Number of available worker threads, or 1 if multithreading is not supported.
    """
    if not is_multithreading_supported():
        return 1
    
    try:
        import js
        return max(1, js.navigator.hardwareConcurrency)
    except (ImportError, AttributeError):
        try:
            import emscripten
            return max(1, int(emscripten.run_script_int("navigator.hardwareConcurrency || 1")))
        except (ImportError, ValueError):
            return 1

class ThreadPool:
    """
    Thread pool for parallel execution of tasks in WebAssembly.
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize a thread pool.
        
        Args:
            num_threads: Number of threads to use. If None, uses the number of available cores.
        """
        if not is_multithreading_supported():
            self.num_threads = 1
            self.enabled = False
            return
            
        self.num_threads = num_threads if num_threads is not None else get_num_workers()
        self.enabled = self.num_threads > 1
        
        if self.enabled:
            self.task_queue = queue.Queue()
            self.result_queue = queue.Queue()
            self.threads = []
            
            for _ in range(self.num_threads):
                thread = threading.Thread(target=self._worker_loop, daemon=True)
                thread.start()
                self.threads.append(thread)
    
    def _worker_loop(self):
        """
        Worker thread loop.
        """
        while True:
            task_id, func, args, kwargs = self.task_queue.get()
            
            try:
                result = func(*args, **kwargs)
                self.result_queue.put((task_id, result, None))
            except Exception as e:
                self.result_queue.put((task_id, None, e))
            
            self.task_queue.task_done()
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Apply a function to each item in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            
        Returns:
            List of results
        """
        if not self.enabled or len(items) <= 1:
            return [func(item) for item in items]
        
        results = [None] * len(items)
        exceptions = []
        
        for i, item in enumerate(items):
            self.task_queue.put((i, func, (item,), {}))
        
        for _ in range(len(items)):
            task_id, result, error = self.result_queue.get()
            
            if error is not None:
                exceptions.append(error)
            else:
                results[task_id] = result
            
            self.result_queue.task_done()
        
        if exceptions:
            raise exceptions[0]
        
        return results

def parallel_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Perform matrix multiplication using multiple threads if available.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Tensor: Result of matrix multiplication
    """
    if not is_multithreading_supported() or a.device.device_type != DeviceType.WEBGPU:
        return a @ b
    
    try:
        from neurenix.binding import wasm_parallel_matmul
        return wasm_parallel_matmul(a, b)
    except (ImportError, AttributeError):
        return a @ b

def parallel_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, 
                   stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
                   dilation: Tuple[int, int] = (1, 1), groups: int = 1) -> Tensor:
    """
    Perform 2D convolution using multiple threads if available.
    
    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Weight tensor of shape (C_out, C_in/groups, H_kernel, W_kernel)
        bias: Optional bias tensor of shape (C_out)
        stride: Stride of the convolution
        padding: Padding added to all sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input to output channels
        
    Returns:
        Tensor: Result of convolution
    """
    if not is_multithreading_supported() or input.device.device_type != DeviceType.WEBGPU:
        from neurenix.nn.functional import conv2d
        return conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    try:
        from neurenix.binding import wasm_parallel_conv2d
        return wasm_parallel_conv2d(input, weight, bias, stride, padding, dilation, groups)
    except (ImportError, AttributeError):
        from neurenix.nn.functional import conv2d
        return conv2d(input, weight, bias, stride, padding, dilation, groups)

def parallel_map(func: Callable, tensors: List[Tensor]) -> List[Tensor]:
    """
    Apply a function to each tensor in parallel.
    
    Args:
        func: Function to apply
        tensors: List of tensors to process
        
    Returns:
        List of processed tensors
    """
    if not is_multithreading_supported():
        return [func(tensor) for tensor in tensors]
    
    try:
        pool = ThreadPool()
        return pool.map(func, tensors)
    except Exception:
        return [func(tensor) for tensor in tensors]

def parallel_batch_processing(model: Module, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple batches in parallel using a model.
    
    Args:
        model: Model to use for processing
        batches: List of input batches
        
    Returns:
        List of model outputs
    """
    if not is_multithreading_supported():
        return [model(batch) for batch in batches]
    
    try:
        pool = ThreadPool()
        return pool.map(model, batches)
    except Exception:
        return [model(batch) for batch in batches]
