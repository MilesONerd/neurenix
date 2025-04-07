"""
WebAssembly module for the Neurenix framework.

This module provides functionality for running Neurenix models in the browser
using WebAssembly.
"""

from .browser import run_in_browser, export_to_wasm
from .multithreaded import (
    is_multithreading_supported, 
    enable_multithreading, 
    get_num_workers,
    parallel_matmul,
    parallel_conv2d,
    parallel_map,
    parallel_batch_processing,
    ThreadPool
)

__all__ = [
    'run_in_browser',
    'export_to_wasm',
    'is_multithreading_supported',
    'enable_multithreading',
    'get_num_workers',
    'parallel_matmul',
    'parallel_conv2d',
    'parallel_map',
    'parallel_batch_processing',
    'ThreadPool'
]
