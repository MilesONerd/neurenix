"""
Hardware acceleration module for the Neurenix framework.

This module provides support for various hardware acceleration technologies:
- Vulkan
- OpenCL
- oneAPI
- DirectML
- oneDNN
- MKL-DNN
- TensorRT
"""

from .vulkan import VulkanBackend, is_vulkan_available
from .opencl import OpenCLBackend, is_opencl_available
from .oneapi import OneAPIBackend, is_oneapi_available
from .directml import DirectMLBackend, is_directml_available
from .onednn import OneDNNBackend, is_onednn_available
from .mkldnn import MKLDNNBackend, is_mkldnn_available
from .tensorrt import TensorRTBackend, is_tensorrt_available

__all__ = [
    'VulkanBackend',
    'is_vulkan_available',
    'OpenCLBackend',
    'is_opencl_available',
    'OneAPIBackend',
    'is_oneapi_available',
    'DirectMLBackend',
    'is_directml_available',
    'OneDNNBackend',
    'is_onednn_available',
    'MKLDNNBackend',
    'is_mkldnn_available',
    'TensorRTBackend',
    'is_tensorrt_available',
]
