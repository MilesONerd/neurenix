"""
Tests for hardware backends in Neurenix.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import neurenix
from neurenix.device import Device, DeviceType, get_available_devices, get_device_count


class TestHardwareBackends(unittest.TestCase):
    """Test hardware backends functionality."""

    def test_device_types(self):
        """Test that all device types are defined."""
        self.assertEqual(DeviceType.CPU.value, "cpu")
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.ROCM.value, "rocm")
        self.assertEqual(DeviceType.WEBGPU.value, "webgpu")
        self.assertEqual(DeviceType.TPU.value, "tpu")
        self.assertEqual(DeviceType.VULKAN.value, "vulkan")
        self.assertEqual(DeviceType.OPENCL.value, "opencl")
        self.assertEqual(DeviceType.ONEAPI.value, "oneapi")
        self.assertEqual(DeviceType.DIRECTML.value, "directml")
        self.assertEqual(DeviceType.ONEDNN.value, "onednn")
        self.assertEqual(DeviceType.MKLDNN.value, "mkldnn")
        self.assertEqual(DeviceType.TENSORRT.value, "tensorrt")

    def test_device_creation(self):
        """Test device creation for all device types."""
        cpu_device = Device(DeviceType.CPU)
        self.assertEqual(cpu_device.type, DeviceType.CPU)
        self.assertEqual(cpu_device.name, "CPU")

        for device_type in DeviceType:
            device = Device(device_type)
            self.assertEqual(device.type, device_type)
            
            if device_type == DeviceType.CPU:
                self.assertEqual(device.name, "CPU")
            else:
                self.assertEqual(device.name, f"{device_type.name.title()}:0")

    def test_get_device_count(self):
        """Test getting device count for all device types."""
        self.assertGreaterEqual(get_device_count(DeviceType.CPU), 1)
        
        for device_type in DeviceType:
            count = get_device_count(device_type)
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)

    def test_get_available_devices(self):
        """Test getting available devices."""
        devices = get_available_devices()
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)  # At least CPU should be available
        
        for device in devices:
            self.assertIsInstance(device, Device)
            
        cpu_found = False
        for device in devices:
            if device.type == DeviceType.CPU:
                cpu_found = True
                break
        self.assertTrue(cpu_found)


if __name__ == '__main__':
    unittest.main()
