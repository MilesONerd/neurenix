"""
Test script for verifying TPU support in the Neurenix framework.
"""

import numpy as np
from neurenix.device import DeviceType, Device
from neurenix.tensor import Tensor
from neurenix.core import set_config, get_config

def test_tpu_device_type():
    """Test that TPU device type is defined."""
    print("Testing TPU device type...")
    assert hasattr(DeviceType, "TPU")
    assert DeviceType.TPU.value == "tpu"
    print("✓ TPU device type is properly defined")

def test_tpu_device_creation():
    """Test TPU device creation."""
    print("Testing TPU device creation...")
    device = Device(DeviceType.TPU)
    assert device.type == DeviceType.TPU
    assert device.index == 0
    print("✓ TPU device creation works correctly")

def test_tpu_tensor_creation():
    """Test tensor creation on TPU."""
    print("Testing tensor creation on TPU...")
    try:
        tensor = Tensor(np.random.randn(2, 3), device=Device(DeviceType.TPU))
        print(f"✓ Successfully created tensor on TPU: {tensor}")
    except Exception as e:
        print(f"! TPU tensor creation failed (expected if no TPU available): {e}")

def test_tpu_configuration():
    """Test TPU configuration."""
    print("Testing TPU configuration...")
    set_config("tpu_visible_devices", "0")
    config = get_config()
    assert "tpu_visible_devices" in config
    assert config["tpu_visible_devices"] == "0"
    print("✓ TPU configuration works correctly")

def test_tpu_binding_functions():
    """Test TPU binding functions."""
    print("Testing TPU binding functions...")
    from neurenix.binding import is_tpu_available, TPU
    assert TPU == "tpu"
    # is_tpu_available() may return False if no TPU hardware is present
    print(f"✓ TPU availability: {is_tpu_available()}")

if __name__ == "__main__":
    print("Testing TPU support in Neurenix...")
    test_tpu_device_type()
    test_tpu_device_creation()
    test_tpu_tensor_creation()
    test_tpu_configuration()
    test_tpu_binding_functions()
    print("All TPU support tests completed!")
