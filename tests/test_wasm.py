"""
Tests for WebAssembly functionality in the Neurenix framework.
"""

import os
import tempfile
import unittest

import neurenix as nx
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.device import Device, DeviceType
from neurenix.wasm import export_to_wasm, run_in_browser

class TestWebAssembly(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        self.input = nx.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        
    def test_run_in_browser(self):
        """Test that run_in_browser works as a no-op in Python."""
        # This should be a no-op in Python
        output = run_in_browser(self.model, self.input)
        
        # Compare with normal execution
        expected = self.model(self.input)
        self.assertTrue(nx.allclose(output, expected))
        
    def test_run_in_browser_with_device(self):
        """Test run_in_browser with explicit device specification."""
        # Test with explicit WebGPU device
        device = Device(DeviceType.WEBGPU)
        output = run_in_browser(self.model, self.input, device=device)
        
        # Compare with normal execution
        expected = self.model(self.input)
        self.assertTrue(nx.allclose(output, expected))
        
    def test_export_to_wasm(self):
        """Test exporting a model for WebAssembly execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export the model
            export_path = export_to_wasm(self.model, temp_dir, "test_model")
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test_model.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test_model.js")))
            
            # Check for parameter files
            for name, _ in self.model.named_parameters():
                self.assertTrue(os.path.exists(os.path.join(temp_dir, f"{name}.bin")))
    
    def test_export_to_wasm_default_name(self):
        """Test exporting a model with default name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export the model with default name
            export_path = export_to_wasm(self.model, temp_dir)
            
            # Default name should be the class name
            model_name = self.model.__class__.__name__
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, f"{model_name}.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, f"{model_name}.js")))

if __name__ == "__main__":
    unittest.main()
