"""
Tests for the Multi-Scale Models module in Neurenix.

This module tests the functionality of multi-scale models, pooling operations,
fusion mechanisms, and transformations.
"""

import unittest
import numpy as np

import neurenix
from neurenix.tensor import Tensor
from neurenix.nn import Module
from neurenix.multiscale import (
    MultiScaleModel, PyramidNetwork, UNet,
    MultiScalePooling, PyramidPooling, SpatialPyramidPooling,
    FeatureFusion, ScaleFusion, AttentionFusion,
    MultiScaleTransform, Rescale, PyramidDownsample
)


class TestMultiScaleModels(unittest.TestCase):
    """Test cases for multi-scale model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.input_tensor = Tensor.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_pyramid_network(self):
        """Test PyramidNetwork model."""
        hidden_channels = [16, 32, 64]
        num_scales = 3
        model = PyramidNetwork(
            input_channels=self.channels,
            hidden_channels=hidden_channels,
            num_scales=num_scales
        )
        
        output = model(self.input_tensor)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
    
    def test_unet(self):
        """Test UNet model."""
        hidden_channels = [16, 32, 64, 128]
        output_channels = 1
        model = UNet(
            input_channels=self.channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels
        )
        
        output = model(self.input_tensor)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], output_channels)
        self.assertEqual(output.shape[2], self.height)
        self.assertEqual(output.shape[3], self.width)


class TestMultiScalePooling(unittest.TestCase):
    """Test cases for multi-scale pooling operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.input_tensor = Tensor.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_pyramid_pooling(self):
        """Test PyramidPooling module."""
        pool_sizes = [1, 2, 3, 6]
        pooling = PyramidPooling(
            in_channels=self.channels,
            out_channels=self.channels,
            pool_sizes=pool_sizes
        )
        
        output = pooling(self.input_tensor)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.channels * (1 + len(pool_sizes)))
        self.assertEqual(output.shape[2], self.height)
        self.assertEqual(output.shape[3], self.width)
    
    def test_spatial_pyramid_pooling(self):
        """Test SpatialPyramidPooling module."""
        output_sizes = [1, 2, 4]
        pooling = SpatialPyramidPooling(output_sizes=output_sizes)
        
        output = pooling(self.input_tensor)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        expected_features = self.channels * sum([size * size for size in output_sizes])
        self.assertEqual(output.shape[1], expected_features)


class TestMultiScaleFusion(unittest.TestCase):
    """Test cases for multi-scale fusion mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.channels = 3
        self.scales = [
            Tensor.randn(self.batch_size, self.channels, 64, 64),
            Tensor.randn(self.batch_size, self.channels, 32, 32),
            Tensor.randn(self.batch_size, self.channels, 16, 16)
        ]
    
    def test_scale_fusion_concat(self):
        """Test ScaleFusion with concat mode."""
        fusion = ScaleFusion(
            in_channels=self.channels,
            out_channels=self.channels,
            fusion_mode="concat",
            target_scale="largest"
        )
        
        output = fusion(self.scales)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.channels)  # After projection
        self.assertEqual(output.shape[2], 64)  # Largest height
        self.assertEqual(output.shape[3], 64)  # Largest width
    
    def test_scale_fusion_sum(self):
        """Test ScaleFusion with sum mode."""
        fusion = ScaleFusion(
            in_channels=self.channels,
            out_channels=self.channels,
            fusion_mode="sum",
            target_scale="largest"
        )
        
        output = fusion(self.scales)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.channels)
        self.assertEqual(output.shape[2], 64)  # Largest height
        self.assertEqual(output.shape[3], 64)  # Largest width
    
    def test_attention_fusion(self):
        """Test AttentionFusion."""
        fusion = AttentionFusion(
            in_channels=self.channels,
            out_channels=self.channels,
            num_scales=len(self.scales),
            attention_type="both"
        )
        
        output = fusion(self.scales)
        
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.channels)


class TestMultiScaleTransforms(unittest.TestCase):
    """Test cases for multi-scale transformations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.input_tensor = Tensor.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_rescale(self):
        """Test Rescale transform."""
        scales = [0.5, 1.0, 2.0]
        transform = Rescale(scales=scales)
        
        outputs = transform(self.input_tensor)
        
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), len(scales))
        
        for i, scale in enumerate(scales):
            output = outputs[i]
            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[1], self.channels)
            
            if scale == 1.0:
                self.assertEqual(output.shape[2], self.height)
                self.assertEqual(output.shape[3], self.width)
            else:
                expected_h = int(self.height * scale)
                expected_w = int(self.width * scale)
                self.assertEqual(output.shape[2], expected_h)
                self.assertEqual(output.shape[3], expected_w)
    
    def test_pyramid_downsample(self):
        """Test PyramidDownsample transform."""
        num_levels = 3
        transform = PyramidDownsample(num_levels=num_levels)
        
        outputs = transform(self.input_tensor)
        
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), num_levels)
        
        for i in range(num_levels):
            output = outputs[i]
            self.assertIsInstance(output, Tensor)
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[1], self.channels)
            
            expected_h = self.height // (2 ** i)
            expected_w = self.width // (2 ** i)
            self.assertEqual(output.shape[2], expected_h)
            self.assertEqual(output.shape[3], expected_w)


if __name__ == '__main__':
    unittest.main()
