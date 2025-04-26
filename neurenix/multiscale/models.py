"""
Multi-Scale Models for Neurenix.

This module provides implementations of neural network models that operate
on multiple scales or resolutions of input data simultaneously.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable

from neurenix.nn import Module, Conv2d, MaxPool2d, AvgPool2d, Linear, ReLU, BatchNorm2d
from neurenix.tensor import Tensor


class MultiScaleModel(Module):
    """
    Base class for multi-scale models.
    
    Multi-scale models process inputs at multiple resolutions or scales,
    allowing them to capture both fine-grained details and global context.
    """
    
    def __init__(self, 
                 input_channels: int,
                 num_scales: int = 3,
                 scale_factor: float = 0.5):
        """
        Initialize a multi-scale model.
        
        Args:
            input_channels: Number of input channels
            num_scales: Number of scales to process
            scale_factor: Factor by which to reduce dimensions at each scale
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        
        self.scale_branches = self._create_scale_branches()
        
        self.fusion = self._create_fusion_mechanism()
    
    def _create_scale_branches(self) -> List[Module]:
        """
        Create processing branches for each scale.
        
        Returns:
            List of modules, one for each scale
        """
        raise NotImplementedError("Subclasses must implement _create_scale_branches")
    
    def _create_fusion_mechanism(self) -> Module:
        """
        Create mechanism to fuse features from different scales.
        
        Returns:
            Module that fuses multi-scale features
        """
        raise NotImplementedError("Subclasses must implement _create_fusion_mechanism")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the multi-scale model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after multi-scale processing
        """
        scale_inputs = self._generate_multi_scale_inputs(x)
        
        scale_features = []
        for i, scale_input in enumerate(scale_inputs):
            scale_features.append(self.scale_branches[i](scale_input))
        
        output = self.fusion(scale_features)
        
        return output
    
    def _generate_multi_scale_inputs(self, x: Tensor) -> List[Tensor]:
        """
        Generate inputs at different scales.
        
        Args:
            x: Original input tensor
            
        Returns:
            List of tensors at different scales
        """
        scale_inputs = [x]  # Original scale
        
        current_input = x
        for i in range(1, self.num_scales):
            h, w = current_input.shape[-2:]
            new_h = int(h * self.scale_factor)
            new_w = int(w * self.scale_factor)
            
            pool = AvgPool2d((h // new_h, w // new_w))
            downsampled = pool(current_input)
            
            scale_inputs.append(downsampled)
            current_input = downsampled
        
        return scale_inputs


class PyramidNetwork(MultiScaleModel):
    """
    Pyramid Network for multi-scale feature extraction.
    
    Processes input at multiple scales and combines features in a pyramid-like structure.
    """
    
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 num_scales: int = 3,
                 scale_factor: float = 0.5):
        """
        Initialize a pyramid network.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: List of channel dimensions for each layer
            num_scales: Number of scales to process
            scale_factor: Factor by which to reduce dimensions at each scale
        """
        self.hidden_channels = hidden_channels
        super().__init__(input_channels, num_scales, scale_factor)
    
    def _create_scale_branches(self) -> List[Module]:
        """
        Create processing branches for each scale.
        
        Returns:
            List of modules, one for each scale
        """
        branches = []
        
        for i in range(self.num_scales):
            layers = []
            in_channels = self.input_channels
            
            for out_channels in self.hidden_channels:
                layers.extend([
                    Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    BatchNorm2d(out_channels),
                    ReLU()
                ])
                in_channels = out_channels
            
            branches.append(Module(layers))
        
        return branches
    
    def _create_fusion_mechanism(self) -> Module:
        """
        Create mechanism to fuse features from different scales.
        
        Returns:
            Module that fuses multi-scale features
        """
        from neurenix.multiscale.fusion import PyramidFusion
        return PyramidFusion(
            in_channels=self.hidden_channels[-1],
            num_scales=self.num_scales
        )


class UNet(MultiScaleModel):
    """
    U-Net architecture for multi-scale processing.
    
    Features an encoder-decoder structure with skip connections between
    corresponding encoder and decoder layers.
    """
    
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 output_channels: int,
                 num_scales: int = 4,
                 scale_factor: float = 0.5):
        """
        Initialize a U-Net model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: List of channel dimensions for encoder layers
            output_channels: Number of output channels
            num_scales: Number of scales (depth of U-Net)
            scale_factor: Factor by which to reduce dimensions at each scale
        """
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        super().__init__(input_channels, num_scales, scale_factor)
    
    def _create_scale_branches(self) -> List[Module]:
        """
        Create encoder branches for each scale.
        
        Returns:
            List of encoder modules, one for each scale
        """
        encoders = []
        
        for i in range(self.num_scales):
            layers = []
            in_channels = self.input_channels if i == 0 else self.hidden_channels[i-1]
            out_channels = self.hidden_channels[i]
            
            layers.extend([
                Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                BatchNorm2d(out_channels),
                ReLU(),
                Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                BatchNorm2d(out_channels),
                ReLU()
            ])
            
            encoders.append(Module(layers))
        
        return encoders
    
    def _create_fusion_mechanism(self) -> Module:
        """
        Create decoder with skip connections for feature fusion.
        
        Returns:
            Module that implements U-Net decoder with skip connections
        """
        from neurenix.multiscale.fusion import UNetDecoder
        return UNetDecoder(
            encoder_channels=self.hidden_channels,
            output_channels=self.output_channels,
            num_scales=self.num_scales
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after U-Net processing
        """
        encoder_features = []
        current = x
        
        for i in range(self.num_scales):
            current = self.scale_branches[i](current)
            encoder_features.append(current)
            
            if i < self.num_scales - 1:
                current = MaxPool2d(kernel_size=2)(current)
        
        output = self.fusion(encoder_features)
        
        return output
