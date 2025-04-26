"""
Multi-Scale Fusion operations for Neurenix.

This module provides mechanisms for fusing features from multiple scales
or resolutions, enabling more effective multi-scale model architectures.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable

from neurenix.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sigmoid
from neurenix.tensor import Tensor


class FeatureFusion(Module):
    """
    Base class for feature fusion operations.
    
    Feature fusion combines features from multiple scales or sources,
    allowing models to leverage complementary information.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int):
        """
        Initialize a feature fusion module.
        
        Args:
            in_channels: Number of input channels per feature
            out_channels: Number of output channels after fusion
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Forward pass through the feature fusion module.
        
        Args:
            features: List of feature tensors to fuse
            
        Returns:
            Fused feature tensor
        """
        raise NotImplementedError("Subclasses must implement forward")


class ScaleFusion(FeatureFusion):
    """
    Scale Fusion module for combining features from different scales.
    
    Resizes all features to a common scale and combines them through
    element-wise operations or concatenation.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 fusion_mode: str = 'concat',
                 target_scale: str = 'largest'):
        """
        Initialize a scale fusion module.
        
        Args:
            in_channels: Number of input channels per feature
            out_channels: Number of output channels after fusion
            fusion_mode: How to fuse features ('concat', 'sum', 'max', 'avg')
            target_scale: Which scale to resize features to ('largest', 'smallest', or index)
        """
        super().__init__(in_channels, out_channels)
        self.fusion_mode = fusion_mode
        self.target_scale = target_scale
        
        if fusion_mode == 'concat':
            self.projection = Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
            self.bn = BatchNorm2d(out_channels)
            self.relu = ReLU()
    
    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Forward pass through the scale fusion module.
        
        Args:
            features: List of feature tensors from different scales
            
        Returns:
            Fused feature tensor
        """
        if not features:
            raise ValueError("Empty feature list provided to ScaleFusion")
        
        target_h, target_w = self._get_target_size(features)
        
        resized_features = []
        for feature in features:
            if feature.shape[2:] != (target_h, target_w):
                resized = feature.resize((target_h, target_w))
            else:
                resized = feature
            resized_features.append(resized)
        
        if self.fusion_mode == 'concat':
            fused = Tensor.cat(resized_features, dim=1)
            return self.relu(self.bn(self.projection(fused)))
        elif self.fusion_mode == 'sum':
            return sum(resized_features)
        elif self.fusion_mode == 'max':
            fused = resized_features[0]
            for feature in resized_features[1:]:
                fused = Tensor.maximum(fused, feature)
            return fused
        elif self.fusion_mode == 'avg':
            return sum(resized_features) / len(resized_features)
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
    
    def _get_target_size(self, features: List[Tensor]) -> Tuple[int, int]:
        """
        Determine the target size for feature resizing.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Target height and width
        """
        if self.target_scale == 'largest':
            max_h = max(f.shape[2] for f in features)
            max_w = max(f.shape[3] for f in features)
            return max_h, max_w
        elif self.target_scale == 'smallest':
            min_h = min(f.shape[2] for f in features)
            min_w = min(f.shape[3] for f in features)
            return min_h, min_w
        elif isinstance(self.target_scale, int):
            idx = self.target_scale
            if idx < 0 or idx >= len(features):
                raise IndexError(f"Target scale index {idx} out of range for {len(features)} features")
            return features[idx].shape[2], features[idx].shape[3]
        else:
            raise ValueError(f"Unsupported target_scale: {self.target_scale}")


class AttentionFusion(FeatureFusion):
    """
    Attention-based Fusion module for combining features from different scales.
    
    Uses attention mechanisms to weight the importance of different features
    before combining them.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_scales: int = 2,
                 attention_type: str = 'channel'):
        """
        Initialize an attention fusion module.
        
        Args:
            in_channels: Number of input channels per feature
            out_channels: Number of output channels after fusion
            num_scales: Number of scales to fuse
            attention_type: Type of attention ('channel', 'spatial', 'both')
        """
        super().__init__(in_channels, out_channels)
        self.num_scales = num_scales
        self.attention_type = attention_type
        
        if attention_type in ['channel', 'both']:
            self.channel_attention = []
            for i in range(num_scales):
                self.channel_attention.append(self._create_channel_attention(in_channels))
        
        if attention_type in ['spatial', 'both']:
            self.spatial_attention = []
            for i in range(num_scales):
                self.spatial_attention.append(self._create_spatial_attention())
        
        self.projection = Conv2d(in_channels * num_scales, out_channels, kernel_size=1, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.relu = ReLU()
    
    def _create_channel_attention(self, channels: int) -> Module:
        """
        Create a channel attention module.
        
        Args:
            channels: Number of input channels
            
        Returns:
            Channel attention module
        """
        return Module([
            Conv2d(channels, channels // 16, kernel_size=1),
            ReLU(),
            Conv2d(channels // 16, channels, kernel_size=1),
            Sigmoid()
        ])
    
    def _create_spatial_attention(self) -> Module:
        """
        Create a spatial attention module.
        
        Returns:
            Spatial attention module
        """
        return Module([
            Conv2d(2, 1, kernel_size=7, padding=3),
            Sigmoid()
        ])
    
    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Forward pass through the attention fusion module.
        
        Args:
            features: List of feature tensors from different scales
            
        Returns:
            Fused feature tensor
        """
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} features, got {len(features)}")
        
        attended_features = []
        
        for i, feature in enumerate(features):
            attended = feature
            
            if self.attention_type in ['channel', 'both']:
                channel_avg = attended.mean(dim=(2, 3), keepdim=True)
                channel_weights = self.channel_attention[i](channel_avg)
                attended = attended * channel_weights
            
            if self.attention_type in ['spatial', 'both']:
                avg_pool = attended.mean(dim=1, keepdim=True)
                max_pool, _ = attended.max(dim=1, keepdim=True)
                spatial_features = Tensor.cat([avg_pool, max_pool], dim=1)
                spatial_weights = self.spatial_attention[i](spatial_features)
                attended = attended * spatial_weights
            
            attended_features.append(attended)
        
        fused = Tensor.cat(attended_features, dim=1)
        
        return self.relu(self.bn(self.projection(fused)))


class PyramidFusion(FeatureFusion):
    """
    Pyramid Fusion module for feature pyramid networks.
    
    Implements top-down pathway with lateral connections as used in FPN.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int = None,
                 num_scales: int = 4):
        """
        Initialize a pyramid fusion module.
        
        Args:
            in_channels: Number of input channels per feature
            out_channels: Number of output channels after fusion (defaults to in_channels)
            num_scales: Number of scales in the pyramid
        """
        out_channels = out_channels or in_channels
        super().__init__(in_channels, out_channels)
        self.num_scales = num_scales
        
        self.lateral_convs = []
        for i in range(num_scales):
            self.lateral_convs.append(Conv2d(in_channels, out_channels, kernel_size=1))
        
        self.top_down_convs = []
        for i in range(num_scales - 1):
            self.top_down_convs.append(Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Forward pass through the pyramid fusion module.
        
        Args:
            features: List of feature tensors from different scales (coarsest to finest)
            
        Returns:
            List of fused feature tensors at each scale
        """
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} features, got {len(features)}")
        
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        
        outputs = [laterals[-1]]  # Start with the coarsest level
        
        for i in range(self.num_scales - 2, -1, -1):
            upsampled = outputs[-1].resize(laterals[i].shape[2:])
            merged = laterals[i] + upsampled
            refined = self.top_down_convs[i](merged)
            outputs.append(refined)
        
        return outputs[::-1]


class UNetDecoder(Module):
    """
    U-Net decoder with skip connections.
    
    Implements the decoder path of a U-Net with skip connections from the encoder.
    """
    
    def __init__(self, 
                 encoder_channels: List[int],
                 output_channels: int,
                 num_scales: int = 4):
        """
        Initialize a U-Net decoder.
        
        Args:
            encoder_channels: List of channel dimensions for each encoder layer
            output_channels: Number of output channels
            num_scales: Number of scales (depth of U-Net)
        """
        super().__init__()
        self.encoder_channels = encoder_channels
        self.output_channels = output_channels
        self.num_scales = num_scales
        
        self.up_blocks = []
        
        for i in range(num_scales - 1, 0, -1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i-1]
            
            block = Module([
                ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),  # *2 for skip connection
                BatchNorm2d(out_ch),
                ReLU(),
                Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                BatchNorm2d(out_ch),
                ReLU()
            ])
            
            self.up_blocks.append(block)
        
        self.final = Conv2d(encoder_channels[0], output_channels, kernel_size=1)
    
    def forward(self, encoder_features: List[Tensor]) -> Tensor:
        """
        Forward pass through the U-Net decoder.
        
        Args:
            encoder_features: List of feature tensors from the encoder (finest to coarsest)
            
        Returns:
            Output tensor after U-Net decoding
        """
        if len(encoder_features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} features, got {len(encoder_features)}")
        
        x = encoder_features[-1]
        
        for i in range(self.num_scales - 1):
            skip = encoder_features[-(i+2)]
            
            x = self.up_blocks[i](x)
            
            x = Tensor.cat([x, skip], dim=1)
        
        return self.final(x)
