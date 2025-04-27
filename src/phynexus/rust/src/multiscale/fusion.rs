
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sigmoid};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct FeatureFusion {
    in_channels: usize,
    out_channels: usize,
}

impl FeatureFusion {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
        }
    }
    
    pub fn forward(&self, features: &[Tensor]) -> Tensor {
        if features.is_empty() {
            panic!("Empty feature list provided to FeatureFusion");
        }
        
        features[0].clone()
    }
}

pub struct ScaleFusion {
    fusion: FeatureFusion,
    fusion_mode: String,
    target_scale: String,
    projection: Option<Box<dyn Module>>,
    bn: Option<BatchNorm2d>,
    relu: Option<ReLU>,
}

impl ScaleFusion {
    pub fn new(in_channels: usize, out_channels: usize, fusion_mode: &str, target_scale: &str) -> Self {
        let fusion = FeatureFusion::new(in_channels, out_channels);
        
        let (projection, bn, relu) = if fusion_mode == "concat" {
            let proj = Conv2d::new(in_channels * 2, out_channels, 1, 1, 0, false);
            let bn = BatchNorm2d::new(out_channels);
            let relu = ReLU::new();
            (Some(Module::from(proj)), Some(bn), Some(relu))
        } else {
            (None, None, None)
        };
        
        Self {
            fusion,
            fusion_mode: fusion_mode.to_string(),
            target_scale: target_scale.to_string(),
            projection,
            bn,
            relu,
        }
    }
    
    pub fn forward(&self, features: &[Tensor]) -> Tensor {
        if features.is_empty() {
            panic!("Empty feature list provided to ScaleFusion");
        }
        
        let (target_h, target_w) = self.get_target_size(features);
        
        let mut resized_features = Vec::new();
        for feature in features {
            let shape = feature.shape();
            let h = shape[shape.len() - 2];
            let w = shape[shape.len() - 1];
            
            if h != target_h || w != target_w {
                let resized = feature.resize(&[target_h, target_w]);
                resized_features.push(resized);
            } else {
                resized_features.push(feature.clone());
            }
        }
        
        match self.fusion_mode.as_str() {
            "concat" => {
                let fused = Tensor::cat(&resized_features, 1);
                let projected = self.projection.as_ref().unwrap().forward(&fused);
                let normalized = self.bn.as_ref().unwrap().forward(&projected);
                self.relu.as_ref().unwrap().forward(&normalized)
            },
            "sum" => {
                let mut result = resized_features[0].clone();
                for feature in &resized_features[1..] {
                    result = result + feature;
                }
                result
            },
            "max" => {
                let mut result = resized_features[0].clone();
                for feature in &resized_features[1..] {
                    result = Tensor::maximum(&result, feature);
                }
                result
            },
            "avg" => {
                let mut sum = resized_features[0].clone();
                for feature in &resized_features[1..] {
                    sum = sum + feature;
                }
                sum / (resized_features.len() as f32)
            },
            _ => panic!("Unsupported fusion_mode: {}", self.fusion_mode),
        }
    }
    
    fn get_target_size(&self, features: &[Tensor]) -> (usize, usize) {
        match self.target_scale.as_str() {
            "largest" => {
                let mut max_h = 0;
                let mut max_w = 0;
                
                for feature in features {
                    let shape = feature.shape();
                    let h = shape[shape.len() - 2];
                    let w = shape[shape.len() - 1];
                    
                    max_h = std::cmp::max(max_h, h);
                    max_w = std::cmp::max(max_w, w);
                }
                
                (max_h, max_w)
            },
            "smallest" => {
                let mut min_h = std::usize::MAX;
                let mut min_w = std::usize::MAX;
                
                for feature in features {
                    let shape = feature.shape();
                    let h = shape[shape.len() - 2];
                    let w = shape[shape.len() - 1];
                    
                    min_h = std::cmp::min(min_h, h);
                    min_w = std::cmp::min(min_w, w);
                }
                
                (min_h, min_w)
            },
            _ => {
                if let Ok(idx) = self.target_scale.parse::<usize>() {
                    if idx >= features.len() {
                        panic!("Target scale index {} out of range for {} features", idx, features.len());
                    }
                    
                    let shape = features[idx].shape();
                    let h = shape[shape.len() - 2];
                    let w = shape[shape.len() - 1];
                    
                    (h, w)
                } else {
                    panic!("Unsupported target_scale: {}", self.target_scale);
                }
            }
        }
    }
}

pub struct AttentionFusion {
    fusion: FeatureFusion,
    num_scales: usize,
    attention_type: String,
    channel_attention: Vec<Box<dyn Module>>,
    spatial_attention: Vec<Box<dyn Module>>,
    projection: Box<dyn Module>,
    bn: BatchNorm2d,
    relu: ReLU,
}

impl AttentionFusion {
    pub fn new(in_channels: usize, out_channels: usize, num_scales: usize, attention_type: &str) -> Self {
        let fusion = FeatureFusion::new(in_channels, out_channels);
        
        let mut channel_attention = Vec::new();
        if attention_type == "channel" || attention_type == "both" {
            for _ in 0..num_scales {
                channel_attention.push(Self::create_channel_attention(in_channels));
            }
        }
        
        let mut spatial_attention = Vec::new();
        if attention_type == "spatial" || attention_type == "both" {
            for _ in 0..num_scales {
                spatial_attention.push(Self::create_spatial_attention());
            }
        }
        
        let projection = Conv2d::new(in_channels * num_scales, out_channels, 1, 1, 0, false);
        let bn = BatchNorm2d::new(out_channels);
        let relu = ReLU::new();
        
        Self {
            fusion,
            num_scales,
            attention_type: attention_type.to_string(),
            channel_attention,
            spatial_attention,
            projection: Box::new(projection) as Box<dyn Module>,
            bn,
            relu,
        }
    }
    
    fn create_channel_attention(channels: usize) -> Box<dyn Module> {
        let mut layers = Vec::new();
        
        layers.push(Conv2d::new(channels, channels / 16, 1, 1, 0, false));
        layers.push(ReLU::new());
        layers.push(Conv2d::new(channels / 16, channels, 1, 1, 0, false));
        layers.push(Sigmoid::new());
        
        Box::new(Module::from_layers(layers))
    }
    
    fn create_spatial_attention() -> Box<dyn Module> {
        let mut layers = Vec::new();
        
        layers.push(Conv2d::new(2, 1, 7, 1, 3, false));
        layers.push(Sigmoid::new());
        
        Box::new(Module::from_layers(layers))
    }
    
    pub fn forward(&self, features: &[Tensor]) -> Tensor {
        if features.len() != self.num_scales {
            panic!("Expected {} features, got {}", self.num_scales, features.len());
        }
        
        let mut attended_features = Vec::new();
        
        for (i, feature) in features.iter().enumerate() {
            let mut attended = feature.clone();
            
            if self.attention_type == "channel" || self.attention_type == "both" {
                let channel_avg = attended.mean(&[2, 3], true);
                let channel_weights = self.channel_attention[i].forward(&channel_avg);
                attended = attended * &channel_weights;
            }
            
            if self.attention_type == "spatial" || self.attention_type == "both" {
                let avg_pool = attended.mean(&[1], true);
                let max_pool = attended.max(&[1], true).0;
                let spatial_features = Tensor::cat(&[avg_pool, max_pool], 1);
                let spatial_weights = self.spatial_attention[i].forward(&spatial_features);
                attended = attended * &spatial_weights;
            }
            
            attended_features.push(attended);
        }
        
        let fused = Tensor::cat(&attended_features, 1);
        
        let projected = self.projection.forward(&fused);
        let normalized = self.bn.forward(&projected);
        self.relu.forward(&normalized)
    }
}

pub struct PyramidFusion {
    fusion: FeatureFusion,
    num_scales: usize,
    lateral_convs: Vec<Conv2d>,
    top_down_convs: Vec<Conv2d>,
}

impl PyramidFusion {
    pub fn new(in_channels: usize, out_channels: Option<usize>, num_scales: usize) -> Self {
        let out_channels = out_channels.unwrap_or(in_channels);
        let fusion = FeatureFusion::new(in_channels, out_channels);
        
        let mut lateral_convs = Vec::new();
        for _ in 0..num_scales {
            lateral_convs.push(Conv2d::new(in_channels, out_channels, 1, 1, 0, false));
        }
        
        let mut top_down_convs = Vec::new();
        for _ in 0..(num_scales - 1) {
            top_down_convs.push(Conv2d::new(out_channels, out_channels, 3, 1, 1, false));
        }
        
        Self {
            fusion,
            num_scales,
            lateral_convs,
            top_down_convs,
        }
    }
    
    pub fn forward(&self, features: &[Tensor]) -> Vec<Tensor> {
        if features.len() != self.num_scales {
            panic!("Expected {} features, got {}", self.num_scales, features.len());
        }
        
        let mut laterals = Vec::new();
        for (i, feature) in features.iter().enumerate() {
            laterals.push(Module::from(self.lateral_convs[i].clone()).forward(feature));
        }
        
        let mut outputs = vec![laterals[laterals.len() - 1].clone()];  // Start with the coarsest level
        
        for i in (0..(self.num_scales - 1)).rev() {
            let shape = laterals[i].shape();
            let h = shape[shape.len() - 2];
            let w = shape[shape.len() - 1];
            
            let upsampled = outputs[outputs.len() - 1].resize(&[h, w]);
            
            let merged = &laterals[i] + &upsampled;
            
            let refined = Module::from(self.top_down_convs[i].clone()).forward(&merged);
            
            outputs.push(refined);
        }
        
        outputs.reverse();
        
        outputs
    }
}

pub struct UNetDecoder {
    encoder_channels: Vec<usize>,
    output_channels: usize,
    num_scales: usize,
    up_blocks: Vec<Box<dyn Module>>,
    final_conv: Conv2d,
}

impl UNetDecoder {
    pub fn new(encoder_channels: Vec<usize>, output_channels: usize, num_scales: usize) -> Self {
        let mut up_blocks = Vec::new();
        
        for i in (1..num_scales).rev() {
            let in_ch = encoder_channels[i];
            let out_ch = encoder_channels[i-1];
            
            let mut block = Vec::new();
            
            block.push(ConvTranspose2d::new(in_ch, out_ch, 2, 2, 0, false));
            block.push(Conv2d::new(out_ch * 2, out_ch, 3, 1, 1, false));  // *2 for skip connection
            block.push(BatchNorm2d::new(out_ch));
            block.push(ReLU::new());
            block.push(Conv2d::new(out_ch, out_ch, 3, 1, 1, false));
            block.push(BatchNorm2d::new(out_ch));
            block.push(ReLU::new());
            
            up_blocks.push(Module::from_layers(block));
        }
        
        let final_conv = Conv2d::new(encoder_channels[0], output_channels, 1, 1, 0, false);
        
        Self {
            encoder_channels,
            output_channels,
            num_scales,
            up_blocks,
            final_conv,
        }
    }
    
    pub fn forward(&self, encoder_features: &[Tensor]) -> Tensor {
        if encoder_features.len() != self.num_scales {
            panic!("Expected {} features, got {}", self.num_scales, encoder_features.len());
        }
        
        let mut x = encoder_features[encoder_features.len() - 1].clone();
        
        for i in 0..(self.num_scales - 1) {
            let skip = &encoder_features[encoder_features.len() - i - 2];
            
            x = self.up_blocks[i].forward(&x);
            
            x = Tensor::cat(&[x, skip.clone()], 1);
        }
        
        Module::from(self.final_conv.clone()).forward(&x)
    }
}

pub fn register_fusion(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let feature_fusion = PyModule::new(py, "feature_fusion")?;
    m.add_submodule(&feature_fusion)?;
    
    let scale_fusion = PyModule::new(py, "scale_fusion")?;
    m.add_submodule(&scale_fusion)?;
    
    let attention_fusion = PyModule::new(py, "attention_fusion")?;
    m.add_submodule(&attention_fusion)?;
    
    let pyramid_fusion = PyModule::new(py, "pyramid_fusion")?;
    m.add_submodule(&pyramid_fusion)?;
    
    let unet_decoder = PyModule::new(py, "unet_decoder")?;
    m.add_submodule(&unet_decoder)?;
    
    Ok(())
}
