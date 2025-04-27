
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Linear, BatchNorm1d, Dropout, ReLU, Sequential, Embedding as EmbeddingLayer};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub struct EmbeddingModel {
    input_dim: usize,
    embedding_dim: usize,
    normalize: bool,
    embedding_network: Box<dyn Module>,
}

impl EmbeddingModel {
    pub fn new(
        input_dim: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        dropout: f32,
        normalize: bool,
    ) -> Result<Self, PhynexusError> {
        let hidden_dims = hidden_dims.unwrap_or_else(|| vec![input_dim * 2, input_dim]);
        
        let mut layers: Vec<Box<dyn Module>> = Vec::new();
        let mut prev_dim = input_dim;
        
        for hidden_dim in hidden_dims {
            layers.push(Box::new(Linear::new(prev_dim, hidden_dim)?));
            layers.push(Box::new(BatchNorm1d::new(hidden_dim)?));
            layers.push(Box::new(ReLU::new()));
            layers.push(Box::new(Dropout::new(dropout)));
            prev_dim = hidden_dim;
        }
        
        layers.push(Box::new(Linear::new(prev_dim, embedding_dim)?));
        
        let embedding_network = Sequential::new(layers);
        
        Ok(Self {
            input_dim,
            embedding_dim,
            normalize,
            embedding_network: Box::new(embedding_network),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, PhynexusError> {
        let mut embeddings = self.embedding_network.forward(x)?;
        
        if self.normalize {
            let norm = embeddings.norm(2, 1, true)?;
            embeddings = embeddings.div(&norm.add_scalar(1e-10)?)?;
        }
        
        Ok(embeddings)
    }
}

pub struct TextEncoder {
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_length: usize,
    word_embedding: EmbeddingLayer,
    embedding_model: EmbeddingModel,
}

impl TextEncoder {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        max_seq_length: usize,
        dropout: f32,
        normalize: bool,
    ) -> Result<Self, PhynexusError> {
        let word_embedding = EmbeddingLayer::new(vocab_size, embedding_dim)?;
        
        let embedding_model = EmbeddingModel::new(
            embedding_dim,
            embedding_dim,
            hidden_dims,
            dropout,
            normalize,
        )?;
        
        Ok(Self {
            vocab_size,
            embedding_dim,
            max_seq_length,
            word_embedding,
            embedding_model,
        })
    }
    
    pub fn forward(&self, tokens: &Tensor, mask: Option<&Tensor>) -> Result<Tensor, PhynexusError> {
        let word_embeddings = self.word_embedding.forward(tokens)?;
        
        let text_features = if let Some(mask) = mask {
            let mask_expanded = mask.unsqueeze(-1)?;
            let masked_embeddings = word_embeddings.mul(&mask_expanded)?;
            
            let sum = masked_embeddings.sum(1)?;
            let mask_sum = mask.sum(1, true)?;
            sum.div(&mask_sum.add_scalar(1e-10)?)?
        } else {
            word_embeddings.mean(1)?
        };
        
        self.embedding_model.forward(&text_features)
    }
}

pub struct ImageEncoder {
    input_channels: usize,
    embedding_dim: usize,
    backbone_name: String,
    backbone: Box<dyn Module>,
    embedding_model: EmbeddingModel,
}

impl ImageEncoder {
    pub fn new(
        input_channels: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        backbone: &str,
        pretrained: bool,
        dropout: f32,
        normalize: bool,
    ) -> Result<Self, PhynexusError> {
        let (backbone, backbone_dim) = match backbone {
            "resnet50" => {
                let backbone = crate::nn::models::resnet50(pretrained)?;
                (Box::new(backbone) as Box<dyn Module>, 2048)
            },
            "vit" => {
                let backbone = crate::nn::models::vit_base(pretrained)?;
                (Box::new(backbone) as Box<dyn Module>, 768)
            },
            _ => return Err(PhynexusError::InvalidArgument(format!("Unsupported backbone: {}", backbone))),
        };
        
        let embedding_model = EmbeddingModel::new(
            backbone_dim,
            embedding_dim,
            hidden_dims,
            dropout,
            normalize,
        )?;
        
        Ok(Self {
            input_channels,
            embedding_dim,
            backbone_name: backbone.to_string(),
            backbone,
            embedding_model,
        })
    }
    
    pub fn forward(&self, images: &Tensor) -> Result<Tensor, PhynexusError> {
        let features = self.backbone.forward(images)?;
        
        self.embedding_model.forward(&features)
    }
}

pub struct CrossModalEncoder {
    image_encoder: ImageEncoder,
    text_encoder: TextEncoder,
    projection_dim: usize,
    temperature: f32,
    image_projection: Linear,
    text_projection: Linear,
}

impl CrossModalEncoder {
    pub fn new(
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        projection_dim: usize,
        temperature: f32,
    ) -> Result<Self, PhynexusError> {
        let image_projection = Linear::new(
            image_encoder.embedding_dim,
            projection_dim,
        )?;
        
        let text_projection = Linear::new(
            text_encoder.embedding_dim,
            projection_dim,
        )?;
        
        Ok(Self {
            image_encoder,
            text_encoder,
            projection_dim,
            temperature,
            image_projection,
            text_projection,
        })
    }
    
    pub fn encode_image(&self, images: &Tensor) -> Result<Tensor, PhynexusError> {
        let image_features = self.image_encoder.forward(images)?;
        let image_embeddings = self.image_projection.forward(&image_features)?;
        
        let norm = image_embeddings.norm(2, 1, true)?;
        let image_embeddings = image_embeddings.div(&norm.add_scalar(1e-10)?)?;
        
        Ok(image_embeddings)
    }
    
    pub fn encode_text(&self, tokens: &Tensor, mask: Option<&Tensor>) -> Result<Tensor, PhynexusError> {
        let text_features = self.text_encoder.forward(tokens, mask)?;
        let text_embeddings = self.text_projection.forward(&text_features)?;
        
        let norm = text_embeddings.norm(2, 1, true)?;
        let text_embeddings = text_embeddings.div(&norm.add_scalar(1e-10)?)?;
        
        Ok(text_embeddings)
    }
    
    pub fn forward(
        &self,
        images: &Tensor,
        tokens: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor), PhynexusError> {
        let image_embeddings = self.encode_image(images)?;
        let text_embeddings = self.encode_text(tokens, mask)?;
        
        let logits = image_embeddings.matmul(&text_embeddings.t())?;
        let logits = logits.div_scalar(self.temperature)?;
        
        Ok((image_embeddings, text_embeddings, logits))
    }
}

#[pymodule]
fn embeddings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEmbeddingModel>()?;
    m.add_class::<PyTextEncoder>()?;
    m.add_class::<PyImageEncoder>()?;
    m.add_class::<PyCrossModalEncoder>()?;
    Ok(())
}

pub fn register_embeddings(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "embedding")?;
    embeddings(py, submodule)?;
    m.add_submodule(&submodule)?;
    Ok(())
}

#[pyclass]
struct PyEmbeddingModel {
    inner: EmbeddingModel,
}

#[pymethods]
impl PyEmbeddingModel {
    #[new]
    fn new(
        input_dim: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        dropout: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let inner = EmbeddingModel::new(input_dim, embedding_dim, hidden_dims, dropout, normalize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
    
    fn forward(&self, x: &PyAny) -> PyResult<PyObject> {
        let tensor = Tensor::from_pyany(x)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let embeddings = self.inner.forward(&tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let embeddings_py = embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(embeddings_py)
    }
}

#[pyclass]
struct PyTextEncoder {
    inner: TextEncoder,
}

#[pymethods]
impl PyTextEncoder {
    #[new]
    fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        max_seq_length: usize,
        dropout: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let inner = TextEncoder::new(
            vocab_size,
            embedding_dim,
            hidden_dims,
            max_seq_length,
            dropout,
            normalize,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
    
    fn forward(&self, tokens: &PyAny, mask: Option<&PyAny>) -> PyResult<PyObject> {
        let tokens_tensor = Tensor::from_pyany(tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let mask_tensor = if let Some(mask) = mask {
            Some(Tensor::from_pyany(mask)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let embeddings = self.inner.forward(&tokens_tensor, mask_tensor.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let embeddings_py = embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(embeddings_py)
    }
}

#[pyclass]
struct PyImageEncoder {
    inner: ImageEncoder,
}

#[pymethods]
impl PyImageEncoder {
    #[new]
    fn new(
        input_channels: usize,
        embedding_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        backbone: &str,
        pretrained: bool,
        dropout: f32,
        normalize: bool,
    ) -> PyResult<Self> {
        let inner = ImageEncoder::new(
            input_channels,
            embedding_dim,
            hidden_dims,
            backbone,
            pretrained,
            dropout,
            normalize,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
    
    fn forward(&self, images: &PyAny) -> PyResult<PyObject> {
        let images_tensor = Tensor::from_pyany(images)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let embeddings = self.inner.forward(&images_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let embeddings_py = embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(embeddings_py)
    }
}

#[pyclass]
struct PyCrossModalEncoder {
    inner: CrossModalEncoder,
}

#[pymethods]
impl PyCrossModalEncoder {
    #[new]
    fn new(
        image_encoder: &PyImageEncoder,
        text_encoder: &PyTextEncoder,
        projection_dim: usize,
        temperature: f32,
    ) -> PyResult<Self> {
        let inner = CrossModalEncoder::new(
            image_encoder.inner.clone(),
            text_encoder.inner.clone(),
            projection_dim,
            temperature,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
    
    fn encode_image(&self, images: &PyAny) -> PyResult<PyObject> {
        let images_tensor = Tensor::from_pyany(images)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let embeddings = self.inner.encode_image(&images_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let embeddings_py = embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(embeddings_py)
    }
    
    fn encode_text(&self, tokens: &PyAny, mask: Option<&PyAny>) -> PyResult<PyObject> {
        let tokens_tensor = Tensor::from_pyany(tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let mask_tensor = if let Some(mask) = mask {
            Some(Tensor::from_pyany(mask)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let embeddings = self.inner.encode_text(&tokens_tensor, mask_tensor.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let embeddings_py = embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(embeddings_py)
    }
    
    fn forward(
        &self,
        images: &PyAny,
        tokens: &PyAny,
        mask: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject, PyObject)> {
        let images_tensor = Tensor::from_pyany(images)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let tokens_tensor = Tensor::from_pyany(tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let mask_tensor = if let Some(mask) = mask {
            Some(Tensor::from_pyany(mask)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let (image_embeddings, text_embeddings, logits) = self.inner.forward(
            &images_tensor,
            &tokens_tensor,
            mask_tensor.as_ref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let image_embeddings_py = image_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let text_embeddings_py = text_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let logits_py = logits.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((image_embeddings_py, text_embeddings_py, logits_py))
    }
}
