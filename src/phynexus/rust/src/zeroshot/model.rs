
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Linear, BatchNorm1d, Dropout, ReLU, Sequential};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub struct ZeroShotModel {
    visual_dim: usize,
    semantic_dim: usize,
    hidden_dim: usize,
    visual_embedding: Box<dyn Module>,
    semantic_embedding: Box<dyn Module>,
}

impl ZeroShotModel {
    pub fn new(
        visual_dim: usize,
        semantic_dim: usize,
        hidden_dim: usize,
        dropout: f32,
    ) -> Result<Self, PhynexusError> {
        let visual_embedding = Sequential::new(vec![
            Box::new(Linear::new(visual_dim, hidden_dim)?),
            Box::new(BatchNorm1d::new(hidden_dim)?),
            Box::new(ReLU::new()),
            Box::new(Dropout::new(dropout)),
            Box::new(Linear::new(hidden_dim, hidden_dim)?),
        ]);

        let semantic_embedding = Sequential::new(vec![
            Box::new(Linear::new(semantic_dim, hidden_dim)?),
            Box::new(BatchNorm1d::new(hidden_dim)?),
            Box::new(ReLU::new()),
            Box::new(Dropout::new(dropout)),
            Box::new(Linear::new(hidden_dim, hidden_dim)?),
        ]);

        Ok(Self {
            visual_dim,
            semantic_dim,
            hidden_dim,
            visual_embedding: Box::new(visual_embedding),
            semantic_embedding: Box::new(semantic_embedding),
        })
    }

    pub fn forward(
        &self,
        visual_features: &Tensor,
        semantic_features: Option<&Tensor>,
    ) -> Result<(Option<Tensor>, Tensor), PhynexusError> {
        let visual_embeddings = self.visual_embedding.forward(visual_features)?;

        if let Some(semantic_features) = semantic_features {
            let semantic_embeddings = self.semantic_embedding.forward(semantic_features)?;

            let compatibility = cosine_similarity(&visual_embeddings, &semantic_embeddings)?;

            Ok((Some(compatibility), visual_embeddings))
        } else {
            Ok((None, visual_embeddings))
        }
    }

    pub fn predict(
        &self,
        visual_features: &Tensor,
        semantic_features: &Tensor,
    ) -> Result<Tensor, PhynexusError> {
        let (compatibility, _) = self.forward(visual_features, Some(semantic_features))?;
        let predictions = compatibility.unwrap().argmax(1)?;
        Ok(predictions)
    }
}

pub struct ZeroShotTransformer {
    base_model: ZeroShotModel,
    visual_transformer: Box<dyn Module>,
    semantic_transformer: Box<dyn Module>,
}

impl ZeroShotTransformer {
    pub fn new(
        visual_dim: usize,
        semantic_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout: f32,
    ) -> Result<Self, PhynexusError> {
        let base_model = ZeroShotModel::new(visual_dim, semantic_dim, hidden_dim, dropout)?;

        let encoder_layer = TransformerEncoderLayer::new(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
        )?;
        let visual_transformer = TransformerEncoder::new(encoder_layer, num_layers)?;

        let encoder_layer = TransformerEncoderLayer::new(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
        )?;
        let semantic_transformer = TransformerEncoder::new(encoder_layer, num_layers)?;

        Ok(Self {
            base_model,
            visual_transformer: Box::new(visual_transformer),
            semantic_transformer: Box::new(semantic_transformer),
        })
    }

    pub fn forward(
        &self,
        visual_features: &Tensor,
        semantic_features: Option<&Tensor>,
    ) -> Result<(Option<Tensor>, Tensor), PhynexusError> {
        let (_, visual_embeddings) = self.base_model.forward(visual_features, None)?;

        let visual_embeddings = visual_embeddings.unsqueeze(1)?;
        let visual_embeddings = self.visual_transformer.forward(&visual_embeddings)?;
        let visual_embeddings = visual_embeddings.squeeze(1)?;

        if let Some(semantic_features) = semantic_features {
            let (_, semantic_embeddings) = self.base_model.forward(semantic_features, None)?;

            let semantic_embeddings = semantic_embeddings.unsqueeze(1)?;
            let semantic_embeddings = self.semantic_transformer.forward(&semantic_embeddings)?;
            let semantic_embeddings = semantic_embeddings.squeeze(1)?;

            let compatibility = cosine_similarity(&visual_embeddings, &semantic_embeddings)?;

            Ok((Some(compatibility), visual_embeddings))
        } else {
            Ok((None, visual_embeddings))
        }
    }

    pub fn predict(
        &self,
        visual_features: &Tensor,
        semantic_features: &Tensor,
    ) -> Result<Tensor, PhynexusError> {
        let (compatibility, _) = self.forward(visual_features, Some(semantic_features))?;
        let predictions = compatibility.unwrap().argmax(1)?;
        Ok(predictions)
    }
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor, PhynexusError> {
    let a_norm = a.norm(2, 1, true)?;
    let b_norm = b.norm(2, 1, true)?;

    let a_normalized = a.div(&a_norm.add_scalar(1e-10)?)?;
    let b_normalized = b.div(&b_norm.add_scalar(1e-10)?)?;

    let a_unsqueezed = a_normalized.unsqueeze(1)?;
    let b_unsqueezed = b_normalized.unsqueeze(0)?;
    let dot_product = a_unsqueezed.bmm(&b_unsqueezed.transpose(1, 2)?)?;

    Ok(dot_product.squeeze(1)?)
}

#[pymodule]
fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyZeroShotModel>()?;
    m.add_class::<PyZeroShotTransformer>()?;
    Ok(())
}

pub fn register_models(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "model")?;
    models(py, submodule)?;
    m.add_submodule(&submodule)?;
    Ok(())
}

#[pyclass]
struct PyZeroShotModel {
    inner: ZeroShotModel,
}

#[pymethods]
impl PyZeroShotModel {
    #[new]
    fn new(
        visual_dim: usize,
        semantic_dim: usize,
        hidden_dim: usize,
        dropout: f32,
    ) -> PyResult<Self> {
        let inner = ZeroShotModel::new(visual_dim, semantic_dim, hidden_dim, dropout)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn forward(
        &self,
        visual_features: &PyAny,
        semantic_features: Option<&PyAny>,
    ) -> PyResult<(Option<PyObject>, PyObject)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let semantic_tensor = if let Some(semantic_features) = semantic_features {
            Some(Tensor::from_pyany(semantic_features)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };

        let (compatibility, visual_embeddings) = self.inner.forward(
            &visual_tensor,
            semantic_tensor.as_ref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let gil = Python::acquire_gil();
        let py = gil.python();

        let visual_embeddings_py = visual_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let compatibility_py = if let Some(compatibility) = compatibility {
            Some(compatibility.to_pyobject(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };

        Ok((compatibility_py, visual_embeddings_py))
    }

    fn predict(
        &self,
        visual_features: &PyAny,
        semantic_features: &PyAny,
    ) -> PyResult<PyObject> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let semantic_tensor = Tensor::from_pyany(semantic_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let predictions = self.inner.predict(&visual_tensor, &semantic_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let gil = Python::acquire_gil();
        let py = gil.python();

        let predictions_py = predictions.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions_py)
    }
}

#[pyclass]
struct PyZeroShotTransformer {
    inner: ZeroShotTransformer,
}

#[pymethods]
impl PyZeroShotTransformer {
    #[new]
    fn new(
        visual_dim: usize,
        semantic_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout: f32,
    ) -> PyResult<Self> {
        let inner = ZeroShotTransformer::new(
            visual_dim,
            semantic_dim,
            hidden_dim,
            num_heads,
            num_layers,
            dropout,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn forward(
        &self,
        visual_features: &PyAny,
        semantic_features: Option<&PyAny>,
    ) -> PyResult<(Option<PyObject>, PyObject)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let semantic_tensor = if let Some(semantic_features) = semantic_features {
            Some(Tensor::from_pyany(semantic_features)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };

        let (compatibility, visual_embeddings) = self.inner.forward(
            &visual_tensor,
            semantic_tensor.as_ref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let gil = Python::acquire_gil();
        let py = gil.python();

        let visual_embeddings_py = visual_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let compatibility_py = if let Some(compatibility) = compatibility {
            Some(compatibility.to_pyobject(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };

        Ok((compatibility_py, visual_embeddings_py))
    }

    fn predict(
        &self,
        visual_features: &PyAny,
        semantic_features: &PyAny,
    ) -> PyResult<PyObject> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let semantic_tensor = Tensor::from_pyany(semantic_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let predictions = self.inner.predict(&visual_tensor, &semantic_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let gil = Python::acquire_gil();
        let py = gil.python();

        let predictions_py = predictions.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions_py)
    }
}
