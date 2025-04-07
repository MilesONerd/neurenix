
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use super::model::ZeroShotModel;

pub struct ZeroShotClassifier {
    zero_shot_model: ZeroShotModel,
    class_names: Option<Vec<String>>,
    class_embeddings: Option<Tensor>,
}

impl ZeroShotClassifier {
    pub fn new(
        zero_shot_model: ZeroShotModel,
        class_names: Option<Vec<String>>,
        class_embeddings: Option<Tensor>,
    ) -> Self {
        Self {
            zero_shot_model,
            class_names,
            class_embeddings,
        }
    }
    
    pub fn forward(&self, visual_features: &Tensor) -> Result<(Tensor, Tensor), PhynexusError> {
        if self.class_embeddings.is_none() {
            return Err(PhynexusError::InvalidArgument(
                "Class embeddings must be set before inference".to_string()
            ));
        }
        
        let (compatibility, visual_embeddings) = self.zero_shot_model.forward(
            visual_features,
            self.class_embeddings.as_ref(),
        )?;
        
        Ok((compatibility.unwrap(), visual_embeddings))
    }
    
    pub fn predict(&self, visual_features: &Tensor) -> Result<(Tensor, Option<Vec<String>>), PhynexusError> {
        let (logits, _) = self.forward(visual_features)?;
        let predictions = logits.argmax(1)?;
        
        if let Some(class_names) = &self.class_names {
            let predicted_classes = predictions
                .to_vec1::<i64>()?
                .iter()
                .map(|&idx| class_names[idx as usize].clone())
                .collect();
            
            Ok((predictions, Some(predicted_classes)))
        } else {
            Ok((predictions, None))
        }
    }
    
    pub fn set_class_embeddings(
        &mut self,
        class_embeddings: Tensor,
        class_names: Option<Vec<String>>,
    ) {
        self.class_embeddings = Some(class_embeddings);
        if let Some(names) = class_names {
            self.class_names = Some(names);
        }
    }
}

pub struct AttributeClassifier {
    classifier: ZeroShotClassifier,
    attribute_names: Option<Vec<String>>,
    class_attribute_matrix: Option<Tensor>,
}

impl AttributeClassifier {
    pub fn new(
        zero_shot_model: ZeroShotModel,
        class_names: Option<Vec<String>>,
        attribute_names: Option<Vec<String>>,
        class_attribute_matrix: Option<Tensor>,
    ) -> Self {
        let mut classifier = ZeroShotClassifier::new(
            zero_shot_model,
            class_names,
            None,
        );
        
        if let Some(matrix) = &class_attribute_matrix {
            classifier.set_class_embeddings(matrix.clone(), None);
        }
        
        Self {
            classifier,
            attribute_names,
            class_attribute_matrix,
        }
    }
    
    pub fn forward(&self, visual_features: &Tensor) -> Result<(Tensor, Tensor), PhynexusError> {
        self.classifier.forward(visual_features)
    }
    
    pub fn predict(&self, visual_features: &Tensor) -> Result<(Tensor, Option<Vec<String>>), PhynexusError> {
        self.classifier.predict(visual_features)
    }
    
    pub fn set_class_attributes(
        &mut self,
        class_attribute_matrix: Tensor,
        class_names: Option<Vec<String>>,
        attribute_names: Option<Vec<String>>,
    ) {
        self.class_attribute_matrix = Some(class_attribute_matrix.clone());
        self.classifier.set_class_embeddings(class_attribute_matrix, class_names);
        
        if let Some(names) = attribute_names {
            self.attribute_names = Some(names);
        }
    }
    
    pub fn predict_attributes(
        &self,
        visual_features: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Vec<String>>>), PhynexusError> {
        let (_, visual_embeddings) = self.zero_shot_model.forward(visual_features, None)?;
        
        let num_attributes = self.class_attribute_matrix.as_ref().unwrap().size(1)?;
        let attribute_embeddings = Tensor::eye(num_attributes)?;
        
        let visual_embeddings_unsqueezed = visual_embeddings.unsqueeze(1)?;
        let attribute_embeddings_unsqueezed = attribute_embeddings.unsqueeze(0)?;
        
        let attribute_scores = cosine_similarity(
            &visual_embeddings_unsqueezed,
            &attribute_embeddings_unsqueezed,
        )?;
        
        let predicted_attributes = attribute_scores.gt_scalar(0.5)?;
        
        if let Some(attribute_names) = &self.attribute_names {
            let batch_size = predicted_attributes.size(0)?;
            let mut attribute_lists = Vec::with_capacity(batch_size);
            
            for i in 0..batch_size {
                let mut attrs = Vec::new();
                let pred_attrs = predicted_attributes.select(0, i)?;
                
                for j in 0..num_attributes {
                    if pred_attrs.get_item(j)?.to_scalar::<bool>()? {
                        attrs.push(attribute_names[j].clone());
                    }
                }
                
                attribute_lists.push(attrs);
            }
            
            Ok((attribute_scores, Some(attribute_lists)))
        } else {
            Ok((attribute_scores, None))
        }
    }
}

pub struct SemanticClassifier {
    classifier: ZeroShotClassifier,
    text_encoder: Option<PyObject>,
}

impl SemanticClassifier {
    pub fn new(
        zero_shot_model: ZeroShotModel,
        class_names: Option<Vec<String>>,
        class_embeddings: Option<Tensor>,
        text_encoder: Option<PyObject>,
    ) -> Result<Self, PhynexusError> {
        let mut classifier = ZeroShotClassifier::new(
            zero_shot_model,
            class_names.clone(),
            class_embeddings.clone(),
        );
        
        let mut semantic_classifier = Self {
            classifier,
            text_encoder,
        };
        
        if class_names.is_some() && text_encoder.is_some() && class_embeddings.is_none() {
            semantic_classifier.compute_class_embeddings()?;
        }
        
        Ok(semantic_classifier)
    }
    
    pub fn forward(&self, visual_features: &Tensor) -> Result<(Tensor, Tensor), PhynexusError> {
        self.classifier.forward(visual_features)
    }
    
    pub fn predict(&self, visual_features: &Tensor) -> Result<(Tensor, Option<Vec<String>>), PhynexusError> {
        self.classifier.predict(visual_features)
    }
    
    pub fn compute_class_embeddings(&mut self) -> Result<(), PhynexusError> {
        if self.classifier.class_names.is_none() || self.text_encoder.is_none() {
            return Err(PhynexusError::InvalidArgument(
                "Class names and text encoder must be set".to_string()
            ));
        }
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let tokenize = py.import("neurenix.text")?.getattr("tokenize")?;
        let class_names = self.classifier.class_names.as_ref().unwrap();
        
        let mut tokens = Vec::new();
        for name in class_names {
            let token = tokenize.call1((name,))?;
            tokens.push(token);
        }
        
        let mut max_length = 0;
        for token in &tokens {
            let len = token.len()?;
            if len > max_length {
                max_length = len;
            }
        }
        
        let mut padded_tokens = Vec::new();
        let mut masks = Vec::new();
        
        for token in tokens {
            let len = token.len()?;
            let padding = vec![0; max_length - len];
            
            let mut padded = token.extract::<Vec<i64>>()?;
            padded.extend(padding.clone());
            padded_tokens.push(padded);
            
            let mut mask = vec![1; len];
            mask.extend(vec![0; max_length - len]);
            masks.push(mask);
        }
        
        let token_tensor = Tensor::from_vec2(padded_tokens)?;
        let mask_tensor = Tensor::from_vec2(masks)?;
        
        let text_encoder = self.text_encoder.as_ref().unwrap();
        let class_embeddings = text_encoder.call(py, (token_tensor, mask_tensor), None)?;
        let class_embeddings = Tensor::from_pyobject(class_embeddings)?;
        
        self.classifier.set_class_embeddings(class_embeddings, None);
        
        Ok(())
    }
    
    pub fn set_text_encoder(&mut self, text_encoder: PyObject) -> Result<(), PhynexusError> {
        self.text_encoder = Some(text_encoder);
        
        if self.classifier.class_names.is_some() && self.classifier.class_embeddings.is_none() {
            self.compute_class_embeddings()?;
        }
        
        Ok(())
    }
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor, PhynexusError> {
    let a_norm = a.norm(2, -1, true)?;
    let b_norm = b.norm(2, -1, true)?;
    
    let a_normalized = a.div(&a_norm.add_scalar(1e-10)?)?;
    let b_normalized = b.div(&b_norm.add_scalar(1e-10)?)?;
    
    let dot_product = a_normalized.bmm(&b_normalized.transpose(-2, -1)?)?;
    
    Ok(dot_product)
}

#[pymodule]
fn classifiers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyZeroShotClassifier>()?;
    m.add_class::<PyAttributeClassifier>()?;
    m.add_class::<PySemanticClassifier>()?;
    Ok(())
}

pub fn register_classifiers(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "classifier")?;
    classifiers(py, submodule)?;
    m.add_submodule(submodule)?;
    Ok(())
}

#[pyclass]
struct PyZeroShotClassifier {
    inner: ZeroShotClassifier,
}

#[pymethods]
impl PyZeroShotClassifier {
    #[new]
    fn new(
        zero_shot_model: &PyAny,
        class_names: Option<Vec<String>>,
        class_embeddings: Option<&PyAny>,
    ) -> PyResult<Self> {
        let zero_shot_model = PyZeroShotModel::extract(zero_shot_model)?;
        
        let class_embeddings = if let Some(embeddings) = class_embeddings {
            Some(Tensor::from_pyany(embeddings)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let inner = ZeroShotClassifier::new(
            zero_shot_model.inner,
            class_names,
            class_embeddings,
        );
        
        Ok(Self { inner })
    }
    
    fn forward(&self, visual_features: &PyAny) -> PyResult<(PyObject, PyObject)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (logits, visual_embeddings) = self.inner.forward(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let logits_py = logits.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let visual_embeddings_py = visual_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((logits_py, visual_embeddings_py))
    }
    
    fn predict(&self, visual_features: &PyAny) -> PyResult<(PyObject, Option<Vec<String>>)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (predictions, class_names) = self.inner.predict(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let predictions_py = predictions.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((predictions_py, class_names))
    }
    
    fn set_class_embeddings(
        &mut self,
        class_embeddings: &PyAny,
        class_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let embeddings = Tensor::from_pyany(class_embeddings)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.inner.set_class_embeddings(embeddings, class_names);
        
        Ok(())
    }
}

#[pyclass]
struct PyAttributeClassifier {
    inner: AttributeClassifier,
}

#[pymethods]
impl PyAttributeClassifier {
    #[new]
    fn new(
        zero_shot_model: &PyAny,
        class_names: Option<Vec<String>>,
        attribute_names: Option<Vec<String>>,
        class_attribute_matrix: Option<&PyAny>,
    ) -> PyResult<Self> {
        let zero_shot_model = PyZeroShotModel::extract(zero_shot_model)?;
        
        let class_attribute_matrix = if let Some(matrix) = class_attribute_matrix {
            Some(Tensor::from_pyany(matrix)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let inner = AttributeClassifier::new(
            zero_shot_model.inner,
            class_names,
            attribute_names,
            class_attribute_matrix,
        );
        
        Ok(Self { inner })
    }
    
    fn forward(&self, visual_features: &PyAny) -> PyResult<(PyObject, PyObject)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (logits, visual_embeddings) = self.inner.forward(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let logits_py = logits.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let visual_embeddings_py = visual_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((logits_py, visual_embeddings_py))
    }
    
    fn predict(&self, visual_features: &PyAny) -> PyResult<(PyObject, Option<Vec<String>>)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (predictions, class_names) = self.inner.predict(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let predictions_py = predictions.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((predictions_py, class_names))
    }
    
    fn set_class_attributes(
        &mut self,
        class_attribute_matrix: &PyAny,
        class_names: Option<Vec<String>>,
        attribute_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        let matrix = Tensor::from_pyany(class_attribute_matrix)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.inner.set_class_attributes(matrix, class_names, attribute_names);
        
        Ok(())
    }
    
    fn predict_attributes(
        &self,
        visual_features: &PyAny,
    ) -> PyResult<(PyObject, Option<Vec<Vec<String>>>)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (attribute_scores, attribute_lists) = self.inner.predict_attributes(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let attribute_scores_py = attribute_scores.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((attribute_scores_py, attribute_lists))
    }
}

#[pyclass]
struct PySemanticClassifier {
    inner: SemanticClassifier,
}

#[pymethods]
impl PySemanticClassifier {
    #[new]
    fn new(
        zero_shot_model: &PyAny,
        class_names: Option<Vec<String>>,
        class_embeddings: Option<&PyAny>,
        text_encoder: Option<&PyAny>,
    ) -> PyResult<Self> {
        let zero_shot_model = PyZeroShotModel::extract(zero_shot_model)?;
        
        let class_embeddings = if let Some(embeddings) = class_embeddings {
            Some(Tensor::from_pyany(embeddings)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            None
        };
        
        let text_encoder = if let Some(encoder) = text_encoder {
            Some(encoder.to_object(Python::acquire_gil().python()))
        } else {
            None
        };
        
        let inner = SemanticClassifier::new(
            zero_shot_model.inner,
            class_names,
            class_embeddings,
            text_encoder,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { inner })
    }
    
    fn forward(&self, visual_features: &PyAny) -> PyResult<(PyObject, PyObject)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (logits, visual_embeddings) = self.inner.forward(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let logits_py = logits.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let visual_embeddings_py = visual_embeddings.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((logits_py, visual_embeddings_py))
    }
    
    fn predict(&self, visual_features: &PyAny) -> PyResult<(PyObject, Option<Vec<String>>)> {
        let visual_tensor = Tensor::from_pyany(visual_features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let (predictions, class_names) = self.inner.predict(&visual_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let predictions_py = predictions.to_pyobject(py)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok((predictions_py, class_names))
    }
    
    fn compute_class_embeddings(&mut self) -> PyResult<()> {
        self.inner.compute_class_embeddings()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn set_text_encoder(&mut self, text_encoder: &PyAny) -> PyResult<()> {
        let encoder = text_encoder.to_object(Python::acquire_gil().python());
        
        self.inner.set_text_encoder(encoder)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
