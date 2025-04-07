
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn semantic_similarity(embeddings1: &Tensor, embeddings2: &Tensor) -> Result<Tensor, PhynexusError> {
    let norm1 = embeddings1.norm(2, 1, true)?;
    let norm2 = embeddings2.norm(2, 1, true)?;
    
    let embeddings1_normalized = embeddings1.div(&norm1.add_scalar(1e-10)?)?;
    let embeddings2_normalized = embeddings2.div(&norm2.add_scalar(1e-10)?)?;
    
    let embeddings1_unsqueezed = embeddings1_normalized.unsqueeze(1)?;
    let embeddings2_unsqueezed = embeddings2_normalized.unsqueeze(0)?;
    
    let similarities = cosine_similarity(&embeddings1_unsqueezed, &embeddings2_unsqueezed)?;
    
    Ok(similarities)
}

pub fn attribute_mapping(
    class_names: Vec<String>,
    attribute_names: Vec<String>,
    class_attribute_map: &PyDict,
) -> Result<Tensor, PhynexusError> {
    let num_classes = class_names.len();
    let num_attributes = attribute_names.len();
    
    let mut attribute_indices = std::collections::HashMap::new();
    for (i, attr) in attribute_names.iter().enumerate() {
        attribute_indices.insert(attr.clone(), i);
    }
    
    let mut class_attribute_matrix = Tensor::zeros(&[num_classes, num_attributes])?;
    
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    for (i, class_name) in class_names.iter().enumerate() {
        if let Some(attrs) = class_attribute_map.get_item(class_name) {
            let attrs = attrs.extract::<Vec<String>>()?;
            for attr in attrs {
                if let Some(&idx) = attribute_indices.get(&attr) {
                    class_attribute_matrix.set_item(&[i, idx], 1.0)?;
                }
            }
        }
    }
    
    Ok(class_attribute_matrix)
}

pub fn class_embedding(
    class_names: Vec<String>,
    text_encoder: &PyAny,
    tokenizer: Option<&PyAny>,
) -> Result<Tensor, PhynexusError> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    let token_tensor: PyObject;
    let mask_tensor: PyObject;
    
    if let Some(tokenizer) = tokenizer {
        let tokenized = tokenizer.call1((class_names,))?;
        token_tensor = tokenized.getattr("input_ids")?.extract()?;
        mask_tensor = tokenized.getattr("attention_mask")?.extract()?;
    } else {
        let tokenize = py.import("neurenix.text")?.getattr("tokenize")?;
        let vocab = py.import("neurenix.text")?.call_method0("get_vocab")?;
        let unk_token = vocab.get_item("<unk>")?;
        
        let mut tokens = Vec::new();
        for name in &class_names {
            let name_tokens = name.to_lowercase().split_whitespace().collect::<Vec<_>>();
            let mut token_indices = Vec::new();
            
            for token in name_tokens {
                let token_idx = match vocab.get_item(token) {
                    Ok(idx) => idx.extract::<i64>()?,
                    Err(_) => unk_token.extract::<i64>()?,
                };
                token_indices.push(token_idx);
            }
            
            tokens.push(token_indices);
        }
        
        let max_length = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let mut padded_tokens = Vec::new();
        let mut masks = Vec::new();
        
        for token_indices in tokens {
            let padding = vec![0; max_length - token_indices.len()];
            let mut padded = token_indices.clone();
            padded.extend(padding.clone());
            padded_tokens.push(padded);
            
            let mut mask = vec![1; token_indices.len()];
            mask.extend(vec![0; max_length - token_indices.len()]);
            masks.push(mask);
        }
        
        token_tensor = Tensor::from_vec2(padded_tokens)?.to_pyobject(py)?;
        mask_tensor = Tensor::from_vec2(masks)?.to_pyobject(py)?;
    }
    
    let no_grad = py.import("neurenix")?.getattr("no_grad")?;
    let context_manager = no_grad.call0()?;
    let enter = context_manager.getattr("__enter__")?.call0()?;
    
    let class_embeddings_py = text_encoder.call1((token_tensor, mask_tensor))?;
    let class_embeddings = Tensor::from_pyobject(class_embeddings_py)?;
    
    let exit = context_manager.getattr("__exit__")?.call1((py.None(), py.None(), py.None()))?;
    
    Ok(class_embeddings)
}

pub fn generate_class_descriptions(
    class_names: Vec<String>,
    template: Option<String>,
) -> Vec<String> {
    let template = template.unwrap_or_else(|| "A photo of a {}.".to_string());
    
    class_names.iter()
        .map(|name| template.replace("{}", name))
        .collect()
}

pub fn compute_attribute_importance(
    model: &PyAny,
    visual_features: &Tensor,
    attribute_matrix: &Tensor,
) -> Result<Tensor, PhynexusError> {
    let visual_embeddings_py = model.call1((visual_features.to_pyobject(Python::acquire_gil().python())?,))?;
    let visual_embeddings = Tensor::from_pyobject(visual_embeddings_py)?;
    
    let num_attributes = attribute_matrix.size(1)?;
    let batch_size = visual_features.size(0)?;
    let mut attribute_importance = Tensor::zeros(&[batch_size, num_attributes])?;
    
    for i in 0..num_attributes {
        let attribute_mask = attribute_matrix.slice(1, i, i + 1, 1)?;
        
        let visual_embeddings_unsqueezed = visual_embeddings.unsqueeze(1)?;
        let attribute_mask_unsqueezed = attribute_mask.unsqueeze(0)?;
        
        let similarity = cosine_similarity(&visual_embeddings_unsqueezed, &attribute_mask_unsqueezed)?;
        
        let mean_similarity = similarity.mean(1)?;
        
        for j in 0..batch_size {
            let score = mean_similarity.get_item(j)?.to_scalar::<f32>()?;
            attribute_importance.set_item(&[j, i], score)?;
        }
    }
    
    Ok(attribute_importance)
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
fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_semantic_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_attribute_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(py_class_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_class_descriptions, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_attribute_importance, m)?)?;
    Ok(())
}

pub fn register_utils(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "utils")?;
    utils(py, submodule)?;
    m.add_submodule(submodule)?;
    Ok(())
}

#[pyfunction]
fn py_semantic_similarity(embeddings1: &PyAny, embeddings2: &PyAny) -> PyResult<PyObject> {
    let embeddings1_tensor = Tensor::from_pyany(embeddings1)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let embeddings2_tensor = Tensor::from_pyany(embeddings2)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let similarities = semantic_similarity(&embeddings1_tensor, &embeddings2_tensor)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    let similarities_py = similarities.to_pyobject(py)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(similarities_py)
}

#[pyfunction]
fn py_attribute_mapping(
    class_names: Vec<String>,
    attribute_names: Vec<String>,
    class_attribute_map: &PyDict,
) -> PyResult<PyObject> {
    let class_attribute_matrix = attribute_mapping(class_names, attribute_names, class_attribute_map)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    let matrix_py = class_attribute_matrix.to_pyobject(py)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(matrix_py)
}

#[pyfunction]
fn py_class_embedding(
    class_names: Vec<String>,
    text_encoder: &PyAny,
    tokenizer: Option<&PyAny>,
) -> PyResult<PyObject> {
    let class_embeddings = class_embedding(class_names, text_encoder, tokenizer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    let embeddings_py = class_embeddings.to_pyobject(py)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(embeddings_py)
}

#[pyfunction]
fn py_generate_class_descriptions(
    class_names: Vec<String>,
    template: Option<String>,
) -> PyResult<Vec<String>> {
    let descriptions = generate_class_descriptions(class_names, template);
    Ok(descriptions)
}

#[pyfunction]
fn py_compute_attribute_importance(
    model: &PyAny,
    visual_features: &PyAny,
    attribute_matrix: &PyAny,
) -> PyResult<PyObject> {
    let visual_tensor = Tensor::from_pyany(visual_features)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let attribute_tensor = Tensor::from_pyany(attribute_matrix)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let importance = compute_attribute_importance(model, &visual_tensor, &attribute_tensor)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let gil = Python::acquire_gil();
    let py = gil.python();
    
    let importance_py = importance.to_pyobject(py)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(importance_py)
}
