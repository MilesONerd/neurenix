
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use numpy::{PyArray, PyArray1, PyArray2};

use crate::tensor::Tensor;

#[pyclass]
#[derive(Clone)]
pub struct MultiAgentLearning {
    #[pyo3(get)]
    agent_ids: Vec<String>,
    #[pyo3(get)]
    state_dim: usize,
    #[pyo3(get)]
    action_dim: usize,
    #[pyo3(get, set)]
    learning_rate: f64,
    #[pyo3(get, set)]
    policies: HashMap<String, PyObject>,
}

#[pymethods]
impl MultiAgentLearning {
    #[new]
    fn new(
        agent_ids: Vec<String>,
        state_dim: usize,
        action_dim: usize,
        learning_rate: Option<f64>,
    ) -> Self {
        MultiAgentLearning {
            agent_ids,
            state_dim,
            action_dim,
            learning_rate: learning_rate.unwrap_or(0.01),
            policies: HashMap::new(),
        }
    }

    fn forward(&self, py: Python, states: &PyDict) -> PyResult<PyObject> {
        let result = PyDict::new(py);
        Ok(result.into())
    }

    fn update(
        &mut self,
        py: Python,
        states: &PyDict,
        actions: &PyDict,
        rewards: &PyDict,
        next_states: &PyDict,
        dones: &PyDict,
    ) -> PyResult<PyObject> {
        let result = PyDict::new(py);
        Ok(result.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.policies.clear();
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct IndependentLearners {
    #[pyo3(get)]
    base: MultiAgentLearning,
    #[pyo3(get, set)]
    discount_factor: f64,
    #[pyo3(get, set)]
    q_tables: HashMap<String, HashMap<String, Vec<f64>>>,
}

#[pymethods]
impl IndependentLearners {
    #[new]
    fn new(
        agent_ids: Vec<String>,
        state_dim: usize,
        action_dim: usize,
        learning_rate: Option<f64>,
        discount_factor: Option<f64>,
    ) -> Self {
        let base = MultiAgentLearning::new(
            agent_ids.clone(),
            state_dim,
            action_dim,
            learning_rate,
        );
        
        let mut q_tables = HashMap::new();
        for agent_id in &agent_ids {
            q_tables.insert(agent_id.clone(), HashMap::new());
        }
        
        IndependentLearners {
            base,
            discount_factor: discount_factor.unwrap_or(0.99),
            q_tables,
        }
    }

    fn _get_state_key(&self, py: Python, state: &PyAny) -> PyResult<String> {
        let array = state.extract::<&PyArray1<f64>>()?;
        let data = array.readonly();
        let state_vec: Vec<f64> = data.as_slice()?.to_vec();
        Ok(format!("{:?}", state_vec))
    }

    fn forward(&self, py: Python, states: &PyDict) -> PyResult<PyObject> {
        let action_probs = PyDict::new(py);
        
        for (agent_id_obj, state_obj) in states.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            
            if !self.q_tables.contains_key(&agent_id) {
                continue;
            }
            
            let state_key = self._get_state_key(py, state_obj)?;
            let q_table = self.q_tables.get(&agent_id).unwrap();
            
            let q_values = if let Some(values) = q_table.get(&state_key) {
                values.clone()
            } else {
                vec![0.0; self.base.action_dim]
            };
            
            let array = PyArray1::from_vec(py, q_values);
            let tensor = Tensor::from_array(py, array.as_ref())?;
            let softmax_result = tensor.call_method0(py, "softmax")?;
            
            action_probs.set_item(agent_id, softmax_result)?;
        }
        
        Ok(action_probs.into())
    }

    fn update(
        &mut self,
        py: Python,
        states: &PyDict,
        actions: &PyDict,
        rewards: &PyDict,
        next_states: &PyDict,
        dones: &PyDict,
    ) -> PyResult<PyObject> {
        let losses = PyDict::new(py);
        
        for agent_id_obj in self.base.agent_ids.iter() {
            let agent_id = agent_id_obj;
            
            if !states.contains(agent_id)? {
                continue;
            }
            
            let state = states.get_item(agent_id).unwrap();
            let action = actions.get_item(agent_id).unwrap().extract::<i64>()?;
            let reward = rewards.get_item(agent_id).unwrap().extract::<f64>()?;
            let next_state = next_states.get_item(agent_id).unwrap();
            let done = dones.get_item(agent_id).unwrap().extract::<bool>()?;
            
            let state_key = self._get_state_key(py, state)?;
            let next_state_key = self._get_state_key(py, next_state)?;
            
            let q_table = self.q_tables.get_mut(agent_id).unwrap();
            
            if !q_table.contains_key(&state_key) {
                q_table.insert(state_key.clone(), vec![0.0; self.base.action_dim]);
            }
            
            if !q_table.contains_key(&next_state_key) {
                q_table.insert(next_state_key.clone(), vec![0.0; self.base.action_dim]);
            }
            
            let current_q = q_table.get(&state_key).unwrap()[action as usize];
            
            let next_q_max = q_table.get(&next_state_key).unwrap().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            let target_q = if done {
                reward
            } else {
                reward + self.discount_factor * next_q_max
            };
            
            let new_q = current_q + self.base.learning_rate * (target_q - current_q);
            q_table.get_mut(&state_key).unwrap()[action as usize] = new_q;
            
            let loss = (target_q - current_q).powi(2);
            losses.set_item(agent_id, loss)?;
        }
        
        Ok(losses.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.base.reset()?;
        
        self.q_tables.clear();
        for agent_id in &self.base.agent_ids {
            self.q_tables.insert(agent_id.clone(), HashMap::new());
        }
        
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct JointActionLearners {
    #[pyo3(get)]
    base: MultiAgentLearning,
    #[pyo3(get, set)]
    discount_factor: f64,
    #[pyo3(get, set)]
    joint_q_tables: HashMap<String, HashMap<String, HashMap<String, f64>>>,
    #[pyo3(get, set)]
    agent_models: HashMap<String, HashMap<String, HashMap<String, Vec<f64>>>>,
}

#[pymethods]
impl JointActionLearners {
    #[new]
    fn new(
        agent_ids: Vec<String>,
        state_dim: usize,
        action_dim: usize,
        learning_rate: Option<f64>,
        discount_factor: Option<f64>,
    ) -> Self {
        let base = MultiAgentLearning::new(
            agent_ids.clone(),
            state_dim,
            action_dim,
            learning_rate,
        );
        
        let mut joint_q_tables = HashMap::new();
        let mut agent_models = HashMap::new();
        
        for agent_id in &agent_ids {
            joint_q_tables.insert(agent_id.clone(), HashMap::new());
            
            let mut other_agents = HashMap::new();
            for other_id in &agent_ids {
                if other_id != agent_id {
                    other_agents.insert(other_id.clone(), HashMap::new());
                }
            }
            
            agent_models.insert(agent_id.clone(), other_agents);
        }
        
        JointActionLearners {
            base,
            discount_factor: discount_factor.unwrap_or(0.99),
            joint_q_tables,
            agent_models,
        }
    }

    fn _get_state_key(&self, py: Python, state: &PyAny) -> PyResult<String> {
        let array = state.extract::<&PyArray1<f64>>()?;
        let data = array.readonly();
        let state_vec: Vec<f64> = data.as_slice()?.to_vec();
        Ok(format!("{:?}", state_vec))
    }

    fn _get_joint_action_key(&self, py: Python, actions: &PyDict) -> PyResult<String> {
        let mut action_pairs = Vec::new();
        
        for (agent_id_obj, action_obj) in actions.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            let action = action_obj.extract::<i64>()?;
            action_pairs.push((agent_id, action));
        }
        
        action_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(format!("{:?}", action_pairs))
    }

    fn forward(&self, py: Python, states: &PyDict) -> PyResult<PyObject> {
        let action_probs = PyDict::new(py);
        
        for (agent_id_obj, _) in states.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            
            let probs = vec![1.0 / self.base.action_dim as f64; self.base.action_dim];
            let array = PyArray1::from_vec(py, probs);
            let tensor = Tensor::from_array(py, array.as_ref())?;
            
            action_probs.set_item(agent_id, tensor)?;
        }
        
        Ok(action_probs.into())
    }

    fn update(
        &mut self,
        py: Python,
        states: &PyDict,
        actions: &PyDict,
        rewards: &PyDict,
        next_states: &PyDict,
        dones: &PyDict,
    ) -> PyResult<PyObject> {
        let losses = PyDict::new(py);
        
        for agent_id in &self.base.agent_ids {
            if states.contains(agent_id)? {
                losses.set_item(agent_id, 0.0)?;
            }
        }
        
        Ok(losses.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.base.reset()?;
        
        self.joint_q_tables.clear();
        self.agent_models.clear();
        
        for agent_id in &self.base.agent_ids {
            self.joint_q_tables.insert(agent_id.clone(), HashMap::new());
            
            let mut other_agents = HashMap::new();
            for other_id in &self.base.agent_ids {
                if other_id != agent_id {
                    other_agents.insert(other_id.clone(), HashMap::new());
                }
            }
            
            self.agent_models.insert(agent_id.clone(), other_agents);
        }
        
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct TeamLearning {
    #[pyo3(get)]
    base: MultiAgentLearning,
    #[pyo3(get, set)]
    discount_factor: f64,
    #[pyo3(get, set)]
    team_q_table: HashMap<String, HashMap<String, f64>>,
}

#[pymethods]
impl TeamLearning {
    #[new]
    fn new(
        agent_ids: Vec<String>,
        state_dim: usize,
        action_dim: usize,
        learning_rate: Option<f64>,
        discount_factor: Option<f64>,
    ) -> Self {
        let base = MultiAgentLearning::new(
            agent_ids,
            state_dim,
            action_dim,
            learning_rate,
        );
        
        TeamLearning {
            base,
            discount_factor: discount_factor.unwrap_or(0.99),
            team_q_table: HashMap::new(),
        }
    }

    fn _get_state_key(&self, py: Python, states: &PyDict) -> PyResult<String> {
        let mut state_pairs = Vec::new();
        
        for (agent_id_obj, state_obj) in states.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            let array = state_obj.extract::<&PyArray1<f64>>()?;
            let data = array.readonly();
            let state_vec: Vec<f64> = data.as_slice()?.to_vec();
            state_pairs.push((agent_id, state_vec));
        }
        
        state_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(format!("{:?}", state_pairs))
    }

    fn _get_joint_action_key(&self, py: Python, actions: &PyDict) -> PyResult<String> {
        let mut action_pairs = Vec::new();
        
        for (agent_id_obj, action_obj) in actions.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            let action = action_obj.extract::<i64>()?;
            action_pairs.push((agent_id, action));
        }
        
        action_pairs.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(format!("{:?}", action_pairs))
    }

    fn forward(&self, py: Python, states: &PyDict) -> PyResult<PyObject> {
        let action_probs = PyDict::new(py);
        
        for (agent_id_obj, _) in states.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            
            let probs = vec![1.0 / self.base.action_dim as f64; self.base.action_dim];
            let array = PyArray1::from_vec(py, probs);
            let tensor = Tensor::from_array(py, array.as_ref())?;
            
            action_probs.set_item(agent_id, tensor)?;
        }
        
        Ok(action_probs.into())
    }

    fn update(
        &mut self,
        py: Python,
        states: &PyDict,
        actions: &PyDict,
        rewards: &PyDict,
        next_states: &PyDict,
        dones: &PyDict,
    ) -> PyResult<PyObject> {
        let losses = PyDict::new(py);
        
        for agent_id in &self.base.agent_ids {
            if states.contains(agent_id)? {
                losses.set_item(agent_id, 0.0)?;
            }
        }
        
        Ok(losses.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.base.reset()?;
        self.team_q_table.clear();
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct OpponentModeling {
    #[pyo3(get)]
    base: MultiAgentLearning,
    #[pyo3(get, set)]
    discount_factor: f64,
    #[pyo3(get, set)]
    q_tables: HashMap<String, HashMap<String, Vec<f64>>>,
    #[pyo3(get, set)]
    opponent_models: HashMap<String, HashMap<String, HashMap<String, Vec<f64>>>>,
}

#[pymethods]
impl OpponentModeling {
    #[new]
    fn new(
        agent_ids: Vec<String>,
        state_dim: usize,
        action_dim: usize,
        learning_rate: Option<f64>,
        discount_factor: Option<f64>,
    ) -> Self {
        let base = MultiAgentLearning::new(
            agent_ids.clone(),
            state_dim,
            action_dim,
            learning_rate,
        );
        
        let mut q_tables = HashMap::new();
        let mut opponent_models = HashMap::new();
        
        for agent_id in &agent_ids {
            q_tables.insert(agent_id.clone(), HashMap::new());
            
            let mut other_agents = HashMap::new();
            for other_id in &agent_ids {
                if other_id != agent_id {
                    other_agents.insert(other_id.clone(), HashMap::new());
                }
            }
            
            opponent_models.insert(agent_id.clone(), other_agents);
        }
        
        OpponentModeling {
            base,
            discount_factor: discount_factor.unwrap_or(0.99),
            q_tables,
            opponent_models,
        }
    }

    fn _get_state_key(&self, py: Python, state: &PyAny) -> PyResult<String> {
        let array = state.extract::<&PyArray1<f64>>()?;
        let data = array.readonly();
        let state_vec: Vec<f64> = data.as_slice()?.to_vec();
        Ok(format!("{:?}", state_vec))
    }

    fn forward(&self, py: Python, states: &PyDict) -> PyResult<PyObject> {
        let action_probs = PyDict::new(py);
        
        for (agent_id_obj, state_obj) in states.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            
            if !self.q_tables.contains_key(&agent_id) {
                continue;
            }
            
            let state_key = self._get_state_key(py, state_obj)?;
            let q_table = self.q_tables.get(&agent_id).unwrap();
            
            let q_values = if let Some(values) = q_table.get(&state_key) {
                values.clone()
            } else {
                vec![0.0; self.base.action_dim]
            };
            
            let array = PyArray1::from_vec(py, q_values);
            let tensor = Tensor::from_array(py, array.as_ref())?;
            let softmax_result = tensor.call_method0(py, "softmax")?;
            
            action_probs.set_item(agent_id, softmax_result)?;
        }
        
        Ok(action_probs.into())
    }

    fn update(
        &mut self,
        py: Python,
        states: &PyDict,
        actions: &PyDict,
        rewards: &PyDict,
        next_states: &PyDict,
        dones: &PyDict,
    ) -> PyResult<PyObject> {
        let losses = PyDict::new(py);
        
        for agent_id in &self.base.agent_ids {
            if !states.contains(agent_id)? {
                continue;
            }
            
            let state = states.get_item(agent_id).unwrap();
            let state_key = self._get_state_key(py, state)?;
            
            for (other_id_obj, action_obj) in actions.iter() {
                let other_id = other_id_obj.extract::<String>()?;
                
                if other_id == *agent_id {
                    continue;
                }
                
                let action = action_obj.extract::<i64>()?;
                
                let agent_models = self.opponent_models.get_mut(agent_id).unwrap();
                let other_model = agent_models.get_mut(&other_id).unwrap();
                
                if !other_model.contains_key(&state_key) {
                    other_model.insert(state_key.clone(), vec![0.0; self.base.action_dim]);
                }
                
                other_model.get_mut(&state_key).unwrap()[action as usize] += 1.0;
            }
            
            losses.set_item(agent_id, 0.0)?;
        }
        
        Ok(losses.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.base.reset()?;
        
        self.q_tables.clear();
        self.opponent_models.clear();
        
        for agent_id in &self.base.agent_ids {
            self.q_tables.insert(agent_id.clone(), HashMap::new());
            
            let mut other_agents = HashMap::new();
            for other_id in &self.base.agent_ids {
                if other_id != agent_id {
                    other_agents.insert(other_id.clone(), HashMap::new());
                }
            }
            
            self.opponent_models.insert(agent_id.clone(), other_agents);
        }
        
        Ok(())
    }
}

pub fn register_learning(py: Python, m: &PyModule) -> PyResult<()> {
    let learning_module = PyModule::new(py, "learning")?;
    
    learning_module.add_class::<MultiAgentLearning>()?;
    learning_module.add_class::<IndependentLearners>()?;
    learning_module.add_class::<JointActionLearners>()?;
    learning_module.add_class::<TeamLearning>()?;
    learning_module.add_class::<OpponentModeling>()?;
    
    m.add_submodule(learning_module)?;
    
    Ok(())
}
