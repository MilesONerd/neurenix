
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::tensor::Tensor;

#[pyclass]
#[derive(Clone, Debug)]
pub struct AgentState {
    #[pyo3(get, set)]
    state: HashMap<String, PyObject>,
    history: Vec<HashMap<String, PyObject>>,
}

#[pymethods]
impl AgentState {
    #[new]
    fn new(initial_state: Option<HashMap<String, PyObject>>) -> Self {
        let state = initial_state.unwrap_or_default();
        AgentState {
            state,
            history: Vec::new(),
        }
    }

    fn update(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            self.history.push(self.state.clone());
            
            for (key, value) in kwargs.iter() {
                let key_str = key.extract::<String>()?;
                self.state.insert(key_str, value.into_py(py));
            }
        }
        Ok(())
    }

    fn get(&self, py: Python, key: &str, default: Option<PyObject>) -> PyObject {
        match self.state.get(key) {
            Some(value) => value.clone_ref(py),
            None => default.unwrap_or_else(|| py.None()),
        }
    }

    fn as_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.state {
            dict.set_item(key, value.clone_ref(py))?;
        }
        Ok(dict.into())
    }

    fn history(&self, py: Python, key: Option<&str>) -> PyResult<PyObject> {
        if let Some(key) = key {
            let values = PyList::empty(py);
            for state in &self.history {
                if let Some(value) = state.get(key) {
                    values.append(value.clone_ref(py))?;
                } else {
                    values.append(py.None())?;
                }
            }
            Ok(values.into())
        } else {
            let history_list = PyList::empty(py);
            for state in &self.history {
                let state_dict = PyDict::new(py);
                for (k, v) in state {
                    state_dict.set_item(k, v.clone_ref(py))?;
                }
                history_list.append(state_dict)?;
            }
            Ok(history_list.into())
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Agent {
    #[pyo3(get)]
    agent_id: String,
    #[pyo3(get, set)]
    observation_space: Option<PyObject>,
    #[pyo3(get, set)]
    action_space: Option<PyObject>,
    #[pyo3(get, set)]
    state: AgentState,
    #[pyo3(get, set)]
    mailbox: Vec<PyObject>,
}

#[pymethods]
impl Agent {
    #[new]
    fn new(
        agent_id: String,
        observation_space: Option<PyObject>,
        action_space: Option<PyObject>,
    ) -> Self {
        Agent {
            agent_id,
            observation_space,
            action_space,
            state: AgentState::new(None),
            mailbox: Vec::new(),
        }
    }

    fn observe(&mut self, _observation: PyObject) -> PyResult<()> {
        Ok(())
    }

    fn act(&self, py: Python) -> PyObject {
        py.None()
    }

    fn receive_message(&mut self, message: PyObject) -> PyResult<()> {
        self.mailbox.push(message);
        Ok(())
    }

    fn send_message(&self, py: Python, recipient_id: &str, content: PyObject) -> PyResult<PyObject> {
        let message = PyDict::new(py);
        message.set_item("sender", &self.agent_id)?;
        message.set_item("recipient", recipient_id)?;
        message.set_item("content", content)?;
        
        let datetime = py.import("datetime")?;
        let now = datetime.getattr("datetime")?.getattr("now")?.call0()?;
        message.set_item("timestamp", now)?;
        
        Ok(message.into())
    }

    fn update_state(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<()> {
        self.state.update(py, kwargs)
    }

    fn reset(&mut self) -> PyResult<()> {
        self.state = AgentState::new(None);
        self.mailbox.clear();
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ReactiveAgent {
    #[pyo3(get)]
    base: Agent,
    #[pyo3(get, set)]
    policy_function: PyObject,
    #[pyo3(get, set)]
    current_observation: Option<PyObject>,
}

#[pymethods]
impl ReactiveAgent {
    #[new]
    fn new(
        agent_id: String,
        policy_function: PyObject,
        observation_space: Option<PyObject>,
        action_space: Option<PyObject>,
    ) -> Self {
        ReactiveAgent {
            base: Agent::new(agent_id, observation_space, action_space),
            policy_function,
            current_observation: None,
        }
    }

    fn observe(&mut self, observation: PyObject) -> PyResult<()> {
        self.current_observation = Some(observation);
        Ok(())
    }

    fn act(&self, py: Python) -> PyResult<PyObject> {
        if let Some(observation) = &self.current_observation {
            let args = PyTuple::new(py, &[observation.clone_ref(py)]);
            Ok(self.policy_function.call1(py, args)?)
        } else {
            let err = PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Agent must receive an observation before acting",
            );
            Err(err)
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DeliberativeAgent {
    #[pyo3(get)]
    base: Agent,
    #[pyo3(get, set)]
    world_model: PyObject,
    #[pyo3(get, set)]
    planner: PyObject,
    #[pyo3(get, set)]
    current_plan: Vec<PyObject>,
}

#[pymethods]
impl DeliberativeAgent {
    #[new]
    fn new(
        agent_id: String,
        world_model: PyObject,
        planner: PyObject,
        observation_space: Option<PyObject>,
        action_space: Option<PyObject>,
    ) -> Self {
        DeliberativeAgent {
            base: Agent::new(agent_id, observation_space, action_space),
            world_model,
            planner,
            current_plan: Vec::new(),
        }
    }

    fn observe(&mut self, py: Python, observation: PyObject) -> PyResult<()> {
        self.world_model.call_method1(py, "update", (observation,))?;
        
        if !self._is_plan_valid(py)? {
            self._replan(py)?;
        }
        
        Ok(())
    }

    fn act(&mut self, py: Python) -> PyResult<PyObject> {
        if self.current_plan.is_empty() {
            self._replan(py)?;
        }
        
        if self.current_plan.is_empty() {
            let err = PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Failed to generate a plan",
            );
            return Err(err);
        }
        
        Ok(self.current_plan.remove(0))
    }

    fn _is_plan_valid(&self, _py: Python) -> PyResult<bool> {
        Ok(!self.current_plan.is_empty())
    }

    fn _replan(&mut self, py: Python) -> PyResult<()> {
        let plan = self.planner.call_method1(py, "plan", (&self.world_model,))?;
        let plan_list = plan.extract::<Vec<PyObject>>(py)?;
        self.current_plan = plan_list;
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HybridAgent {
    #[pyo3(get)]
    base: Agent,
    #[pyo3(get, set)]
    reactive_component: PyObject,
    #[pyo3(get, set)]
    deliberative_component: PyObject,
    #[pyo3(get, set)]
    meta_controller: PyObject,
}

#[pymethods]
impl HybridAgent {
    #[new]
    fn new(
        agent_id: String,
        reactive_component: PyObject,
        deliberative_component: PyObject,
        meta_controller: Option<PyObject>,
        observation_space: Option<PyObject>,
        action_space: Option<PyObject>,
        py: Python,
    ) -> PyResult<Self> {
        let meta_controller = match meta_controller {
            Some(controller) => controller,
            None => {
                let locals = PyDict::new(py);
                py.run(
                    "def default_meta_controller(reactive_action, deliberative_action):\n    return 0.5",
                    None,
                    Some(locals),
                )?;
                locals.get_item("default_meta_controller")?.into_py(py)
            }
        };
        
        Ok(HybridAgent {
            base: Agent::new(agent_id, observation_space, action_space),
            reactive_component,
            deliberative_component,
            meta_controller,
        })
    }

    fn observe(&self, py: Python, observation: PyObject) -> PyResult<()> {
        self.reactive_component.call_method1(py, "observe", (observation.clone_ref(py),))?;
        self.deliberative_component.call_method1(py, "observe", (observation,))?;
        Ok(())
    }

    fn act(&self, py: Python) -> PyResult<PyObject> {
        let reactive_action = self.reactive_component.call_method0(py, "act")?;
        let deliberative_action = self.deliberative_component.call_method0(py, "act")?;
        
        let weight = self.meta_controller.call1(
            py,
            (reactive_action.clone_ref(py), deliberative_action.clone_ref(py)),
        )?;
        let weight_float = weight.extract::<f64>(py)?;
        
        if weight_float > 0.5 {
            Ok(deliberative_action)
        } else {
            Ok(reactive_action)
        }
    }
}

pub fn register_agent(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let agent_module = PyModule::new(py, "agent")?;
    
    agent_module.add_class::<AgentState>()?;
    agent_module.add_class::<Agent>()?;
    agent_module.add_class::<ReactiveAgent>()?;
    agent_module.add_class::<DeliberativeAgent>()?;
    agent_module.add_class::<HybridAgent>()?;
    
    m.add_submodule(&agent_module)?;
    
    Ok(())
}
