
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use numpy::{PyArray, PyArray1, PyArray2};

use crate::tensor::Tensor;

#[pyclass]
#[derive(Clone, Debug)]
pub struct StateSpace {
    #[pyo3(get)]
    dimensions: HashMap<String, (f64, f64, bool)>,
}

#[pymethods]
impl StateSpace {
    #[new]
    fn new(dimensions: HashMap<String, (f64, f64, Option<bool>)>) -> Self {
        let dimensions = dimensions
            .into_iter()
            .map(|(k, (min, max, is_discrete))| (k, (min, max, is_discrete.unwrap_or(false))))
            .collect();
        
        StateSpace { dimensions }
    }

    fn sample(&self, py: Python) -> PyResult<PyObject> {
        let state = PyDict::new(py);
        
        for (dim_name, (min_val, max_val, is_discrete)) in &self.dimensions {
            let value = if *is_discrete {
                let min_int = *min_val as i64;
                let max_int = *max_val as i64;
                let random = py.import("random")?;
                let value = random.call_method1("randint", (min_int, max_int))?;
                value
            } else {
                let random = py.import("random")?;
                let value = random.call_method1("uniform", (*min_val, *max_val))?;
                value
            };
            
            state.set_item(dim_name, value)?;
        }
        
        Ok(state.into())
    }

    fn contains(&self, py: Python, state: &PyDict) -> PyResult<bool> {
        for (dim_name, value) in state.iter() {
            let dim_name = dim_name.extract::<String>()?;
            
            if !self.dimensions.contains_key(&dim_name) {
                return Ok(false);
            }
            
            let (min_val, max_val, is_discrete) = self.dimensions.get(&dim_name).unwrap();
            
            if *is_discrete {
                if !value.is_instance_of::<pyo3::types::PyInt>(py)? {
                    return Ok(false);
                }
            }
            
            let value_float = value.extract::<f64>()?;
            if value_float < *min_val || value_float > *max_val {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    fn to_tensor(&self, py: Python, state: &PyDict) -> PyResult<Tensor> {
        let mut sorted_dims: Vec<&String> = self.dimensions.keys().collect();
        sorted_dims.sort();
        
        let mut values = Vec::new();
        for dim in &sorted_dims {
            let value = state.get_item(*dim).unwrap();
            let value_float = value.extract::<f64>()?;
            values.push(value_float);
        }
        
        let array = PyArray1::from_vec(py, values);
        let tensor = Tensor::from_array(py, array.as_ref())?;
        
        Ok(tensor)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ActionSpace {
    #[pyo3(get)]
    dimensions: HashMap<String, (f64, f64, bool)>,
}

#[pymethods]
impl ActionSpace {
    #[new]
    fn new(dimensions: HashMap<String, (f64, f64, Option<bool>)>) -> Self {
        let dimensions = dimensions
            .into_iter()
            .map(|(k, (min, max, is_discrete))| (k, (min, max, is_discrete.unwrap_or(false))))
            .collect();
        
        ActionSpace { dimensions }
    }

    fn sample(&self, py: Python) -> PyResult<PyObject> {
        let action = PyDict::new(py);
        
        for (dim_name, (min_val, max_val, is_discrete)) in &self.dimensions {
            let value = if *is_discrete {
                let min_int = *min_val as i64;
                let max_int = *max_val as i64;
                let random = py.import("random")?;
                let value = random.call_method1("randint", (min_int, max_int))?;
                value
            } else {
                let random = py.import("random")?;
                let value = random.call_method1("uniform", (*min_val, *max_val))?;
                value
            };
            
            action.set_item(dim_name, value)?;
        }
        
        Ok(action.into())
    }

    fn contains(&self, py: Python, action: &PyDict) -> PyResult<bool> {
        for (dim_name, value) in action.iter() {
            let dim_name = dim_name.extract::<String>()?;
            
            if !self.dimensions.contains_key(&dim_name) {
                return Ok(false);
            }
            
            let (min_val, max_val, is_discrete) = self.dimensions.get(&dim_name).unwrap();
            
            if *is_discrete {
                if !value.is_instance_of::<pyo3::types::PyInt>(py)? {
                    return Ok(false);
                }
            }
            
            let value_float = value.extract::<f64>()?;
            if value_float < *min_val || value_float > *max_val {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    fn to_tensor(&self, py: Python, action: &PyDict) -> PyResult<Tensor> {
        let mut sorted_dims: Vec<&String> = self.dimensions.keys().collect();
        sorted_dims.sort();
        
        let mut values = Vec::new();
        for dim in &sorted_dims {
            let value = action.get_item(*dim).unwrap();
            let value_float = value.extract::<f64>()?;
            values.push(value_float);
        }
        
        let array = PyArray1::from_vec(py, values);
        let tensor = Tensor::from_array(py, array.as_ref())?;
        
        Ok(tensor)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Environment {
    #[pyo3(get, set)]
    state_space: Option<StateSpace>,
    #[pyo3(get, set)]
    action_space: Option<ActionSpace>,
    #[pyo3(get, set)]
    agents: HashMap<String, PyObject>,
    #[pyo3(get, set)]
    current_state: Option<PyObject>,
    #[pyo3(get, set)]
    timestep: usize,
}

#[pymethods]
impl Environment {
    #[new]
    fn new(state_space: Option<StateSpace>, action_space: Option<ActionSpace>) -> Self {
        Environment {
            state_space,
            action_space,
            agents: HashMap::new(),
            current_state: None,
            timestep: 0,
        }
    }

    fn reset(&mut self, py: Python) -> PyResult<PyObject> {
        self.timestep = 0;
        let observations = PyDict::new(py);
        Ok(observations.into())
    }

    fn step(&mut self, py: Python, actions: &PyDict) -> PyResult<PyObject> {
        self.timestep += 1;
        
        let observations = PyDict::new(py);
        let rewards = PyDict::new(py);
        let dones = PyDict::new(py);
        let infos = PyDict::new(py);
        
        let result = PyTuple::new(py, &[observations, rewards, dones, infos]);
        Ok(result.into())
    }

    fn render(&self, py: Python, mode: Option<&str>) -> PyResult<PyObject> {
        Ok(py.None())
    }

    fn close(&mut self, _py: Python) -> PyResult<()> {
        Ok(())
    }

    fn add_agent(&mut self, agent_id: String, agent: PyObject) -> PyResult<()> {
        self.agents.insert(agent_id, agent);
        Ok(())
    }

    fn remove_agent(&mut self, agent_id: &str) -> PyResult<()> {
        self.agents.remove(agent_id);
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GridEnvironment {
    #[pyo3(get)]
    base: Environment,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
    #[pyo3(get, set)]
    obstacles: Vec<(usize, usize)>,
    #[pyo3(get, set)]
    agent_positions: HashMap<String, (usize, usize)>,
    #[pyo3(get, set)]
    grid: Vec<Vec<i32>>,
}

#[pymethods]
impl GridEnvironment {
    #[new]
    fn new(width: usize, height: usize, obstacles: Option<Vec<(usize, usize)>>) -> Self {
        let mut state_space_dims = HashMap::new();
        state_space_dims.insert("x".to_string(), (0.0, (width - 1) as f64, Some(true)));
        state_space_dims.insert("y".to_string(), (0.0, (height - 1) as f64, Some(true)));
        let state_space = StateSpace::new(state_space_dims);
        
        let mut action_space_dims = HashMap::new();
        action_space_dims.insert("direction".to_string(), (0.0, 3.0, Some(true)));
        let action_space = ActionSpace::new(action_space_dims);
        
        let base = Environment::new(Some(state_space), Some(action_space));
        
        let mut grid = vec![vec![0; width]; height];
        
        let obstacles = obstacles.unwrap_or_default();
        for (x, y) in &obstacles {
            if *x < width && *y < height {
                grid[*y][*x] = 1;
            }
        }
        
        GridEnvironment {
            base,
            width,
            height,
            obstacles,
            agent_positions: HashMap::new(),
            grid,
        }
    }

    fn reset(&mut self, py: Python) -> PyResult<PyObject> {
        self.base.timestep = 0;
        self.agent_positions.clear();
        
        let random = py.import("random")?;
        
        for agent_id in self.base.agents.keys() {
            loop {
                let x = random.call_method1("randint", (0, self.width - 1))?.extract::<usize>()?;
                let y = random.call_method1("randint", (0, self.height - 1))?.extract::<usize>()?;
                
                if self.grid[y][x] == 0 && !self.agent_positions.values().any(|&pos| pos == (x, y)) {
                    self.agent_positions.insert(agent_id.clone(), (x, y));
                    break;
                }
            }
        }
        
        let observations = PyDict::new(py);
        
        for (agent_id, position) in &self.agent_positions {
            let obs = PyDict::new(py);
            obs.set_item("x", position.0)?;
            obs.set_item("y", position.1)?;
            observations.set_item(agent_id, obs)?;
        }
        
        Ok(observations.into())
    }

    fn step(&mut self, py: Python, actions: &PyDict) -> PyResult<PyObject> {
        self.base.timestep += 1;
        
        let mut new_positions = HashMap::new();
        
        for (agent_id_obj, action_obj) in actions.iter() {
            let agent_id = agent_id_obj.extract::<String>()?;
            
            if !self.agent_positions.contains_key(&agent_id) {
                continue;
            }
            
            let (x, y) = self.agent_positions[&agent_id];
            let action = action_obj.extract::<&PyDict>()?;
            let direction = action.get_item("direction").unwrap_or_else(|| py.None()).extract::<usize>().unwrap_or(0);
            
            let (new_x, new_y) = match direction {
                0 => (x, y.saturating_sub(1)), // Up
                1 => (std::cmp::min(self.width - 1, x + 1), y), // Right
                2 => (x, std::cmp::min(self.height - 1, y + 1)), // Down
                3 => (x.saturating_sub(1), y), // Left
                _ => (x, y), // Invalid direction
            };
            
            if new_y < self.grid.len() && new_x < self.grid[new_y].len() && self.grid[new_y][new_x] == 0 {
                new_positions.insert(agent_id, (new_x, new_y));
            } else {
                new_positions.insert(agent_id, (x, y));
            }
        }
        
        let mut final_positions = HashMap::new();
        let mut position_agents = HashMap::new();
        
        for (agent_id, position) in new_positions {
            position_agents
                .entry(position)
                .or_insert_with(Vec::new)
                .push(agent_id);
        }
        
        for (position, agent_ids) in position_agents {
            if agent_ids.len() == 1 {
                final_positions.insert(agent_ids[0].clone(), position);
            } else {
                for agent_id in agent_ids {
                    final_positions.insert(agent_id.clone(), self.agent_positions[&agent_id]);
                }
            }
        }
        
        self.agent_positions = final_positions;
        
        let observations = PyDict::new(py);
        let rewards = PyDict::new(py);
        let dones = PyDict::new(py);
        let infos = PyDict::new(py);
        
        for (agent_id, position) in &self.agent_positions {
            let obs = PyDict::new(py);
            obs.set_item("x", position.0)?;
            obs.set_item("y", position.1)?;
            observations.set_item(agent_id, obs)?;
            
            rewards.set_item(agent_id, 0.0)?;
            
            dones.set_item(agent_id, false)?;
            
            infos.set_item(agent_id, PyDict::new(py))?;
        }
        
        let result = PyTuple::new(py, &[observations, rewards, dones, infos]);
        Ok(result.into())
    }

    fn render(&self, py: Python, mode: Option<&str>) -> PyResult<PyObject> {
        let mode = mode.unwrap_or("human");
        
        let mut render_grid = self.grid.clone();
        
        let mut agent_indices = HashMap::new();
        for (i, agent_id) in self.base.agents.keys().enumerate() {
            agent_indices.insert(agent_id, i + 2); // 0 is empty, 1 is obstacle
        }
        
        for (agent_id, (x, y)) in &self.agent_positions {
            if *y < render_grid.len() && *x < render_grid[*y].len() {
                render_grid[*y][*x] = agent_indices[agent_id] as i32;
            }
        }
        
        if mode == "human" {
            for row in &render_grid {
                let row_str = row.iter().map(|cell| cell.to_string()).collect::<Vec<_>>().join(" ");
                println!("{}", row_str);
            }
            Ok(py.None())
        } else if mode == "rgb_array" {
            let numpy = py.import("numpy")?;
            let height = render_grid.len();
            let width = if height > 0 { render_grid[0].len() } else { 0 };
            
            let image = numpy.call_method1("zeros", ((height, width, 3), "uint8"))?;
            
            for y in 0..height {
                for x in 0..width {
                    let cell = render_grid[y][x];
                    let color = if cell == 0 {
                        [255, 255, 255]
                    } else if cell == 1 {
                        [128, 128, 128]
                    } else {
                        let agent_idx = cell - 2;
                        [
                            ((agent_idx * 50) % 256) as u8,
                            ((agent_idx * 100) % 256) as u8,
                            ((agent_idx * 150) % 256) as u8,
                        ]
                    };
                    
                    image.call_method("itemset", ((y, x, 0), color[0]))?;
                    image.call_method("itemset", ((y, x, 1), color[1]))?;
                    image.call_method("itemset", ((y, x, 2), color[2]))?;
                }
            }
            
            Ok(image.into())
        } else {
            let err = PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported render mode: {}", mode),
            );
            Err(err)
        }
    }
}

pub fn register_environment(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let environment_module = PyModule::new(py, "environment")?;
    
    environment_module.add_class::<StateSpace>()?;
    environment_module.add_class::<ActionSpace>()?;
    environment_module.add_class::<Environment>()?;
    environment_module.add_class::<GridEnvironment>()?;
    
    m.add_submodule(&environment_module)?;
    
    Ok(())
}
