
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;
use std::f64::consts::PI;
use num_complex::{Complex64, Complex};
use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};

pub type ComplexF64 = Complex64;

pub type StateVector = Vec<ComplexF64>;

pub type UnitaryMatrix = Vec<Vec<ComplexF64>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    H,      // Hadamard
    X,      // Pauli-X
    Y,      // Pauli-Y
    Z,      // Pauli-Z
    CX,     // CNOT
    CZ,     // Controlled-Z
    SWAP,   // SWAP
    T,      // T gate
    S,      // S gate
    RX,     // RX rotation
    RY,     // RY rotation
    RZ,     // RZ rotation
    U1,     // U1 gate
    U2,     // U2 gate
    U3,     // U3 gate
    MEASURE, // Measurement
}

#[derive(Debug, Clone)]
pub struct GateOperation {
    pub gate_type: GateType,
    pub qubits: Vec<usize>,
    pub parameters: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub num_qubits: usize,
    pub name: String,
    pub operations: Vec<GateOperation>,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| format!("circuit_{}q", num_qubits));
        Self {
            num_qubits,
            name,
            operations: Vec::new(),
        }
    }
    
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::H,
            qubits: vec![qubit],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::X,
            qubits: vec![qubit],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::Y,
            qubits: vec![qubit],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::Z,
            qubits: vec![qubit],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::CX,
            qubits: vec![control, target],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::CZ,
            qubits: vec![control, target],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::RX,
            qubits: vec![qubit],
            parameters: vec![theta],
        });
        self
    }
    
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::RY,
            qubits: vec![qubit],
            parameters: vec![theta],
        });
        self
    }
    
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        self.operations.push(GateOperation {
            gate_type: GateType::RZ,
            qubits: vec![qubit],
            parameters: vec![theta],
        });
        self
    }
    
    pub fn measure(&mut self, qubit: usize, classical_bit: Option<usize>) -> &mut Self {
        let classical_bit = classical_bit.unwrap_or(qubit);
        self.operations.push(GateOperation {
            gate_type: GateType::MEASURE,
            qubits: vec![qubit, classical_bit],
            parameters: Vec::new(),
        });
        self
    }
    
    pub fn to_matrix(&self) -> UnitaryMatrix {
        let simulator = SimulatorBackend::new();
        simulator.to_matrix(self)
    }
    
    pub fn run(&self, shots: usize) -> HashMap<String, usize> {
        let simulator = SimulatorBackend::new();
        simulator.run(self, shots)
    }
}

#[derive(Debug, Clone)]
pub struct ParameterizedCircuit {
    pub circuit: QuantumCircuit,
    pub parameters: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

impl ParameterizedCircuit {
    pub fn new(num_qubits: usize, parameters: Option<Vec<String>>, name: Option<String>) -> Self {
        let circuit = QuantumCircuit::new(num_qubits, name);
        let parameters = parameters.unwrap_or_else(Vec::new);
        let parameter_values = parameters.iter().map(|p| (p.clone(), 0.0)).collect();
        
        Self {
            circuit,
            parameters,
            parameter_values,
        }
    }
    
    pub fn rx_param(&mut self, qubit: usize, param_name: &str) -> &mut Self {
        if !self.parameters.contains(&param_name.to_string()) {
            self.parameters.push(param_name.to_string());
            self.parameter_values.insert(param_name.to_string(), 0.0);
        }
        
        self.circuit.operations.push(GateOperation {
            gate_type: GateType::RX,
            qubits: vec![qubit],
            parameters: vec![0.0], // Placeholder, will be replaced when binding
        });
        
        self
    }
    
    pub fn ry_param(&mut self, qubit: usize, param_name: &str) -> &mut Self {
        if !self.parameters.contains(&param_name.to_string()) {
            self.parameters.push(param_name.to_string());
            self.parameter_values.insert(param_name.to_string(), 0.0);
        }
        
        self.circuit.operations.push(GateOperation {
            gate_type: GateType::RY,
            qubits: vec![qubit],
            parameters: vec![0.0], // Placeholder, will be replaced when binding
        });
        
        self
    }
    
    pub fn rz_param(&mut self, qubit: usize, param_name: &str) -> &mut Self {
        if !self.parameters.contains(&param_name.to_string()) {
            self.parameters.push(param_name.to_string());
            self.parameter_values.insert(param_name.to_string(), 0.0);
        }
        
        self.circuit.operations.push(GateOperation {
            gate_type: GateType::RZ,
            qubits: vec![qubit],
            parameters: vec![0.0], // Placeholder, will be replaced when binding
        });
        
        self
    }
    
    pub fn bind_parameters(&self, parameter_values: &HashMap<String, f64>) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.circuit.num_qubits, Some(self.circuit.name.clone()));
        
        for op in &self.circuit.operations {
            match op.gate_type {
                GateType::H => {
                    circuit.h(op.qubits[0]);
                },
                GateType::X => {
                    circuit.x(op.qubits[0]);
                },
                GateType::Y => {
                    circuit.y(op.qubits[0]);
                },
                GateType::Z => {
                    circuit.z(op.qubits[0]);
                },
                GateType::CX => {
                    circuit.cx(op.qubits[0], op.qubits[1]);
                },
                GateType::CZ => {
                    circuit.cz(op.qubits[0], op.qubits[1]);
                },
                GateType::RX => {
                    if let Some(param_name) = self.parameters.get(op.parameters[0] as usize) {
                        let param_value = parameter_values.get(param_name).unwrap_or_else(|| &0.0);
                        circuit.rx(op.qubits[0], *param_value);
                    } else {
                        circuit.rx(op.qubits[0], op.parameters[0]);
                    }
                },
                GateType::RY => {
                    if let Some(param_name) = self.parameters.get(op.parameters[0] as usize) {
                        let param_value = parameter_values.get(param_name).unwrap_or_else(|| &0.0);
                        circuit.ry(op.qubits[0], *param_value);
                    } else {
                        circuit.ry(op.qubits[0], op.parameters[0]);
                    }
                },
                GateType::RZ => {
                    if let Some(param_name) = self.parameters.get(op.parameters[0] as usize) {
                        let param_value = parameter_values.get(param_name).unwrap_or_else(|| &0.0);
                        circuit.rz(op.qubits[0], *param_value);
                    } else {
                        circuit.rz(op.qubits[0], op.parameters[0]);
                    }
                },
                GateType::MEASURE => {
                    circuit.measure(op.qubits[0], Some(op.qubits[1]));
                },
                _ => {
                }
            }
        }
        
        circuit
    }
    
    pub fn set_parameter(&mut self, param_name: &str, value: f64) {
        if !self.parameters.contains(&param_name.to_string()) {
            self.parameters.push(param_name.to_string());
        }
        
        self.parameter_values.insert(param_name.to_string(), value);
    }
}

pub trait QuantumBackendTrait {
    fn run(&self, circuit: &QuantumCircuit, shots: usize) -> HashMap<String, usize>;
    
    fn to_matrix(&self, circuit: &QuantumCircuit) -> UnitaryMatrix;
}

#[derive(Debug, Clone)]
pub struct SimulatorBackend;

impl SimulatorBackend {
    pub fn new() -> Self {
        Self
    }
    
    fn apply_gate(&self, gate: &GateOperation, state: &mut StateVector) {
        let gate_matrix = self.get_gate_matrix(gate);
        let dim = state.len();
        
        let mut result = vec![Complex64::new(0.0, 0.0); dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i] += gate_matrix[i][j] * state[j];
            }
        }
        
        *state = result;
    }
    
    fn get_gate_matrix(&self, gate: &GateOperation) -> UnitaryMatrix {
        let mut n = 0;
        for &qubit in &gate.qubits {
            n = std::cmp::max(n, qubit + 1);
        }
        
        let dim = 1 << n;
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        
        match gate.gate_type {
            GateType::H => {
                let qubit = gate.qubits[0];
                let mask = 1 << qubit;
                
                for i in 0..dim {
                    let j = i ^ mask;
                    if i & mask != 0 {
                        matrix[i][i] = Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0);
                        matrix[i][j] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                    } else {
                        matrix[i][i] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                        matrix[i][j] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                    }
                }
            },
            GateType::X => {
                let qubit = gate.qubits[0];
                let mask = 1 << qubit;
                
                for i in 0..dim {
                    let j = i ^ mask;
                    matrix[i][j] = Complex64::new(1.0, 0.0);
                }
            },
            GateType::Y => {
                let qubit = gate.qubits[0];
                let mask = 1 << qubit;
                
                for i in 0..dim {
                    let j = i ^ mask;
                    if i & mask != 0 {
                        matrix[i][j] = Complex64::new(0.0, -1.0);
                    } else {
                        matrix[i][j] = Complex64::new(0.0, 1.0);
                    }
                }
            },
            GateType::Z => {
                let qubit = gate.qubits[0];
                let mask = 1 << qubit;
                
                for i in 0..dim {
                    if i & mask != 0 {
                        matrix[i][i] = Complex64::new(-1.0, 0.0);
                    } else {
                        matrix[i][i] = Complex64::new(1.0, 0.0);
                    }
                }
            },
            GateType::RX => {
                let qubit = gate.qubits[0];
                let theta = gate.parameters[0];
                let mask = 1 << qubit;
                
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                
                for i in 0..dim {
                    let j = i ^ mask;
                    if i & mask != 0 {
                        matrix[i][i] = Complex64::new(cos, 0.0);
                        matrix[i][j] = Complex64::new(0.0, -sin);
                    } else {
                        matrix[i][i] = Complex64::new(cos, 0.0);
                        matrix[i][j] = Complex64::new(0.0, -sin);
                    }
                }
            },
            GateType::RY => {
                let qubit = gate.qubits[0];
                let theta = gate.parameters[0];
                let mask = 1 << qubit;
                
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                
                for i in 0..dim {
                    let j = i ^ mask;
                    if i & mask != 0 {
                        matrix[i][i] = Complex64::new(cos, 0.0);
                        matrix[i][j] = Complex64::new(-sin, 0.0);
                    } else {
                        matrix[i][i] = Complex64::new(cos, 0.0);
                        matrix[i][j] = Complex64::new(sin, 0.0);
                    }
                }
            },
            GateType::RZ => {
                let qubit = gate.qubits[0];
                let theta = gate.parameters[0];
                let mask = 1 << qubit;
                
                let phase_pos = Complex64::new(0.0, -theta / 2.0).exp();
                let phase_neg = Complex64::new(0.0, theta / 2.0).exp();
                
                for i in 0..dim {
                    if i & mask != 0 {
                        matrix[i][i] = phase_neg;
                    } else {
                        matrix[i][i] = phase_pos;
                    }
                }
            },
            GateType::CX => {
                let control = gate.qubits[0];
                let target = gate.qubits[1];
                let control_mask = 1 << control;
                let target_mask = 1 << target;
                
                for i in 0..dim {
                    if i & control_mask != 0 {
                        let j = i ^ target_mask;
                        matrix[i][j] = Complex64::new(1.0, 0.0);
                    } else {
                        matrix[i][i] = Complex64::new(1.0, 0.0);
                    }
                }
            },
            GateType::CZ => {
                let control = gate.qubits[0];
                let target = gate.qubits[1];
                let control_mask = 1 << control;
                let target_mask = 1 << target;
                
                for i in 0..dim {
                    if (i & control_mask != 0) && (i & target_mask != 0) {
                        matrix[i][i] = Complex64::new(-1.0, 0.0);
                    } else {
                        matrix[i][i] = Complex64::new(1.0, 0.0);
                    }
                }
            },
            _ => {
                for i in 0..dim {
                    matrix[i][i] = Complex64::new(1.0, 0.0);
                }
            }
        }
        
        matrix
    }
}

impl QuantumBackendTrait for SimulatorBackend {
    fn run(&self, circuit: &QuantumCircuit, shots: usize) -> HashMap<String, usize> {
        let n = circuit.num_qubits;
        let dim = 1 << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        
        for op in &circuit.operations {
            if op.gate_type != GateType::MEASURE {
                self.apply_gate(op, &mut state);
            }
        }
        
        let mut counts = HashMap::new();
        let mut rng = rand::thread_rng();
        
        let probabilities: Vec<f64> = state.iter().map(|&x| x.norm_sqr()).collect();
        
        let dist = WeightedIndex::new(&probabilities).unwrap();
        
        for _ in 0..shots {
            let outcome = dist.sample(&mut rng);
            let bitstring = format!("{:0width$b}", outcome, width = n);
            *counts.entry(bitstring).or_insert(0) += 1;
        }
        
        counts
    }
    
    fn to_matrix(&self, circuit: &QuantumCircuit) -> UnitaryMatrix {
        let n = circuit.num_qubits;
        let dim = 1 << n;
        
        let mut unitary = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            unitary[i][i] = Complex64::new(1.0, 0.0);
        }
        
        for op in &circuit.operations {
            if op.gate_type != GateType::MEASURE {
                let gate_matrix = self.get_gate_matrix(op);
                
                let mut result = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
                for i in 0..dim {
                    for j in 0..dim {
                        for k in 0..dim {
                            result[i][j] += gate_matrix[i][k] * unitary[k][j];
                        }
                    }
                }
                
                unitary = result;
            }
        }
        
        unitary
    }
}

pub mod utils {
    use super::*;
    
    pub fn state_fidelity(state1: &StateVector, state2: &StateVector) -> f64 {
        let mut inner_product = Complex64::new(0.0, 0.0);
        for i in 0..state1.len() {
            inner_product += state1[i].conj() * state2[i];
        }
        
        inner_product.norm_sqr()
    }
    
    pub fn density_matrix(state: &StateVector) -> Vec<Vec<ComplexF64>> {
        let dim = state.len();
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        
        for i in 0..dim {
            for j in 0..dim {
                rho[i][j] = state[i] * state[j].conj();
            }
        }
        
        rho
    }
    
    pub fn bell_pair() -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(2, Some("bell_pair".to_string()));
        circuit.h(0).cx(0, 1);
        circuit
    }
    
    pub fn ghz_state(num_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits, Some(format!("ghz_{}", num_qubits)));
        circuit.h(0);
        for i in 0..num_qubits-1 {
            circuit.cx(i, i+1);
        }
        circuit
    }
    
    pub fn w_state(num_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits, Some(format!("w_{}", num_qubits)));
        circuit.x(0);
        for i in 0..num_qubits-1 {
            let theta = ((1.0 / (num_qubits - i) as f64).sqrt()).acos();
            circuit.ry(i+1, theta);
            circuit.cx(i, i+1);
        }
        circuit
    }
    
    pub fn qft(num_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(num_qubits, Some(format!("qft_{}", num_qubits)));
        for i in 0..num_qubits {
            circuit.h(i);
            for j in i+1..num_qubits {
                let angle = PI / (1 << (j - i)) as f64;
                circuit.cz(i, j);
            }
        }
        
        for i in 0..num_qubits/2 {
            let j = num_qubits - i - 1;
            circuit.cx(i, j);
            circuit.cx(j, i);
            circuit.cx(i, j);
        }
        
        circuit
    }
}

pub fn register_quantum(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let quantum = PyModule::new(py, "quantum")?;
    
    quantum.add_function(wrap_pyfunction!(py_is_quantum_available, quantum)?)?;
    quantum.add_function(wrap_pyfunction!(py_get_quantum_device_count, quantum)?)?;
    
    m.add_submodule(&quantum)?;
    
    Ok(())
}

#[pyfunction]
fn py_is_quantum_available() -> PyResult<bool> {
    Ok(true)
}

#[pyfunction]
fn py_get_quantum_device_count() -> PyResult<usize> {
    Ok(1)
}
