
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

#[pyclass]
#[derive(Clone)]
pub struct SymbolicKnowledgeBase {
    facts: HashSet<String>,
    rules: Vec<(String, Vec<String>)>,
    predicates: HashSet<String>,
}

#[pymethods]
impl SymbolicKnowledgeBase {
    #[new]
    fn new() -> Self {
        SymbolicKnowledgeBase {
            facts: HashSet::new(),
            rules: Vec::new(),
            predicates: HashSet::new(),
        }
    }

    fn add_fact(&mut self, fact: String) -> PyResult<()> {
        self.facts.insert(fact.clone());
        
        if let Some(predicate_end) = fact.find('(') {
            let predicate = fact[..predicate_end].to_string();
            self.predicates.insert(predicate);
        }
        
        Ok(())
    }

    fn add_rule(&mut self, head: String, body: Vec<String>) -> PyResult<()> {
        if let Some(predicate_end) = head.find('(') {
            let predicate = head[..predicate_end].to_string();
            self.predicates.insert(predicate);
        }
        
        for literal in &body {
            if let Some(predicate_end) = literal.find('(') {
                let predicate = literal[..predicate_end].to_string();
                self.predicates.insert(predicate);
            }
        }
        
        self.rules.push((head, body));
        Ok(())
    }

    fn get_facts(&self) -> PyResult<HashSet<String>> {
        Ok(self.facts.clone())
    }

    fn get_rules(&self) -> PyResult<Vec<(String, Vec<String>)>> {
        Ok(self.rules.clone())
    }

    fn get_predicates(&self) -> PyResult<HashSet<String>> {
        Ok(self.predicates.clone())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct RuleSet {
    rules: Vec<(String, Vec<String>)>,
    rule_weights: Vec<f64>,
}

#[pymethods]
impl RuleSet {
    #[new]
    fn new() -> Self {
        RuleSet {
            rules: Vec::new(),
            rule_weights: Vec::new(),
        }
    }

    fn add_rule(&mut self, rule: (String, Vec<String>), weight: Option<f64>) -> PyResult<()> {
        self.rules.push(rule);
        self.rule_weights.push(weight.unwrap_or(1.0));
        Ok(())
    }

    fn get_rules(&self) -> PyResult<Vec<(String, Vec<String>)>> {
        Ok(self.rules.clone())
    }

    fn get_rule_weights(&self) -> PyResult<Vec<f64>> {
        Ok(self.rule_weights.clone())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.rules.len())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct LogicProgram {
    kb: SymbolicKnowledgeBase,
    inferred_facts: HashSet<String>,
}

#[pymethods]
impl LogicProgram {
    #[new]
    fn new(kb: Option<SymbolicKnowledgeBase>) -> Self {
        LogicProgram {
            kb: kb.unwrap_or_else(SymbolicKnowledgeBase::new),
            inferred_facts: HashSet::new(),
        }
    }

    fn add_knowledge_base(&mut self, kb: SymbolicKnowledgeBase) -> PyResult<()> {
        self.kb = kb;
        Ok(())
    }

    fn query(&mut self, query: String) -> PyResult<bool> {
        let mut visited = HashSet::new();
        Ok(self.backward_chain(query, &mut visited))
    }

    fn get_inferred_facts(&self) -> PyResult<HashSet<String>> {
        Ok(self.inferred_facts.clone())
    }
}

impl LogicProgram {
    fn backward_chain(&mut self, query: String, visited: &mut HashSet<String>) -> bool {
        if visited.contains(&query) {
            return false;
        }
        
        visited.insert(query.clone());
        
        if self.kb.facts.contains(&query) || self.inferred_facts.contains(&query) {
            return true;
        }
        
        if let Some(predicate_end) = query.find('(') {
            let query_predicate = query[..predicate_end].to_string();
            let query_args_str = query[predicate_end + 1..query.len() - 1].to_string();
            let query_args: Vec<String> = query_args_str.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            
            for (head, body) in &self.kb.rules {
                if let Some(head_predicate_end) = head.find('(') {
                    let head_predicate = head[..head_predicate_end].to_string();
                    
                    if head_predicate == query_predicate {
                        let head_args_str = head[head_predicate_end + 1..head.len() - 1].to_string();
                        let head_args: Vec<String> = head_args_str.split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        
                        if let Some(substitution) = self.unify(&query_args, &head_args) {
                            let mut all_satisfied = true;
                            
                            for literal in body {
                                if let Some(literal_predicate_end) = literal.find('(') {
                                    let literal_predicate = literal[..literal_predicate_end].to_string();
                                    let literal_args_str = literal[literal_predicate_end + 1..literal.len() - 1].to_string();
                                    let literal_args: Vec<String> = literal_args_str.split(',')
                                        .map(|s| s.trim().to_string())
                                        .collect();
                                    
                                    let substituted_args: Vec<String> = literal_args.iter()
                                        .map(|arg| substitution.get(arg).cloned().unwrap_or_else(|| arg.clone()))
                                        .collect();
                                    
                                    let substituted_literal = format!("{}({})", literal_predicate, substituted_args.join(", "));
                                    
                                    let mut new_visited = visited.clone();
                                    if !self.backward_chain(substituted_literal, &mut new_visited) {
                                        all_satisfied = false;
                                        break;
                                    }
                                }
                            }
                            
                            if all_satisfied {
                                self.inferred_facts.insert(query.clone());
                                return true;
                            }
                        }
                    }
                }
            }
        }
        
        false
    }

    fn unify(&self, query_args: &[String], head_args: &[String]) -> Option<HashMap<String, String>> {
        if query_args.len() != head_args.len() {
            return None;
        }
        
        let mut substitution = HashMap::new();
        
        for (i, head_arg) in head_args.iter().enumerate() {
            let query_arg = &query_args[i];
            
            if head_arg.chars().all(|c| c.is_uppercase()) {
                if let Some(existing_value) = substitution.get(head_arg) {
                    if existing_value != query_arg {
                        return None;
                    }
                } else {
                    substitution.insert(head_arg.clone(), query_arg.clone());
                }
            } else if query_arg != head_arg {
                return None;
            }
        }
        
        Some(substitution)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct SymbolicReasoner {
    logic_program: LogicProgram,
}

#[pymethods]
impl SymbolicReasoner {
    #[new]
    fn new(logic_program: Option<LogicProgram>) -> Self {
        SymbolicReasoner {
            logic_program: logic_program.unwrap_or_else(|| LogicProgram::new(None)),
        }
    }

    fn set_logic_program(&mut self, logic_program: LogicProgram) -> PyResult<()> {
        self.logic_program = logic_program;
        Ok(())
    }

    fn reason(&mut self, query: String) -> PyResult<bool> {
        self.logic_program.query(query)
    }

    fn batch_reason(&mut self, queries: Vec<String>) -> PyResult<Vec<bool>> {
        let mut results = Vec::with_capacity(queries.len());
        
        for query in queries {
            results.push(self.logic_program.query(query)?);
        }
        
        Ok(results)
    }

    fn to_tensor(&mut self, queries: Vec<String>) -> PyResult<Tensor> {
        let results = self.batch_reason(queries)?;
        let values: Vec<f64> = results.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        
        let shape = vec![values.len() as i64];
        Tensor::new(values, shape)
    }
}

pub fn register_symbolic(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "symbolic")?;
    
    submodule.add_class::<SymbolicKnowledgeBase>()?;
    submodule.add_class::<RuleSet>()?;
    submodule.add_class::<LogicProgram>()?;
    submodule.add_class::<SymbolicReasoner>()?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
