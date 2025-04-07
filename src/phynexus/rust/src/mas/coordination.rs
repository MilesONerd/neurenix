
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

#[pyclass]
#[derive(Clone)]
pub struct Coordinator {
    #[pyo3(get)]
    coordinator_id: String,
    #[pyo3(get, set)]
    agents: Vec<String>,
    #[pyo3(get, set)]
    tasks: HashMap<String, PyObject>,
    #[pyo3(get, set)]
    assignments: HashMap<String, String>,
    #[pyo3(get, set)]
    coordination_strategy: String,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl Coordinator {
    #[new]
    fn new(
        coordinator_id: String,
        agents: Option<Vec<String>>,
        coordination_strategy: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        Coordinator {
            coordinator_id,
            agents: agents.unwrap_or_default(),
            tasks: HashMap::new(),
            assignments: HashMap::new(),
            coordination_strategy: coordination_strategy.unwrap_or_else(|| "round_robin".to_string()),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn add_agent(&mut self, agent_id: String) -> PyResult<()> {
        if !self.agents.contains(&agent_id) {
            self.agents.push(agent_id);
        }
        Ok(())
    }

    fn remove_agent(&mut self, agent_id: &str) -> PyResult<()> {
        self.agents.retain(|id| id != agent_id);
        
        let tasks_to_reassign: Vec<String> = self.assignments
            .iter()
            .filter_map(|(task_id, assigned_agent)| {
                if assigned_agent == agent_id {
                    Some(task_id.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for task_id in tasks_to_reassign {
            self.assignments.remove(&task_id);
        }
        
        Ok(())
    }

    fn add_task(&mut self, py: Python, task_id: String, task: PyObject) -> PyResult<()> {
        self.tasks.insert(task_id, task);
        Ok(())
    }

    fn remove_task(&mut self, task_id: &str) -> PyResult<()> {
        self.tasks.remove(task_id);
        self.assignments.remove(task_id);
        Ok(())
    }

    fn assign_task(&mut self, task_id: String, agent_id: String) -> PyResult<bool> {
        if !self.tasks.contains_key(&task_id) || !self.agents.contains(&agent_id) {
            return Ok(false);
        }
        
        self.assignments.insert(task_id, agent_id);
        Ok(true)
    }

    fn assign_all_tasks(&mut self, py: Python) -> PyResult<HashMap<String, String>> {
        match self.coordination_strategy.as_str() {
            "round_robin" => self._assign_round_robin(),
            "random" => self._assign_random(py),
            "load_balanced" => self._assign_load_balanced(),
            _ => self._assign_round_robin(),
        }
    }

    fn _assign_round_robin(&mut self) -> PyResult<HashMap<String, String>> {
        if self.agents.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut assignments = HashMap::new();
        let mut agent_idx = 0;
        
        for task_id in self.tasks.keys() {
            let agent_id = &self.agents[agent_idx];
            assignments.insert(task_id.clone(), agent_id.clone());
            
            agent_idx = (agent_idx + 1) % self.agents.len();
        }
        
        self.assignments = assignments.clone();
        Ok(assignments)
    }

    fn _assign_random(&mut self, py: Python) -> PyResult<HashMap<String, String>> {
        if self.agents.is_empty() {
            return Ok(HashMap::new());
        }
        
        let random = py.import("random")?;
        let mut assignments = HashMap::new();
        
        for task_id in self.tasks.keys() {
            let agent_idx = random.call_method1("randint", (0, self.agents.len() - 1))?.extract::<usize>()?;
            let agent_id = &self.agents[agent_idx];
            assignments.insert(task_id.clone(), agent_id.clone());
        }
        
        self.assignments = assignments.clone();
        Ok(assignments)
    }

    fn _assign_load_balanced(&mut self) -> PyResult<HashMap<String, String>> {
        if self.agents.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut assignments = HashMap::new();
        let mut agent_loads = HashMap::new();
        
        for agent_id in &self.agents {
            agent_loads.insert(agent_id.clone(), 0);
        }
        
        for task_id in self.tasks.keys() {
            let min_load_agent = agent_loads
                .iter()
                .min_by_key(|(_, load)| *load)
                .map(|(agent_id, _)| agent_id.clone())
                .unwrap_or_else(|| self.agents[0].clone());
            
            assignments.insert(task_id.clone(), min_load_agent.clone());
            
            *agent_loads.get_mut(&min_load_agent).unwrap() += 1;
        }
        
        self.assignments = assignments.clone();
        Ok(assignments)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Auction {
    #[pyo3(get)]
    auction_id: String,
    #[pyo3(get, set)]
    auctioneer: String,
    #[pyo3(get, set)]
    bidders: Vec<String>,
    #[pyo3(get, set)]
    items: HashMap<String, PyObject>,
    #[pyo3(get, set)]
    bids: HashMap<String, HashMap<String, f64>>,
    #[pyo3(get, set)]
    allocations: HashMap<String, String>,
    #[pyo3(get, set)]
    auction_type: String,
    #[pyo3(get, set)]
    status: String,
}

#[pymethods]
impl Auction {
    #[new]
    fn new(
        auction_id: String,
        auctioneer: String,
        auction_type: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        Auction {
            auction_id,
            auctioneer,
            bidders: Vec::new(),
            items: HashMap::new(),
            bids: HashMap::new(),
            allocations: HashMap::new(),
            auction_type: auction_type.unwrap_or_else(|| "first_price".to_string()),
            status: "pending".to_string(),
        }
    }

    fn add_bidder(&mut self, bidder_id: String) -> PyResult<()> {
        if !self.bidders.contains(&bidder_id) {
            self.bidders.push(bidder_id);
        }
        Ok(())
    }

    fn add_item(&mut self, py: Python, item_id: String, item: PyObject) -> PyResult<()> {
        self.items.insert(item_id.clone(), item);
        self.bids.insert(item_id, HashMap::new());
        Ok(())
    }

    fn place_bid(&mut self, item_id: &str, bidder_id: &str, bid_amount: f64) -> PyResult<bool> {
        if !self.items.contains_key(item_id) || !self.bidders.contains(bidder_id) {
            return Ok(false);
        }
        
        if let Some(item_bids) = self.bids.get_mut(item_id) {
            item_bids.insert(bidder_id.to_string(), bid_amount);
            return Ok(true);
        }
        
        Ok(false)
    }

    fn run_auction(&mut self) -> PyResult<HashMap<String, String>> {
        if self.status != "pending" {
            return Ok(self.allocations.clone());
        }
        
        match self.auction_type.as_str() {
            "first_price" => self._run_first_price_auction()?,
            "second_price" => self._run_second_price_auction()?,
            _ => self._run_first_price_auction()?,
        };
        
        self.status = "completed".to_string();
        Ok(self.allocations.clone())
    }

    fn _run_first_price_auction(&mut self) -> PyResult<()> {
        for (item_id, item_bids) in &self.bids {
            if item_bids.is_empty() {
                continue;
            }
            
            let (highest_bidder, _) = item_bids
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap();
            
            self.allocations.insert(item_id.clone(), highest_bidder.clone());
        }
        
        Ok(())
    }

    fn _run_second_price_auction(&mut self) -> PyResult<()> {
        for (item_id, item_bids) in &self.bids {
            if item_bids.is_empty() {
                continue;
            }
            
            let mut sorted_bids: Vec<(&String, &f64)> = item_bids.iter().collect();
            sorted_bids.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            
            if sorted_bids.len() >= 1 {
                let (highest_bidder, _) = sorted_bids[0];
                self.allocations.insert(item_id.clone(), highest_bidder.clone());
            }
        }
        
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ContractNet {
    #[pyo3(get)]
    contract_net_id: String,
    #[pyo3(get, set)]
    manager: String,
    #[pyo3(get, set)]
    contractors: Vec<String>,
    #[pyo3(get, set)]
    tasks: HashMap<String, PyObject>,
    #[pyo3(get, set)]
    bids: HashMap<String, HashMap<String, f64>>,
    #[pyo3(get, set)]
    awards: HashMap<String, String>,
    #[pyo3(get, set)]
    status: String,
}

#[pymethods]
impl ContractNet {
    #[new]
    fn new(
        contract_net_id: String,
        manager: String,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        ContractNet {
            contract_net_id,
            manager,
            contractors: Vec::new(),
            tasks: HashMap::new(),
            bids: HashMap::new(),
            awards: HashMap::new(),
            status: "announcement".to_string(),
        }
    }

    fn add_contractor(&mut self, contractor_id: String) -> PyResult<()> {
        if !self.contractors.contains(&contractor_id) {
            self.contractors.push(contractor_id);
        }
        Ok(())
    }

    fn add_task(&mut self, py: Python, task_id: String, task: PyObject) -> PyResult<()> {
        self.tasks.insert(task_id.clone(), task);
        self.bids.insert(task_id, HashMap::new());
        Ok(())
    }

    fn submit_bid(&mut self, task_id: &str, contractor_id: &str, bid_value: f64) -> PyResult<bool> {
        if !self.tasks.contains_key(task_id) || !self.contractors.contains(contractor_id) {
            return Ok(false);
        }
        
        if self.status != "announcement" && self.status != "bidding" {
            return Ok(false);
        }
        
        if let Some(task_bids) = self.bids.get_mut(task_id) {
            task_bids.insert(contractor_id.to_string(), bid_value);
            self.status = "bidding".to_string();
            return Ok(true);
        }
        
        Ok(false)
    }

    fn award_tasks(&mut self) -> PyResult<HashMap<String, String>> {
        if self.status != "bidding" {
            return Ok(self.awards.clone());
        }
        
        for (task_id, task_bids) in &self.bids {
            if task_bids.is_empty() {
                continue;
            }
            
            let (best_contractor, _) = task_bids
                .iter()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap();
            
            self.awards.insert(task_id.clone(), best_contractor.clone());
        }
        
        self.status = "awarded".to_string();
        Ok(self.awards.clone())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct VotingMechanism {
    #[pyo3(get)]
    voting_id: String,
    #[pyo3(get, set)]
    voters: Vec<String>,
    #[pyo3(get, set)]
    candidates: Vec<String>,
    #[pyo3(get, set)]
    votes: HashMap<String, Vec<String>>,
    #[pyo3(get, set)]
    weights: HashMap<String, f64>,
    #[pyo3(get, set)]
    voting_method: String,
    #[pyo3(get, set)]
    status: String,
    #[pyo3(get, set)]
    results: HashMap<String, f64>,
    #[pyo3(get, set)]
    winner: Option<String>,
}

#[pymethods]
impl VotingMechanism {
    #[new]
    fn new(
        voting_id: String,
        voting_method: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        VotingMechanism {
            voting_id,
            voters: Vec::new(),
            candidates: Vec::new(),
            votes: HashMap::new(),
            weights: HashMap::new(),
            voting_method: voting_method.unwrap_or_else(|| "plurality".to_string()),
            status: "open".to_string(),
            results: HashMap::new(),
            winner: None,
        }
    }

    fn add_voter(&mut self, voter_id: String, weight: Option<f64>) -> PyResult<()> {
        if !self.voters.contains(&voter_id) {
            self.voters.push(voter_id.clone());
            self.weights.insert(voter_id, weight.unwrap_or(1.0));
        }
        Ok(())
    }

    fn add_candidate(&mut self, candidate_id: String) -> PyResult<()> {
        if !self.candidates.contains(&candidate_id) {
            self.candidates.push(candidate_id);
        }
        Ok(())
    }

    fn cast_vote(&mut self, voter_id: &str, preferences: Vec<String>) -> PyResult<bool> {
        if !self.voters.contains(voter_id) || self.status != "open" {
            return Ok(false);
        }
        
        for candidate_id in &preferences {
            if !self.candidates.contains(candidate_id) {
                return Ok(false);
            }
        }
        
        self.votes.insert(voter_id.to_string(), preferences);
        Ok(true)
    }

    fn count_votes(&mut self) -> PyResult<HashMap<String, f64>> {
        if self.status != "open" {
            return Ok(self.results.clone());
        }
        
        match self.voting_method.as_str() {
            "plurality" => self._count_plurality()?,
            "borda" => self._count_borda()?,
            _ => self._count_plurality()?,
        };
        
        self.status = "closed".to_string();
        Ok(self.results.clone())
    }

    fn _count_plurality(&mut self) -> PyResult<()> {
        self.results.clear();
        for candidate_id in &self.candidates {
            self.results.insert(candidate_id.clone(), 0.0);
        }
        
        for (voter_id, preferences) in &self.votes {
            if preferences.is_empty() {
                continue;
            }
            
            let first_preference = &preferences[0];
            let weight = self.weights.get(voter_id).unwrap_or(&1.0);
            
            if let Some(count) = self.results.get_mut(first_preference) {
                *count += weight;
            }
        }
        
        self.winner = self.results
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(candidate_id, _)| candidate_id.clone());
        
        Ok(())
    }

    fn _count_borda(&mut self) -> PyResult<()> {
        self.results.clear();
        for candidate_id in &self.candidates {
            self.results.insert(candidate_id.clone(), 0.0);
        }
        
        let n = self.candidates.len();
        
        for (voter_id, preferences) in &self.votes {
            let weight = self.weights.get(voter_id).unwrap_or(&1.0);
            
            for (i, candidate_id) in preferences.iter().enumerate() {
                let points = (n - i) as f64;
                
                if let Some(count) = self.results.get_mut(candidate_id) {
                    *count += points * weight;
                }
            }
        }
        
        self.winner = self.results
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(candidate_id, _)| candidate_id.clone());
        
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CoalitionFormation {
    #[pyo3(get)]
    coalition_id: String,
    #[pyo3(get, set)]
    agents: Vec<String>,
    #[pyo3(get, set)]
    coalitions: HashMap<String, Vec<String>>,
    #[pyo3(get, set)]
    values: HashMap<String, f64>,
    #[pyo3(get, set)]
    payoffs: HashMap<String, f64>,
    #[pyo3(get, set)]
    formation_method: String,
    #[pyo3(get, set)]
    status: String,
}

#[pymethods]
impl CoalitionFormation {
    #[new]
    fn new(
        coalition_id: String,
        formation_method: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        CoalitionFormation {
            coalition_id,
            agents: Vec::new(),
            coalitions: HashMap::new(),
            values: HashMap::new(),
            payoffs: HashMap::new(),
            formation_method: formation_method.unwrap_or_else(|| "greedy".to_string()),
            status: "forming".to_string(),
        }
    }

    fn add_agent(&mut self, agent_id: String) -> PyResult<()> {
        if !self.agents.contains(&agent_id) {
            self.agents.push(agent_id);
        }
        Ok(())
    }

    fn set_coalition_value(&mut self, coalition_key: String, value: f64) -> PyResult<()> {
        self.values.insert(coalition_key, value);
        Ok(())
    }

    fn form_coalitions(&mut self, py: Python) -> PyResult<HashMap<String, Vec<String>>> {
        if self.status != "forming" {
            return Ok(self.coalitions.clone());
        }
        
        match self.formation_method.as_str() {
            "greedy" => self._form_greedy_coalitions()?,
            _ => self._form_greedy_coalitions()?,
        };
        
        self.status = "formed".to_string();
        Ok(self.coalitions.clone())
    }

    fn _form_greedy_coalitions(&mut self) -> PyResult<()> {
        let coalition_key = "coalition_1".to_string();
        self.coalitions.insert(coalition_key, self.agents.clone());
        
        let total_value = self.values.values().sum::<f64>();
        let agent_payoff = if !self.agents.is_empty() {
            total_value / self.agents.len() as f64
        } else {
            0.0
        };
        
        for agent_id in &self.agents {
            self.payoffs.insert(agent_id.clone(), agent_payoff);
        }
        
        Ok(())
    }
}

pub fn register_coordination(py: Python, m: &PyModule) -> PyResult<()> {
    let coordination_module = PyModule::new(py, "coordination")?;
    
    coordination_module.add_class::<Coordinator>()?;
    coordination_module.add_class::<Auction>()?;
    coordination_module.add_class::<ContractNet>()?;
    coordination_module.add_class::<VotingMechanism>()?;
    coordination_module.add_class::<CoalitionFormation>()?;
    
    m.add_submodule(coordination_module)?;
    
    Ok(())
}
