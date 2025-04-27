
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

#[pyclass]
#[derive(Clone, Debug)]
pub struct Message {
    #[pyo3(get, set)]
    sender: String,
    #[pyo3(get, set)]
    recipient: String,
    #[pyo3(get, set)]
    content: PyObject,
    #[pyo3(get, set)]
    timestamp: u64,
    #[pyo3(get, set)]
    message_id: String,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl Message {
    #[new]
    fn new(
        sender: String,
        recipient: String,
        content: PyObject,
        timestamp: Option<u64>,
        message_id: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        let timestamp = timestamp.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });
        
        let message_id = message_id.unwrap_or_else(|| {
            format!("{}-{}-{}", sender, recipient, timestamp)
        });
        
        let metadata = metadata.unwrap_or_default();
        
        Message {
            sender,
            recipient,
            content,
            timestamp,
            message_id,
            metadata,
        }
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("sender", &self.sender)?;
        dict.set_item("recipient", &self.recipient)?;
        dict.set_item("content", self.content.clone_ref(py))?;
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("message_id", &self.message_id)?;
        
        let metadata_dict = PyDict::new(py);
        for (key, value) in &self.metadata {
            metadata_dict.set_item(key, value.clone_ref(py))?;
        }
        dict.set_item("metadata", metadata_dict)?;
        
        Ok(dict.into())
    }

    fn create_reply(&self, py: Python, content: PyObject, metadata: Option<HashMap<String, PyObject>>) -> PyResult<Message> {
        let sender = self.recipient.clone();
        let recipient = self.sender.clone();
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let message_id = format!("{}-{}-{}-reply", sender, recipient, timestamp);
        
        let mut new_metadata = metadata.unwrap_or_default();
        
        new_metadata.insert("in_reply_to".to_string(), self.message_id.clone().into_py(py));
        
        Ok(Message {
            sender,
            recipient,
            content,
            timestamp,
            message_id,
            metadata: new_metadata,
        })
    }

    fn is_reply(&self, py: Python) -> bool {
        self.metadata.contains_key("in_reply_to")
    }

    fn get_reply_to(&self, py: Python) -> Option<String> {
        if let Some(reply_to) = self.metadata.get("in_reply_to") {
            reply_to.extract::<String>(py).ok()
        } else {
            None
        }
    }

    fn add_metadata(&mut self, key: String, value: PyObject) -> PyResult<()> {
        self.metadata.insert(key, value);
        Ok(())
    }

    fn get_metadata(&self, py: Python, key: &str) -> Option<PyObject> {
        self.metadata.get(key).map(|value| value.clone_ref(py))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Channel {
    #[pyo3(get)]
    channel_id: String,
    #[pyo3(get, set)]
    participants: Vec<String>,
    #[pyo3(get, set)]
    messages: VecDeque<Message>,
    #[pyo3(get, set)]
    max_messages: usize,
    #[pyo3(get, set)]
    is_reliable: bool,
    #[pyo3(get, set)]
    latency: f64,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl Channel {
    #[new]
    fn new(
        channel_id: String,
        participants: Vec<String>,
        max_messages: Option<usize>,
        is_reliable: Option<bool>,
        latency: Option<f64>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        Channel {
            channel_id,
            participants,
            messages: VecDeque::new(),
            max_messages: max_messages.unwrap_or(1000),
            is_reliable: is_reliable.unwrap_or(true),
            latency: latency.unwrap_or(0.0),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn send_message(&mut self, py: Python, message: Message) -> PyResult<bool> {
        if !self.participants.contains(&message.sender) || !self.participants.contains(&message.recipient) {
            return Ok(false);
        }
        
        if !self.is_reliable {
            let random = py.import("random")?;
            let random_value = random.call_method0("random")?.extract::<f64>()?;
            if random_value < 0.1 {
                return Ok(false);
            }
        }
        
        self.messages.push_back(message);
        
        while self.messages.len() > self.max_messages {
            self.messages.pop_front();
        }
        
        Ok(true)
    }

    fn receive_messages(&mut self, py: Python, agent_id: &str) -> PyResult<Vec<Message>> {
        if !self.participants.contains(&agent_id.to_string()) {
            return Ok(Vec::new());
        }
        
        let mut received_messages = Vec::new();
        let mut remaining_messages = VecDeque::new();
        
        while let Some(message) = self.messages.pop_front() {
            if message.recipient == agent_id {
                received_messages.push(message);
            } else {
                remaining_messages.push_back(message);
            }
        }
        
        self.messages = remaining_messages;
        
        Ok(received_messages)
    }

    fn peek_messages(&self, py: Python, agent_id: &str) -> PyResult<Vec<Message>> {
        if !self.participants.contains(&agent_id.to_string()) {
            return Ok(Vec::new());
        }
        
        let received_messages = self.messages
            .iter()
            .filter(|message| message.recipient == agent_id)
            .cloned()
            .collect();
        
        Ok(received_messages)
    }

    fn get_all_messages(&self, py: Python) -> PyResult<Vec<Message>> {
        Ok(self.messages.iter().cloned().collect())
    }

    fn clear(&mut self) -> PyResult<()> {
        self.messages.clear();
        Ok(())
    }

    fn add_participant(&mut self, agent_id: String) -> PyResult<()> {
        if !self.participants.contains(&agent_id) {
            self.participants.push(agent_id);
        }
        Ok(())
    }

    fn remove_participant(&mut self, agent_id: &str) -> PyResult<()> {
        self.participants.retain(|id| id != agent_id);
        Ok(())
    }

    fn has_participant(&self, agent_id: &str) -> bool {
        self.participants.contains(&agent_id.to_string())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Protocol {
    #[pyo3(get)]
    protocol_id: String,
    #[pyo3(get, set)]
    message_types: Vec<String>,
    #[pyo3(get, set)]
    validation_rules: HashMap<String, PyObject>,
    #[pyo3(get, set)]
    state_machine: HashMap<String, Vec<String>>,
    #[pyo3(get, set)]
    current_state: String,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl Protocol {
    #[new]
    fn new(
        protocol_id: String,
        message_types: Vec<String>,
        validation_rules: Option<HashMap<String, PyObject>>,
        state_machine: Option<HashMap<String, Vec<String>>>,
        initial_state: Option<String>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        Protocol {
            protocol_id,
            message_types,
            validation_rules: validation_rules.unwrap_or_default(),
            state_machine: state_machine.unwrap_or_default(),
            current_state: initial_state.unwrap_or_else(|| "initial".to_string()),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn validate_message(&self, py: Python, message: &Message) -> PyResult<bool> {
        if let Some(message_type) = message.metadata.get("type") {
            let message_type_str = message_type.extract::<String>(py)?;
            
            if !self.message_types.contains(&message_type_str) {
                return Ok(false);
            }
            
            if let Some(allowed_types) = self.state_machine.get(&self.current_state) {
                if !allowed_types.contains(&message_type_str) {
                    return Ok(false);
                }
            }
            
            if let Some(validation_rule) = self.validation_rules.get(&message_type_str) {
                let args = PyTuple::new(py, &[message.to_dict(py)?]);
                let result = validation_rule.call1(py, args)?;
                return result.extract::<bool>(py);
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn transition(&mut self, py: Python, message: &Message) -> PyResult<bool> {
        if !self.validate_message(py, message)? {
            return Ok(false);
        }
        
        if let Some(message_type) = message.metadata.get("type") {
            let message_type_str = message_type.extract::<String>(py)?;
            
            if let Some(allowed_types) = self.state_machine.get(&self.current_state) {
                if allowed_types.contains(&message_type_str) {
                    if let Some(next_state) = message.metadata.get("next_state") {
                        let next_state_str = next_state.extract::<String>(py)?;
                        self.current_state = next_state_str;
                        return Ok(true);
                    }
                }
            }
        }
        
        Ok(false)
    }

    fn reset(&mut self, initial_state: Option<String>) -> PyResult<()> {
        self.current_state = initial_state.unwrap_or_else(|| "initial".to_string());
        Ok(())
    }

    fn add_message_type(&mut self, message_type: String) -> PyResult<()> {
        if !self.message_types.contains(&message_type) {
            self.message_types.push(message_type);
        }
        Ok(())
    }

    fn add_validation_rule(&mut self, message_type: String, validation_rule: PyObject) -> PyResult<()> {
        self.validation_rules.insert(message_type, validation_rule);
        Ok(())
    }

    fn add_state_transition(&mut self, from_state: String, allowed_message_types: Vec<String>) -> PyResult<()> {
        self.state_machine.insert(from_state, allowed_message_types);
        Ok(())
    }

    fn get_state(&self) -> String {
        self.current_state.clone()
    }

    fn is_terminal(&self) -> bool {
        !self.state_machine.contains_key(&self.current_state)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Mailbox {
    #[pyo3(get)]
    owner_id: String,
    #[pyo3(get, set)]
    inbox: VecDeque<Message>,
    #[pyo3(get, set)]
    outbox: VecDeque<Message>,
    #[pyo3(get, set)]
    max_messages: usize,
    #[pyo3(get, set)]
    filters: Vec<PyObject>,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl Mailbox {
    #[new]
    fn new(
        owner_id: String,
        max_messages: Option<usize>,
        filters: Option<Vec<PyObject>>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        Mailbox {
            owner_id,
            inbox: VecDeque::new(),
            outbox: VecDeque::new(),
            max_messages: max_messages.unwrap_or(1000),
            filters: filters.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn receive(&mut self, py: Python, message: Message) -> PyResult<bool> {
        if message.recipient != self.owner_id {
            return Ok(false);
        }
        
        for filter in &self.filters {
            let args = PyTuple::new(py, &[message.to_dict(py)?]);
            let result = filter.call1(py, args)?;
            if !result.extract::<bool>(py)? {
                return Ok(false);
            }
        }
        
        self.inbox.push_back(message);
        
        while self.inbox.len() > self.max_messages {
            self.inbox.pop_front();
        }
        
        Ok(true)
    }

    fn queue_message(&mut self, message: Message) -> PyResult<bool> {
        if message.sender != self.owner_id {
            return Ok(false);
        }
        
        self.outbox.push_back(message);
        
        while self.outbox.len() > self.max_messages {
            self.outbox.pop_front();
        }
        
        Ok(true)
    }

    fn get_next_message(&mut self) -> Option<Message> {
        self.inbox.pop_front()
    }

    fn peek_next_message(&self) -> Option<Message> {
        self.inbox.front().cloned()
    }

    fn get_next_outgoing_message(&mut self) -> Option<Message> {
        self.outbox.pop_front()
    }

    fn get_all_messages(&self, py: Python) -> PyResult<Vec<Message>> {
        Ok(self.inbox.iter().cloned().collect())
    }

    fn get_all_outgoing_messages(&self, py: Python) -> PyResult<Vec<Message>> {
        Ok(self.outbox.iter().cloned().collect())
    }

    fn clear_inbox(&mut self) -> PyResult<()> {
        self.inbox.clear();
        Ok(())
    }

    fn clear_outbox(&mut self) -> PyResult<()> {
        self.outbox.clear();
        Ok(())
    }

    fn add_filter(&mut self, filter: PyObject) -> PyResult<()> {
        self.filters.push(filter);
        Ok(())
    }

    fn clear_filters(&mut self) -> PyResult<()> {
        self.filters.clear();
        Ok(())
    }

    fn count_inbox(&self) -> usize {
        self.inbox.len()
    }

    fn count_outbox(&self) -> usize {
        self.outbox.len()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CommunicationNetwork {
    #[pyo3(get)]
    network_id: String,
    #[pyo3(get, set)]
    channels: HashMap<String, Channel>,
    #[pyo3(get, set)]
    mailboxes: HashMap<String, Mailbox>,
    #[pyo3(get, set)]
    routing_table: HashMap<String, String>,
    #[pyo3(get, set)]
    protocols: HashMap<String, Protocol>,
    #[pyo3(get, set)]
    metadata: HashMap<String, PyObject>,
}

#[pymethods]
impl CommunicationNetwork {
    #[new]
    fn new(
        network_id: String,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        CommunicationNetwork {
            network_id,
            channels: HashMap::new(),
            mailboxes: HashMap::new(),
            routing_table: HashMap::new(),
            protocols: HashMap::new(),
            metadata: metadata.unwrap_or_default(),
        }
    }

    fn add_channel(&mut self, channel: Channel) -> PyResult<()> {
        self.channels.insert(channel.channel_id.clone(), channel);
        Ok(())
    }

    fn remove_channel(&mut self, channel_id: &str) -> PyResult<()> {
        self.channels.remove(channel_id);
        Ok(())
    }

    fn add_mailbox(&mut self, mailbox: Mailbox) -> PyResult<()> {
        self.mailboxes.insert(mailbox.owner_id.clone(), mailbox);
        Ok(())
    }

    fn remove_mailbox(&mut self, agent_id: &str) -> PyResult<()> {
        self.mailboxes.remove(agent_id);
        Ok(())
    }

    fn add_protocol(&mut self, protocol: Protocol) -> PyResult<()> {
        self.protocols.insert(protocol.protocol_id.clone(), protocol);
        Ok(())
    }

    fn remove_protocol(&mut self, protocol_id: &str) -> PyResult<()> {
        self.protocols.remove(protocol_id);
        Ok(())
    }

    fn add_routing_rule(&mut self, agent_id: String, channel_id: String) -> PyResult<()> {
        self.routing_table.insert(agent_id, channel_id);
        Ok(())
    }

    fn remove_routing_rule(&mut self, agent_id: &str) -> PyResult<()> {
        self.routing_table.remove(agent_id);
        Ok(())
    }

    fn send_message(&mut self, py: Python, message: Message) -> PyResult<bool> {
        if let Some(channel_id) = self.routing_table.get(&message.recipient) {
            if let Some(channel) = self.channels.get_mut(channel_id) {
                return channel.send_message(py, message);
            }
        }
        
        Ok(false)
    }

    fn deliver_messages(&mut self, py: Python) -> PyResult<usize> {
        let mut delivered_count = 0;
        
        for (_, channel) in &mut self.channels {
            let messages = channel.get_all_messages(py)?;
            
            channel.clear()?;
            
            for message in messages {
                if let Some(mailbox) = self.mailboxes.get_mut(&message.recipient) {
                    if mailbox.receive(py, message)? {
                        delivered_count += 1;
                    }
                }
            }
        }
        
        Ok(delivered_count)
    }

    fn process_outgoing_messages(&mut self, py: Python) -> PyResult<usize> {
        let mut sent_count = 0;
        
        for (_, mailbox) in &mut self.mailboxes {
            while let Some(message) = mailbox.get_next_outgoing_message() {
                if self.send_message(py, message)? {
                    sent_count += 1;
                }
            }
        }
        
        Ok(sent_count)
    }

    fn get_channel(&self, channel_id: &str) -> Option<Channel> {
        self.channels.get(channel_id).cloned()
    }

    fn get_mailbox(&self, agent_id: &str) -> Option<Mailbox> {
        self.mailboxes.get(agent_id).cloned()
    }

    fn get_protocol(&self, protocol_id: &str) -> Option<Protocol> {
        self.protocols.get(protocol_id).cloned()
    }

    fn get_channel_for_agent(&self, agent_id: &str) -> Option<String> {
        self.routing_table.get(agent_id).cloned()
    }

    fn get_all_agents(&self, py: Python) -> PyResult<Vec<String>> {
        Ok(self.mailboxes.keys().cloned().collect())
    }

    fn get_all_channels(&self, py: Python) -> PyResult<Vec<String>> {
        Ok(self.channels.keys().cloned().collect())
    }

    fn get_all_protocols(&self, py: Python) -> PyResult<Vec<String>> {
        Ok(self.protocols.keys().cloned().collect())
    }

    fn reset(&mut self) -> PyResult<()> {
        for (_, channel) in &mut self.channels {
            channel.clear()?;
        }
        
        for (_, mailbox) in &mut self.mailboxes {
            mailbox.clear_inbox()?;
            mailbox.clear_outbox()?;
        }
        
        for (_, protocol) in &mut self.protocols {
            protocol.reset(None)?;
        }
        
        Ok(())
    }
}

pub fn register_communication(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let communication_module = PyModule::new(py, "communication")?;
    
    communication_module.add_class::<Message>()?;
    communication_module.add_class::<Channel>()?;
    communication_module.add_class::<Protocol>()?;
    communication_module.add_class::<Mailbox>()?;
    communication_module.add_class::<CommunicationNetwork>()?;
    
    m.add_submodule(&communication_module)?;
    
    Ok(())
}
