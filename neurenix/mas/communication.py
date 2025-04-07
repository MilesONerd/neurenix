"""
Communication components for Multi-Agent Systems in Neurenix.

This module provides implementations for agent communication in multi-agent systems,
including message passing, protocols, and communication networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict, deque

from neurenix.tensor import Tensor


class Message:
    """Represents a message between agents."""
    
    def __init__(self, sender_id: str, recipient_id: str, content: Any,
                timestamp: Optional[np.datetime64] = None):
        """Initialize a message.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the recipient agent
            content: Message content
            timestamp: Optional message timestamp
        """
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.timestamp = timestamp or np.datetime64('now')
        self.id = f"{sender_id}_{recipient_id}_{self.timestamp}"
        
    def __str__(self) -> str:
        """Get string representation of the message.
        
        Returns:
            String representation
        """
        return f"Message(from={self.sender_id}, to={self.recipient_id}, time={self.timestamp})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': self.content,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Message object
        """
        return cls(
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            content=data['content'],
            timestamp=data['timestamp']
        )


class Channel:
    """Communication channel between agents."""
    
    def __init__(self, channel_id: str, reliability: float = 1.0,
                latency: float = 0.0, bandwidth: float = float('inf')):
        """Initialize a communication channel.
        
        Args:
            channel_id: Unique identifier for the channel
            reliability: Probability of successful message delivery (0.0-1.0)
            latency: Message delivery delay in time units
            bandwidth: Maximum number of messages per time unit
        """
        self.channel_id = channel_id
        self.reliability = max(0.0, min(1.0, reliability))
        self.latency = max(0.0, latency)
        self.bandwidth = max(0.0, bandwidth)
        self.message_queue = deque()
        self.delivered_messages = []
        
    def send(self, message: Message) -> bool:
        """Send a message through the channel.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was accepted, False otherwise
        """
        if len(self.message_queue) >= self.bandwidth:
            return False
            
        self.message_queue.append((message, np.datetime64('now')))
        return True
    
    def update(self, current_time: np.datetime64) -> List[Message]:
        """Update channel and deliver messages.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of delivered messages
        """
        delivered = []
        remaining = deque()
        
        while self.message_queue:
            message, send_time = self.message_queue.popleft()
            delivery_time = send_time + np.timedelta64(int(self.latency * 1e9), 'ns')
            
            if delivery_time <= current_time:
                if np.random.random() < self.reliability:
                    delivered.append(message)
                    self.delivered_messages.append(message)
            else:
                remaining.append((message, send_time))
                
        self.message_queue = remaining
        return delivered


class Protocol:
    """Communication protocol for agent interactions."""
    
    def __init__(self, protocol_id: str, message_types: List[str],
                validation_rules: Optional[Dict[str, Callable[[Message], bool]]] = None):
        """Initialize a communication protocol.
        
        Args:
            protocol_id: Unique identifier for the protocol
            message_types: List of valid message types
            validation_rules: Optional dictionary mapping message types to validation functions
        """
        self.protocol_id = protocol_id
        self.message_types = message_types
        self.validation_rules = validation_rules or {}
        self.conversation_history = defaultdict(list)
        
    def validate_message(self, message: Message) -> bool:
        """Validate a message against the protocol.
        
        Args:
            message: Message to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        if not isinstance(message.content, dict) or 'type' not in message.content:
            return False
            
        message_type = message.content['type']
        
        if message_type not in self.message_types:
            return False
            
        if message_type in self.validation_rules:
            return self.validation_rules[message_type](message)
            
        return True
    
    def record_message(self, message: Message, conversation_id: str) -> None:
        """Record a message in the conversation history.
        
        Args:
            message: Message to record
            conversation_id: ID of the conversation
        """
        self.conversation_history[conversation_id].append(message)
    
    def get_conversation(self, conversation_id: str) -> List[Message]:
        """Get the history of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of messages in the conversation
        """
        return self.conversation_history[conversation_id]


class Mailbox:
    """Message storage for agents."""
    
    def __init__(self, owner_id: str, capacity: int = 100):
        """Initialize a mailbox.
        
        Args:
            owner_id: ID of the agent that owns this mailbox
            capacity: Maximum number of messages to store
        """
        self.owner_id = owner_id
        self.capacity = capacity
        self.inbox = deque(maxlen=capacity)
        self.outbox = deque(maxlen=capacity)
        self.read_messages = set()
        
    def receive(self, message: Message) -> bool:
        """Receive a message into the inbox.
        
        Args:
            message: Message to receive
            
        Returns:
            True if message was accepted, False otherwise
        """
        if len(self.inbox) >= self.capacity:
            return False
            
        if message.recipient_id != self.owner_id:
            return False
            
        self.inbox.append(message)
        return True
    
    def send(self, message: Message) -> None:
        """Add a message to the outbox.
        
        Args:
            message: Message to send
        """
        if message.sender_id != self.owner_id:
            message.sender_id = self.owner_id
            
        self.outbox.append(message)
    
    def get_unread_messages(self) -> List[Message]:
        """Get all unread messages from the inbox.
        
        Returns:
            List of unread messages
        """
        unread = []
        for message in self.inbox:
            if message.id not in self.read_messages:
                unread.append(message)
                self.read_messages.add(message.id)
        return unread
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages from the inbox.
        
        Returns:
            List of all messages
        """
        return list(self.inbox)
    
    def clear_outbox(self) -> List[Message]:
        """Clear the outbox and return all messages.
        
        Returns:
            List of messages from the outbox
        """
        messages = list(self.outbox)
        self.outbox.clear()
        return messages
    
    def clear_inbox(self) -> None:
        """Clear the inbox and read messages."""
        self.inbox.clear()
        self.read_messages.clear()


class CommunicationNetwork:
    """Network for agent communication."""
    
    def __init__(self):
        """Initialize a communication network."""
        self.agents = {}  # agent_id -> mailbox
        self.channels = {}  # channel_id -> channel
        self.connections = defaultdict(list)  # agent_id -> list of channel_ids
        self.protocols = {}  # protocol_id -> protocol
        self.current_time = np.datetime64('now')
        
    def add_agent(self, agent_id: str, mailbox: Optional[Mailbox] = None) -> None:
        """Add an agent to the network.
        
        Args:
            agent_id: ID of the agent
            mailbox: Optional mailbox for the agent
        """
        self.agents[agent_id] = mailbox or Mailbox(agent_id)
    
    def add_channel(self, channel: Channel) -> None:
        """Add a communication channel to the network.
        
        Args:
            channel: Channel to add
        """
        self.channels[channel.channel_id] = channel
    
    def add_protocol(self, protocol: Protocol) -> None:
        """Add a communication protocol to the network.
        
        Args:
            protocol: Protocol to add
        """
        self.protocols[protocol.protocol_id] = protocol
    
    def connect_agents(self, agent_id1: str, agent_id2: str, 
                     channel_id: str, bidirectional: bool = True) -> bool:
        """Connect two agents with a channel.
        
        Args:
            agent_id1: ID of the first agent
            agent_id2: ID of the second agent
            channel_id: ID of the channel
            bidirectional: Whether the connection is bidirectional
            
        Returns:
            True if connection was successful, False otherwise
        """
        if agent_id1 not in self.agents or agent_id2 not in self.agents:
            return False
            
        if channel_id not in self.channels:
            return False
            
        self.connections[agent_id1].append((agent_id2, channel_id))
        
        if bidirectional:
            self.connections[agent_id2].append((agent_id1, channel_id))
            
        return True
    
    def send_message(self, message: Message, protocol_id: Optional[str] = None) -> bool:
        """Send a message from one agent to another.
        
        Args:
            message: Message to send
            protocol_id: Optional protocol to use
            
        Returns:
            True if message was sent, False otherwise
        """
        sender_id = message.sender_id
        recipient_id = message.recipient_id
        
        if sender_id not in self.agents or recipient_id not in self.agents:
            return False
            
        if protocol_id is not None:
            if protocol_id not in self.protocols:
                return False
                
            protocol = self.protocols[protocol_id]
            if not protocol.validate_message(message):
                return False
                
            conversation_id = f"{sender_id}_{recipient_id}"
            protocol.record_message(message, conversation_id)
        
        channel_id = None
        for target_id, chan_id in self.connections[sender_id]:
            if target_id == recipient_id:
                channel_id = chan_id
                break
                
        if channel_id is None:
            return False
            
        channel = self.channels[channel_id]
        return channel.send(message)
    
    def update(self, time_step: float = 1.0) -> None:
        """Update the network and deliver messages.
        
        Args:
            time_step: Time step for the update in seconds
        """
        self.current_time += np.timedelta64(int(time_step * 1e9), 'ns')
        
        for channel_id, channel in self.channels.items():
            delivered_messages = channel.update(self.current_time)
            
            for message in delivered_messages:
                recipient_id = message.recipient_id
                if recipient_id in self.agents:
                    self.agents[recipient_id].receive(message)
    
    def broadcast(self, sender_id: str, content: Any, 
                protocol_id: Optional[str] = None) -> List[bool]:
        """Broadcast a message to all connected agents.
        
        Args:
            sender_id: ID of the sending agent
            content: Message content
            protocol_id: Optional protocol to use
            
        Returns:
            List of booleans indicating success for each recipient
        """
        if sender_id not in self.agents:
            return []
            
        results = []
        
        for recipient_id, _ in self.connections[sender_id]:
            message = Message(sender_id, recipient_id, content)
            result = self.send_message(message, protocol_id)
            results.append(result)
            
        return results
