"""
Communication module for Multi-Agent Systems in Neurenix.

This module provides implementations of communication protocols and
message passing infrastructure for multi-agent systems.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np

class Message:
    """Message class for agent communication."""
    
    def __init__(self, sender_id: str, receiver_id: str, content: Any,
                 message_type: str = "inform"):
        """
        Initialize a message.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            content: Content of the message
            message_type: Type of message (inform, request, etc.)
        """
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.message_type = message_type
        self.timestamp = None  # Will be set when sent
        
    def __str__(self) -> str:
        """String representation of the message."""
        return f"Message({self.message_type}) from {self.sender_id} to {self.receiver_id}: {self.content}"


class Channel:
    """Communication channel between agents."""
    
    def __init__(self, name: str, latency: float = 0.0, reliability: float = 1.0):
        """
        Initialize a communication channel.
        
        Args:
            name: Name of the channel
            latency: Communication latency in seconds
            reliability: Probability of successful message delivery (0-1)
        """
        self.name = name
        self.latency = latency
        self.reliability = reliability
        self.messages = []
        self.subscribers = set()
        
    def send(self, message: Message) -> bool:
        """
        Send a message through the channel.
        
        Args:
            message: Message to send
            
        Returns:
            Whether the message was successfully sent
        """
        if np.random.random() > self.reliability:
            return False
            
        self.messages.append(message)
        
        return True
        
    def receive(self, agent_id: str) -> List[Message]:
        """
        Receive messages for a specific agent.
        
        Args:
            agent_id: ID of the agent receiving messages
            
        Returns:
            List of messages for the agent
        """
        received_messages = []
        remaining_messages = []
        
        for message in self.messages:
            if message.receiver_id == agent_id:
                received_messages.append(message)
            else:
                remaining_messages.append(message)
                
        self.messages = remaining_messages
        
        return received_messages
    
    def subscribe(self, agent_id: str) -> None:
        """
        Subscribe an agent to the channel.
        
        Args:
            agent_id: ID of the agent to subscribe
        """
        self.subscribers.add(agent_id)
        
    def unsubscribe(self, agent_id: str) -> None:
        """
        Unsubscribe an agent from the channel.
        
        Args:
            agent_id: ID of the agent to unsubscribe
        """
        if agent_id in self.subscribers:
            self.subscribers.remove(agent_id)


class Protocol:
    """Base class for communication protocols."""
    
    def __init__(self, name: str):
        """
        Initialize a communication protocol.
        
        Args:
            name: Name of the protocol
        """
        self.name = name
        
    def validate_message(self, message: Message) -> bool:
        """
        Validate a message according to the protocol.
        
        Args:
            message: Message to validate
            
        Returns:
            Whether the message is valid
        """
        return True
        
    def process_message(self, message: Message) -> List[Message]:
        """
        Process a message and generate responses if needed.
        
        Args:
            message: Message to process
            
        Returns:
            List of response messages
        """
        return []
