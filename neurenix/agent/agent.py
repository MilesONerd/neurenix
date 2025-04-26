"""
Base agent class for the Neurenix framework.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import uuid

from neurenix.tensor import Tensor

class Agent:
    """
    Base class for all AI agents.
    
    This provides a foundation for implementing various types of agents,
    such as reinforcement learning agents or autonomous agents.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new agent.
        
        Args:
            name: The name of the agent. If None, a random name is generated.
        """
        self._name = name or f"Agent-{str(uuid.uuid4())[:8]}"
        self._state: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name
    
    def act(self, observation: Any) -> Any:
        """
        Choose an action based on the current observation.
        
        This method should be overridden by all subclasses.
        
        Args:
            observation: The current observation of the environment.
            
        Returns:
            The action to take.
        """
        raise NotImplementedError("Subclasses must implement act()")
    
    def learn(self, experience: Any) -> None:
        """
        Learn from experience.
        
        This method should be overridden by all subclasses that support learning.
        
        Args:
            experience: The experience to learn from.
        """
        raise NotImplementedError("Subclasses must implement learn()")
    
    def reset(self) -> None:
        """
        Reset the agent's state.
        
        This is typically called at the beginning of a new episode.
        """
        self._state = {}
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            path: The path to save the agent to.
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: The path to load the agent from.
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def __repr__(self) -> str:
        """Get a string representation of the agent."""
        return f"{self.__class__.__name__}(name={self._name})"
