"""
Environment implementation for the Neurenix framework.
"""

from typing import Dict, Any, List, Optional, Union, Callable

class Environment:
    """
    Base class for environments in which agents operate.
    
    An environment defines the world in which agents exist and interact.
    It provides observations to agents and processes their actions.
    """
    
    def __init__(self):
        """Initialize a new environment."""
        self._state = {}
        self._agents = {}
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Initial state of the environment
        """
        self._state = self._get_initial_state()
        return self._state
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        This method should be overridden by subclasses to define
        the initial state of the environment.
        
        Returns:
            Initial state of the environment
        """
        return {}
    
    def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply actions to the environment and update its state.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions
            
        Returns:
            Dictionary containing the results of the actions, including:
            - rewards: Dictionary mapping agent IDs to their rewards
            - done: Whether the episode is complete
            - info: Additional information
        """
        raise NotImplementedError("Subclasses must implement step()")
    
    def observe(self, agent: Any) -> Dict[str, Any]:
        """
        Get an observation of the environment for a specific agent.
        
        Args:
            agent: The agent requesting the observation
            
        Returns:
            Observation for the agent
        """
        raise NotImplementedError("Subclasses must implement observe()")
    
    def register_agent(self, agent: Any) -> None:
        """
        Register an agent with the environment.
        
        Args:
            agent: The agent to register
        """
        self._agents[agent.id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the environment.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
    
    @property
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Current state of the environment
        """
        return self._state.copy()
    
    @property
    def agents(self) -> Dict[str, Any]:
        """
        Get the agents registered with the environment.
        
        Returns:
            Dictionary mapping agent IDs to agents
        """
        return self._agents.copy()
