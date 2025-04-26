"""
Multi-agent system implementation for the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable

from neurenix.agent.agent import Agent
from neurenix.agent.environment import Environment

class MultiAgent:
    """
    A system of multiple agents interacting in a shared environment.
    
    This class provides functionality for coordinating multiple agents,
    enabling them to interact with each other and with a shared environment.
    """
    
    def __init__(self, agents: List[Agent], environment: Environment):
        """
        Initialize a multi-agent system.
        
        Args:
            agents: List of agents in the system
            environment: Shared environment for the agents
        """
        self.agents = agents
        self.environment = environment
        self._step_count = 0
    
    def step(self) -> Dict[str, Any]:
        """
        Perform a single step of the multi-agent system.
        
        This method:
        1. Gets observations for each agent from the environment
        2. Has each agent select an action based on its observation
        3. Applies all actions to the environment
        4. Returns the results
        
        Returns:
            Dictionary containing observations, actions, rewards, and done flags for each agent
        """
        # Get observations for each agent
        observations = {}
        for agent in self.agents:
            observations[agent.id] = self.environment.observe(agent)
        
        # Have each agent select an action
        actions = {}
        for agent in self.agents:
            actions[agent.id] = agent.act(observations[agent.id])
        
        # Apply all actions to the environment
        results = self.environment.step(actions)
        
        # Update step count
        self._step_count += 1
        
        return {
            "observations": observations,
            "actions": actions,
            "rewards": results.get("rewards", {}),
            "done": results.get("done", False),
            "info": results.get("info", {})
        }
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the multi-agent system.
        
        This method:
        1. Resets the environment
        2. Resets each agent
        3. Returns the initial observations
        
        Returns:
            Dictionary containing initial observations for each agent
        """
        # Reset the environment
        self.environment.reset()
        
        # Reset each agent
        for agent in self.agents:
            agent.reset()
        
        # Get initial observations
        observations = {}
        for agent in self.agents:
            observations[agent.id] = self.environment.observe(agent)
        
        # Reset step count
        self._step_count = 0
        
        return observations
    
    @property
    def step_count(self) -> int:
        """
        Get the number of steps taken in the current episode.
        
        Returns:
            Number of steps
        """
        return self._step_count
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add a new agent to the system.
        
        Args:
            agent: Agent to add
        """
        self.agents.append(agent)
    
    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Remove an agent from the system.
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            The removed agent, or None if no agent with the given ID was found
        """
        for i, agent in enumerate(self.agents):
            if agent.id == agent_id:
                return self.agents.pop(i)
        
        return None
    
    def __len__(self) -> int:
        """
        Get the number of agents in the system.
        
        Returns:
            Number of agents
        """
        return len(self.agents)
