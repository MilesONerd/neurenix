"""
Agent implementations for Multi-Agent Systems in Neurenix.

This module provides various agent implementations for multi-agent systems,
including reactive, deliberative, and hybrid agents.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict, deque

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.functional import sigmoid, tanh, relu, softmax


class AgentState:
    """Represents the internal state of an agent."""
    
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize agent state.
        
        Args:
            initial_state: Optional dictionary of initial state values
        """
        self._state = initial_state or {}
        self._history = deque(maxlen=100)  # Keep track of state history
        
    def update(self, **kwargs) -> None:
        """Update agent state with new values.
        
        Args:
            **kwargs: Key-value pairs to update in the state
        """
        self._history.append(self._state.copy())
        
        for key, value in kwargs.items():
            self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent state.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with the key or default
        """
        return self._state.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a value from the agent state using dictionary syntax.
        
        Args:
            key: State key to retrieve
            
        Returns:
            Value associated with the key
            
        Raises:
            KeyError: If key doesn't exist in state
        """
        return self._state[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value in the agent state using dictionary syntax.
        
        Args:
            key: State key to set
            value: Value to associate with the key
        """
        self._history.append(self._state.copy())
        
        self._state[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the agent state.
        
        Args:
            key: State key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._state
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the entire state as a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return self._state.copy()
    
    def history(self, key: Optional[str] = None) -> List[Any]:
        """Get state history.
        
        Args:
            key: Optional key to get history for specific state variable
            
        Returns:
            List of historical state values
        """
        if key is None:
            return list(self._history)
        else:
            return [state.get(key) for state in self._history if key in state]


class Agent(Module):
    """Base class for all agents in a multi-agent system."""
    
    def __init__(self, agent_id: str, observation_space: Optional[Dict[str, Any]] = None,
                action_space: Optional[Dict[str, Any]] = None):
        """Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            observation_space: Optional specification of observation space
            action_space: Optional specification of action space
        """
        super().__init__()
        self.agent_id = agent_id
        self.observation_space = observation_space or {}
        self.action_space = action_space or {}
        self.state = AgentState()
        self.mailbox = []
        
    def observe(self, observation: Any) -> None:
        """Process an observation from the environment.
        
        Args:
            observation: Observation data from the environment
        """
        raise NotImplementedError("Agents must implement observe method")
    
    def act(self) -> Any:
        """Determine the next action based on current state.
        
        Returns:
            Action to take in the environment
        """
        raise NotImplementedError("Agents must implement act method")
    
    def receive_message(self, message: Any) -> None:
        """Receive a message from another agent.
        
        Args:
            message: Message content
        """
        self.mailbox.append(message)
    
    def send_message(self, recipient_id: str, content: Any) -> Any:
        """Send a message to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            content: Message content
            
        Returns:
            Message object
        """
        message = {
            'sender': self.agent_id,
            'recipient': recipient_id,
            'content': content,
            'timestamp': np.datetime64('now')
        }
        return message
    
    def update_state(self, **kwargs) -> None:
        """Update the agent's internal state.
        
        Args:
            **kwargs: Key-value pairs to update in the state
        """
        self.state.update(**kwargs)
    
    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.state = AgentState()
        self.mailbox = []


class ReactiveAgent(Agent):
    """Reactive agent that maps observations directly to actions."""
    
    def __init__(self, agent_id: str, policy_function: Callable[[Any], Any],
                observation_space: Optional[Dict[str, Any]] = None,
                action_space: Optional[Dict[str, Any]] = None):
        """Initialize a reactive agent.
        
        Args:
            agent_id: Unique identifier for the agent
            policy_function: Function that maps observations to actions
            observation_space: Optional specification of observation space
            action_space: Optional specification of action space
        """
        super().__init__(agent_id, observation_space, action_space)
        self.policy_function = policy_function
        self.current_observation = None
    
    def observe(self, observation: Any) -> None:
        """Process an observation from the environment.
        
        Args:
            observation: Observation data from the environment
        """
        self.current_observation = observation
    
    def act(self) -> Any:
        """Determine the next action based on current observation.
        
        Returns:
            Action to take in the environment
        """
        if self.current_observation is None:
            raise ValueError("Agent must receive an observation before acting")
        
        return self.policy_function(self.current_observation)


class DeliberativeAgent(Agent):
    """Deliberative agent that plans actions based on a model of the world."""
    
    def __init__(self, agent_id: str, world_model: Any, planner: Any,
                observation_space: Optional[Dict[str, Any]] = None,
                action_space: Optional[Dict[str, Any]] = None):
        """Initialize a deliberative agent.
        
        Args:
            agent_id: Unique identifier for the agent
            world_model: Model of the environment for planning
            planner: Planning algorithm to determine actions
            observation_space: Optional specification of observation space
            action_space: Optional specification of action space
        """
        super().__init__(agent_id, observation_space, action_space)
        self.world_model = world_model
        self.planner = planner
        self.current_plan = []
    
    def observe(self, observation: Any) -> None:
        """Process an observation from the environment.
        
        Args:
            observation: Observation data from the environment
        """
        self.world_model.update(observation)
        
        if not self._is_plan_valid():
            self._replan()
    
    def act(self) -> Any:
        """Determine the next action based on current plan.
        
        Returns:
            Action to take in the environment
        """
        if not self.current_plan:
            self._replan()
            
        if not self.current_plan:
            raise ValueError("Failed to generate a plan")
            
        return self.current_plan.pop(0)
    
    def _is_plan_valid(self) -> bool:
        """Check if the current plan is still valid.
        
        Returns:
            True if plan is valid, False otherwise
        """
        return len(self.current_plan) > 0
    
    def _replan(self) -> None:
        """Generate a new plan based on the current world model."""
        self.current_plan = self.planner.plan(self.world_model)


class HybridAgent(Agent):
    """Hybrid agent that combines reactive and deliberative approaches."""
    
    def __init__(self, agent_id: str, reactive_component: ReactiveAgent,
                deliberative_component: DeliberativeAgent,
                meta_controller: Optional[Callable[[Any, Any], float]] = None,
                observation_space: Optional[Dict[str, Any]] = None,
                action_space: Optional[Dict[str, Any]] = None):
        """Initialize a hybrid agent.
        
        Args:
            agent_id: Unique identifier for the agent
            reactive_component: Reactive component for immediate responses
            deliberative_component: Deliberative component for planning
            meta_controller: Function to decide which component to use
            observation_space: Optional specification of observation space
            action_space: Optional specification of action space
        """
        super().__init__(agent_id, observation_space, action_space)
        self.reactive_component = reactive_component
        self.deliberative_component = deliberative_component
        self.meta_controller = meta_controller or self._default_meta_controller
    
    def observe(self, observation: Any) -> None:
        """Process an observation from the environment.
        
        Args:
            observation: Observation data from the environment
        """
        self.reactive_component.observe(observation)
        self.deliberative_component.observe(observation)
    
    def act(self) -> Any:
        """Determine the next action using either reactive or deliberative component.
        
        Returns:
            Action to take in the environment
        """
        reactive_action = self.reactive_component.act()
        deliberative_action = self.deliberative_component.act()
        
        weight = self.meta_controller(reactive_action, deliberative_action)
        
        if weight > 0.5:
            return deliberative_action
        else:
            return reactive_action
    
    def _default_meta_controller(self, reactive_action: Any, deliberative_action: Any) -> float:
        """Default meta-controller implementation.
        
        Args:
            reactive_action: Action from reactive component
            deliberative_action: Action from deliberative component
            
        Returns:
            Weight between 0 and 1, where higher values favor deliberative action
        """
        return 0.5  # Equal weighting by default
