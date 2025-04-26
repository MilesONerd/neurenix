"""
Environment module for reinforcement learning in Neurenix.

This module provides the base Environment class and implementations of various
reinforcement learning environments.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neurenix.tensor import Tensor


class Environment:
    """
    Base class for reinforcement learning environments.
    
    This class provides the basic functionality for reinforcement learning environments,
    including state representation, action handling, and reward calculation.
    """
    
    def __init__(
        self,
        name: str = "Environment",
        max_steps: int = 1000,
    ):
        """
        Initialize environment.
        
        Args:
            name: Environment name
            max_steps: Maximum number of steps per episode
        """
        self.name = name
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.state = None
        self.reward = 0.0
        self.info = {}
    
    def reset(self) -> Union[np.ndarray, Tensor]:
        """
        Reset the environment.
        
        Returns:
            Initial state
        """
        self.steps = 0
        self.done = False
        self.reward = 0.0
        self.info = {}
        
        # Reset state (to be implemented by subclasses)
        self.state = self._reset_state()
        
        return self.state
    
    def _reset_state(self) -> Union[np.ndarray, Tensor]:
        """
        Reset the state.
        
        Returns:
            Initial state
        """
        raise NotImplementedError("Subclasses must implement _reset_state")
    
    def step(self, action: Any) -> Tuple[Union[np.ndarray, Tensor], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Check if environment is done
        if self.done:
            return self.state, 0.0, True, {"warning": "Environment is already done"}
        
        # Increment step counter
        self.steps += 1
        
        # Check if maximum number of steps is reached
        if self.steps >= self.max_steps:
            self.done = True
        
        # Take step (to be implemented by subclasses)
        self.state, self.reward, done, self.info = self._step(action)
        
        # Update done flag
        self.done = self.done or done
        
        return self.state, self.reward, self.done, self.info
    
    def _step(self, action: Any) -> Tuple[Union[np.ndarray, Tensor], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        raise NotImplementedError("Subclasses must implement _step")
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is "rgb_array", None otherwise
        """
        # To be implemented by subclasses
        return None
    
    def close(self):
        """Close the environment."""
        # To be implemented by subclasses
        pass
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the random seed.
        
        Args:
            seed: Random seed
            
        Returns:
            List of seeds
        """
        # To be implemented by subclasses
        return [seed]
    
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get the observation space.
        
        Returns:
            Observation space specification
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_observation_space")
    
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get the action space.
        
        Returns:
            Action space specification
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_action_space")


class GridWorld(Environment):
    """
    Grid world environment for reinforcement learning.
    
    This class implements a simple grid world environment for reinforcement learning,
    where the agent navigates a grid to reach a goal while avoiding obstacles.
    """
    
    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_steps: int = 100,
        obstacle_density: float = 0.2,
        name: str = "GridWorld",
    ):
        """
        Initialize grid world environment.
        
        Args:
            width: Grid width
            height: Grid height
            max_steps: Maximum number of steps per episode
            obstacle_density: Density of obstacles in the grid
            name: Environment name
        """
        super().__init__(name=name, max_steps=max_steps)
        
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        
        # Grid elements
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.AGENT = 2
        self.GOAL = 3
        
        # Actions
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        
        # Action effects
        self.action_effects = [
            (-1, 0),  # UP
            (0, 1),   # RIGHT
            (1, 0),   # DOWN
            (0, -1),  # LEFT
        ]
        
        # Initialize grid
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # Initialize agent and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        
        # Generate grid
        self._generate_grid()
    
    def _generate_grid(self):
        """Generate the grid."""
        # Clear grid
        self.grid.fill(self.EMPTY)
        
        # Add obstacles
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) != self.agent_pos and (i, j) != self.goal_pos:
                    if np.random.random() < self.obstacle_density:
                        self.grid[i, j] = self.OBSTACLE
        
        # Add agent and goal
        self.grid[self.agent_pos] = self.AGENT
        self.grid[self.goal_pos] = self.GOAL
    
    def _reset_state(self) -> np.ndarray:
        """
        Reset the state.
        
        Returns:
            Initial state
        """
        # Reset agent position
        self.agent_pos = (0, 0)
        
        # Generate new grid
        self._generate_grid()
        
        # Return state
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state.
        
        Returns:
            Current state
        """
        # Return grid as state
        return self.grid.copy()
    
    def _step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Check if action is valid
        if action not in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
            raise ValueError(f"Invalid action: {action}")
        
        # Get action effect
        row_delta, col_delta = self.action_effects[action]
        
        # Compute new position
        new_row = self.agent_pos[0] + row_delta
        new_col = self.agent_pos[1] + col_delta
        
        # Check if new position is valid
        if (
            new_row < 0
            or new_row >= self.height
            or new_col < 0
            or new_col >= self.width
            or self.grid[new_row, new_col] == self.OBSTACLE
        ):
            # Invalid position, stay in place
            reward = -0.1
            done = False
            info = {"valid_move": False}
        else:
            # Valid position, move agent
            self.grid[self.agent_pos] = self.EMPTY
            self.agent_pos = (new_row, new_col)
            
            # Check if agent reached goal
            if self.agent_pos == self.goal_pos:
                reward = 1.0
                done = True
                info = {"valid_move": True, "goal_reached": True}
            else:
                # Agent moved but didn't reach goal
                self.grid[self.agent_pos] = self.AGENT
                reward = -0.01
                done = False
                info = {"valid_move": True}
        
        # Return state, reward, done, info
        return self._get_state(), reward, done, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if mode is "rgb_array", None otherwise
        """
        if mode == "human":
            # Print grid
            for i in range(self.height):
                row = ""
                for j in range(self.width):
                    if self.grid[i, j] == self.EMPTY:
                        row += "."
                    elif self.grid[i, j] == self.OBSTACLE:
                        row += "#"
                    elif self.grid[i, j] == self.AGENT:
                        row += "A"
                    elif self.grid[i, j] == self.GOAL:
                        row += "G"
                print(row)
            print()
            return None
        elif mode == "rgb_array":
            # Create RGB array
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Set colors
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == self.EMPTY:
                        rgb[i, j] = [255, 255, 255]  # White
                    elif self.grid[i, j] == self.OBSTACLE:
                        rgb[i, j] = [0, 0, 0]  # Black
                    elif self.grid[i, j] == self.AGENT:
                        rgb[i, j] = [0, 0, 255]  # Blue
                    elif self.grid[i, j] == self.GOAL:
                        rgb[i, j] = [0, 255, 0]  # Green
            
            return rgb
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get the observation space.
        
        Returns:
            Observation space specification
        """
        return {
            "type": "box",
            "shape": (self.height, self.width),
            "low": 0,
            "high": 3,
            "dtype": np.int32,
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get the action space.
        
        Returns:
            Action space specification
        """
        return {
            "type": "discrete",
            "n": 4,
        }
