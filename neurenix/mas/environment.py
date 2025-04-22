"""
Environment module for Multi-Agent Systems (MAS).

This module provides classes and functions for creating and managing environments
for multi-agent systems, including grid-based and continuous environments.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np

from neurenix.tensor import Tensor

class StateSpace:
    """Base class for state spaces in environments."""
    pass

class ActionSpace:
    """Base class for action spaces in environments."""
    pass


class Environment:
    """Base class for all multi-agent environments."""
    
    def __init__(self, num_agents: int, max_steps: int = 1000):
        """
        Initialize a multi-agent environment.
        
        Args:
            num_agents: Number of agents in the environment
            max_steps: Maximum number of steps before the environment terminates
        """
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.agents = {}
        
    def reset(self) -> Dict[int, Any]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        self.current_step = 0
        self.done = False
        return {}
    
    def step(self, actions: Dict[int, Any]) -> Tuple[Dict[int, Any], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions
            
        Returns:
            Tuple containing:
                - Dictionary mapping agent IDs to their observations
                - Dictionary mapping agent IDs to their rewards
                - Dictionary mapping agent IDs to their done flags
                - Dictionary mapping agent IDs to additional information
        """
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        
        return {}, {}, {}, {}
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            If mode is 'rgb_array', returns a numpy array representing the rendered environment
        """
        return None
    
    def close(self) -> None:
        """Close the environment and release resources."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the random seed for the environment.
        
        Args:
            seed: Random seed
            
        Returns:
            List of seeds used in the environment
        """
        return [seed]
    
    def get_agent_ids(self) -> List[int]:
        """
        Get the IDs of all agents in the environment.
        
        Returns:
            List of agent IDs
        """
        return list(range(self.num_agents))
    
    def register_agent(self, agent_id: int, agent: Any) -> None:
        """
        Register an agent in the environment.
        
        Args:
            agent_id: ID of the agent
            agent: Agent object
        """
        self.agents[agent_id] = agent
    
    def get_agent(self, agent_id: int) -> Any:
        """
        Get an agent by its ID.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent object
        """
        return self.agents.get(agent_id)


class GridEnvironment(Environment):
    """Grid-based environment for multi-agent systems."""
    
    def __init__(
        self,
        num_agents: int,
        grid_size: Tuple[int, int],
        max_steps: int = 1000,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        reward_positions: Optional[Dict[Tuple[int, int], float]] = None
    ):
        """
        Initialize a grid-based environment.
        
        Args:
            num_agents: Number of agents in the environment
            grid_size: Size of the grid (height, width)
            max_steps: Maximum number of steps before the environment terminates
            obstacle_positions: List of positions (row, col) where obstacles are located
            reward_positions: Dictionary mapping positions (row, col) to reward values
        """
        super().__init__(num_agents, max_steps)
        
        self.grid_size = grid_size
        self.obstacle_positions = obstacle_positions or []
        self.reward_positions = reward_positions or {}
        
        self.grid = np.zeros(grid_size, dtype=np.int32)
        for pos in self.obstacle_positions:
            self.grid[pos] = -1
        
        self.agent_positions = {}
        self.initialize_agent_positions()
    
    def initialize_agent_positions(self) -> None:
        """Initialize agent positions randomly on the grid."""
        available_positions = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) not in self.obstacle_positions and self.grid[i, j] == 0:
                    available_positions.append((i, j))
        
        if len(available_positions) < self.num_agents:
            raise ValueError("Not enough available positions for all agents")
        
        positions = np.random.choice(len(available_positions), self.num_agents, replace=False)
        for i, agent_id in enumerate(range(self.num_agents)):
            self.agent_positions[agent_id] = available_positions[positions[i]]
            self.grid[available_positions[positions[i]]] = agent_id + 1
    
    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        super().reset()
        
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        for pos in self.obstacle_positions:
            self.grid[pos] = -1
        
        self.agent_positions = {}
        self.initialize_agent_positions()
        
        return {agent_id: self.get_observation(agent_id) for agent_id in range(self.num_agents)}
    
    def get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get the observation for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Observation for the agent
        """
        return np.copy(self.grid)
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions (0: up, 1: right, 2: down, 3: left)
            
        Returns:
            Tuple containing:
                - Dictionary mapping agent IDs to their observations
                - Dictionary mapping agent IDs to their rewards
                - Dictionary mapping agent IDs to their done flags
                - Dictionary mapping agent IDs to additional information
        """
        super().step(actions)
        
        rewards = {agent_id: 0.0 for agent_id in range(self.num_agents)}
        infos = {agent_id: {} for agent_id in range(self.num_agents)}
        
        for agent_id, action in actions.items():
            if agent_id not in self.agent_positions:
                continue
            
            row, col = self.agent_positions[agent_id]
            new_row, new_col = row, col
            
            if action == 0:  # Up
                new_row = max(0, row - 1)
            elif action == 1:  # Right
                new_col = min(self.grid_size[1] - 1, col + 1)
            elif action == 2:  # Down
                new_row = min(self.grid_size[0] - 1, row + 1)
            elif action == 3:  # Left
                new_col = max(0, col - 1)
            
            if (new_row, new_col) not in self.obstacle_positions and self.grid[new_row, new_col] <= 0:
                self.grid[row, col] = 0
                self.grid[new_row, new_col] = agent_id + 1
                
                self.agent_positions[agent_id] = (new_row, new_col)
                
                if (new_row, new_col) in self.reward_positions:
                    rewards[agent_id] += self.reward_positions[(new_row, new_col)]
        
        observations = {agent_id: self.get_observation(agent_id) for agent_id in range(self.num_agents)}
        
        dones = {agent_id: self.done for agent_id in range(self.num_agents)}
        
        return observations, rewards, dones, infos
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            If mode is 'rgb_array', returns a numpy array representing the rendered environment
        """
        if mode == 'human':
            for row in range(self.grid_size[0]):
                line = ''
                for col in range(self.grid_size[1]):
                    if self.grid[row, col] == -1:
                        line += 'X '  # Obstacle
                    elif self.grid[row, col] == 0:
                        line += '. '  # Empty
                    else:
                        line += f'{self.grid[row, col]} '  # Agent
                print(line)
            print()
            return None
        elif mode == 'rgb_array':
            rgb_grid = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    if self.grid[row, col] == -1:
                        rgb_grid[row, col] = [0, 0, 0]  # Obstacle (black)
                    elif self.grid[row, col] == 0:
                        rgb_grid[row, col] = [255, 255, 255]  # Empty (white)
                    else:
                        agent_id = self.grid[row, col] - 1
                        rgb_grid[row, col] = [
                            (agent_id * 50) % 256,
                            (agent_id * 100) % 256,
                            (agent_id * 150) % 256
                        ]
            return rgb_grid
        else:
            return None


class ContinuousEnvironment(Environment):
    """Continuous environment for multi-agent systems."""
    
    def __init__(
        self,
        num_agents: int,
        dimensions: int = 2,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_steps: int = 1000,
        obstacle_regions: Optional[List[Tuple[np.ndarray, float]]] = None,
        reward_regions: Optional[List[Tuple[np.ndarray, float, float]]] = None
    ):
        """
        Initialize a continuous environment.
        
        Args:
            num_agents: Number of agents in the environment
            dimensions: Number of dimensions in the environment
            bounds: List of (min, max) bounds for each dimension
            max_steps: Maximum number of steps before the environment terminates
            obstacle_regions: List of (center, radius) tuples defining obstacle regions
            reward_regions: List of (center, radius, reward) tuples defining reward regions
        """
        super().__init__(num_agents, max_steps)
        
        self.dimensions = dimensions
        self.bounds = bounds or [(0.0, 1.0) for _ in range(dimensions)]
        self.obstacle_regions = obstacle_regions or []
        self.reward_regions = reward_regions or []
        
        self.agent_positions = {}
        self.agent_velocities = {}
        self.initialize_agent_positions()
    
    def initialize_agent_positions(self) -> None:
        """Initialize agent positions randomly within bounds."""
        for agent_id in range(self.num_agents):
            position = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in self.bounds])
            
            while any(np.linalg.norm(position - center) <= radius for center, radius in self.obstacle_regions):
                position = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in self.bounds])
            
            self.agent_positions[agent_id] = position
            self.agent_velocities[agent_id] = np.zeros(self.dimensions)
    
    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        super().reset()
        
        self.agent_positions = {}
        self.agent_velocities = {}
        self.initialize_agent_positions()
        
        return {agent_id: self.get_observation(agent_id) for agent_id in range(self.num_agents)}
    
    def get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get the observation for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Observation for the agent
        """
        observation = []
        
        observation.extend(self.agent_positions[agent_id])
        observation.extend(self.agent_velocities[agent_id])
        
        for other_id, position in self.agent_positions.items():
            if other_id != agent_id:
                observation.extend(position)
                observation.extend(self.agent_velocities[other_id])
        
        for center, radius in self.obstacle_regions:
            observation.extend(center)
            observation.append(radius)
        
        return np.array(observation)
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to their actions (forces or accelerations)
            
        Returns:
            Tuple containing:
                - Dictionary mapping agent IDs to their observations
                - Dictionary mapping agent IDs to their rewards
                - Dictionary mapping agent IDs to their done flags
                - Dictionary mapping agent IDs to additional information
        """
        super().step(actions)
        
        rewards = {agent_id: 0.0 for agent_id in range(self.num_agents)}
        infos = {agent_id: {} for agent_id in range(self.num_agents)}
        
        for agent_id, action in actions.items():
            if agent_id not in self.agent_positions:
                continue
            
            self.agent_velocities[agent_id] += action
            
            max_velocity = 0.1
            velocity_norm = np.linalg.norm(self.agent_velocities[agent_id])
            if velocity_norm > max_velocity:
                self.agent_velocities[agent_id] = self.agent_velocities[agent_id] * max_velocity / velocity_norm
            
            new_position = self.agent_positions[agent_id] + self.agent_velocities[agent_id]
            
            for i in range(self.dimensions):
                new_position[i] = max(self.bounds[i][0], min(self.bounds[i][1], new_position[i]))
            
            collision = False
            for center, radius in self.obstacle_regions:
                if np.linalg.norm(new_position - center) <= radius:
                    collision = True
                    break
            
            if not collision:
                self.agent_positions[agent_id] = new_position
            else:
                self.agent_velocities[agent_id] = -0.5 * self.agent_velocities[agent_id]
                rewards[agent_id] -= 1.0
            
            for center, radius, reward in self.reward_regions:
                if np.linalg.norm(self.agent_positions[agent_id] - center) <= radius:
                    rewards[agent_id] += reward
        
        observations = {agent_id: self.get_observation(agent_id) for agent_id in range(self.num_agents)}
        
        dones = {agent_id: self.done for agent_id in range(self.num_agents)}
        
        return observations, rewards, dones, infos
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            If mode is 'rgb_array', returns a numpy array representing the rendered environment
        """
        if mode == 'human':
            for agent_id, position in self.agent_positions.items():
                print(f"Agent {agent_id}: {position}")
            print()
            return None
        elif mode == 'rgb_array' and self.dimensions == 2:
            resolution = 500
            rgb_grid = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
            
            for center, radius in self.obstacle_regions:
                center_px = ((center - np.array([self.bounds[i][0] for i in range(2)])) / 
                            np.array([self.bounds[i][1] - self.bounds[i][0] for i in range(2)])) * resolution
                radius_px = radius / np.mean([self.bounds[i][1] - self.bounds[i][0] for i in range(2)]) * resolution
                
                for i in range(resolution):
                    for j in range(resolution):
                        if np.linalg.norm(np.array([i, j]) - center_px) <= radius_px:
                            rgb_grid[i, j] = [0, 0, 0]  # Obstacle (black)
            
            for center, radius, _ in self.reward_regions:
                center_px = ((center - np.array([self.bounds[i][0] for i in range(2)])) / 
                            np.array([self.bounds[i][1] - self.bounds[i][0] for i in range(2)])) * resolution
                radius_px = radius / np.mean([self.bounds[i][1] - self.bounds[i][0] for i in range(2)]) * resolution
                
                for i in range(resolution):
                    for j in range(resolution):
                        if np.linalg.norm(np.array([i, j]) - center_px) <= radius_px:
                            rgb_grid[i, j] = [0, 255, 0]  # Reward (green)
            
            for agent_id, position in self.agent_positions.items():
                position_px = ((position - np.array([self.bounds[i][0] for i in range(2)])) / 
                              np.array([self.bounds[i][1] - self.bounds[i][0] for i in range(2)])) * resolution
                
                agent_radius = 5
                for i in range(max(0, int(position_px[0] - agent_radius)), min(resolution, int(position_px[0] + agent_radius + 1))):
                    for j in range(max(0, int(position_px[1] - agent_radius)), min(resolution, int(position_px[1] + agent_radius + 1))):
                        if np.linalg.norm(np.array([i, j]) - position_px) <= agent_radius:
                            rgb_grid[i, j] = [
                                (agent_id * 50) % 256,
                                (agent_id * 100) % 256,
                                (agent_id * 150) % 256
                            ]
            
            return rgb_grid
        else:
            return None


class EnvironmentManager:
    """Manager for multi-agent environments."""
    
    def __init__(self):
        """Initialize the environment manager."""
        self.environments = {}
    
    def create_grid_environment(
        self,
        env_id: str,
        num_agents: int,
        grid_size: Tuple[int, int],
        max_steps: int = 1000,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        reward_positions: Optional[Dict[Tuple[int, int], float]] = None
    ) -> GridEnvironment:
        """
        Create a grid-based environment.
        
        Args:
            env_id: ID for the environment
            num_agents: Number of agents in the environment
            grid_size: Size of the grid (height, width)
            max_steps: Maximum number of steps before the environment terminates
            obstacle_positions: List of positions (row, col) where obstacles are located
            reward_positions: Dictionary mapping positions (row, col) to reward values
            
        Returns:
            Created grid environment
        """
        env = GridEnvironment(num_agents, grid_size, max_steps, obstacle_positions, reward_positions)
        self.environments[env_id] = env
        return env
    
    def create_continuous_environment(
        self,
        env_id: str,
        num_agents: int,
        dimensions: int = 2,
        bounds: Optional[List[Tuple[float, float]]] = None,
        max_steps: int = 1000,
        obstacle_regions: Optional[List[Tuple[np.ndarray, float]]] = None,
        reward_regions: Optional[List[Tuple[np.ndarray, float, float]]] = None
    ) -> ContinuousEnvironment:
        """
        Create a continuous environment.
        
        Args:
            env_id: ID for the environment
            num_agents: Number of agents in the environment
            dimensions: Number of dimensions in the environment
            bounds: List of (min, max) bounds for each dimension
            max_steps: Maximum number of steps before the environment terminates
            obstacle_regions: List of (center, radius) tuples defining obstacle regions
            reward_regions: List of (center, radius, reward) tuples defining reward regions
            
        Returns:
            Created continuous environment
        """
        env = ContinuousEnvironment(num_agents, dimensions, bounds, max_steps, obstacle_regions, reward_regions)
        self.environments[env_id] = env
        return env
    
    def get_environment(self, env_id: str) -> Optional[Environment]:
        """
        Get an environment by its ID.
        
        Args:
            env_id: ID of the environment
            
        Returns:
            Environment object or None if not found
        """
        return self.environments.get(env_id)
    
    def remove_environment(self, env_id: str) -> bool:
        """
        Remove an environment.
        
        Args:
            env_id: ID of the environment
            
        Returns:
            True if the environment was removed, False otherwise
        """
        if env_id in self.environments:
            self.environments[env_id].close()
            del self.environments[env_id]
            return True
        return False
    
    def close_all(self) -> None:
        """Close all environments."""
        for env in self.environments.values():
            env.close()
        self.environments.clear()
