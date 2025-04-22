"""
Network environment module for Multi-Agent Systems (MAS).

This module provides a network-based environment for multi-agent systems,
where agents are nodes in a network and can communicate with connected agents.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

from neurenix.tensor import Tensor
from neurenix.mas.environment import Environment


class NetworkEnvironment(Environment):
    """
    A network-based environment for multi-agent systems.
    
    Agents are nodes in a network and can communicate with connected agents.
    """
    
    def __init__(
        self,
        num_agents: int,
        connectivity: float = 0.5,
        max_steps: int = 1000,
        directed: bool = False,
        weighted: bool = False,
        features_dim: int = 10
    ):
        """
        Initialize a network environment.
        
        Args:
            num_agents: Number of agents in the environment
            connectivity: Probability of an edge between any two nodes
            max_steps: Maximum number of steps before the environment terminates
            directed: Whether the network is directed
            weighted: Whether the edges have weights
            features_dim: Dimension of node features
        """
        super().__init__(num_agents, max_steps)
        
        self.connectivity = connectivity
        self.directed = directed
        self.weighted = weighted
        self.features_dim = features_dim
        
        self.adjacency_matrix = np.zeros((num_agents, num_agents), dtype=np.float32)
        
        self.node_features = np.zeros((num_agents, features_dim), dtype=np.float32)
        
        if self.weighted:
            self.edge_weights = np.zeros((num_agents, num_agents), dtype=np.float32)
        
        self._initialize_network()
    
    def _initialize_network(self) -> None:
        """Initialize the network structure."""
        self.adjacency_matrix = np.random.binomial(1, self.connectivity, size=(self.num_agents, self.num_agents))
        
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        if not self.directed:
            self.adjacency_matrix = np.maximum(self.adjacency_matrix, self.adjacency_matrix.T)
        
        if self.weighted:
            self.edge_weights = np.random.uniform(0, 1, size=(self.num_agents, self.num_agents))
            self.edge_weights = self.edge_weights * self.adjacency_matrix
        
        self.node_features = np.random.randn(self.num_agents, self.features_dim)
    
    def reset(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Reset the environment to its initial state.
        
        Returns:
            Dictionary mapping agent IDs to their initial observations
        """
        super().reset()
        
        self._initialize_network()
        
        return {agent_id: self.get_observation(agent_id) for agent_id in range(self.num_agents)}
    
    def get_observation(self, agent_id: int) -> Dict[str, np.ndarray]:
        """
        Get the observation for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary containing adjacency matrix, node features, and agent's node index
        """
        neighbors = np.where(self.adjacency_matrix[agent_id] > 0)[0]
        
        nodes = np.append(neighbors, agent_id)
        
        sub_adjacency = self.adjacency_matrix[np.ix_(nodes, nodes)]
        sub_features = self.node_features[nodes]
        
        if self.weighted:
            sub_weights = self.edge_weights[np.ix_(nodes, nodes)]
            return {
                'adjacency': sub_adjacency,
                'features': sub_features,
                'weights': sub_weights,
                'node_idx': np.where(nodes == agent_id)[0][0]  # Index of the agent's node in the subgraph
            }
        else:
            return {
                'adjacency': sub_adjacency,
                'features': sub_features,
                'node_idx': np.where(nodes == agent_id)[0][0]  # Index of the agent's node in the subgraph
            }
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, float], Dict[int, bool], Dict[int, Dict]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to action vectors
            
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
            if agent_id < self.num_agents:
                self.node_features[agent_id] += action
                
                neighbors = np.where(self.adjacency_matrix[agent_id] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_features = self.node_features[neighbors]
                    similarity = np.mean(np.sum(self.node_features[agent_id] * neighbor_features, axis=1))
                    rewards[agent_id] = similarity
        
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
            print("Network Environment:")
            print(f"Number of agents: {self.num_agents}")
            print(f"Number of edges: {np.sum(self.adjacency_matrix) // (1 if self.directed else 2)}")
            print(f"Connectivity: {self.connectivity}")
            print(f"Directed: {self.directed}")
            print(f"Weighted: {self.weighted}")
            print()
            return None
        elif mode == 'rgb_array':
            size = max(500, self.num_agents * 20)
            image = np.ones((size, size, 3), dtype=np.uint8) * 255
            
            node_positions = []
            radius = size // 2 - 50
            center = size // 2
            
            for i in range(self.num_agents):
                angle = 2 * np.pi * i / self.num_agents
                x = int(center + radius * np.cos(angle))
                y = int(center + radius * np.sin(angle))
                node_positions.append((x, y))
                
                node_color = [
                    (i * 50) % 256,
                    (i * 100) % 256,
                    (i * 150) % 256
                ]
                
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx**2 + dy**2 <= 25:  # Circle with radius 5
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                image[ny, nx] = node_color
            
            for i in range(self.num_agents):
                for j in range(self.num_agents if self.directed else i + 1):
                    if self.adjacency_matrix[i, j] > 0:
                        x1, y1 = node_positions[i]
                        x2, y2 = node_positions[j]
                        
                        length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                        for t in range(length):
                            alpha = t / length
                            x = int(x1 + alpha * (x2 - x1))
                            y = int(y1 + alpha * (y2 - y1))
                            if 0 <= x < size and 0 <= y < size:
                                if self.weighted:
                                    weight = self.edge_weights[i, j]
                                    color = [int(255 * (1 - weight)), int(255 * (1 - weight)), int(255 * (1 - weight))]
                                else:
                                    color = [0, 0, 0]  # Black line
                                
                                image[y, x] = color
            
            return image
        else:
            return None
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the network.
        
        Returns:
            Adjacency matrix
        """
        return self.adjacency_matrix.copy()
    
    def get_node_features(self) -> np.ndarray:
        """
        Get the node features of the network.
        
        Returns:
            Node features
        """
        return self.node_features.copy()
    
    def get_edge_weights(self) -> Optional[np.ndarray]:
        """
        Get the edge weights of the network.
        
        Returns:
            Edge weights if the network is weighted, None otherwise
        """
        if self.weighted:
            return self.edge_weights.copy()
        else:
            return None
    
    def add_edge(self, source: int, target: int, weight: float = 1.0) -> bool:
        """
        Add an edge to the network.
        
        Args:
            source: Source node
            target: Target node
            weight: Edge weight (only used if the network is weighted)
            
        Returns:
            True if the edge was added, False otherwise
        """
        if source < 0 or source >= self.num_agents or target < 0 or target >= self.num_agents:
            return False
        
        self.adjacency_matrix[source, target] = 1
        
        if not self.directed:
            self.adjacency_matrix[target, source] = 1
        
        if self.weighted:
            self.edge_weights[source, target] = weight
            
            if not self.directed:
                self.edge_weights[target, source] = weight
        
        return True
    
    def remove_edge(self, source: int, target: int) -> bool:
        """
        Remove an edge from the network.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            True if the edge was removed, False otherwise
        """
        if source < 0 or source >= self.num_agents or target < 0 or target >= self.num_agents:
            return False
        
        self.adjacency_matrix[source, target] = 0
        
        if not self.directed:
            self.adjacency_matrix[target, source] = 0
        
        if self.weighted:
            self.edge_weights[source, target] = 0
            
            if not self.directed:
                self.edge_weights[target, source] = 0
        
        return True
    
    def set_node_features(self, node: int, features: np.ndarray) -> bool:
        """
        Set the features of a node.
        
        Args:
            node: Node index
            features: New features
            
        Returns:
            True if the features were set, False otherwise
        """
        if node < 0 or node >= self.num_agents or features.shape[0] != self.features_dim:
            return False
        
        self.node_features[node] = features
        return True
