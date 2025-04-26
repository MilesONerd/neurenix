"""
NeuroEvolution of Augmenting Topologies (NEAT) implementation.

This module implements the NEAT algorithm for evolving neural network topologies
and weights simultaneously, as described in the paper by Kenneth O. Stanley and
Risto Miikkulainen.
"""

import random
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Callable
from enum import Enum, auto
from collections import defaultdict

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.nn.functional import sigmoid, relu, tanh

class NEATConfig:
    """Configuration parameters for NEAT algorithm."""
    
    def __init__(self):
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.0
        self.weight_coefficient = 0.4
        self.species_elitism = 0.2
        self.species_stagnation_threshold = 15
        self.min_species_count = 2
        
        self.weight_mutation_rate = 0.8
        self.weight_perturb_prob = 0.9
        self.weight_perturb_amount = 0.5
        self.connection_toggle_rate = 0.1
        self.node_activation_mutation_rate = 0.1
        self.node_bias_mutation_rate = 0.1
        self.bias_perturb_amount = 0.5
        self.add_node_mutation_rate = 0.03
        self.add_connection_mutation_rate = 0.05
        
        self.asexual_reproduction_rate = 0.25
        
        self.activation_functions = ['sigmoid', 'tanh', 'relu']


class NodeType(Enum):
    """Enumeration of possible node types in a NEAT network."""
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()
    BIAS = auto()


class NodeGene:
    """Represents a node gene in NEAT."""
    
    def __init__(self, node_id: int, node_type: NodeType, activation: str = 'sigmoid'):
        """Initialize a node gene."""
        self.id = node_id
        self.type = node_type
        self.activation = activation
        self.response = 1.0  # Response multiplier
        self.bias = 0.0 if node_type != NodeType.BIAS else 1.0
        
    def copy(self) -> 'NodeGene':
        """Create a copy of this node gene."""
        new_node = NodeGene(self.id, self.type, self.activation)
        new_node.response = self.response
        new_node.bias = self.bias
        return new_node


class ConnectionGene:
    """Represents a connection gene in NEAT."""
    
    def __init__(self, input_node: int, output_node: int, weight: float, 
                 innovation: int, enabled: bool = True):
        """Initialize a connection gene."""
        self.input = input_node
        self.output = output_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled
        
    def copy(self) -> 'ConnectionGene':
        """Create a copy of this connection gene."""
        return ConnectionGene(
            self.input, 
            self.output, 
            self.weight, 
            self.innovation, 
            self.enabled
        )


class NEATGenome:
    """Represents a genome in NEAT."""
    
    def __init__(self):
        """Initialize an empty genome."""
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[Tuple[int, int], ConnectionGene] = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = None
        
    def add_node(self, node: NodeGene) -> None:
        """Add a node to the genome."""
        self.nodes[node.id] = node
        
    def add_connection(self, conn: ConnectionGene) -> None:
        """Add a connection to the genome."""
        self.connections[(conn.input, conn.output)] = conn
        
    def mutate_weight(self, perturb_prob: float = 0.9, perturb_amount: float = 0.5) -> None:
        """Mutate connection weights."""
        for conn in self.connections.values():
            if random.random() < perturb_prob:
                conn.weight += random.uniform(-perturb_amount, perturb_amount)
                conn.weight = max(-8.0, min(8.0, conn.weight))
            else:
                conn.weight = random.uniform(-4.0, 4.0)
    
    def mutate_add_node(self, innovation_history: Dict) -> None:
        """Add a new node by splitting an existing connection."""
        if not self.connections:
            return
            
        enabled_connections = [c for c in self.connections.values() if c.enabled]
        if not enabled_connections:
            return
            
        conn = random.choice(enabled_connections)
        
        conn.enabled = False
        
        new_node_id = max(self.nodes.keys()) + 1
        new_node = NodeGene(new_node_id, NodeType.HIDDEN)
        self.add_node(new_node)
        
        in_to_new_key = (conn.input, new_node_id)
        if in_to_new_key in innovation_history:
            in_to_new_innovation = innovation_history[in_to_new_key]
        else:
            in_to_new_innovation = max(innovation_history.values()) + 1 if innovation_history else 0
            innovation_history[in_to_new_key] = in_to_new_innovation
            
        in_to_new = ConnectionGene(conn.input, new_node_id, 1.0, in_to_new_innovation)
        self.add_connection(in_to_new)
        
        new_to_out_key = (new_node_id, conn.output)
        if new_to_out_key in innovation_history:
            new_to_out_innovation = innovation_history[new_to_out_key]
        else:
            new_to_out_innovation = max(innovation_history.values()) + 1 if innovation_history else 0
            innovation_history[new_to_out_key] = new_to_out_innovation
            
        new_to_out = ConnectionGene(new_node_id, conn.output, conn.weight, new_to_out_innovation)
        self.add_connection(new_to_out)
    
    def mutate_add_connection(self, innovation_history: Dict) -> None:
        """Add a new connection between two unconnected nodes."""
        possible_connections = []
        
        for in_node_id, in_node in self.nodes.items():
            for out_node_id, out_node in self.nodes.items():
                if in_node.type == NodeType.OUTPUT or out_node.type == NodeType.INPUT:
                    continue
                    
                if (in_node_id, out_node_id) in self.connections:
                    continue
                    
                if in_node_id == out_node_id:
                    continue
                    
                possible_connections.append((in_node_id, out_node_id))
                
        if not possible_connections:
            return
            
        in_node_id, out_node_id = random.choice(possible_connections)
        
        conn_key = (in_node_id, out_node_id)
        if conn_key in innovation_history:
            innovation = innovation_history[conn_key]
        else:
            innovation = max(innovation_history.values()) + 1 if innovation_history else 0
            innovation_history[conn_key] = innovation
            
        weight = random.uniform(-2.0, 2.0)
        conn = ConnectionGene(in_node_id, out_node_id, weight, innovation)
        self.add_connection(conn)
    
    def crossover(self, other: 'NEATGenome') -> 'NEATGenome':
        """Perform crossover with another genome."""
        if self.fitness > other.fitness:
            parent1, parent2 = self, other
        else:
            parent1, parent2 = other, self
            
        child = NEATGenome()
        
        for node_id, node in parent1.nodes.items():
            child.add_node(node.copy())
            
        for node_id, node in parent2.nodes.items():
            if node_id not in child.nodes:
                child.add_node(node.copy())
        
        for conn_key, conn1 in parent1.connections.items():
            if conn_key in parent2.connections:
                conn2 = parent2.connections[conn_key]
                if abs(parent1.fitness - parent2.fitness) < 0.001:
                    conn = random.choice([conn1, conn2]).copy()
                else:
                    conn = conn1.copy()
                    
                if not conn1.enabled or not conn2.enabled:
                    conn.enabled = random.random() < 0.75
                    
                child.add_connection(conn)
            else:
                child.add_connection(conn1.copy())
                
        return child
    
    def is_compatible(self, other: 'NEATGenome', config: NEATConfig) -> bool:
        """Check if this genome is compatible with another genome."""
        disjoint_excess = 0
        matching = 0
        weight_diff = 0.0
        
        all_innovations_self = {c.innovation for c in self.connections.values()}
        all_innovations_other = {c.innovation for c in other.connections.values()}
        
        max_innovation_self = max(all_innovations_self) if all_innovations_self else 0
        max_innovation_other = max(all_innovations_other) if all_innovations_other else 0
        
        for conn in self.connections.values():
            if conn.innovation in all_innovations_other:
                matching += 1
                other_conn = next(c for c in other.connections.values() 
                                if c.innovation == conn.innovation)
                weight_diff += abs(conn.weight - other_conn.weight)
            elif conn.innovation <= max_innovation_other:
                disjoint_excess += 1
            else:
                disjoint_excess += 1
                
        for conn in other.connections.values():
            if conn.innovation not in all_innovations_self:
                if conn.innovation <= max_innovation_self:
                    disjoint_excess += 1
                else:
                    disjoint_excess += 1
        
        n = max(len(self.connections), len(other.connections))
        n = 1 if n < 1 else n  # Avoid division by zero
        
        weight_diff = weight_diff / matching if matching > 0 else 0
        
        compatibility = (
            config.excess_coefficient * disjoint_excess / n +
            config.weight_coefficient * weight_diff
        )
        
        return compatibility < config.compatibility_threshold
    
    def copy(self) -> 'NEATGenome':
        """Create a deep copy of this genome."""
        new_genome = NEATGenome()
        
        for node_id, node in self.nodes.items():
            new_genome.nodes[node_id] = node.copy()
            
        for conn_key, conn in self.connections.items():
            new_genome.connections[conn_key] = conn.copy()
            
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id
        
        return new_genome


class NEATNetwork(Module):
    """Neural network implementation of a NEAT genome."""
    
    def __init__(self, genome: NEATGenome):
        """Initialize a neural network from a NEAT genome."""
        super().__init__()
        self.genome = genome
        self.input_nodes = [n.id for n in genome.nodes.values() if n.type == NodeType.INPUT]
        self.output_nodes = [n.id for n in genome.nodes.values() if n.type == NodeType.OUTPUT]
        self.bias_nodes = [n.id for n in genome.nodes.values() if n.type == NodeType.BIAS]
        self.hidden_nodes = [n.id for n in genome.nodes.values() if n.type == NodeType.HIDDEN]
        
        self.input_nodes.sort()
        self.output_nodes.sort()
        self.bias_nodes.sort()
        
        self.hidden_nodes = self._sort_nodes()
        
        self.activation_functions = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu,
        }
        
    def _sort_nodes(self) -> List[int]:
        """Topologically sort the hidden nodes."""
        connections = defaultdict(list)
        for conn in self.genome.connections.values():
            if conn.enabled:
                connections[conn.output].append(conn.input)
                
        sorted_nodes = []
        unsorted = set(self.hidden_nodes)
        
        while unsorted:
            ready = []
            for node in unsorted:
                deps_satisfied = True
                for dep in connections[node]:
                    if dep in unsorted:
                        deps_satisfied = False
                        break
                        
                if deps_satisfied:
                    ready.append(node)
                    
            if not ready:
                sorted_nodes.extend(list(unsorted))
                break
                
            sorted_nodes.extend(ready)
            unsorted -= set(ready)
            
        return sorted_nodes
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        batch_size = x.shape[0]
        
        values = {}
        
        for i, node_id in enumerate(self.input_nodes):
            if i < x.shape[1]:
                values[node_id] = x[:, i]
            
        for node_id in self.bias_nodes:
            values[node_id] = Tensor([1.0] * batch_size)
            
        for node_id in self.hidden_nodes + self.output_nodes:
            node = self.genome.nodes[node_id]
            incoming_sum = None
            
            for conn_key, conn in self.connections.items():
                if conn.output == node_id and conn.enabled:
                    if conn.input in values:
                        if incoming_sum is None:
                            incoming_sum = values[conn.input] * conn.weight
                        else:
                            incoming_sum = incoming_sum + values[conn.input] * conn.weight
            
            if incoming_sum is not None:
                activation_fn = self.activation_functions.get(node.activation, sigmoid)
                values[node_id] = activation_fn(incoming_sum + node.bias)
            else:
                values[node_id] = Tensor([node.bias] * batch_size)
                
        outputs = []
        for node_id in self.output_nodes:
            if node_id in values:
                outputs.append(values[node_id].unsqueeze(1))
            else:
                outputs.append(Tensor([[0.0]] * batch_size))
                
        if not outputs:
            return Tensor([[0.0]] * batch_size)
            
        return Tensor.cat(outputs, dim=1)


class NEATSpecies:
    """Represents a species in NEAT."""
    
    def __init__(self, id: int):
        """Initialize a species."""
        self.id = id
        self.members = []
        self.representative = None
        self.fitness_history = []
        self.stagnation = 0
        self.max_fitness = 0.0
        self.adjusted_fitness_sum = 0.0
        
    def add(self, genome: NEATGenome) -> None:
        """Add a genome to this species."""
        self.members.append(genome)
        genome.species_id = self.id
        
    def update_representative(self) -> None:
        """Update the representative to a random member."""
        if self.members:
            self.representative = random.choice(self.members).copy()
            
    def calculate_adjusted_fitness(self) -> float:
        """Calculate the adjusted fitness for all members."""
        self.adjusted_fitness_sum = 0.0
        
        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / len(self.members)
            self.adjusted_fitness_sum += genome.adjusted_fitness
            
        return self.adjusted_fitness_sum
    
    def update_fitness(self) -> bool:
        """Update the fitness history and check for stagnation."""
        if not self.members:
            return False
            
        current_max = max(genome.fitness for genome in self.members)
        self.fitness_history.append(current_max)
        
        improved = False
        if current_max > self.max_fitness:
            self.max_fitness = current_max
            self.stagnation = 0
            improved = True
        else:
            self.stagnation += 1
            
        return improved
    
    def cull(self, keep_percent: float = 0.5) -> None:
        """Remove the least fit members of the species."""
        if len(self.members) <= 1:
            return
            
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        
        survivors = max(1, int(len(self.members) * keep_percent))
        self.members = self.members[:survivors]
        
    def breed(self, innovation_history: Dict, config: NEATConfig) -> NEATGenome:
        """Create a new offspring from this species."""
        if len(self.members) == 0:
            return self.representative.copy() if self.representative else None
            
        if len(self.members) == 1 or random.random() < config.asexual_reproduction_rate:
            child = random.choice(self.members).copy()
        else:
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)
            
            while len(self.members) > 1 and parent1 is parent2:
                parent2 = random.choice(self.members)
                
            child = parent1.crossover(parent2)
            
        if random.random() < config.weight_mutation_rate:
            child.mutate_weight(config.weight_perturb_prob, config.weight_perturb_amount)
            
        if random.random() < config.add_node_mutation_rate:
            child.mutate_add_node(innovation_history)
            
        if random.random() < config.add_connection_mutation_rate:
            child.mutate_add_connection(innovation_history)
            
        return child


class NEAT:
    """Implementation of the NEAT algorithm."""
    
    def __init__(self, config: NEATConfig = None):
        """Initialize the NEAT algorithm."""
        self.config = config or NEATConfig()
        self.species = {}
        self.genomes = []
        self.generation = 0
        self.innovation_history = {}
        self.node_innovation = 0
        self.connection_innovation = 0
        self.best_genome = None
        self.best_fitness = 0.0
        self.species_counter = 0
        
    def initialize(self, population_size: int, num_inputs: int, num_outputs: int) -> None:
        """Initialize the population with random genomes."""
        self.genomes = []
        
        for _ in range(population_size):
            genome = self._create_minimal_genome(num_inputs, num_outputs)
            self.genomes.append(genome)
            
        self._speciate()
        
    def _create_minimal_genome(self, num_inputs: int, num_outputs: int) -> NEATGenome:
        """Create a minimal genome with random connections."""
        genome = NEATGenome()
        
        for i in range(num_inputs):
            node = NodeGene(i, NodeType.INPUT)
            genome.add_node(node)
            self.node_innovation = max(self.node_innovation, i)
            
        bias_id = self.node_innovation + 1
        bias_node = NodeGene(bias_id, NodeType.BIAS)
        genome.add_node(bias_node)
        self.node_innovation = bias_id
        
        output_ids = []
        for i in range(num_outputs):
            node_id = self.node_innovation + 1 + i
            node = NodeGene(node_id, NodeType.OUTPUT)
            genome.add_node(node)
            output_ids.append(node_id)
            self.node_innovation = node_id
            
        for input_id in range(num_inputs):
            for output_id in output_ids:
                key = (input_id, output_id)
                if key in self.innovation_history:
                    innovation = self.innovation_history[key]
                else:
                    self.connection_innovation += 1
                    innovation = self.connection_innovation
                    self.innovation_history[key] = innovation
                    
                weight = random.uniform(-2.0, 2.0)
                conn = ConnectionGene(input_id, output_id, weight, innovation)
                genome.add_connection(conn)
                
        for output_id in output_ids:
            key = (bias_id, output_id)
            if key in self.innovation_history:
                innovation = self.innovation_history[key]
            else:
                self.connection_innovation += 1
                innovation = self.connection_innovation
                self.innovation_history[key] = innovation
                
            weight = random.uniform(-2.0, 2.0)
            conn = ConnectionGene(bias_id, output_id, weight, innovation)
            genome.add_connection(conn)
            
        return genome
    
    def _speciate(self) -> None:
        """Assign each genome to a species."""
        for species in self.species.values():
            species.members.clear()
            
        for genome in self.genomes:
            found_species = False
            
            for species in self.species.values():
                if species.representative and genome.is_compatible(species.representative, self.config):
                    species.add(genome)
                    found_species = True
                    break
                    
            if not found_species:
                self.species_counter += 1
                new_species = NEATSpecies(self.species_counter)
                new_species.add(genome)
                new_species.representative = genome.copy()
                self.species[self.species_counter] = new_species
                
        self.species = {sid: species for sid, species in self.species.items() if len(species.members) > 0}
        
    def evolve(self, fitness_function: Callable[[NEATGenome], float]) -> None:
        """Evolve the population for one generation."""
        for genome in self.genomes:
            genome.fitness = fitness_function(genome)
            
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()
                
        self._speciate()
        
        stagnant_species = []
        for species_id, species in self.species.items():
            improved = species.update_fitness()
            if not improved and species.stagnation >= self.config.species_stagnation_threshold:
                stagnant_species.append(species_id)
                
        for species_id in stagnant_species:
            has_best = any(genome.fitness >= self.best_fitness for genome in self.species[species_id].members)
            if not has_best and len(self.species) > self.config.min_species_count:
                del self.species[species_id]
                
        total_adjusted_fitness = sum(species.calculate_adjusted_fitness() for species in self.species.values())
        
        for species in self.species.values():
            species.cull(self.config.species_elitism)
            
        new_population = []
        
        if self.config.species_elitism > 0:
            for species in self.species.values():
                if len(species.members) > 0:
                    champion = max(species.members, key=lambda x: x.fitness)
                    new_population.append(champion.copy())
                    
        while len(new_population) < len(self.genomes):
            if total_adjusted_fitness > 0:
                species = self._select_species(total_adjusted_fitness)
            else:
                species = random.choice(list(self.species.values()))
                
            child = species.breed(self.innovation_history, self.config)
            if child:
                new_population.append(child)
                
        self.genomes = new_population
        self.generation += 1
        
        for species in self.species.values():
            species.update_representative()
            
    def _select_species(self, total_fitness: float) -> NEATSpecies:
        """Select a species based on fitness."""
        if total_fitness <= 0:
            return random.choice(list(self.species.values()))
            
        target = random.uniform(0, total_fitness)
        current_sum = 0
        
        for species in self.species.values():
            current_sum += species.adjusted_fitness_sum
            if current_sum >= target:
                return species
                
        return list(self.species.values())[-1]
    
    def get_best_genome(self) -> NEATGenome:
        """Get the best genome found so far."""
        return self.best_genome
    
    def get_network(self, genome: NEATGenome = None) -> NEATNetwork:
        """Get a neural network from a genome."""
        if genome is None:
            genome = self.best_genome
            
        if genome is None:
            return None
            
        return NEATNetwork(genome)
