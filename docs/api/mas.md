# Multi-Agent Systems API Documentation

## Overview

The Multi-Agent Systems (MAS) module provides implementations for distributed artificial intelligence, agent-based modeling, and collaborative learning. Multi-agent systems consist of multiple interacting intelligent agents that can perceive their environment, make decisions, and take actions to achieve individual or collective goals.

Unlike single-agent systems, multi-agent systems can model complex interactions, emergent behaviors, and social dynamics that arise when multiple autonomous entities operate in a shared environment. This makes them particularly valuable for applications such as distributed problem solving, simulation of social and economic systems, robotics, game theory, and collaborative AI.

## Key Concepts

### Agents

Agents are autonomous entities that can perceive their environment, make decisions, and take actions. In Neurenix, agents can be implemented with different architectures:

- **Reactive Agents**: Simple stimulus-response agents that act based on current perceptions without maintaining internal state
- **Deliberative Agents**: Agents that maintain internal representations of the world and plan ahead
- **Hybrid Agents**: Agents that combine reactive and deliberative approaches

### Environments

Environments define the world in which agents operate. They provide observations to agents and change in response to agent actions. Neurenix supports various environment types:

- **Grid Environments**: Discrete grid-based worlds
- **Continuous Environments**: Continuous state and action spaces
- **Network Environments**: Environments structured as networks or graphs

### Communication

Communication enables agents to exchange information. Neurenix provides mechanisms for agent communication:

- **Messages**: Structured information exchanged between agents
- **Channels**: Communication pathways between agents
- **Protocols**: Rules governing communication
- **Mailboxes**: Storage for received messages
- **Communication Networks**: Topologies defining which agents can communicate with each other

### Coordination

Coordination mechanisms help agents work together effectively. Neurenix implements various coordination strategies:

- **Auctions**: Market-based mechanisms for resource allocation
- **Contract Net**: Task allocation through bidding
- **Voting Mechanisms**: Collective decision making
- **Coalition Formation**: Dynamic team formation

### Multi-Agent Learning

Multi-agent learning enables agents to improve through experience in multi-agent settings. Neurenix supports different learning approaches:

- **Independent Learners**: Agents learn individually without considering others
- **Joint Action Learners**: Agents learn considering the joint action space
- **Team Learning**: Agents learn to optimize team performance
- **Opponent Modeling**: Agents model and adapt to other agents' behaviors

## API Reference

### Agent Components

```python
neurenix.mas.Agent(name=None)
```

Base class for all agents in multi-agent systems.

**Methods:**
- `perceive(observation)`: Process an observation from the environment
- `decide()`: Make a decision based on current state
- `act(action)`: Execute an action in the environment
- `update(reward)`: Update the agent based on received reward
- `send_message(recipient, content)`: Send a message to another agent
- `receive_message(message)`: Process a received message

```python
neurenix.mas.ReactiveAgent(stimulus_response_map=None, name=None)
```

Implements a reactive agent that maps stimuli directly to responses.

```python
neurenix.mas.DeliberativeAgent(planner=None, name=None)
```

Implements a deliberative agent that plans ahead.

```python
neurenix.mas.HybridAgent(reactive_component=None, deliberative_component=None, name=None)
```

Implements a hybrid agent that combines reactive and deliberative approaches.

```python
neurenix.mas.AgentState
```

Dataclass representing the state of an agent.

### Environment Components

```python
neurenix.mas.Environment(name=None)
```

Base class for all environments in multi-agent systems.

**Methods:**
- `reset()`: Reset the environment to its initial state
- `step(actions)`: Update the environment based on agent actions
- `get_observations()`: Get observations for all agents
- `get_rewards(actions)`: Get rewards for all agents
- `is_done()`: Check if the episode is complete

```python
neurenix.mas.GridEnvironment(width, height, num_agents=1, name=None)
```

Implements a grid-based environment.

```python
neurenix.mas.ContinuousEnvironment(bounds, num_agents=1, name=None)
```

Implements a continuous environment.

```python
neurenix.mas.NetworkEnvironment(adjacency_matrix=None, num_agents=1, name=None)
```

Implements a network-based environment.

```python
neurenix.mas.StateSpace(low, high, shape=None, dtype=None)
```

Defines the state space of an environment.

```python
neurenix.mas.ActionSpace(low=None, high=None, shape=None, dtype=None, discrete=False, n=None)
```

Defines the action space of an environment.

### Communication Components

```python
neurenix.mas.Message(sender, recipient, content, timestamp=None)
```

Represents a message exchanged between agents.

```python
neurenix.mas.Channel(sender, recipient, bandwidth=float('inf'), delay=0, reliability=1.0)
```

Represents a communication channel between agents.

```python
neurenix.mas.Protocol(name=None)
```

Defines a communication protocol.

```python
neurenix.mas.Mailbox(owner, capacity=float('inf'))
```

Stores messages received by an agent.

```python
neurenix.mas.CommunicationNetwork(agents, topology='fully_connected')
```

Defines the communication network between agents.

### Coordination Components

```python
neurenix.mas.Coordinator(agents, strategy='centralized', name=None)
```

Coordinates activities among multiple agents.

```python
neurenix.mas.Auction(auctioneer, bidders, auction_type='english', name=None)
```

Implements an auction-based coordination mechanism.

```python
neurenix.mas.ContractNet(manager, contractors, name=None)
```

Implements the Contract Net Protocol for task allocation.

```python
neurenix.mas.VotingMechanism(voters, voting_rule='majority', name=None)
```

Implements voting-based collective decision making.

```python
neurenix.mas.CoalitionFormation(agents, utility_function=None, name=None)
```

Implements dynamic coalition formation among agents.

### Learning Components

```python
neurenix.mas.MultiAgentLearning(agents, learning_rate=0.1, discount_factor=0.9, name=None)
```

Base class for multi-agent learning algorithms.

```python
neurenix.mas.IndependentLearners(agents, learning_rate=0.1, discount_factor=0.9, name=None)
```

Implements independent learning where each agent learns separately.

```python
neurenix.mas.JointActionLearners(agents, learning_rate=0.1, discount_factor=0.9, name=None)
```

Implements joint action learning where agents consider the joint action space.

```python
neurenix.mas.TeamLearning(agents, learning_rate=0.1, discount_factor=0.9, name=None)
```

Implements team learning where agents learn to optimize team performance.

```python
neurenix.mas.OpponentModeling(agents, learning_rate=0.1, discount_factor=0.9, name=None)
```

Implements learning with opponent modeling.

## Framework Comparison

### Neurenix vs. PettingZoo

| Feature | Neurenix | PettingZoo |
|---------|----------|------------|
| **API Design** | Unified, object-oriented API | OpenAI Gym-like API |
| **Agent Types** | Multiple agent architectures | Generic agent interface |
| **Environment Types** | Multiple environment types | Rich collection of pre-built environments |
| **Communication** | Comprehensive communication framework | Limited communication support |
| **Coordination** | Multiple coordination mechanisms | Limited coordination mechanisms |
| **Learning Algorithms** | Multiple multi-agent learning algorithms | No built-in learning algorithms |
| **Integration with Core Framework** | Seamless integration with Neurenix | Requires additional libraries for learning |

### Neurenix vs. RLLIB

| Feature | Neurenix | RLLIB |
|---------|----------|-------|
| **API Design** | Unified, object-oriented API | Ray-based distributed API |
| **Agent Types** | Multiple agent architectures | Primarily reinforcement learning agents |
| **Environment Types** | Multiple environment types | OpenAI Gym-compatible environments |
| **Communication** | Comprehensive communication framework | Limited communication support |
| **Coordination** | Multiple coordination mechanisms | Limited coordination mechanisms |
| **Learning Algorithms** | Multiple multi-agent learning algorithms | Strong reinforcement learning support |
| **Scalability** | Moderate scalability | Excellent scalability with Ray |

### Neurenix vs. MADDPG

| Feature | Neurenix | MADDPG |
|---------|----------|--------|
| **API Design** | Unified, object-oriented API | Algorithm-specific API |
| **Agent Types** | Multiple agent architectures | Actor-critic agents |
| **Environment Types** | Multiple environment types | Limited environment support |
| **Communication** | Comprehensive communication framework | No explicit communication support |
| **Coordination** | Multiple coordination mechanisms | Implicit coordination through learning |
| **Learning Algorithms** | Multiple multi-agent learning algorithms | Specific to MADDPG algorithm |
| **Flexibility** | Highly flexible | Limited to MADDPG algorithm |

## Best Practices

### Agent Design

When designing agents, consider the following:

1. **Appropriate Architecture**: Choose the right agent architecture for your problem
2. **Modularity**: Design agents with clear separation of concerns
3. **Adaptability**: Make agents adaptable to changing environments
4. **Resource Constraints**: Consider computational and memory constraints

### Environment Design

When designing environments, consider the following:

1. **State Representation**: Choose appropriate state representations
2. **Action Space**: Define clear and meaningful action spaces
3. **Reward Design**: Design rewards that align with desired behaviors
4. **Observability**: Consider partial observability when appropriate

### Communication Design

When designing communication systems, consider the following:

1. **Message Structure**: Define clear message structures
2. **Bandwidth Constraints**: Consider bandwidth limitations
3. **Reliability**: Account for potential message loss
4. **Scalability**: Design for scalability with increasing agents

### Coordination Design

When designing coordination mechanisms, consider the following:

1. **Efficiency**: Design for efficient task allocation
2. **Fairness**: Consider fairness in resource allocation
3. **Stability**: Ensure stability of coordination outcomes
4. **Incentive Compatibility**: Design mechanisms that incentivize truthful behavior

### Learning Design

When designing multi-agent learning systems, consider the following:

1. **Exploration-Exploitation**: Balance exploration and exploitation
2. **Credit Assignment**: Address the credit assignment problem
3. **Non-Stationarity**: Handle non-stationarity in multi-agent learning
4. **Scalability**: Design for scalability with increasing agents

## Tutorials

### Building a Predator-Prey Simulation

```python
import neurenix as nx
import numpy as np

# Create a grid environment for predator-prey simulation
class PredatorPreyEnvironment(nx.mas.GridEnvironment):
    def __init__(self, width=20, height=20, num_predators=3, num_prey=10):
        super().__init__(width, height, num_predators + num_prey, name="predator_prey")
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.predator_positions = []
        self.prey_positions = []
        self.prey_alive = [True] * num_prey
    
    def reset(self):
        # Reset predator and prey positions
        self.predator_positions = []
        self.prey_positions = []
        # Initialize positions randomly
        # Return observations
        return self._get_observations()
    
    def step(self, actions):
        # Process predator and prey actions
        # Check for captures
        # Return observations, rewards, and done
        observations = self._get_observations()
        rewards = self._get_rewards()
        done = not any(self.prey_alive)
        return observations, rewards, done

# Create predator and prey agents
class PredatorAgent(nx.mas.ReactiveAgent):
    def decide(self):
        # Simple predator strategy: move towards closest prey
        # Implementation details...
        return action

class PreyAgent(nx.mas.ReactiveAgent):
    def decide(self):
        # Simple prey strategy: move away from closest predator
        # Implementation details...
        return action

# Create environment and agents
env = PredatorPreyEnvironment()
predators = [PredatorAgent(f"predator_{i}") for i in range(env.num_predators)]
prey = [PreyAgent(f"prey_{i}") for i in range(env.num_prey)]

# Simulation loop
observations = env.reset()
done = False
while not done:
    # Get actions from all agents
    actions = {}
    for i, agent in enumerate(predators):
        agent.perceive(observations[i])
        actions[i] = agent.decide()
    
    for i, agent in enumerate(prey):
        agent_idx = i + env.num_predators
        if observations[agent_idx].get('alive', False):
            agent.perceive(observations[agent_idx])
            actions[agent_idx] = agent.decide()
    
    # Step the environment
    observations, rewards, done = env.step(actions)
    
    # Print status
    print(f"Predators: {env.predator_positions}")
    print(f"Prey alive: {sum(env.prey_alive)}/{env.num_prey}")
```

### Implementing a Contract Net Protocol

```python
import neurenix as nx

# Create a contract net for task allocation
class TaskManager(nx.mas.Agent):
    def __init__(self, name=None):
        super().__init__(name)
        self.tasks = []
        self.allocated_tasks = {}
    
    def add_task(self, task):
        self.tasks.append(task)
    
    def announce_tasks(self, contractors):
        for task in self.tasks:
            for contractor in contractors:
                message = nx.mas.Message(
                    sender=self.name,
                    recipient=contractor.name,
                    content={"type": "task_announcement", "task": task}
                )
                self.send_message(message)
    
    def receive_message(self, message):
        if message.content["type"] == "bid":
            task_id = message.content["task_id"]
            bid_value = message.content["bid_value"]
            
            # Award task to best bidder
            if task_id not in self.allocated_tasks or bid_value < self.allocated_tasks[task_id]["bid"]:
                self.allocated_tasks[task_id] = {
                    "contractor": message.sender,
                    "bid": bid_value
                }

class Contractor(nx.mas.Agent):
    def __init__(self, name=None, capabilities=None):
        super().__init__(name)
        self.capabilities = capabilities or {}
        self.assigned_tasks = []
    
    def receive_message(self, message):
        if message.content["type"] == "task_announcement":
            task = message.content["task"]
            
            # Evaluate task and submit bid
            if self.can_perform(task):
                bid_value = self.evaluate_task(task)
                
                response = nx.mas.Message(
                    sender=self.name,
                    recipient=message.sender,
                    content={
                        "type": "bid",
                        "task_id": task["id"],
                        "bid_value": bid_value
                    }
                )
                self.send_message(response)
        
        elif message.content["type"] == "task_award":
            task = message.content["task"]
            self.assigned_tasks.append(task)
    
    def can_perform(self, task):
        return task["required_capability"] in self.capabilities
    
    def evaluate_task(self, task):
        # Calculate bid based on capabilities and current load
        capability_level = self.capabilities.get(task["required_capability"], 0)
        current_load = len(self.assigned_tasks)
        
        # Lower bid is better
        return (1.0 / capability_level) * (1.0 + 0.1 * current_load)

# Create manager and contractors
manager = TaskManager("manager")
contractors = [
    Contractor("contractor_1", {"programming": 5, "design": 3}),
    Contractor("contractor_2", {"programming": 2, "design": 5}),
    Contractor("contractor_3", {"testing": 4, "programming": 3})
]

# Create tasks
tasks = [
    {"id": "task1", "description": "Implement feature X", "required_capability": "programming"},
    {"id": "task2", "description": "Design UI", "required_capability": "design"},
    {"id": "task3", "description": "Test module Y", "required_capability": "testing"}
]

# Add tasks to manager
for task in tasks:
    manager.add_task(task)

# Run contract net protocol
manager.announce_tasks(contractors)

# In a real system, messages would be transmitted through a communication network
# For simplicity, we simulate direct message passing
for contractor in contractors:
    for task in tasks:
        if contractor.can_perform(task):
            bid_value = contractor.evaluate_task(task)
            message = nx.mas.Message(
                sender=contractor.name,
                recipient=manager.name,
                content={
                    "type": "bid",
                    "task_id": task["id"],
                    "bid_value": bid_value
                }
            )
            manager.receive_message(message)

# Print task allocations
print("Task Allocations:")
for task_id, allocation in manager.allocated_tasks.items():
    print(f"Task {task_id} allocated to {allocation['contractor']} with bid {allocation['bid']}")
```

This documentation provides a comprehensive overview of the Multi-Agent Systems module in Neurenix, including key concepts, API reference, framework comparisons, best practices, and tutorials for common multi-agent system tasks.
