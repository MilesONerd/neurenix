# Agent API Documentation

## Overview

The Agent API provides functionality for developing autonomous AI agents within the Neurenix framework. These agents can perceive their environment, make decisions, and take actions to achieve specific goals. The module supports various agent architectures, from simple reactive agents to complex cognitive agents with planning capabilities.

Neurenix's Agent API is designed to be flexible and extensible, allowing developers to create agents for a wide range of applications, including reinforcement learning, multi-agent systems, and autonomous decision-making.

## Key Concepts

### Agent Architecture

An agent in Neurenix consists of several key components:

1. **Perception**: Processing sensory input from the environment
2. **Memory**: Storing and retrieving information
3. **Decision Making**: Selecting actions based on current state
4. **Action Execution**: Implementing chosen actions in the environment

### Agent Types

Neurenix supports multiple agent types:

- **Reactive Agents**: Simple stimulus-response agents without internal state
- **Deliberative Agents**: Agents that maintain internal state and plan ahead
- **Hybrid Agents**: Combining reactive and deliberative approaches
- **Learning Agents**: Agents that improve through experience

### Environments

Agents interact with environments, which provide observations and rewards. Neurenix supports various environment types:

- **Simulated Environments**: Virtual worlds for training and testing
- **Real-World Interfaces**: Connections to physical systems
- **Multi-Agent Environments**: Environments with multiple interacting agents

## API Reference

### Agent Base Class

```python
neurenix.agent.Agent(name: str = None)
```

Base class for all agents in Neurenix.

**Parameters:**
- `name`: Optional name for the agent

**Methods:**
- `perceive(observation)`: Process an observation from the environment
- `decide()`: Make a decision based on current state
- `act(action)`: Execute an action in the environment
- `learn(reward)`: Update the agent based on received reward
- `reset()`: Reset the agent to its initial state

**Example:**
```python
import neurenix as nx
from neurenix.agent import Agent

class MyAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name)
        self.state = None
    
    def perceive(self, observation):
        self.state = observation
        return self.state
    
    def decide(self):
        # Simple decision logic
        if self.state > 0:
            return "move_forward"
        else:
            return "move_backward"
    
    def act(self, action):
        print(f"Executing action: {action}")
        return action
    
    def learn(self, reward):
        print(f"Received reward: {reward}")
    
    def reset(self):
        self.state = None

# Create an agent
agent = MyAgent("my_agent")

# Use the agent
observation = 1.0
state = agent.perceive(observation)
action = agent.decide()
agent.act(action)
agent.learn(0.5)
agent.reset()
```

### Reactive Agent

```python
neurenix.agent.ReactiveAgent(
    policy: Callable[[Any], Any],
    name: str = None
)
```

A simple reactive agent that maps observations directly to actions.

**Parameters:**
- `policy`: Function mapping observations to actions
- `name`: Optional name for the agent

**Example:**
```python
from neurenix.agent import ReactiveAgent

# Define a simple policy
def simple_policy(observation):
    if observation > 0:
        return "move_forward"
    else:
        return "move_backward"

# Create a reactive agent
agent = ReactiveAgent(simple_policy, "reactive_agent")

# Use the agent
observation = 1.0
agent.perceive(observation)
action = agent.decide()
agent.act(action)
```

### Deliberative Agent

```python
neurenix.agent.DeliberativeAgent(
    policy: Callable[[Any], Any],
    planner: Callable[[Any], List[Any]],
    name: str = None
)
```

An agent that plans ahead before taking actions.

**Parameters:**
- `policy`: Function mapping states to actions
- `planner`: Function generating action plans
- `name`: Optional name for the agent

**Methods:**
- `plan()`: Generate a plan based on current state

**Example:**
```python
from neurenix.agent import DeliberativeAgent

# Define a policy
def policy(state):
    return state["plan"][0] if state["plan"] else "wait"

# Define a planner
def planner(state):
    if state["goal"] == "reach_target":
        return ["move_forward", "turn_right", "move_forward"]
    return []

# Create a deliberative agent
agent = DeliberativeAgent(policy, planner, "deliberative_agent")

# Use the agent
state = {"goal": "reach_target", "position": [0, 0], "plan": []}
agent.perceive(state)
agent.plan()  # Generate a plan
action = agent.decide()
agent.act(action)
```

### Learning Agent

```python
neurenix.agent.LearningAgent(
    policy: Callable[[Any], Any],
    learning_algorithm: str,
    learning_rate: float = 0.1,
    name: str = None
)
```

An agent that improves its policy through experience.

**Parameters:**
- `policy`: Initial policy function
- `learning_algorithm`: Algorithm for learning ("q_learning", "sarsa", etc.)
- `learning_rate`: Rate of learning
- `name`: Optional name for the agent

**Methods:**
- `update_policy(state, action, reward, next_state)`: Update the policy based on experience

**Example:**
```python
from neurenix.agent import LearningAgent

# Define an initial policy
def initial_policy(state):
    # Random initial policy
    import random
    actions = ["up", "down", "left", "right"]
    return random.choice(actions)

# Create a learning agent
agent = LearningAgent(
    policy=initial_policy,
    learning_algorithm="q_learning",
    learning_rate=0.1,
    name="learning_agent"
)

# Use the agent
state = {"position": [0, 0]}
agent.perceive(state)
action = agent.decide()
agent.act(action)
reward = 0.5
next_state = {"position": [0, 1]}
agent.update_policy(state, action, reward, next_state)
```

### Environment

```python
neurenix.agent.Environment(name: str = None)
```

Base class for environments in which agents operate.

**Parameters:**
- `name`: Optional name for the environment

**Methods:**
- `reset()`: Reset the environment to its initial state
- `step(action)`: Update the environment based on an action
- `get_observation()`: Get the current observation
- `get_reward(action)`: Get the reward for an action
- `is_done()`: Check if the episode is complete

**Example:**
```python
from neurenix.agent import Environment

class GridWorld(Environment):
    def __init__(self, width=5, height=5, name=None):
        super().__init__(name)
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.target_pos = [width-1, height-1]
    
    def reset(self):
        self.agent_pos = [0, 0]
        return self.get_observation()
    
    def step(self, action):
        if action == "up":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == "down":
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == "left":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == "right":
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)
        
        observation = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        
        return observation, reward, done
    
    def get_observation(self):
        return {
            "agent_position": self.agent_pos,
            "target_position": self.target_pos
        }
    
    def get_reward(self, action):
        if self.agent_pos == self.target_pos:
            return 1.0
        return -0.01  # Small penalty for each step
    
    def is_done(self):
        return self.agent_pos == self.target_pos

# Create an environment
env = GridWorld(width=5, height=5, name="grid_world")

# Reset the environment
observation = env.reset()

# Take a step
action = "right"
next_observation, reward, done = env.step(action)
```

### Agent Manager

```python
neurenix.agent.AgentManager(name: str = None)
```

Manages multiple agents in a shared environment.

**Parameters:**
- `name`: Optional name for the manager

**Methods:**
- `add_agent(agent)`: Add an agent to the manager
- `remove_agent(agent)`: Remove an agent from the manager
- `set_environment(environment)`: Set the shared environment
- `step()`: Advance all agents by one step
- `run(steps)`: Run the simulation for a specified number of steps

**Example:**
```python
from neurenix.agent import AgentManager, ReactiveAgent, Environment

# Create agents
agent1 = ReactiveAgent(lambda obs: "right", "agent1")
agent2 = ReactiveAgent(lambda obs: "left", "agent2")

# Create environment
env = GridWorld(width=10, height=10, name="shared_grid")

# Create manager
manager = AgentManager("grid_manager")
manager.add_agent(agent1)
manager.add_agent(agent2)
manager.set_environment(env)

# Run simulation
manager.run(steps=100)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Agent Framework** | Native | TF-Agents |
| **Agent Types** | Reactive, Deliberative, Hybrid, Learning | Primarily RL-based |
| **Environment Integration** | Flexible, custom environments | Gym-compatible environments |
| **Multi-Agent Support** | Native | Limited |
| **Planning Capabilities** | Built-in | Limited |

Neurenix provides a more comprehensive agent framework compared to TensorFlow's TF-Agents, with support for various agent architectures beyond reinforcement learning. While TF-Agents focuses primarily on RL algorithms, Neurenix's Agent API supports a wider range of agent types, including deliberative agents with planning capabilities and hybrid architectures.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Agent Framework** | Native | Third-party libraries |
| **Agent Types** | Reactive, Deliberative, Hybrid, Learning | Depends on library |
| **Environment Integration** | Flexible, custom environments | Depends on library |
| **Multi-Agent Support** | Native | Limited |
| **Planning Capabilities** | Built-in | Limited |

PyTorch does not provide a native agent framework, relying instead on third-party libraries like RLlib or PyTorch-RL. Neurenix's integrated Agent API offers a more cohesive experience, with consistent interfaces and better integration with the rest of the framework.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Agent Framework** | Native | None |
| **Agent Types** | Reactive, Deliberative, Hybrid, Learning | N/A |
| **Environment Integration** | Flexible, custom environments | N/A |
| **Multi-Agent Support** | Native | N/A |
| **Planning Capabilities** | Built-in | N/A |

Scikit-Learn does not provide agent-based functionality, focusing instead on traditional machine learning algorithms. Neurenix fills this gap with its comprehensive Agent API, enabling the development of autonomous agents for various applications.

## Best Practices

### Agent Design

Design agents with clear separation of concerns:

```python
from neurenix.agent import Agent

class WellDesignedAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name)
        self.perception_module = PerceptionModule()
        self.decision_module = DecisionModule()
        self.action_module = ActionModule()
        self.learning_module = LearningModule()
    
    def perceive(self, observation):
        return self.perception_module.process(observation)
    
    def decide(self):
        return self.decision_module.select_action(self.state)
    
    def act(self, action):
        return self.action_module.execute(action)
    
    def learn(self, reward):
        self.learning_module.update(self.state, self.action, reward)
```

### Environment Interaction

Use a consistent pattern for environment interaction:

```python
# Create agent and environment
agent = MyAgent()
env = MyEnvironment()

# Reset environment
observation = env.reset()

done = False
while not done:
    # Agent perceives the environment
    agent.perceive(observation)
    
    # Agent decides on an action
    action = agent.decide()
    
    # Agent acts in the environment
    agent.act(action)
    
    # Environment updates
    observation, reward, done = env.step(action)
    
    # Agent learns from the experience
    agent.learn(reward)
```

### Multi-Agent Coordination

For multi-agent systems, establish clear communication protocols:

```python
from neurenix.agent import Agent, AgentManager

class CommunicatingAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name)
        self.messages = []
    
    def send_message(self, recipient, content):
        return {"sender": self.name, "recipient": recipient, "content": content}
    
    def receive_message(self, message):
        self.messages.append(message)
    
    def decide(self):
        # Consider messages in decision-making
        # ...
        return action

# Create a manager that handles communication
class CommunicatingManager(AgentManager):
    def __init__(self, name=None):
        super().__init__(name)
        self.message_queue = []
    
    def step(self):
        # Collect messages from all agents
        for agent in self.agents:
            if hasattr(agent, "send_message"):
                messages = agent.decide_messages()
                self.message_queue.extend(messages)
        
        # Deliver messages to recipients
        for message in self.message_queue:
            recipient_name = message["recipient"]
            for agent in self.agents:
                if agent.name == recipient_name and hasattr(agent, "receive_message"):
                    agent.receive_message(message)
        
        
        self.message_queue = []
        
        # Regular step
        super().step()
```

## Tutorials

### Creating a Simple Reactive Agent

```python
import neurenix as nx
from neurenix.agent import ReactiveAgent, Environment

# Define a simple grid environment
class GridEnvironment(Environment):
    def __init__(self, width=5, height=5):
        super().__init__("grid")
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.target_pos = [width-1, height-1]
    
    def reset(self):
        self.agent_pos = [0, 0]
        return self.get_observation()
    
    def step(self, action):
        if action == "up":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == "down":
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == "left":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == "right":
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)
        
        observation = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        
        return observation, reward, done
    
    def get_observation(self):
        return {
            "agent_position": self.agent_pos,
            "target_position": self.target_pos
        }
    
    def get_reward(self, action):
        if self.agent_pos == self.target_pos:
            return 1.0
        return -0.01
    
    def is_done(self):
        return self.agent_pos == self.target_pos

# Define a simple policy
def simple_policy(observation):
    agent_pos = observation["agent_position"]
    target_pos = observation["target_position"]
    
    # Move horizontally first, then vertically
    if agent_pos[0] < target_pos[0]:
        return "right"
    elif agent_pos[0] > target_pos[0]:
        return "left"
    elif agent_pos[1] < target_pos[1]:
        return "down"
    elif agent_pos[1] > target_pos[1]:
        return "up"
    else:
        return "stay"

# Create the agent and environment
agent = ReactiveAgent(simple_policy, "grid_agent")
env = GridEnvironment(width=5, height=5)

# Run the simulation
observation = env.reset()
done = False
total_reward = 0

while not done:
    # Agent perceives the environment
    agent.perceive(observation)
    
    # Agent decides on an action
    action = agent.decide()
    
    # Agent acts in the environment
    agent.act(action)
    
    # Environment updates
    observation, reward, done = env.step(action)
    total_reward += reward
    
    print(f"Position: {env.agent_pos}, Action: {action}, Reward: {reward}")

print(f"Total reward: {total_reward}")
```

### Creating a Learning Agent with Q-Learning

```python
import neurenix as nx
import numpy as np
from neurenix.agent import Agent, Environment

# Define a Q-learning agent
class QLearningAgent(Agent):
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, name=None):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        self.state = None
        self.action = None
    
    def perceive(self, observation):
        self.state = observation
        return self.state
    
    def decide(self):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_rate:
            # Explore: choose a random action
            self.action = np.random.randint(self.action_size)
        else:
            # Exploit: choose the best action based on Q-values
            self.action = np.argmax(self.q_table[self.state])
        
        return self.action
    
    def act(self, action):
        return action
    
    def learn(self, reward, next_state, done):
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[self.state, self.action]
        self.q_table[self.state, self.action] += self.learning_rate * td_error
        
        # Update exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
    
    def reset(self):
        self.state = None
        self.action = None

# Define a simple grid environment
class GridEnvironment(Environment):
    def __init__(self, width=5, height=5):
        super().__init__("grid")
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.target_pos = [width-1, height-1]
        
        # Define state space as a flattened grid
        self.state_size = width * height
        self.action_size = 4  # up, down, left, right
    
    def reset(self):
        self.agent_pos = [0, 0]
        return self.get_state()
    
    def step(self, action):
        # Convert action index to direction
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)
        
        state = self.get_state()
        reward = self.get_reward(action)
        done = self.is_done()
        
        return state, reward, done
    
    def get_state(self):
        # Convert 2D position to 1D state index
        return self.agent_pos[1] * self.width + self.agent_pos[0]
    
    def get_reward(self, action):
        if self.agent_pos == self.target_pos:
            return 1.0
        return -0.01
    
    def is_done(self):
        return self.agent_pos == self.target_pos

# Create the agent and environment
env = GridEnvironment(width=5, height=5)
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=1.0,
    name="q_learning_agent"
)

# Training loop
num_episodes = 1000
max_steps = 100

for episode in range(num_episodes):
    state = env.reset()
    agent.perceive(state)
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Agent decides on an action
        action = agent.decide()
        
        # Agent acts in the environment
        agent.act(action)
        
        # Environment updates
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Agent learns from the experience
        agent.learn(reward, next_state, done)
        
        # Update state
        state = next_state
        agent.perceive(state)
        
        step += 1
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {step}, Exploration Rate: {agent.exploration_rate:.4f}")

# Test the trained agent
state = env.reset()
agent.perceive(state)
agent.exploration_rate = 0  # No exploration during testing

done = False
total_reward = 0
step = 0

print("\nTesting the trained agent:")
while not done and step < max_steps:
    action = agent.decide()
    agent.act(action)
    
    next_state, reward, done = env.step(action)
    total_reward += reward
    
    print(f"Step {step}, Position: {env.agent_pos}, Action: {action}, Reward: {reward}")
    
    state = next_state
    agent.perceive(state)
    
    step += 1

print(f"Total reward: {total_reward}, Steps: {step}")
```

## Conclusion

The Agent API in Neurenix provides a comprehensive framework for developing autonomous AI agents. With support for various agent architectures, from simple reactive agents to complex learning agents, the API enables developers to create intelligent systems for a wide range of applications.

Compared to other frameworks, Neurenix's Agent API offers unique advantages in terms of flexibility, integration with the rest of the framework, and support for advanced agent capabilities like planning and multi-agent coordination. These features make Neurenix particularly well-suited for developing autonomous AI systems.
