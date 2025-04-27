# Federated Learning API Documentation

## Overview

The Federated Learning module provides implementations of federated learning algorithms and utilities for distributed training across multiple devices or clients. Federated Learning enables model training on decentralized data without sharing the raw data, preserving privacy while leveraging collective knowledge.

In traditional machine learning, data is centralized on a server for training. In contrast, federated learning keeps data local to each client, with only model updates being shared. This approach is particularly valuable for privacy-sensitive applications in healthcare, finance, mobile devices, and IoT, where data cannot or should not be centralized.

Neurenix's Federated Learning module offers a comprehensive set of tools for implementing various federated learning strategies, with built-in support for privacy-preserving techniques, efficient communication, and heterogeneous client environments.

## Key Concepts

### Client-Server Architecture

Federated learning typically follows a client-server architecture:

- **Clients**: Entities with local data that train models locally and share updates
- **Server**: Central entity that coordinates the learning process, aggregates updates, and distributes the global model

Neurenix provides flexible implementations of both client and server components, allowing for customization based on specific requirements.

### Aggregation Strategies

Different strategies can be used to aggregate model updates from clients:

- **FedAvg**: Simple weighted averaging of client updates
- **FedProx**: Adds a proximal term to client optimization to handle heterogeneity
- **FedNova**: Normalizes and scales client updates based on their local steps
- **FedOpt**: Uses server-side optimization algorithms like Adam or Adagrad

### Privacy and Security

Federated learning inherently provides some privacy benefits by keeping raw data local, but additional techniques can enhance privacy and security:

- **Secure Aggregation**: Cryptographic techniques to aggregate updates without revealing individual contributions
- **Differential Privacy**: Adding calibrated noise to protect against inference attacks
- **Homomorphic Encryption**: Performing computations on encrypted data

### Communication Efficiency

Communication between clients and the server can be a bottleneck in federated learning. Techniques to improve efficiency include:

- **Model Compression**: Reducing the size of model updates
- **Gradient Compression**: Compressing gradients before transmission
- **Client Selection**: Selecting a subset of clients for each round

## API Reference

### Client Components

```python
neurenix.federated.FederatedClient(
    model: neurenix.nn.Module,
    optimizer: neurenix.optim.Optimizer,
    loss_fn: Callable,
    dataset: neurenix.data.Dataset,
    config: Optional[ClientConfig] = None
)
```

Represents a client in the federated learning system.

**Parameters:**
- `model`: The local model to be trained
- `optimizer`: Optimizer for local training
- `loss_fn`: Loss function for local training
- `dataset`: Local dataset
- `config`: Configuration for the client

**Methods:**
- `train(num_epochs: int) -> ClientState`: Perform local training
- `evaluate() -> Dict[str, float]`: Evaluate the model on local data
- `update_model(global_model: Dict[str, Tensor]) -> None`: Update local model with global model
- `get_model_update() -> Dict[str, Tensor]`: Get model update to send to server

```python
neurenix.federated.ClientConfig(
    batch_size: int = 32,
    local_epochs: int = 1,
    max_samples: Optional[int] = None,
    device: Optional[neurenix.Device] = None,
    privacy_mechanism: Optional[str] = None,
    privacy_budget: float = 1.0,
    compression_rate: float = 1.0
)
```

Configuration for a federated client.

**Parameters:**
- `batch_size`: Batch size for local training
- `local_epochs`: Number of local epochs per round
- `max_samples`: Maximum number of samples to use from local dataset
- `device`: Device to use for computation
- `privacy_mechanism`: Privacy mechanism to use (None, 'differential_privacy', 'homomorphic')
- `privacy_budget`: Privacy budget for differential privacy
- `compression_rate`: Compression rate for model updates

```python
neurenix.federated.ClientState
```

Dataclass representing the state of a client after training.

**Attributes:**
- `model_update`: Model update to send to server
- `num_samples`: Number of samples used for training
- `metrics`: Training metrics (loss, accuracy, etc.)
- `computation_time`: Time taken for local computation

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import SGD
from neurenix.federated import FederatedClient, ClientConfig

# Create a model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Create a dataset
dataset = nx.data.MNIST(root='./data', train=True, download=True)

# Create a client configuration
client_config = ClientConfig(
    batch_size=32,
    local_epochs=5,
    device=nx.Device('cuda:0') if nx.get_device_count(nx.DeviceType.CUDA) > 0 else nx.Device('cpu')
)

# Create a federated client
client = FederatedClient(model, optimizer, loss_fn, dataset, client_config)

# Perform local training
client_state = client.train(num_epochs=5)

# Get model update
model_update = client.get_model_update()
```

### Server Components

```python
neurenix.federated.FederatedServer(
    model: neurenix.nn.Module,
    strategy: AggregationStrategy,
    config: Optional[ServerConfig] = None
)
```

Represents the server in the federated learning system.

**Parameters:**
- `model`: The global model to be trained
- `strategy`: Strategy for aggregating client updates
- `config`: Configuration for the server

**Methods:**
- `select_clients(available_clients: List[str], num_clients: int) -> List[str]`: Select clients for the current round
- `aggregate_updates(updates: List[Tuple[Dict[str, Tensor], int]]) -> Dict[str, Tensor]`: Aggregate client updates
- `update_global_model(aggregated_update: Dict[str, Tensor]) -> None`: Update the global model
- `evaluate(test_data: neurenix.data.Dataset) -> Dict[str, float]`: Evaluate the global model
- `run_round(clients: List[FederatedClient]) -> ServerState`: Run a single federated round

```python
neurenix.federated.ServerConfig(
    num_rounds: int = 100,
    clients_per_round: int = 10,
    client_selection_strategy: str = 'random',
    evaluation_frequency: int = 1,
    secure_aggregation: bool = False,
    compression_enabled: bool = False
)
```

Configuration for a federated server.

**Parameters:**
- `num_rounds`: Number of federated rounds
- `clients_per_round`: Number of clients to select per round
- `client_selection_strategy`: Strategy for selecting clients ('random', 'power_of_choice')
- `evaluation_frequency`: Frequency of global model evaluation
- `secure_aggregation`: Whether to use secure aggregation
- `compression_enabled`: Whether to enable compression for communication

```python
neurenix.federated.ServerState
```

Dataclass representing the state of the server after a round.

**Attributes:**
- `round_num`: Current round number
- `selected_clients`: Clients selected for the round
- `aggregated_update`: Aggregated model update
- `metrics`: Evaluation metrics
- `communication_time`: Time taken for communication

```python
neurenix.federated.AggregationStrategy
```

Base class for aggregation strategies.

**Methods:**
- `aggregate(updates: List[Tuple[Dict[str, Tensor], int]]) -> Dict[str, Tensor]`: Aggregate client updates

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.federated import FederatedServer, ServerConfig, FedAvg

# Create a global model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Create an aggregation strategy
strategy = FedAvg()

# Create a server configuration
server_config = ServerConfig(
    num_rounds=100,
    clients_per_round=10,
    evaluation_frequency=5
)

# Create a federated server
server = FederatedServer(model, strategy, server_config)

# Run a federated learning experiment
for round_num in range(server_config.num_rounds):
    # Select clients
    selected_clients = server.select_clients(available_clients, server_config.clients_per_round)
    
    # Run a round
    server_state = server.run_round([clients[client_id] for client_id in selected_clients])
    
    # Evaluate the global model
    if round_num % server_config.evaluation_frequency == 0:
        metrics = server.evaluate(test_data)
        print(f"Round {round_num}, Accuracy: {metrics['accuracy']:.4f}")
```

### Aggregation Strategies

```python
neurenix.federated.FedAvg()
```

Implements the Federated Averaging algorithm, which computes a weighted average of client updates based on the number of samples.

```python
neurenix.federated.FedProx(mu: float = 0.01)
```

Implements the FedProx algorithm, which adds a proximal term to client optimization to handle heterogeneity.

**Parameters:**
- `mu`: Proximal term coefficient

```python
neurenix.federated.FedNova(tau_eff: Optional[float] = None)
```

Implements the FedNova algorithm, which normalizes and scales client updates based on their local steps.

**Parameters:**
- `tau_eff`: Effective number of local steps (if None, computed automatically)

```python
neurenix.federated.FedOpt(
    server_optimizer: str = 'sgd',
    server_lr: float = 0.1,
    server_momentum: float = 0.9,
    server_weight_decay: float = 0.0
)
```

Base class for server-side optimization algorithms.

**Parameters:**
- `server_optimizer`: Server optimizer type ('sgd', 'adam', 'adagrad')
- `server_lr`: Server learning rate
- `server_momentum`: Server momentum (for SGD)
- `server_weight_decay`: Server weight decay

```python
neurenix.federated.FedAdagrad(
    server_lr: float = 0.1,
    server_weight_decay: float = 0.0,
    server_epsilon: float = 1e-8
)
```

Implements FedOpt with Adagrad as the server optimizer.

```python
neurenix.federated.FedAdam(
    server_lr: float = 0.1,
    server_weight_decay: float = 0.0,
    server_beta1: float = 0.9,
    server_beta2: float = 0.999,
    server_epsilon: float = 1e-8
)
```

Implements FedOpt with Adam as the server optimizer.

```python
neurenix.federated.FedYogi(
    server_lr: float = 0.1,
    server_weight_decay: float = 0.0,
    server_beta1: float = 0.9,
    server_beta2: float = 0.999,
    server_epsilon: float = 1e-8
)
```

Implements FedOpt with Yogi as the server optimizer.

**Example:**
```python
import neurenix as nx
from neurenix.federated import FedProx, FedAdam

# Create a FedProx strategy
fedprox = FedProx(mu=0.01)

# Create a FedAdam strategy
fedadam = FedAdam(server_lr=0.01, server_beta1=0.9, server_beta2=0.999)

# Use the strategy in a federated server
server_with_fedprox = FederatedServer(model, fedprox, server_config)
server_with_fedadam = FederatedServer(model, fedadam, server_config)
```

### Security Components

```python
neurenix.federated.SecureAggregation(
    num_clients: int,
    threshold: int,
    key_size: int = 256
)
```

Implements secure aggregation using threshold homomorphic encryption.

**Parameters:**
- `num_clients`: Total number of clients
- `threshold`: Minimum number of clients required for decryption
- `key_size`: Size of encryption keys in bits

**Methods:**
- `setup() -> Dict[str, Any]`: Set up the secure aggregation protocol
- `encrypt_update(client_id: int, update: Dict[str, Tensor], keys: Dict[str, Any]) -> Dict[str, bytes]`: Encrypt a client update
- `aggregate_encrypted_updates(encrypted_updates: List[Dict[str, bytes]]) -> Dict[str, Tensor]`: Aggregate encrypted updates

```python
neurenix.federated.DifferentialPrivacy(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    mechanism: str = 'gaussian',
    clip_norm: float = 1.0
)
```

Implements differential privacy for federated learning.

**Parameters:**
- `epsilon`: Privacy parameter (smaller values provide stronger privacy)
- `delta`: Probability of privacy violation
- `mechanism`: Noise mechanism ('gaussian' or 'laplace')
- `clip_norm`: Maximum L2 norm for gradient clipping

**Methods:**
- `privatize_update(update: Dict[str, Tensor]) -> Dict[str, Tensor]`: Add noise to a model update
- `get_privacy_spent() -> Tuple[float, float]`: Get the current privacy budget spent

```python
neurenix.federated.HomomorphicEncryption(
    key_size: int = 2048,
    precision: int = 16
)
```

Implements homomorphic encryption for federated learning.

**Parameters:**
- `key_size`: Size of encryption keys in bits
- `precision`: Precision for fixed-point encoding

**Methods:**
- `generate_keys() -> Tuple[Any, Any]`: Generate encryption keys
- `encrypt(data: Tensor, public_key: Any) -> bytes`: Encrypt data
- `decrypt(encrypted_data: bytes, private_key: Any) -> Tensor`: Decrypt data
- `aggregate_encrypted(encrypted_updates: List[bytes]) -> bytes`: Aggregate encrypted updates

**Example:**
```python
import neurenix as nx
from neurenix.federated import DifferentialPrivacy

# Create a differential privacy mechanism
dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5, mechanism='gaussian', clip_norm=1.0)

# Privatize a model update
privatized_update = dp.privatize_update(model_update)

# Get the privacy budget spent
epsilon_spent, delta_spent = dp.get_privacy_spent()
print(f"Privacy budget spent: ε={epsilon_spent:.4f}, δ={delta_spent:.8f}")
```

### Utility Components

```python
neurenix.federated.ClientSelector
```

Base class for client selection strategies.

**Methods:**
- `select(available_clients: List[str], num_clients: int) -> List[str]`: Select clients for a round

```python
neurenix.federated.RandomClientSelector()
```

Selects clients uniformly at random.

```python
neurenix.federated.PowerOfChoiceSelector(num_choices: int = 2)
```

Implements the power-of-choice selection strategy, which selects the best client from a random subset.

**Parameters:**
- `num_choices`: Number of random choices to consider

```python
neurenix.federated.ModelCompressor(
    compression_rate: float = 0.1,
    method: str = 'threshold'
)
```

Compresses model updates to reduce communication costs.

**Parameters:**
- `compression_rate`: Fraction of parameters to keep
- `method`: Compression method ('threshold', 'topk', 'random')

**Methods:**
- `compress(update: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Any]]`: Compress a model update
- `decompress(compressed_update: Dict[str, Tensor], metadata: Dict[str, Any]) -> Dict[str, Tensor]`: Decompress a model update

```python
neurenix.federated.GradientCompressor(
    compression_rate: float = 0.1,
    method: str = 'threshold'
)
```

Compresses gradients to reduce communication costs.

**Parameters:**
- `compression_rate`: Fraction of gradients to keep
- `method`: Compression method ('threshold', 'topk', 'random')

**Methods:**
- `compress(gradients: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Any]]`: Compress gradients
- `decompress(compressed_gradients: Dict[str, Tensor], metadata: Dict[str, Any]) -> Dict[str, Tensor]`: Decompress gradients

**Example:**
```python
import neurenix as nx
from neurenix.federated import ModelCompressor

# Create a model compressor
compressor = ModelCompressor(compression_rate=0.1, method='topk')

# Compress a model update
compressed_update, metadata = compressor.compress(model_update)

# Decompress the model update
decompressed_update = compressor.decompress(compressed_update, metadata)

# Check compression ratio
original_size = sum(param.numel() for param in model_update.values())
compressed_size = sum(param.numel() for param in compressed_update.values())
compression_ratio = compressed_size / original_size
print(f"Compression ratio: {compression_ratio:.4f}")
```

## Framework Comparison

### Neurenix vs. TensorFlow Federated (TFF)

| Feature | Neurenix | TensorFlow Federated |
|---------|----------|----------------------|
| **API Design** | High-level and low-level APIs | Primarily low-level API with complex abstractions |
| **Aggregation Strategies** | Multiple strategies (FedAvg, FedProx, FedNova, FedOpt) | Primarily FedAvg with limited alternatives |
| **Privacy Mechanisms** | Comprehensive (Secure Aggregation, Differential Privacy, Homomorphic Encryption) | Basic Differential Privacy |
| **Communication Efficiency** | Multiple compression techniques | Limited compression options |
| **Client Selection** | Multiple strategies | Basic random selection |
| **Integration with Core Framework** | Seamless integration with Neurenix | Tightly coupled with TensorFlow |
| **Edge Device Support** | Optimized for edge devices | Limited edge device optimization |

Neurenix provides a more comprehensive federated learning solution compared to TensorFlow Federated, with a wider range of aggregation strategies, privacy mechanisms, and communication efficiency techniques. The API is also more intuitive and easier to use, with better integration with the core framework and optimization for edge devices.

### Neurenix vs. PySyft/PyTorch

| Feature | Neurenix | PySyft/PyTorch |
|---------|----------|----------------|
| **API Design** | Unified, consistent API | Complex, multi-layered API |
| **Aggregation Strategies** | Multiple strategies with unified interface | Limited strategies with inconsistent interfaces |
| **Privacy Mechanisms** | Comprehensive, integrated privacy mechanisms | Strong privacy focus but complex implementation |
| **Communication Efficiency** | Built-in compression techniques | Limited compression options |
| **Client Selection** | Multiple strategies | Basic selection strategies |
| **Integration with Core Framework** | Seamless integration with Neurenix | Requires additional setup with PyTorch |
| **Edge Device Support** | Optimized for edge devices | Limited edge device optimization |

While PySyft has a strong focus on privacy-preserving machine learning, Neurenix provides a more comprehensive and integrated federated learning solution with better usability and performance on edge devices. The unified API and seamless integration with the core framework make Neurenix easier to use for federated learning applications.

### Neurenix vs. Flower

| Feature | Neurenix | Flower |
|---------|----------|--------|
| **API Design** | Unified, framework-specific API | Framework-agnostic but more verbose API |
| **Aggregation Strategies** | Multiple strategies with unified interface | Basic strategies with extensible interface |
| **Privacy Mechanisms** | Comprehensive, integrated privacy mechanisms | Limited privacy mechanisms |
| **Communication Efficiency** | Built-in compression techniques | Basic compression options |
| **Client Selection** | Multiple strategies | Basic selection strategies |
| **Integration with Core Framework** | Seamless integration with Neurenix | Framework-agnostic but requires more boilerplate |
| **Edge Device Support** | Optimized for edge devices | Good edge device support |

Flower is a framework-agnostic federated learning library that works with multiple deep learning frameworks, while Neurenix provides a more integrated solution specifically for the Neurenix framework. Neurenix offers more comprehensive privacy mechanisms and aggregation strategies, with better optimization for edge devices and a more intuitive API.

## Best Practices

### Client Configuration

Properly configuring clients is crucial for effective federated learning:

```python
import neurenix as nx
from neurenix.federated import ClientConfig

# For resource-constrained devices
mobile_config = ClientConfig(
    batch_size=8,  # Smaller batch size for limited memory
    local_epochs=1,  # Fewer local epochs to reduce computation
    privacy_mechanism='differential_privacy',  # Enable privacy for sensitive data
    privacy_budget=0.5,  # Strict privacy budget
    compression_rate=0.1  # High compression to reduce communication
)

# For more powerful devices
server_config = ClientConfig(
    batch_size=32,  # Larger batch size for better utilization
    local_epochs=5,  # More local epochs for better convergence
    privacy_mechanism=None,  # No privacy mechanism for non-sensitive data
    compression_rate=0.5  # Moderate compression
)

# Adapt configuration based on device capabilities
def get_client_config(device_type):
    if device_type == 'mobile':
        return mobile_config
    elif device_type == 'server':
        return server_config
    else:
        return ClientConfig()  # Default configuration
```

### Handling Non-IID Data

Federated learning often deals with non-IID (non-independent and identically distributed) data across clients:

```python
import neurenix as nx
from neurenix.federated import FedProx, FedNova

# For highly heterogeneous data
if data_heterogeneity == 'high':
    # Use FedProx with a higher mu value
    strategy = FedProx(mu=0.1)
elif data_heterogeneity == 'medium':
    # Use FedProx with a moderate mu value
    strategy = FedProx(mu=0.01)
else:
    # Use FedNova for mild heterogeneity
    strategy = FedNova()

# Create a federated server with the appropriate strategy
server = FederatedServer(model, strategy, server_config)
```

### Privacy-Utility Trade-off

Balancing privacy and utility is important in federated learning:

```python
import neurenix as nx
from neurenix.federated import DifferentialPrivacy

# For highly sensitive data
if data_sensitivity == 'high':
    # Strong privacy guarantees
    dp = DifferentialPrivacy(epsilon=0.1, delta=1e-6, clip_norm=0.5)
elif data_sensitivity == 'medium':
    # Moderate privacy guarantees
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
else:
    # Basic privacy guarantees
    dp = DifferentialPrivacy(epsilon=10.0, delta=1e-4, clip_norm=5.0)

# Monitor privacy budget
for round_num in range(num_rounds):
    # Privatize updates
    privatized_updates = [dp.privatize_update(update) for update in client_updates]
    
    # Check privacy budget
    epsilon_spent, delta_spent = dp.get_privacy_spent()
    if epsilon_spent > max_epsilon:
        print(f"Privacy budget exceeded at round {round_num}")
        break
```

### Communication Efficiency

Optimizing communication is critical for federated learning:

```python
import neurenix as nx
from neurenix.federated import ModelCompressor, GradientCompressor

# Create compressors with different settings
model_compressor = ModelCompressor(compression_rate=0.1, method='topk')
gradient_compressor = GradientCompressor(compression_rate=0.05, method='threshold')

# Compress model updates
compressed_update, metadata = model_compressor.compress(model_update)

# Compress gradients
compressed_gradients, grad_metadata = gradient_compressor.compress(gradients)

# Adaptive compression based on network conditions
def get_compression_rate(network_bandwidth):
    if network_bandwidth < 1:  # < 1 Mbps
        return 0.01  # High compression
    elif network_bandwidth < 10:  # < 10 Mbps
        return 0.1  # Moderate compression
    else:
        return 0.5  # Low compression

# Update compressor based on network conditions
compression_rate = get_compression_rate(current_bandwidth)
model_compressor.compression_rate = compression_rate
```

## Tutorials

### Basic Federated Learning with FedAvg

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import SGD
from neurenix.federated import (
    FederatedClient, ClientConfig,
    FederatedServer, ServerConfig,
    FedAvg
)
import numpy as np

# Initialize Neurenix
nx.init()

# Create a model architecture
def create_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )

# Load MNIST dataset
mnist = nx.data.MNIST(root='./data', train=True, download=True)
test_mnist = nx.data.MNIST(root='./data', train=False, download=True)

# Split the dataset into 10 clients (simulating federated setting)
def split_data(dataset, num_clients):
    # For simplicity, we'll split the data equally
    # In a real scenario, the data would be naturally distributed
    data_per_client = len(dataset) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = start_idx + data_per_client
        indices = list(range(start_idx, end_idx))
        client_datasets.append(nx.data.Subset(dataset, indices))
    
    return client_datasets

num_clients = 10
client_datasets = split_data(mnist, num_clients)

# Create clients
clients = []
for i in range(num_clients):
    # Create a model for this client
    model = create_model()
    
    # Create an optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Create a loss function
    loss_fn = nx.nn.CrossEntropyLoss()
    
    # Create a client configuration
    client_config = ClientConfig(
        batch_size=32,
        local_epochs=5,
        device=nx.Device('cpu')
    )
    
    # Create a federated client
    client = FederatedClient(model, optimizer, loss_fn, client_datasets[i], client_config)
    clients.append(client)

# Create a global model
global_model = create_model()

# Create an aggregation strategy
strategy = FedAvg()

# Create a server configuration
server_config = ServerConfig(
    num_rounds=50,
    clients_per_round=5,
    evaluation_frequency=5
)

# Create a federated server
server = FederatedServer(global_model, strategy, server_config)

# Run federated learning
print("Starting federated learning...")
for round_num in range(server_config.num_rounds):
    print(f"Round {round_num+1}/{server_config.num_rounds}")
    
    # Select clients for this round
    available_clients = list(range(num_clients))
    selected_indices = server.select_clients(available_clients, server_config.clients_per_round)
    selected_clients = [clients[i] for i in selected_indices]
    
    # Update clients with the global model
    for client in selected_clients:
        client.update_model(global_model.state_dict())
    
    # Train clients locally
    client_states = []
    for client in selected_clients:
        client_state = client.train(num_epochs=client.config.local_epochs)
        client_states.append(client_state)
    
    # Collect model updates and sample counts
    updates = [(state.model_update, state.num_samples) for state in client_states]
    
    # Aggregate updates
    aggregated_update = server.aggregate_updates(updates)
    
    # Update global model
    server.update_global_model(aggregated_update)
    
    # Evaluate global model
    if round_num % server_config.evaluation_frequency == 0 or round_num == server_config.num_rounds - 1:
        # Create a test dataloader
        test_dataloader = nx.data.DataLoader(test_mnist, batch_size=100, shuffle=False)
        
        # Evaluate the model
        global_model.eval()
        correct = 0
        total = 0
        
        for inputs, targets in test_dataloader:
            outputs = global_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        print(f"Round {round_num+1}, Test Accuracy: {accuracy:.4f}")

print("Federated learning completed!")
```

### Privacy-Preserving Federated Learning

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import SGD
from neurenix.federated import (
    FederatedClient, ClientConfig,
    FederatedServer, ServerConfig,
    FedAvg, DifferentialPrivacy
)
import numpy as np

# Initialize Neurenix
nx.init()

# Create a model architecture
def create_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )

# Load MNIST dataset
mnist = nx.data.MNIST(root='./data', train=True, download=True)
test_mnist = nx.data.MNIST(root='./data', train=False, download=True)

# Split the dataset into 10 clients
num_clients = 10
client_datasets = []

# Create a non-IID split (each client gets mostly examples of certain digits)
for i in range(num_clients):
    # Each client gets examples of 2 digits predominantly
    primary_digits = [(i * 2) % 10, (i * 2 + 1) % 10]
    
    # Indices of examples with the primary digits (80% of client's data)
    primary_indices = [idx for idx, (_, label) in enumerate(mnist) if label in primary_digits]
    primary_indices = np.random.choice(primary_indices, size=1000, replace=False)
    
    # Indices of examples with other digits (20% of client's data)
    other_indices = [idx for idx, (_, label) in enumerate(mnist) if label not in primary_digits]
    other_indices = np.random.choice(other_indices, size=200, replace=False)
    
    # Combine indices
    indices = np.concatenate([primary_indices, other_indices])
    np.random.shuffle(indices)
    
    # Create a subset dataset
    client_datasets.append(nx.data.Subset(mnist, indices))

# Create a differential privacy mechanism
dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, mechanism='gaussian', clip_norm=1.0)

# Create clients with privacy
clients = []
for i in range(num_clients):
    # Create a model for this client
    model = create_model()
    
    # Create an optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Create a loss function
    loss_fn = nx.nn.CrossEntropyLoss()
    
    # Create a client configuration with privacy
    client_config = ClientConfig(
        batch_size=32,
        local_epochs=5,
        device=nx.Device('cpu'),
        privacy_mechanism='differential_privacy',
        privacy_budget=1.0
    )
    
    # Create a federated client
    client = FederatedClient(model, optimizer, loss_fn, client_datasets[i], client_config)
    clients.append(client)

# Create a global model
global_model = create_model()

# Create an aggregation strategy
strategy = FedAvg()

# Create a server configuration
server_config = ServerConfig(
    num_rounds=50,
    clients_per_round=5,
    evaluation_frequency=5,
    secure_aggregation=True
)

# Create a federated server
server = FederatedServer(global_model, strategy, server_config)

# Run federated learning with privacy
print("Starting privacy-preserving federated learning...")
for round_num in range(server_config.num_rounds):
    print(f"Round {round_num+1}/{server_config.num_rounds}")
    
    # Select clients for this round
    available_clients = list(range(num_clients))
    selected_indices = server.select_clients(available_clients, server_config.clients_per_round)
    selected_clients = [clients[i] for i in selected_indices]
    
    # Update clients with the global model
    for client in selected_clients:
        client.update_model(global_model.state_dict())
    
    # Train clients locally
    client_states = []
    for client in selected_clients:
        client_state = client.train(num_epochs=client.config.local_epochs)
        client_states.append(client_state)
    
    # Collect model updates and sample counts
    updates = []
    for state in client_states:
        # Apply differential privacy to the model update
        privatized_update = dp.privatize_update(state.model_update)
        updates.append((privatized_update, state.num_samples))
    
    # Aggregate updates
    aggregated_update = server.aggregate_updates(updates)
    
    # Update global model
    server.update_global_model(aggregated_update)
    
    # Check privacy budget
    epsilon_spent, delta_spent = dp.get_privacy_spent()
    print(f"Privacy budget spent: ε={epsilon_spent:.4f}, δ={delta_spent:.8f}")
    
    # Evaluate global model
    if round_num % server_config.evaluation_frequency == 0 or round_num == server_config.num_rounds - 1:
        # Create a test dataloader
        test_dataloader = nx.data.DataLoader(test_mnist, batch_size=100, shuffle=False)
        
        # Evaluate the model
        global_model.eval()
        correct = 0
        total = 0
        
        for inputs, targets in test_dataloader:
            outputs = global_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        print(f"Round {round_num+1}, Test Accuracy: {accuracy:.4f}")

print("Privacy-preserving federated learning completed!")
```

### Communication-Efficient Federated Learning

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import SGD
from neurenix.federated import (
    FederatedClient, ClientConfig,
    FederatedServer, ServerConfig,
    FedAvg, ModelCompressor
)
import numpy as np

# Initialize Neurenix
nx.init()

# Create a model architecture
def create_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )

# Load MNIST dataset
mnist = nx.data.MNIST(root='./data', train=True, download=True)
test_mnist = nx.data.MNIST(root='./data', train=False, download=True)

# Split the dataset into 10 clients
num_clients = 10
client_datasets = []

# Create a simple split
data_per_client = len(mnist) // num_clients
for i in range(num_clients):
    start_idx = i * data_per_client
    end_idx = start_idx + data_per_client
    indices = list(range(start_idx, end_idx))
    client_datasets.append(nx.data.Subset(mnist, indices))

# Create a model compressor
compressor = ModelCompressor(compression_rate=0.1, method='topk')

# Create clients with compression
clients = []
for i in range(num_clients):
    # Create a model for this client
    model = create_model()
    
    # Create an optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Create a loss function
    loss_fn = nx.nn.CrossEntropyLoss()
    
    # Create a client configuration with compression
    client_config = ClientConfig(
        batch_size=32,
        local_epochs=5,
        device=nx.Device('cpu'),
        compression_rate=0.1
    )
    
    # Create a federated client
    client = FederatedClient(model, optimizer, loss_fn, client_datasets[i], client_config)
    clients.append(client)

# Create a global model
global_model = create_model()

# Create an aggregation strategy
strategy = FedAvg()

# Create a server configuration
server_config = ServerConfig(
    num_rounds=50,
    clients_per_round=5,
    evaluation_frequency=5,
    compression_enabled=True
)

# Create a federated server
server = FederatedServer(global_model, strategy, server_config)

# Run federated learning with compression
print("Starting communication-efficient federated learning...")
for round_num in range(server_config.num_rounds):
    print(f"Round {round_num+1}/{server_config.num_rounds}")
    
    # Select clients for this round
    available_clients = list(range(num_clients))
    selected_indices = server.select_clients(available_clients, server_config.clients_per_round)
    selected_clients = [clients[i] for i in selected_indices]
    
    # Update clients with the global model
    for client in selected_clients:
        client.update_model(global_model.state_dict())
    
    # Train clients locally
    client_states = []
    for client in selected_clients:
        client_state = client.train(num_epochs=client.config.local_epochs)
        client_states.append(client_state)
    
    # Collect model updates and sample counts
    updates = []
    for state in client_states:
        # Compress the model update
        compressed_update, metadata = compressor.compress(state.model_update)
        
        # In a real system, only the compressed update would be sent
        # Here we decompress it immediately for simplicity
        decompressed_update = compressor.decompress(compressed_update, metadata)
        
        updates.append((decompressed_update, state.num_samples))
        
        # Calculate compression statistics
        original_size = sum(param.numel() for param in state.model_update.values())
        compressed_size = sum(param.numel() for param in compressed_update.values())
        compression_ratio = compressed_size / original_size
        print(f"Client compression ratio: {compression_ratio:.4f}")
    
    # Aggregate updates
    aggregated_update = server.aggregate_updates(updates)
    
    # Update global model
    server.update_global_model(aggregated_update)
    
    # Evaluate global model
    if round_num % server_config.evaluation_frequency == 0 or round_num == server_config.num_rounds - 1:
        # Create a test dataloader
        test_dataloader = nx.data.DataLoader(test_mnist, batch_size=100, shuffle=False)
        
        # Evaluate the model
        global_model.eval()
        correct = 0
        total = 0
        
        for inputs, targets in test_dataloader:
            outputs = global_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        print(f"Round {round_num+1}, Test Accuracy: {accuracy:.4f}")

print("Communication-efficient federated learning completed!")
```

## Conclusion

The Federated Learning module in Neurenix provides a comprehensive set of tools for implementing federated learning systems, enabling model training on decentralized data without sharing the raw data. With support for various aggregation strategies, privacy-preserving techniques, and communication efficiency optimizations, the module enables developers to build federated learning applications that are privacy-preserving, efficient, and effective.

Compared to other frameworks, Neurenix's Federated Learning module offers advantages in terms of API consistency, integration with the core framework, and optimization for edge devices. These features make Neurenix particularly well-suited for developing federated learning systems in privacy-sensitive domains such as healthcare, finance, mobile devices, and IoT.
