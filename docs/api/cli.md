# Command Line Interface (CLI)

## Overview

The Command Line Interface (CLI) module in Neurenix provides a comprehensive set of command-line tools for interacting with the framework. It enables users to create, train, evaluate, and deploy machine learning models without writing code, making the framework accessible to a wider audience including data scientists, researchers, and DevOps engineers.

The CLI is designed with a consistent interface and extensive documentation, allowing users to quickly learn and utilize the full capabilities of the Neurenix framework. It supports all major functionalities of the framework, from data preprocessing to model deployment, and integrates seamlessly with the Python API for advanced use cases.

Built on Neurenix's multi-language architecture, the CLI leverages the high-performance Rust and C++ backends while providing a user-friendly interface for common machine learning workflows.

## Key Concepts

### Command Structure

The Neurenix CLI follows a consistent command structure:

```
neurenix <command> [subcommand] [options]
```

Commands are organized into logical groups based on their functionality:

- **Project Management**: Commands for creating and managing Neurenix projects
- **Model Operations**: Commands for training, evaluating, and using models
- **Data Management**: Commands for preprocessing and managing datasets
- **Deployment**: Commands for deploying models to various environments
- **Utilities**: Helper commands for common tasks

### Configuration Management

The CLI uses a hierarchical configuration system:

- **Global Configuration**: System-wide settings stored in `~/.neurenix/config.yaml`
- **Project Configuration**: Project-specific settings stored in `neurenix.yaml`
- **Command-line Options**: Override configurations for specific commands
- **Environment Variables**: Can be used to set configurations in CI/CD environments

### Project Structure

The CLI enforces a standardized project structure:

```
project/
├── neurenix.yaml       # Project configuration
├── data/               # Data directory
│   ├── raw/            # Raw data
│   └── processed/      # Processed data
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── scripts/            # Custom scripts
└── outputs/            # Outputs (logs, visualizations, etc.)
```

This structure ensures consistency across projects and enables the CLI to automatically locate resources.

### Hardware Abstraction

The CLI provides hardware abstraction, allowing users to run the same commands on different hardware:

- Automatic device selection based on availability
- Explicit device specification through options
- Distributed execution across multiple devices
- Edge device deployment with optimized configurations

## API Reference

### Project Management Commands

#### `neurenix init`

Creates a new Neurenix project with the standard directory structure.

```bash
neurenix init [project_name] [--template <template_name>] [--dataset <dataset_name>]
```

Options:
- `--template`: Template to use (default, vision, nlp, rl, etc.)
- `--dataset`: Download and include a dataset
- `--git`: Initialize a git repository
- `--docker`: Include Dockerfile and docker-compose.yml

#### `neurenix save`

Saves the current project state, including model, configuration, and data.

```bash
neurenix save [--name <checkpoint_name>] [--include-data] [--compress]
```

Options:
- `--name`: Name of the checkpoint
- `--include-data`: Include dataset in the checkpoint
- `--compress`: Compress the checkpoint
- `--cloud <provider>`: Upload to cloud storage

#### `neurenix help`

Displays help information for commands.

```bash
neurenix help [command]
```

### Model Operations Commands

#### `neurenix run`

Trains a model with the provided configuration.

```bash
neurenix run [--config <config_file>] [--data <data_dir>] [--output <output_dir>]
```

Options:
- `--config`: Configuration file
- `--data`: Data directory
- `--output`: Output directory
- `--device`: Device to use (cpu, cuda, etc.)
- `--distributed`: Enable distributed training
- `--resume`: Resume from checkpoint

#### `neurenix predict`

Makes predictions using a trained model.

```bash
neurenix predict [--model <model_path>] [--input <input_path>] [--output <output_path>]
```

Options:
- `--model`: Path to the model
- `--input`: Input data
- `--output`: Output path
- `--batch-size`: Batch size
- `--device`: Device to use

#### `neurenix eval`

Evaluates a trained model with specific metrics.

```bash
neurenix eval [--model <model_path>] [--data <data_path>] [--metrics <metrics>]
```

Options:
- `--model`: Path to the model
- `--data`: Test data
- `--metrics`: Metrics to evaluate (comma-separated)
- `--output`: Output path for evaluation results
- `--device`: Device to use

#### `neurenix export`

Exports a trained model to a specific format.

```bash
neurenix export [--model <model_path>] [--format <format>] [--output <output_path>]
```

Options:
- `--model`: Path to the model
- `--format`: Export format (onnx, torchscript, etc.)
- `--output`: Output path
- `--optimize`: Apply optimizations (quantization, pruning, etc.)

#### `neurenix optimize`

Optimizes a model through hyperparameter tuning or techniques like quantization.

```bash
neurenix optimize [--model <model_path>] [--data <data_path>] [--strategy <strategy>]
```

Options:
- `--model`: Path to the model
- `--data`: Validation data
- `--strategy`: Optimization strategy (hyperparameter, quantization, pruning)
- `--trials`: Number of trials for hyperparameter optimization
- `--output`: Output path for optimized model

### Data Management Commands

#### `neurenix preprocess`

Performs data preprocessing according to the configuration.

```bash
neurenix preprocess [--config <config_file>] [--input <input_dir>] [--output <output_dir>]
```

Options:
- `--config`: Preprocessing configuration
- `--input`: Input data directory
- `--output`: Output directory
- `--steps`: Preprocessing steps to run (comma-separated)
- `--workers`: Number of worker processes

#### `neurenix dataset`

Manages datasets for Neurenix projects.

```bash
neurenix dataset [list|download|info|split|convert] [options]
```

Subcommands:
- `list`: Lists available datasets
- `download`: Downloads a dataset
- `info`: Displays information about a dataset
- `split`: Splits a dataset into train/val/test
- `convert`: Converts a dataset to a different format

### Deployment Commands

#### `neurenix serve`

Serves a trained model as a RESTful API.

```bash
neurenix serve [--model <model_path>] [--host <host>] [--port <port>]
```

Options:
- `--model`: Path to the model
- `--host`: Host to bind to
- `--port`: Port to listen on
- `--workers`: Number of worker processes
- `--device`: Device to use
- `--auth`: Enable authentication

#### `neurenix monitor`

Monitors model training in real-time.

```bash
neurenix monitor [--run <run_id>] [--metrics <metrics>] [--refresh <interval>]
```

Options:
- `--run`: Run ID to monitor
- `--metrics`: Metrics to display (comma-separated)
- `--refresh`: Refresh interval in seconds
- `--output`: Output file for metrics

### Hardware Management Commands

#### `neurenix hardware`

Manages hardware configurations and optimizations.

```bash
neurenix hardware [list|info|auto|benchmark] [options]
```

Subcommands:
- `list`: Lists available hardware
- `info`: Displays information about hardware
- `auto`: Enables automatic hardware selection
- `benchmark`: Benchmarks hardware performance

## Framework Comparison

### Neurenix CLI vs. TensorFlow CLI

| Feature | Neurenix CLI | TensorFlow CLI |
|---------|--------------|----------------|
| Project Management | Comprehensive project creation and management | Limited project management |
| Model Export | Multiple export formats with optimizations | Primarily focused on SavedModel and TFLite |
| Hardware Abstraction | Unified interface for CPU, GPU, TPU, and edge devices | Primarily focused on CPU, GPU, and TPU |
| Deployment Options | Built-in serving, monitoring, and deployment tools | Requires additional tools like TF Serving |
| Edge Device Support | Native support for edge devices with optimizations | Limited to TFLite with separate workflow |
| Multi-language Support | Seamless integration with Rust/C++/Python components | Primarily Python-focused |

Neurenix CLI provides a more comprehensive set of tools for the entire machine learning lifecycle, with better support for edge devices and a more consistent interface across different hardware platforms.

### Neurenix CLI vs. PyTorch CLI

| Feature | Neurenix CLI | PyTorch CLI |
|---------|--------------|-------------|
| Command Structure | Unified command structure with consistent options | Limited CLI capabilities, primarily through torchrun |
| Project Templates | Multiple project templates for different domains | No built-in project templates |
| Distributed Training | Simple commands for distributed training | Requires complex torchrun commands |
| Model Optimization | Built-in commands for quantization, pruning, etc. | Requires separate scripts |
| Deployment | Integrated deployment workflows | Requires additional tools |
| Hardware Support | Comprehensive hardware support | Primarily focused on CUDA devices |

While PyTorch offers flexibility through its Python API, it lacks a comprehensive CLI. Neurenix CLI provides a more user-friendly interface for common machine learning workflows, making it accessible to users without extensive programming experience.

### Neurenix CLI vs. Scikit-Learn CLI

| Feature | Neurenix CLI | Scikit-Learn CLI |
|---------|--------------|------------------|
| Command Coverage | Comprehensive set of commands for all ML tasks | Limited CLI capabilities |
| Hardware Acceleration | Native support for GPUs and specialized hardware | Primarily CPU-focused |
| Distributed Computing | Built-in commands for distributed training | Limited distributed capabilities |
| Model Deployment | Integrated deployment workflows | Requires additional tools |
| Edge Device Support | Native support for edge devices | Limited edge support |
| Scalability | Scales from laptops to clusters | Primarily designed for single-node usage |

Scikit-Learn has limited CLI capabilities, focusing primarily on its Python API. Neurenix CLI offers a more comprehensive set of tools for training, evaluating, and deploying models, with better support for hardware acceleration and distributed computing.

## Best Practices

### Efficient CLI Usage

1. **Use Project Templates**: Start with a template that matches your use case.

```bash
neurenix init my-vision-project --template vision --dataset cifar10
```

2. **Leverage Configuration Files**: Store common settings in configuration files.

```yaml
# neurenix.yaml
model:
  type: resnet50
  pretrained: true
training:
  batch_size: 64
  epochs: 100
  optimizer: adam
  learning_rate: 0.001
data:
  train: data/train
  val: data/val
  test: data/test
  augmentation: true
```

3. **Create Command Aliases**: Define aliases for frequently used commands.

```bash
# Add to ~/.bashrc or ~/.zshrc
alias nx-train="neurenix run --config config/train.yaml"
alias nx-eval="neurenix eval --metrics accuracy,f1,precision,recall"
alias nx-deploy="neurenix serve --workers 4 --device cuda"
```

4. **Use Environment Variables for CI/CD**: Configure the CLI using environment variables in CI/CD pipelines.

```bash
export NEURENIX_DEVICE=cuda
export NEURENIX_LOG_LEVEL=info
export NEURENIX_OUTPUT_DIR=/path/to/outputs
neurenix run --config config.yaml
```

5. **Implement Hooks for Custom Logic**: Use hooks to extend the CLI with custom logic.

```bash
# hooks/pre_train.sh
#!/bin/bash
echo "Preparing for training..."
# Custom logic here

# hooks/post_train.sh
#!/bin/bash
echo "Training completed!"
# Custom logic here
```

### Hardware Optimization

1. **Automatic Hardware Selection**: Let Neurenix choose the optimal hardware.

```bash
neurenix hardware auto
neurenix run --config config.yaml
```

2. **Benchmark Hardware**: Identify the best hardware for your workload.

```bash
neurenix hardware benchmark --model resnet50 --batch-sizes 16,32,64,128
```

3. **Distributed Training**: Utilize multiple GPUs for faster training.

```bash
neurenix run --distributed --nodes 2 --gpus-per-node 4 --config config.yaml
```

4. **Edge Device Optimization**: Optimize models for edge deployment.

```bash
neurenix export --model models/my_model.nrx --format edge --target raspberry-pi --optimize
```

## Tutorials

### Creating and Training a Computer Vision Model

```bash
# Create a new project
neurenix init vision-project --template vision

# Download and prepare a dataset
cd vision-project
neurenix dataset download cifar10 --output data/raw
neurenix preprocess --config configs/preprocess.yaml --input data/raw --output data/processed

# Create a training configuration
cat > configs/train.yaml << EOF
model:
  type: resnet18
  pretrained: true
  num_classes: 10
training:
  batch_size: 64
  epochs: 50
  optimizer: adam
  learning_rate: 0.001
  scheduler: cosine
  early_stopping: true
  patience: 5
data:
  train: data/processed/train
  val: data/processed/val
  augmentation: true
EOF

# Train the model
neurenix run --config configs/train.yaml --output models/resnet18_cifar10

# Evaluate the model
neurenix eval --model models/resnet18_cifar10 --data data/processed/test --metrics accuracy,f1,precision,recall

# Export the model for deployment
neurenix export --model models/resnet18_cifar10 --format onnx --output models/resnet18_cifar10.onnx --optimize

# Serve the model
neurenix serve --model models/resnet18_cifar10.onnx --host 0.0.0.0 --port 8000
```

### Hyperparameter Optimization

```bash
# Create a hyperparameter optimization configuration
cat > configs/hpo.yaml << EOF
model:
  type: mlp
  hidden_layers: [128, 64]
training:
  batch_size: 32
  epochs: 20
  optimizer: adam
data:
  train: data/processed/train
  val: data/processed/val
optimization:
  strategy: bayesian
  metric: val_accuracy
  direction: maximize
  trials: 50
  parameters:
    learning_rate:
      type: float
      min: 0.0001
      max: 0.1
      log: true
    dropout:
      type: float
      min: 0.1
      max: 0.5
    hidden_layers:
      type: categorical
      values: [[64], [128], [256], [128, 64], [256, 128], [512, 256, 128]]
    optimizer:
      type: categorical
      values: [adam, sgd, rmsprop]
EOF

# Run hyperparameter optimization
neurenix optimize --config configs/hpo.yaml --output models/optimized_mlp

# Train with the best parameters
neurenix run --config models/optimized_mlp/best_params.yaml --output models/best_mlp
```

### Deploying a Model to Edge Devices

```bash
# Create an edge deployment configuration
cat > configs/edge.yaml << EOF
model:
  path: models/my_model.nrx
optimization:
  quantization: int8
  pruning: true
  pruning_ratio: 0.5
  distillation: false
target:
  device: raspberry-pi-4
  accelerator: coral-edge-tpu
packaging:
  format: docker
  include_runtime: true
  include_examples: true
EOF

# Export the model for edge deployment
neurenix export --config configs/edge.yaml --output deployments/edge_model

# Test the edge deployment locally
cd deployments/edge_model
./test_locally.sh

# Deploy to the edge device
./deploy.sh --target 192.168.1.100 --username pi --key ~/.ssh/id_rsa
```

## Conclusion

The Command Line Interface (CLI) module of Neurenix provides a comprehensive set of tools for interacting with the framework without writing code. Its consistent interface, extensive documentation, and support for the entire machine learning lifecycle make it accessible to a wide range of users, from data scientists to DevOps engineers.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix CLI offers advantages in terms of command coverage, hardware abstraction, and deployment options. These features make it particularly well-suited for users who prefer command-line tools or need to integrate machine learning workflows into scripts and CI/CD pipelines.

By following the best practices and tutorials outlined in this documentation, users can leverage the full power of the Neurenix framework through its CLI, streamlining their machine learning workflows and improving productivity.
