# Docker Integration Documentation

## Overview

The Docker Integration module in Neurenix provides tools and utilities for containerizing AI applications and managing Docker containers. This module enables users to package their machine learning models and applications in containers for consistent deployment across different environments, from development to production.

Neurenix's Docker integration is implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance while providing the convenience of containerization.

## Key Concepts

### Container Management

Container management involves creating, starting, stopping, and removing Docker containers. The Docker Integration module provides a simple API for these operations, allowing users to manage containers programmatically from within their Neurenix applications.

### Image Building and Management

Docker images are the blueprints for containers. The Docker Integration module provides tools for building, pulling, pushing, and managing Docker images, enabling users to create reproducible environments for their AI applications.

### Volume Management

Docker volumes provide persistent storage for containers. The Docker Integration module includes utilities for creating, mounting, and managing volumes, allowing users to persist data across container restarts and share data between containers.

### Network Configuration

Network configuration is essential for containerized applications, especially in distributed settings. The Docker Integration module provides tools for configuring container networks, enabling communication between containers and with external services.

### Registry Integration

Docker registries store and distribute Docker images. The Docker Integration module includes utilities for interacting with Docker registries, allowing users to publish and retrieve images from public or private registries.

## API Reference

### Container Management

```python
neurenix.docker.Container(
    image,
    name=None,
    command=None,
    environment=None,
    volumes=None,
    ports=None,
    network=None,
    gpu=False
)
```

Create and manage a Docker container.

**Parameters:**
- `image`: Docker image to use
- `name`: Container name (optional)
- `command`: Command to run in the container (optional)
- `environment`: Environment variables as a dictionary (optional)
- `volumes`: Volume mappings as a dictionary (optional)
- `ports`: Port mappings as a dictionary (optional)
- `network`: Network configuration (optional)
- `gpu`: Whether to use GPU acceleration (default: False)

**Methods:**
- `start()`: Start the container
- `stop()`: Stop the container
- `restart()`: Restart the container
- `remove()`: Remove the container
- `exec(command)`: Execute a command in the container
- `logs()`: Get container logs
- `status()`: Get container status
- `is_running()`: Check if the container is running

### Image Management

```python
neurenix.docker.Image(
    name,
    tag="latest",
    dockerfile=None,
    context=None
)
```

Create and manage a Docker image.

**Parameters:**
- `name`: Image name
- `tag`: Image tag (default: "latest")
- `dockerfile`: Path to Dockerfile (optional)
- `context`: Path to build context (optional)

**Methods:**
- `build()`: Build the image
- `pull()`: Pull the image from a registry
- `push()`: Push the image to a registry
- `tag(new_tag)`: Tag the image
- `remove()`: Remove the image
- `exists()`: Check if the image exists locally
- `get_info()`: Get image information

### Volume Management

```python
neurenix.docker.Volume(
    name,
    driver="local",
    labels=None
)
```

Create and manage a Docker volume.

**Parameters:**
- `name`: Volume name
- `driver`: Volume driver (default: "local")
- `labels`: Volume labels as a dictionary (optional)

**Methods:**
- `create()`: Create the volume
- `remove()`: Remove the volume
- `exists()`: Check if the volume exists
- `get_info()`: Get volume information

### Network Management

```python
neurenix.docker.Network(
    name,
    driver="bridge",
    subnet=None,
    gateway=None,
    labels=None
)
```

Create and manage a Docker network.

**Parameters:**
- `name`: Network name
- `driver`: Network driver (default: "bridge")
- `subnet`: Network subnet (optional)
- `gateway`: Network gateway (optional)
- `labels`: Network labels as a dictionary (optional)

**Methods:**
- `create()`: Create the network
- `remove()`: Remove the network
- `connect(container)`: Connect a container to the network
- `disconnect(container)`: Disconnect a container from the network
- `exists()`: Check if the network exists
- `get_info()`: Get network information

### Registry Management

```python
neurenix.docker.Registry(
    url,
    username=None,
    password=None,
    token=None
)
```

Interact with a Docker registry.

**Parameters:**
- `url`: Registry URL
- `username`: Registry username (optional)
- `password`: Registry password (optional)
- `token`: Registry token (optional)

**Methods:**
- `login()`: Log in to the registry
- `logout()`: Log out from the registry
- `list_repositories()`: List repositories in the registry
- `list_tags(repository)`: List tags for a repository
- `search(query)`: Search for images in the registry

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Docker Integration** | Native Docker integration with comprehensive API | TensorFlow Serving for containerized models |
| **Container Management** | Built-in container management | Requires external tools or scripts |
| **GPU Support** | Native GPU support for containerized models | Requires specific TensorFlow Docker images |
| **Edge Device Containers** | Optimized containers for edge devices | TensorFlow Lite containers for edge devices |
| **Multi-Container Orchestration** | Built-in support for multi-container setups | Requires external orchestration tools |
| **Registry Integration** | Native registry integration | Requires external tools or scripts |

Neurenix's Docker integration offers a more comprehensive solution compared to TensorFlow, with native support for container management, GPU acceleration, and registry integration. While TensorFlow provides Docker images for deployment, it lacks a programmatic API for container management, requiring users to rely on external tools or scripts. Neurenix's integrated approach simplifies the containerization and deployment of AI applications, particularly for edge devices and multi-container setups.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Docker Integration** | Native Docker integration with comprehensive API | Official Docker images without programmatic API |
| **Container Management** | Built-in container management | Requires external tools or scripts |
| **GPU Support** | Native GPU support for containerized models | Requires specific PyTorch Docker images |
| **Edge Device Containers** | Optimized containers for edge devices | PyTorch Mobile containers for edge devices |
| **Multi-Container Orchestration** | Built-in support for multi-container setups | Requires external orchestration tools |
| **Registry Integration** | Native registry integration | Requires external tools or scripts |

Neurenix and PyTorch both provide Docker support, but Neurenix offers a more integrated solution with a programmatic API for container management. PyTorch provides official Docker images but lacks built-in tools for container orchestration and registry integration. Neurenix's Docker integration simplifies the deployment of AI applications, particularly for edge devices and distributed systems, by providing a consistent API for container management across different environments.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Docker Integration** | Native Docker integration with comprehensive API | No official Docker integration |
| **Container Management** | Built-in container management | Requires external tools or scripts |
| **GPU Support** | Native GPU support for containerized models | No native GPU support |
| **Edge Device Containers** | Optimized containers for edge devices | No specific edge device support |
| **Multi-Container Orchestration** | Built-in support for multi-container setups | Requires external orchestration tools |
| **Registry Integration** | Native registry integration | Requires external tools or scripts |

Neurenix provides much more comprehensive Docker integration compared to Scikit-Learn, which has no official Docker support. While Scikit-Learn models can be containerized using external tools, Neurenix's native Docker integration simplifies the process with a programmatic API for container management, GPU acceleration, and registry integration. This makes Neurenix particularly well-suited for deploying AI applications in production environments, where containerization and orchestration are essential.

## Best Practices

### Optimizing Docker Images

When building Docker images for Neurenix applications, follow these best practices:

1. **Use Multi-Stage Builds**: Reduce image size by using multi-stage builds to include only necessary components
2. **Minimize Layer Count**: Combine related commands to reduce the number of layers
3. **Use Specific Base Images**: Choose appropriate base images for different deployment scenarios
4. **Include Only Required Dependencies**: Avoid installing unnecessary packages
5. **Set Up Proper Environment Variables**: Configure environment variables for optimal performance

```dockerfile
# Example multi-stage build for a Neurenix application
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV NEURENIX_DEVICE=cpu

CMD ["python", "app.py"]
```

### GPU Acceleration in Containers

To optimize GPU acceleration in containerized Neurenix applications:

1. **Use NVIDIA Container Toolkit**: Ensure the host has NVIDIA Container Toolkit installed
2. **Specify GPU Requirements**: Configure container to use specific GPUs or memory limits
3. **Use Appropriate CUDA Version**: Match CUDA version with host driver compatibility
4. **Monitor GPU Usage**: Implement monitoring for GPU utilization and memory usage

```python
import neurenix
from neurenix.docker import Container

# Create a GPU-accelerated container
container = Container(
    image="neurenix/gpu:latest",
    name="neurenix-gpu",
    gpu=True,
    environment={
        "NEURENIX_DEVICE": "cuda",
        "CUDA_VISIBLE_DEVICES": "0,1"  # Use first two GPUs
    }
)

# Start the container
container.start()
```

### Container Orchestration

For orchestrating multiple containers in a Neurenix application:

1. **Define Container Dependencies**: Specify dependencies between containers
2. **Use Shared Volumes**: Share data between containers using volumes
3. **Configure Networking**: Set up appropriate network configurations
4. **Implement Health Checks**: Add health checks to monitor container status
5. **Use Resource Limits**: Set CPU and memory limits for containers

```python
import neurenix
from neurenix.docker import Container, Volume, Network

# Create a shared volume
volume = Volume("model-data")
volume.create()

# Create a network
network = Network("neurenix-net")
network.create()

# Create model server container
model_server = Container(
    image="neurenix/model-server:latest",
    name="model-server",
    volumes={volume.name: "/data"},
    network=network.name,
    environment={"MODEL_PATH": "/data/model.nrx"}
)

# Create API container
api_server = Container(
    image="neurenix/api-server:latest",
    name="api-server",
    volumes={volume.name: "/data"},
    network=network.name,
    ports={"8000": 8000},
    environment={"MODEL_SERVER": "model-server:5000"}
)

# Start containers
model_server.start()
api_server.start()
```

### Edge Device Deployment

When deploying containerized Neurenix applications to edge devices:

1. **Optimize Image Size**: Create minimal images for resource-constrained devices
2. **Use Hardware-Specific Optimizations**: Leverage device-specific acceleration
3. **Implement Offline Operation**: Design for intermittent connectivity
4. **Monitor Resource Usage**: Track CPU, memory, and storage usage
5. **Implement Update Strategies**: Plan for container updates in the field

```python
import neurenix
from neurenix.docker import Container, Image

# Build optimized edge device image
edge_image = Image(
    name="neurenix/edge",
    tag="arm64",
    dockerfile="Dockerfile.edge",
    context="."
)
edge_image.build()

# Create edge device container
edge_container = Container(
    image=edge_image.name + ":" + edge_image.tag,
    name="neurenix-edge",
    volumes={"/data": "/data"},
    environment={
        "NEURENIX_DEVICE": "cpu",
        "NEURENIX_OPTIMIZE_FOR_EDGE": "true"
    }
)

# Start container
edge_container.start()
```

## Tutorials

### Creating a Containerized Neurenix Application

```python
import neurenix
from neurenix.docker import Container, Image, Volume
import os

# Define application directory
app_dir = os.path.abspath("./my_neurenix_app")

# Create a Dockerfile
dockerfile = """
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
"""

with open(os.path.join(app_dir, "Dockerfile"), "w") as f:
    f.write(dockerfile)

# Build the image
image = Image(
    name="my-neurenix-app",
    tag="latest",
    dockerfile=os.path.join(app_dir, "Dockerfile"),
    context=app_dir
)
image.build()

# Create a volume for model data
volume = Volume("model-data")
volume.create()

# Create and start the container
container = Container(
    image=image.name + ":" + image.tag,
    name="my-neurenix-app",
    volumes={volume.name: "/app/data"},
    ports={"8000": 8000},
    environment={
        "NEURENIX_DEVICE": "cpu",
        "MODEL_PATH": "/app/data/model.nrx"
    }
)
container.start()

# Check container status
print(f"Container status: {container.status()}")
print(f"Container logs: {container.logs()}")
```

### Deploying a GPU-Accelerated Model Server

```python
import neurenix
from neurenix.docker import Container, Image, Volume
import os

# Create a volume for model data
volume = Volume("model-data")
volume.create()

# Create and start the container
container = Container(
    image="neurenix/model-server:latest",
    name="neurenix-model-server",
    volumes={volume.name: "/models"},
    ports={"5000": 5000},
    environment={
        "NEURENIX_DEVICE": "cuda",
        "MODEL_PATH": "/models/model.nrx",
        "BATCH_SIZE": "32",
        "NUM_WORKERS": "4"
    },
    gpu=True
)
container.start()

# Load a model
model = neurenix.load_model("my_model.nrx")

# Save the model to the volume
model.save(os.path.join("/var/lib/docker/volumes", volume.name, "_data", "model.nrx"))

# Check container status
print(f"Container status: {container.status()}")
print(f"Container logs: {container.logs()}")

# Test the model server
import requests
import numpy as np

# Create test data
test_data = np.random.randn(1, 10).tolist()

# Send request to model server
response = requests.post(
    "http://localhost:5000/predict",
    json={"data": test_data}
)

# Print response
print(f"Prediction: {response.json()}")
```

### Creating a Multi-Container Application with Docker Compose

```python
import neurenix
from neurenix.docker import Container, Volume, Network
import os

# Create a docker-compose.yml file
docker_compose = """
version: '3'

services:
  model-server:
    image: neurenix/model-server:latest
    volumes:
      - model-data:/models
    environment:
      - NEURENIX_DEVICE=cuda
      - MODEL_PATH=/models/model.nrx
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api-server:
    image: neurenix/api-server:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_SERVER=model-server:5000
    depends_on:
      - model-server

volumes:
  model-data:
"""

with open("docker-compose.yml", "w") as f:
    f.write(docker_compose)

# Create a volume for model data
volume = Volume("model-data")
volume.create()

# Load a model
model = neurenix.load_model("my_model.nrx")

# Save the model to the volume
model.save(os.path.join("/var/lib/docker/volumes", volume.name, "_data", "model.nrx"))

# Start the application using docker-compose
import subprocess
subprocess.run(["docker-compose", "up", "-d"])

# Test the API server
import requests
import numpy as np

# Create test data
test_data = np.random.randn(1, 10).tolist()

# Send request to API server
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": test_data}
)

# Print response
print(f"Prediction: {response.json()}")
```

## Conclusion

The Docker Integration module of Neurenix provides a comprehensive set of tools for containerizing AI applications and managing Docker containers. Its intuitive API makes it easy for researchers and developers to package and deploy their machine learning models in containers, ensuring consistency across different environments.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Docker Integration module offers advantages in terms of API design, container management, and edge device optimization. These features make Neurenix particularly well-suited for deploying AI applications in production environments, where containerization and orchestration are essential.
