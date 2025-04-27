# Kubernetes Integration Documentation

## Overview

The Kubernetes Integration module in Neurenix provides tools and utilities for deploying and managing AI applications on Kubernetes clusters. This module enables users to orchestrate containerized machine learning workloads at scale, from development to production, with features for automated deployment, scaling, and management of AI applications.

Neurenix's Kubernetes integration is implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance while providing the convenience of Kubernetes orchestration.

## Key Concepts

### Deployment Management

Deployment management involves creating, updating, and deleting Kubernetes deployments. The Kubernetes Integration module provides a simple API for these operations, allowing users to manage deployments programmatically from within their Neurenix applications.

### Service Configuration

Kubernetes services expose applications to the network. The Kubernetes Integration module includes utilities for creating and configuring services, enabling communication between components of distributed AI applications and with external clients.

### Pod Management

Pods are the smallest deployable units in Kubernetes. The Kubernetes Integration module provides tools for managing pods, including creation, monitoring, and deletion, allowing fine-grained control over the execution environment of AI workloads.

### ConfigMap and Secret Management

ConfigMaps and Secrets store configuration data and sensitive information. The Kubernetes Integration module includes utilities for creating and managing these resources, enabling secure and flexible configuration of AI applications.

### Job Scheduling

Jobs and CronJobs in Kubernetes run tasks to completion. The Kubernetes Integration module provides tools for scheduling and managing jobs, enabling batch processing and periodic tasks for AI workloads.

### Cluster Management

Cluster management involves interacting with the Kubernetes cluster itself. The Kubernetes Integration module includes utilities for cluster operations, enabling users to manage resources and monitor cluster health.

## API Reference

### Deployment Management

```python
neurenix.kubernetes.Deployment(
    name,
    image,
    namespace="default",
    replicas=1,
    labels=None,
    annotations=None,
    resources=None,
    env=None,
    volumes=None,
    ports=None,
    command=None,
    args=None
)
```

Create and manage a Kubernetes deployment.

**Parameters:**
- `name`: Deployment name
- `image`: Container image
- `namespace`: Kubernetes namespace (default: "default")
- `replicas`: Number of replicas (default: 1)
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)
- `resources`: Resource requirements as a dictionary (optional)
- `env`: Environment variables as a dictionary (optional)
- `volumes`: Volume mappings as a dictionary (optional)
- `ports`: Container ports as a list (optional)
- `command`: Container command as a list (optional)
- `args`: Container arguments as a list (optional)

**Methods:**
- `create()`: Create the deployment
- `update()`: Update the deployment
- `delete()`: Delete the deployment
- `scale(replicas)`: Scale the deployment
- `status()`: Get deployment status
- `is_ready()`: Check if the deployment is ready
- `logs()`: Get deployment logs
- `restart()`: Restart the deployment

### Service Management

```python
neurenix.kubernetes.Service(
    name,
    selector,
    namespace="default",
    ports=None,
    type="ClusterIP",
    labels=None,
    annotations=None
)
```

Create and manage a Kubernetes service.

**Parameters:**
- `name`: Service name
- `selector`: Pod selector as a dictionary
- `namespace`: Kubernetes namespace (default: "default")
- `ports`: Service ports as a list of dictionaries (optional)
- `type`: Service type (default: "ClusterIP")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)

**Methods:**
- `create()`: Create the service
- `update()`: Update the service
- `delete()`: Delete the service
- `status()`: Get service status
- `get_endpoints()`: Get service endpoints
- `get_external_ip()`: Get external IP (for LoadBalancer type)

### Pod Management

```python
neurenix.kubernetes.Pod(
    name,
    image,
    namespace="default",
    labels=None,
    annotations=None,
    resources=None,
    env=None,
    volumes=None,
    ports=None,
    command=None,
    args=None
)
```

Create and manage a Kubernetes pod.

**Parameters:**
- `name`: Pod name
- `image`: Container image
- `namespace`: Kubernetes namespace (default: "default")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)
- `resources`: Resource requirements as a dictionary (optional)
- `env`: Environment variables as a dictionary (optional)
- `volumes`: Volume mappings as a dictionary (optional)
- `ports`: Container ports as a list (optional)
- `command`: Container command as a list (optional)
- `args`: Container arguments as a list (optional)

**Methods:**
- `create()`: Create the pod
- `delete()`: Delete the pod
- `status()`: Get pod status
- `is_running()`: Check if the pod is running
- `logs()`: Get pod logs
- `exec(command)`: Execute a command in the pod
- `port_forward(local_port, remote_port)`: Forward a local port to a pod port

### ConfigMap Management

```python
neurenix.kubernetes.ConfigMap(
    name,
    data,
    namespace="default",
    labels=None,
    annotations=None
)
```

Create and manage a Kubernetes ConfigMap.

**Parameters:**
- `name`: ConfigMap name
- `data`: ConfigMap data as a dictionary
- `namespace`: Kubernetes namespace (default: "default")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)

**Methods:**
- `create()`: Create the ConfigMap
- `update()`: Update the ConfigMap
- `delete()`: Delete the ConfigMap
- `get_data()`: Get ConfigMap data

### Secret Management

```python
neurenix.kubernetes.Secret(
    name,
    data,
    namespace="default",
    type="Opaque",
    labels=None,
    annotations=None
)
```

Create and manage a Kubernetes Secret.

**Parameters:**
- `name`: Secret name
- `data`: Secret data as a dictionary
- `namespace`: Kubernetes namespace (default: "default")
- `type`: Secret type (default: "Opaque")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)

**Methods:**
- `create()`: Create the Secret
- `update()`: Update the Secret
- `delete()`: Delete the Secret
- `get_data()`: Get Secret data (decoded)

### Job Management

```python
neurenix.kubernetes.Job(
    name,
    image,
    namespace="default",
    labels=None,
    annotations=None,
    resources=None,
    env=None,
    volumes=None,
    command=None,
    args=None,
    backoff_limit=6,
    completions=1,
    parallelism=1
)
```

Create and manage a Kubernetes Job.

**Parameters:**
- `name`: Job name
- `image`: Container image
- `namespace`: Kubernetes namespace (default: "default")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)
- `resources`: Resource requirements as a dictionary (optional)
- `env`: Environment variables as a dictionary (optional)
- `volumes`: Volume mappings as a dictionary (optional)
- `command`: Container command as a list (optional)
- `args`: Container arguments as a list (optional)
- `backoff_limit`: Number of retries before considering a Job as failed (default: 6)
- `completions`: Number of successful pod completions required (default: 1)
- `parallelism`: Number of pods that can run in parallel (default: 1)

**Methods:**
- `create()`: Create the Job
- `delete()`: Delete the Job
- `status()`: Get Job status
- `is_complete()`: Check if the Job is complete
- `logs()`: Get Job logs
- `wait_for_completion(timeout=None)`: Wait for Job completion

### CronJob Management

```python
neurenix.kubernetes.CronJob(
    name,
    image,
    schedule,
    namespace="default",
    labels=None,
    annotations=None,
    resources=None,
    env=None,
    volumes=None,
    command=None,
    args=None,
    concurrency_policy="Allow",
    successful_jobs_history_limit=3,
    failed_jobs_history_limit=1
)
```

Create and manage a Kubernetes CronJob.

**Parameters:**
- `name`: CronJob name
- `image`: Container image
- `schedule`: Cron schedule expression
- `namespace`: Kubernetes namespace (default: "default")
- `labels`: Labels as a dictionary (optional)
- `annotations`: Annotations as a dictionary (optional)
- `resources`: Resource requirements as a dictionary (optional)
- `env`: Environment variables as a dictionary (optional)
- `volumes`: Volume mappings as a dictionary (optional)
- `command`: Container command as a list (optional)
- `args`: Container arguments as a list (optional)
- `concurrency_policy`: Concurrency policy (default: "Allow")
- `successful_jobs_history_limit`: Number of successful jobs to keep (default: 3)
- `failed_jobs_history_limit`: Number of failed jobs to keep (default: 1)

**Methods:**
- `create()`: Create the CronJob
- `update()`: Update the CronJob
- `delete()`: Delete the CronJob
- `status()`: Get CronJob status
- `suspend()`: Suspend the CronJob
- `resume()`: Resume the CronJob
- `trigger()`: Trigger a job manually

### Cluster Management

```python
neurenix.kubernetes.Cluster(
    context=None,
    config_file=None
)
```

Interact with a Kubernetes cluster.

**Parameters:**
- `context`: Kubernetes context (optional)
- `config_file`: Path to kubeconfig file (optional)

**Methods:**
- `get_nodes()`: Get cluster nodes
- `get_namespaces()`: Get cluster namespaces
- `create_namespace(name)`: Create a namespace
- `delete_namespace(name)`: Delete a namespace
- `get_events(namespace=None)`: Get cluster events
- `get_resource_usage()`: Get resource usage statistics
- `get_version()`: Get cluster version
- `get_api_resources()`: Get available API resources

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Kubernetes Integration** | Native Kubernetes integration with comprehensive API | TensorFlow Serving with Kubernetes manifests |
| **Deployment Management** | Built-in deployment management | Requires external tools or manifests |
| **GPU Support** | Native GPU support for Kubernetes deployments | Requires specific configuration |
| **Edge Device Orchestration** | Optimized for edge device orchestration | Limited edge device support |
| **Multi-Model Serving** | Built-in support for multi-model serving | Requires custom configuration |
| **Distributed Training** | Integrated distributed training on Kubernetes | Requires manual configuration |

Neurenix's Kubernetes integration offers a more comprehensive solution compared to TensorFlow, with native support for deployment management, GPU acceleration, and edge device orchestration. While TensorFlow can be deployed on Kubernetes using manifests or Kubeflow, it lacks a programmatic API for Kubernetes resource management, requiring users to rely on external tools or manifests. Neurenix's integrated approach simplifies the deployment and orchestration of AI applications on Kubernetes, particularly for edge devices and distributed training scenarios.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Kubernetes Integration** | Native Kubernetes integration with comprehensive API | No official Kubernetes integration |
| **Deployment Management** | Built-in deployment management | Requires external tools or manifests |
| **GPU Support** | Native GPU support for Kubernetes deployments | Requires specific configuration |
| **Edge Device Orchestration** | Optimized for edge device orchestration | Limited edge device support |
| **Multi-Model Serving** | Built-in support for multi-model serving | Requires custom configuration |
| **Distributed Training** | Integrated distributed training on Kubernetes | Requires manual configuration |

Neurenix provides a much more comprehensive Kubernetes integration compared to PyTorch, which has no official Kubernetes support. While PyTorch models can be deployed on Kubernetes using external tools like KServe or custom manifests, Neurenix's native Kubernetes integration simplifies the process with a programmatic API for resource management, GPU acceleration, and edge device orchestration. This makes Neurenix particularly well-suited for deploying AI applications in production environments, where Kubernetes orchestration is essential.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Kubernetes Integration** | Native Kubernetes integration with comprehensive API | No official Kubernetes integration |
| **Deployment Management** | Built-in deployment management | Requires external tools or manifests |
| **GPU Support** | Native GPU support for Kubernetes deployments | No native GPU support |
| **Edge Device Orchestration** | Optimized for edge device orchestration | No specific edge device support |
| **Multi-Model Serving** | Built-in support for multi-model serving | Requires custom configuration |
| **Distributed Training** | Integrated distributed training on Kubernetes | Limited distributed training capabilities |

Neurenix provides a much more comprehensive Kubernetes integration compared to Scikit-Learn, which has no official Kubernetes support. While Scikit-Learn models can be deployed on Kubernetes using external tools or custom manifests, Neurenix's native Kubernetes integration simplifies the process with a programmatic API for resource management, GPU acceleration, and edge device orchestration. This makes Neurenix particularly well-suited for deploying AI applications in production environments, where Kubernetes orchestration is essential.

## Best Practices

### Resource Management

When deploying Neurenix applications on Kubernetes, follow these best practices for resource management:

1. **Set Resource Requests and Limits**: Specify CPU and memory requests and limits for containers
2. **Use Node Selectors and Affinity**: Place workloads on appropriate nodes based on hardware requirements
3. **Implement Pod Disruption Budgets**: Ensure high availability during cluster operations
4. **Use Horizontal Pod Autoscaling**: Scale deployments based on CPU or custom metrics
5. **Monitor Resource Usage**: Track resource utilization and adjust as needed

```python
import neurenix
from neurenix.kubernetes import Deployment

# Create a deployment with resource management
deployment = Deployment(
    name="neurenix-model",
    image="neurenix/model-server:latest",
    namespace="ai-models",
    replicas=3,
    resources={
        "requests": {
            "cpu": "500m",
            "memory": "1Gi"
        },
        "limits": {
            "cpu": "2",
            "memory": "4Gi"
        }
    },
    labels={
        "app": "neurenix",
        "component": "model-server"
    },
    annotations={
        "prometheus.io/scrape": "true",
        "prometheus.io/port": "8080"
    }
)

# Create the deployment
deployment.create()

# Create horizontal pod autoscaler
from neurenix.kubernetes import HorizontalPodAutoscaler

hpa = HorizontalPodAutoscaler(
    name="neurenix-model-hpa",
    namespace="ai-models",
    deployment_name="neurenix-model",
    min_replicas=2,
    max_replicas=10,
    target_cpu_utilization=80
)
hpa.create()
```

### GPU Acceleration

To optimize GPU acceleration in Kubernetes deployments:

1. **Use Node Labels for GPU Nodes**: Label nodes with GPU capabilities
2. **Specify GPU Resource Requirements**: Request specific GPU types and quantities
3. **Use Device Plugins**: Ensure appropriate device plugins are installed
4. **Implement GPU Sharing**: Use GPU sharing for better utilization
5. **Monitor GPU Usage**: Track GPU utilization and memory usage

```python
import neurenix
from neurenix.kubernetes import Deployment

# Create a GPU-accelerated deployment
deployment = Deployment(
    name="neurenix-gpu-model",
    image="neurenix/model-server:latest",
    namespace="ai-models",
    replicas=1,
    resources={
        "requests": {
            "cpu": "1",
            "memory": "2Gi",
            "nvidia.com/gpu": "1"
        },
        "limits": {
            "cpu": "4",
            "memory": "8Gi",
            "nvidia.com/gpu": "1"
        }
    },
    env={
        "NEURENIX_DEVICE": "cuda",
        "CUDA_VISIBLE_DEVICES": "0"
    },
    node_selector={
        "accelerator": "nvidia-tesla-v100"
    }
)

# Create the deployment
deployment.create()
```

### High Availability

For high availability of Neurenix applications on Kubernetes:

1. **Use Multiple Replicas**: Deploy multiple instances of your application
2. **Implement Pod Anti-Affinity**: Distribute pods across nodes
3. **Use Pod Disruption Budgets**: Limit voluntary disruptions
4. **Configure Liveness and Readiness Probes**: Ensure proper health checking
5. **Implement Rolling Updates**: Minimize downtime during updates

```python
import neurenix
from neurenix.kubernetes import Deployment, PodDisruptionBudget

# Create a highly available deployment
deployment = Deployment(
    name="neurenix-ha-model",
    image="neurenix/model-server:latest",
    namespace="ai-models",
    replicas=3,
    pod_anti_affinity={
        "requiredDuringSchedulingIgnoredDuringExecution": [
            {
                "labelSelector": {
                    "matchExpressions": [
                        {
                            "key": "app",
                            "operator": "In",
                            "values": ["neurenix-ha-model"]
                        }
                    ]
                },
                "topologyKey": "kubernetes.io/hostname"
            }
        ]
    },
    liveness_probe={
        "httpGet": {
            "path": "/health",
            "port": 8080
        },
        "initialDelaySeconds": 30,
        "periodSeconds": 10
    },
    readiness_probe={
        "httpGet": {
            "path": "/ready",
            "port": 8080
        },
        "initialDelaySeconds": 5,
        "periodSeconds": 5
    },
    strategy={
        "type": "RollingUpdate",
        "rollingUpdate": {
            "maxSurge": "25%",
            "maxUnavailable": "25%"
        }
    }
)

# Create the deployment
deployment.create()

# Create pod disruption budget
pdb = PodDisruptionBudget(
    name="neurenix-ha-model-pdb",
    namespace="ai-models",
    selector={"app": "neurenix-ha-model"},
    min_available="50%"
)
pdb.create()
```

### Edge Device Orchestration

When orchestrating Neurenix applications on edge devices with Kubernetes:

1. **Use Lightweight Kubernetes Distributions**: Consider K3s or MicroK8s for edge devices
2. **Implement Resource Constraints**: Set strict resource limits for edge deployments
3. **Use Local Storage**: Leverage local storage for data persistence
4. **Implement Offline Operation**: Design for intermittent connectivity
5. **Use Device-Specific Optimizations**: Leverage hardware-specific features

```python
import neurenix
from neurenix.kubernetes import Deployment, PersistentVolumeClaim

# Create a persistent volume claim for local storage
pvc = PersistentVolumeClaim(
    name="edge-data",
    namespace="edge",
    storage_class="local-storage",
    access_modes=["ReadWriteOnce"],
    storage="10Gi"
)
pvc.create()

# Create an edge-optimized deployment
deployment = Deployment(
    name="neurenix-edge",
    image="neurenix/edge-model:latest",
    namespace="edge",
    replicas=1,
    resources={
        "requests": {
            "cpu": "200m",
            "memory": "256Mi"
        },
        "limits": {
            "cpu": "500m",
            "memory": "512Mi"
        }
    },
    env={
        "NEURENIX_DEVICE": "cpu",
        "NEURENIX_OPTIMIZE_FOR_EDGE": "true",
        "OFFLINE_MODE": "true"
    },
    volumes={
        "edge-data": {
            "persistentVolumeClaim": {
                "claimName": "edge-data"
            },
            "mountPath": "/data"
        }
    },
    node_selector={
        "edge-device": "true"
    },
    tolerations=[
        {
            "key": "node-role.kubernetes.io/edge",
            "operator": "Exists",
            "effect": "NoSchedule"
        }
    ]
)

# Create the deployment
deployment.create()
```

## Tutorials

### Deploying a Neurenix Model Server on Kubernetes

```python
import neurenix
from neurenix.kubernetes import Namespace, ConfigMap, Secret, Deployment, Service

# Create a namespace
namespace = Namespace("neurenix-models")
namespace.create()

# Create a ConfigMap for model configuration
model_config = ConfigMap(
    name="model-config",
    namespace="neurenix-models",
    data={
        "config.yaml": """
        model:
          name: my-model
          version: 1.0.0
          batch_size: 32
          num_workers: 4
        server:
          port: 5000
          max_queue_size: 100
          timeout: 60
        """
    }
)
model_config.create()

# Create a Secret for API keys
api_keys = Secret(
    name="api-keys",
    namespace="neurenix-models",
    data={
        "api-key": "${API_KEY_BASE64}"  # Use environment variable for the base64 encoded API key
    }
)
api_keys.create()

# Create a Deployment
deployment = Deployment(
    name="model-server",
    namespace="neurenix-models",
    image="neurenix/model-server:latest",
    replicas=3,
    resources={
        "requests": {
            "cpu": "500m",
            "memory": "1Gi"
        },
        "limits": {
            "cpu": "2",
            "memory": "4Gi"
        }
    },
    env={
        "MODEL_CONFIG_PATH": "/config/config.yaml",
        "API_KEY": {
            "secretKeyRef": {
                "name": "api-keys",
                "key": "api-key"
            }
        }
    },
    volumes={
        "config-volume": {
            "configMap": {
                "name": "model-config"
            },
            "mountPath": "/config"
        }
    },
    ports=[
        {"containerPort": 5000, "name": "http"}
    ],
    liveness_probe={
        "httpGet": {
            "path": "/health",
            "port": 5000
        },
        "initialDelaySeconds": 30,
        "periodSeconds": 10
    },
    readiness_probe={
        "httpGet": {
            "path": "/ready",
            "port": 5000
        },
        "initialDelaySeconds": 5,
        "periodSeconds": 5
    }
)
deployment.create()

# Create a Service
service = Service(
    name="model-server",
    namespace="neurenix-models",
    selector={"app": "model-server"},
    ports=[
        {"port": 80, "targetPort": 5000, "name": "http"}
    ],
    type="ClusterIP"
)
service.create()

# Wait for deployment to be ready
import time
print("Waiting for deployment to be ready...")
while not deployment.is_ready():
    time.sleep(5)
    print(".", end="", flush=True)
print("\nDeployment is ready!")

# Get service endpoint
print(f"Service endpoint: {service.get_endpoints()}")
```

### Distributed Training on Kubernetes

```python
import neurenix
from neurenix.kubernetes import Namespace, ConfigMap, Job, Service

# Create a namespace
namespace = Namespace("neurenix-training")
namespace.create()

# Create a ConfigMap for training configuration
training_config = ConfigMap(
    name="training-config",
    namespace="neurenix-training",
    data={
        "config.yaml": """
        training:
          epochs: 100
          batch_size: 64
          learning_rate: 0.001
          optimizer: adam
        data:
          train_path: /data/train
          val_path: /data/val
          test_path: /data/test
        distributed:
          backend: nccl
          world_size: 4
        """
    }
)
training_config.create()

# Create a Service for the master node
master_service = Service(
    name="training-master",
    namespace="neurenix-training",
    selector={"app": "training-master"},
    ports=[
        {"port": 29500, "targetPort": 29500, "name": "dist"}
    ],
    type="ClusterIP"
)
master_service.create()

# Create a Job for the master node
master_job = Job(
    name="training-master",
    namespace="neurenix-training",
    image="neurenix/training:latest",
    resources={
        "requests": {
            "cpu": "1",
            "memory": "2Gi",
            "nvidia.com/gpu": "1"
        },
        "limits": {
            "cpu": "4",
            "memory": "8Gi",
            "nvidia.com/gpu": "1"
        }
    },
    env={
        "CONFIG_PATH": "/config/config.yaml",
        "NEURENIX_DEVICE": "cuda",
        "NEURENIX_DISTRIBUTED_RANK": "0",
        "NEURENIX_DISTRIBUTED_WORLD_SIZE": "4",
        "NEURENIX_DISTRIBUTED_MASTER_ADDR": "training-master",
        "NEURENIX_DISTRIBUTED_MASTER_PORT": "29500"
    },
    volumes={
        "config-volume": {
            "configMap": {
                "name": "training-config"
            },
            "mountPath": "/config"
        },
        "data-volume": {
            "persistentVolumeClaim": {
                "claimName": "training-data"
            },
            "mountPath": "/data"
        }
    },
    ports=[
        {"containerPort": 29500, "name": "dist"}
    ],
    command=["python", "-m", "neurenix.distributed.train"],
    args=["--config", "/config/config.yaml", "--master"],
    labels={"app": "training-master"}
)
master_job.create()

# Create Jobs for worker nodes
for i in range(1, 4):
    worker_job = Job(
        name=f"training-worker-{i}",
        namespace="neurenix-training",
        image="neurenix/training:latest",
        resources={
            "requests": {
                "cpu": "1",
                "memory": "2Gi",
                "nvidia.com/gpu": "1"
            },
            "limits": {
                "cpu": "4",
                "memory": "8Gi",
                "nvidia.com/gpu": "1"
            }
        },
        env={
            "CONFIG_PATH": "/config/config.yaml",
            "NEURENIX_DEVICE": "cuda",
            "NEURENIX_DISTRIBUTED_RANK": str(i),
            "NEURENIX_DISTRIBUTED_WORLD_SIZE": "4",
            "NEURENIX_DISTRIBUTED_MASTER_ADDR": "training-master",
            "NEURENIX_DISTRIBUTED_MASTER_PORT": "29500"
        },
        volumes={
            "config-volume": {
                "configMap": {
                    "name": "training-config"
                },
                "mountPath": "/config"
            },
            "data-volume": {
                "persistentVolumeClaim": {
                    "claimName": "training-data"
                },
                "mountPath": "/data"
            }
        },
        command=["python", "-m", "neurenix.distributed.train"],
        args=["--config", "/config/config.yaml"],
        labels={"app": f"training-worker-{i}"}
    )
    worker_job.create()

# Wait for master job to start
print("Waiting for master job to start...")
while not master_job.status().get("active", 0):
    time.sleep(5)
    print(".", end="", flush=True)
print("\nMaster job started!")

# Monitor training progress
import time
print("Monitoring training progress...")
while not master_job.is_complete():
    time.sleep(30)
    print(f"Master job logs: {master_job.logs()[-200:]}")

print("Training complete!")
```

### Creating a Scheduled Model Update Job

```python
import neurenix
from neurenix.kubernetes import Namespace, ConfigMap, CronJob, Secret

# Create a namespace
namespace = Namespace("neurenix-scheduled")
namespace.create()

# Create a ConfigMap for update configuration
update_config = ConfigMap(
    name="update-config",
    namespace="neurenix-scheduled",
    data={
        "config.yaml": """
        model:
          name: my-model
          version: latest
          source: s3://models/my-model
        update:
          strategy: rolling
          validation: true
          rollback: true
        notification:
          enabled: true
          channels: ["slack", "email"]
        """
    }
)
update_config.create()

# Create a Secret for credentials
credentials = Secret(
    name="update-credentials",
    namespace="neurenix-scheduled",
    data={
        "aws-access-key": "${AWS_ACCESS_KEY_BASE64}",  # Use environment variable for base64 encoded access key
        "aws-secret-key": "${AWS_SECRET_KEY_BASE64}"   # Use environment variable for base64 encoded secret key
    }
)
credentials.create()

# Create a CronJob for scheduled updates
cronjob = CronJob(
    name="model-update",
    namespace="neurenix-scheduled",
    image="neurenix/model-updater:latest",
    schedule="0 2 * * *",  # Run at 2 AM every day
    resources={
        "requests": {
            "cpu": "500m",
            "memory": "1Gi"
        },
        "limits": {
            "cpu": "1",
            "memory": "2Gi"
        }
    },
    env={
        "CONFIG_PATH": "/config/config.yaml",
        "AWS_ACCESS_KEY_ID": {
            "secretKeyRef": {
                "name": "update-credentials",
                "key": "aws-access-key"
            }
        },
        "AWS_SECRET_ACCESS_KEY": {
            "secretKeyRef": {
                "name": "update-credentials",
                "key": "aws-secret-key"
            }
        }
    },
    volumes={
        "config-volume": {
            "configMap": {
                "name": "update-config"
            },
            "mountPath": "/config"
        }
    },
    command=["python", "-m", "neurenix.kubernetes.update"],
    args=["--config", "/config/config.yaml"],
    concurrency_policy="Forbid",
    successful_jobs_history_limit=3,
    failed_jobs_history_limit=1
)
cronjob.create()

print(f"CronJob created: {cronjob.name}")
print(f"Schedule: {cronjob.schedule}")
print(f"Next scheduled run: {cronjob.status().get('next_scheduled_time')}")
```

## Conclusion

The Kubernetes Integration module of Neurenix provides a comprehensive set of tools for deploying and managing AI applications on Kubernetes clusters. Its intuitive API makes it easy for researchers and developers to orchestrate containerized machine learning workloads at scale, ensuring consistent deployment across different environments.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Kubernetes Integration module offers advantages in terms of API design, resource management, and edge device orchestration. These features make Neurenix particularly well-suited for deploying AI applications in production environments, where Kubernetes orchestration is essential.
