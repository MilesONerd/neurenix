# Neurenix

Neurenix is ​​an AI framework optimized for embedded devices (Edge AI), with support for multiple GPUs and distributed clusters. The framework specializes in AI agents, with native support for multi-agent, reinforcement learning, and autonomous AI.

## Social

[![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&style=for-the-badge)](https://bsky.app/profile/neurenix.bsky.social)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Eqnhr8tK2G)
[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/neurenix)
[![Mastodon](https://img.shields.io/badge/Mastodon-6364FF?style=for-the-badge&logo=Mastodon&logoColor=white)](https://fosstodon.org/@neurenix)
[![X/Twitter](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/neurenix)

## Main Features

- **Hot-swappable backend functionality**:  
  - Added DeviceManager class for runtime device switching  
  - Created Genesis system for automatic hardware detection and selection  
  - Modified Tensor class to support hot-swapping between devices  

- **ONNX support**:  
  - Implemented ONNXConverter for model import/export  
  - Added convenience functions for easy ONNX integration  
  - Support for converting between Neurenix and other ML frameworks  

- **API support**:  
  - Added RESTful, WebSocket, and gRPC server implementations  
  - Created APIManager for centralized server management  
  - Provided convenience functions for serving models

- **Dynamic imports from neurenix.binding with NumPy fallbacks for activation functions**:  
  - relu, sigmoid, tanh, softmax, log_softmax, leaky_relu, elu, selu, gelu  

- **CPU implementations for BLAS operations**:  
  - GEMM, dot product, GEMV  

- **CPU implementations for convolution operations**:  
  - conv2d, conv_transpose2d  

- **Conditional compilation for hardware-specific operations**:  
  - CUDA, ROCm, and WebGPU support for BLAS and convolution operations  
  - Proper error handling for unsupported hardware configurations  

- **Binding functions for tensor operations**:  
  - backward, no_grad, zero_grad, weight_decay
 
- **WebAssembly SIMD and WASI-NN support for browser-based tensor operations**  
- **Hardware acceleration backends**:  
  - Vulkan for cross-platform GPU acceleration  
  - OpenCL for heterogeneous computing  
  - oneAPI for Intel hardware acceleration  
  - DirectML for Windows DirectX 12 acceleration  
  - oneDNN for optimized deep learning primitives  
  - MKL-DNN for Intel CPU optimization  
  - TensorRT for NVIDIA GPU optimization

- **Automatic quantization support**:  
  - INT8, FP16, and FP8 precision  
  - Model pruning capabilities  
  - Quantization-aware training  
  - Post-training quantization with calibration  
 
- **Graph Neural Networks (GNNs)**:  
  - Implemented various GNN layers (GCN, GAT, GraphSAGE, etc.)  
  - Added pooling operations for graph data  
  - Provided graph data structures and utilities  
  - Implemented common GNN models  

- **Fuzzy Logic**:  
  - Added fuzzy sets with various membership functions  
  - Implemented fuzzy variables and linguistic variables  
  - Created fuzzy rule systems with different operators  
  - Implemented Mamdani, Sugeno, and Tsukamoto inference systems  
  - Added multiple defuzzification methods  

- **Federated Learning**:  
  - Implemented client-server architecture for federated learning  
  - Added various aggregation strategies (FedAvg, FedProx, FedNova, etc.)  
  - Implemented security mechanisms (secure aggregation, differential privacy)  
  - Added utilities for client selection and model compression  

- **AutoML & Meta-learning**:  
  - Implemented hyperparameter search strategies (Grid, Random, Bayesian, Evolutionary)  
  - Added neural architecture search capabilities  
  - Implemented model selection and evaluation utilities  
  - Created pipeline optimization tools  
  - Added meta-learning algorithms for few-shot learning  
 
- **Distributed training technologies**:  
  - MPI for high-performance computing clusters  
  - Horovod for distributed deep learning  
  - DeepSpeed for large-scale model training  

- **Memory management technologies**:  
  - Unified Memory (UM) for seamless CPU-GPU memory sharing  
  - Heterogeneous Memory Management (HMM) for advanced memory optimization  

- **Specialized hardware acceleration**:  
  - GraphCore IPU support for intelligence processing  
  - FPGA support via multiple frameworks:  
    - OpenCL for cross-vendor FPGA programming  
    - Xilinx Vitis for Xilinx FPGAs  
    - Intel OpenVINO for Intel FPGAs

- **DatasetHub**: mechanism that allows users to easily load datasets by providing a URL or file path
- **CLI**
- **Continual Learning Module**: Allows models to be retrained and updated with new data without forgetting previously learned information. Implements several techniques:
  - Elastic Weight Consolidation (EWC)
  - Experience Replay
  - L2 Regularization
  - Knowledge Distillation
  - Synaptic Intelligence

- **Asynchronous and Interruptible Training Module**: Provides functionality for asynchronous training with continuous checkpointing and automatic resume, even in unstable environments. Features include:
  - Continuous checkpointing with atomic writes
  - Automatic resume after interruptions
  - Resource monitoring and proactive checkpointing
  - Signal handling for graceful interruptions
  - Distributed checkpointing for multi-node training
 
- **Docker Support**:
  - Container management
  - Image building and management
  - Volume management
  - Network configuration
  - Registry integration
- **Kubernetes Support**:
  - Deployment management
  - Service configuration
  - Pod management
  - ConfigMap handling
  - Secret management
  - Job scheduling
  - Cluster management

## Documentation

[Neurenix Documentation](https://neurenix.readthedocs.io/en/latest/)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
