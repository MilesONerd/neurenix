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

- **Apache Arrow Support**:
  - Efficient in-memory columnar data format
  - Seamless conversion between Arrow tables and Neurenix tensors
  - Support for various data types and operations

- **Parquet Support**:
  - High-performance columnar storage format
  - Reading and writing Parquet files
  - Dataset management with partitioning support
  - Integration with Arrow for efficient data processing

- **SHAP (SHapley Additive exPlanations)**:
  - KernelSHAP for model-agnostic explanations
  - TreeSHAP for tree-based models
  - DeepSHAP for deep learning models

- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Support for tabular, text, and image data
  - Customizable sampling and kernel functions

- **Additional Explanation Techniques**
  - Feature importance analysis
  - Partial dependence plots
  - Counterfactual explanations
  - Activation visualization

- **Multi-Scale Model Architectures**:
  - MultiScaleModel: Base class for multi-scale architectures
  - PyramidNetwork: Feature pyramid network implementation
  - UNet: U-Net architecture with skip connections

- **Multi-Scale Pooling Operations**:
  - MultiScalePooling: Base class for multi-scale pooling
  - PyramidPooling: Pyramid pooling module (PPM)
  - SpatialPyramidPooling: Spatial pyramid pooling (SPP)

- **Feature Fusion Mechanisms**:
  - FeatureFusion: Base class for feature fusion
  - ScaleFusion: Fusion of features from different scales
  - AttentionFusion: Attention-based fusion of multi-scale features

- **Multi-Scale Transformations**:
  - MultiScaleTransform: Base class for multi-scale transforms
  - Rescale: Rescaling to multiple scales
  - PyramidDownsample: Pyramid downsampling for multi-scale representations
  - GaussianPyramid: Gaussian pyramid implementation
  - LaplacianPyramid: Laplacian pyramid implementation
 
- **Zero-shot Learning**
- **NVIDIA Tensor Cores support**
- **WebAssembly multithreaded support**
- **gRPC-Streaming support**
- **Neuroevolution + Evolutionary Algorithms Support**:
  - Genetic Algorithms: Implementation of population-based optimization with selection, crossover, and mutation operators
  - NEAT (NeuroEvolution of Augmenting Topologies): Algorithm for evolving both neural network topologies and weights
  - HyperNEAT: Extension of NEAT that uses CPPNs to generate large-scale neural networks with geometric regularities
  - CMA-ES (Covariance Matrix Adaptation Evolution Strategy): State-of-the-art evolutionary algorithm for continuous optimization
  - Evolution Strategies: Implementation of various ES variants with adaptive learning rates and population-based optimization
 
- **Hybrid Neuro-Symbolic Models Support**:
  - Symbolic reasoning components with rule-based inference
  - Neural-symbolic integration with multiple interaction modes
  - Differentiable logic with support for fuzzy and probabilistic logic
  - Knowledge distillation between symbolic and neural systems
  - Advanced reasoning paradigms (constraint satisfaction, logical inference, abductive/deductive/inductive reasoning)
 
- **Multi-Agent Systems (MAS) Support**:
  - Agent framework with reactive, deliberative, and hybrid agent types
  - Communication protocols and message passing infrastructure
  - Coordination mechanisms (task allocation, auctions, contract nets, voting)
  - Multi-agent learning algorithms (independent learners, joint action learners, team learning)
  - Environment abstractions for agent interaction
 
- **Quantum Computing (Qiskit and Cirq) Support**:
  - Quantum circuit interfaces for both Qiskit and Cirq
  - Hybrid quantum-classical computing support
  - Parameterized quantum circuits for variational algorithms
  - Quantum utility functions for state manipulation and measurement
  - Integration with the Neurenix tensor and device management systems
 
- **.NET distributed computing with ASP.NET and Orleans**
- **NPU hardware support**
- **ARM architecture support**:
  - ARM Compute Library for optimized neural network operations
  - Ethos-U/NPU for dedicated neural processing
  - Neon SIMD for vectorized operations
  - SVE (Scalable Vector Extensions) for variable-length vector processing

## Documentation

[Neurenix Documentation](https://neurenix.readthedocs.io/en/latest/)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
