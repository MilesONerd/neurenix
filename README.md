# Neurenix

Neurenix is ​​an AI framework optimized for embedded devices (Edge AI), with support for multiple GPUs and distributed clusters. The framework specializes in AI agents, with native support for multi-agent, reinforcement learning, and autonomous AI.

## Social

[![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&style=for-the-badge)](https://bsky.app/profile/neurenix.bsky.social)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Eqnhr8tK2G)
[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/neurenix)
[![Mastodon](https://img.shields.io/badge/Mastodon-6364FF?style=for-the-badge&logo=Mastodon&logoColor=white)](https://fosstodon.org/@neurenix)
[![X/Twitter](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/neurenix)

## Main Features

- **Phynexus Engine**: High-performance engine for tensor operations
- **Optimized for Edge AI**: Efficient execution on embedded devices
- **Multi-GPU and Distributed Support**: Scalability to clusters and multiple GPUs
- **TPU Support**: Tensor Processing Unit acceleration for machine learning workloads
- **Agent-Specialized**: Native support for multi-agent and autonomous AI
- **Declarative API**: Simpler and more intuitive interface than traditional frameworks
- **LLM Fine-tuning**: Simplified tools for tuning language models
- **Hugging Face Integration**: Compatibility with pre-existing models
- **Hot-swappable backend functionality**:  
  1- Added DeviceManager class for runtime device switching  
  2- Created Genesis system for automatic hardware detection and selection  
  3- Modified Tensor class to support hot-swapping between devices  

- **ONNX support**:  
  1- Implemented ONNXConverter for model import/export  
  2- Added convenience functions for easy ONNX integration  
  3- Support for converting between Neurenix and other ML frameworks  

- **API support**:  
  1- Added RESTful, WebSocket, and gRPC server implementations  
  2- Created APIManager for centralized server management  
  3- Provided convenience functions for serving models

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

## Documentation

[Neurenix Documentation](https://neurenix.readthedocs.io/en/latest/)

## License

This project is licensed under the [Apache License 2.0](LICENSE).
