"""
Neurenix - Artificial Intelligence Framework Optimized for Edge AI

Neurenix is ​​an artificial intelligence framework optimized for embedded devices (Edge AI),
with support for multiple GPUs and distributed clusters. The framework is specialized for AI agents,
with native support for multi-agents, reinforcement learning, and autonomous AI.
"""

__version__ = "2.0.1"

from neurenix.core import init, version
from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module, Linear, Conv2d, LSTM, Sequential
from neurenix.optim import Optimizer, SGD, Adam
from neurenix.agent import Agent, MultiAgent, Environment

# Initialize the framework
init()
