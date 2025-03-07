"""
Neurenix - Framework de Inteligência Artificial otimizado para Edge AI

Neurenix é um framework de inteligência artificial otimizado para dispositivos embarcados (Edge AI),
com suporte para múltiplas GPUs e clusters distribuídos. O framework é especializado em agentes de IA,
com suporte nativo para multi-agentes, aprendizado por reforço e IA autônoma.
"""

__version__ = "0.1.0"

from neurenix.core import init, version
from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module, Linear, Conv2d, LSTM, Sequential
from neurenix.optim import Optimizer, SGD, Adam
from neurenix.agent import Agent, MultiAgent, Environment

# Initialize the framework
init()
