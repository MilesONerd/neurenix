import numpy as np
import neurenix
from neurenix.tensor import Tensor
from neurenix.nn import ReLU, Sigmoid, Tanh, Softmax, Linear, Sequential
from neurenix.optim import SGD, Adam

# Test tensor creation and basic operations
print("Testing tensor creation...")
t1 = Tensor([1, 2, 3, 4])
print(f"Tensor shape: {t1.shape}")

# Test activation functions
print("\nTesting activation functions...")
relu_result = t1.relu()
sigmoid_result = t1.sigmoid()
tanh_result = t1.tanh()
softmax_result = t1.softmax()

print(f"ReLU: {relu_result.numpy()}")
print(f"Sigmoid: {sigmoid_result.numpy()}")
print(f"Tanh: {tanh_result.numpy()}")
print(f"Softmax: {softmax_result.numpy()}")

# Test neural network modules
print("\nTesting neural network modules...")
relu_module = ReLU()
sigmoid_module = Sigmoid()
tanh_module = Tanh()
softmax_module = Softmax()

print(f"ReLU module: {relu_module(t1).numpy()}")
print(f"Sigmoid module: {sigmoid_module(t1).numpy()}")
print(f"Tanh module: {tanh_module(t1).numpy()}")
print(f"Softmax module: {softmax_module(t1).numpy()}")

# Test optimizers
print("\nTesting optimizers...")
params = [Tensor([1.0, 2.0, 3.0], requires_grad=True)]
sgd = SGD(params, lr=0.01)
adam = Adam(params, lr=0.001)

print("SGD optimizer created successfully")
print("Adam optimizer created successfully")

print("\nAll tests completed successfully!")
