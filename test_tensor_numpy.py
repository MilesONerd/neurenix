import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurenix', 'python'))

from neurenix.tensor import Tensor

t = Tensor([1, 2, 3, 4])
print(f"Tensor created: {t}")

print(f"Has numpy method: {hasattr(t, 'numpy')}")

try:
    result = t.numpy()
    print(f"numpy() result: {result}")
except Exception as e:
    print(f"Error calling numpy(): {e}")

print("\nTesting activation functions...")
relu_result = t.relu()
print(f"ReLU result type: {type(relu_result)}")
print(f"ReLU result: {relu_result}")
print(f"ReLU has numpy method: {hasattr(relu_result, 'numpy')}")

try:
    numpy_result = relu_result.numpy()
    print(f"ReLU numpy() result: {numpy_result}")
except Exception as e:
    print(f"Error calling ReLU numpy(): {e}")
