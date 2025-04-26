import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurenix', 'python'))

try:
    from neurenix.tensor import Tensor
    print("Successfully imported Tensor from neurenix.tensor")
    
    t = Tensor([1, 2, 3])
    print(f"Tensor created: {t}")
    
    if hasattr(t, 'numpy'):
        print("numpy() method exists")
        result = t.numpy()
        print(f"numpy() result: {result}")
    else:
        print("numpy() method does not exist")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
