"""
Verify the Neurenix package installation from Conda.
"""
import sys

try:
    import neurenix
    print(f"Neurenix version: {neurenix.__version__}")
    
    from neurenix import nn, optim, agent, tensor, device
    print("All modules imported successfully")
    
    sys.exit(0)
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
