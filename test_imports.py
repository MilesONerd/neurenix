"""
Test script to verify all Neurenix modules can be imported correctly.
"""

import sys
import importlib

def test_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing {module_name}: {e}")
        return False

if __name__ == "__main__":
    modules = [
        "neurenix",
        "neurenix.nn",
        "neurenix.optim",
        "neurenix.agent",
        "neurenix.rl",
        "neurenix.meta",
        "neurenix.distributed",
        "neurenix.transfer",
        "neurenix.unsupervised",
        "neurenix.wasm",
        "neurenix.huggingface"
    ]
    
    success = True
    for module in modules:
        if not test_import(module):
            success = False
    
    sys.exit(0 if success else 1)
