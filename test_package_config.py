"""
Test script to verify that all Neurenix modules can be imported correctly.
This helps ensure the package configuration in pyproject.toml is working properly.
"""

import sys
import importlib

def test_import(module_name):
    """Test importing a module and print the result."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Test importing all Neurenix modules."""
    print("Testing Neurenix package imports...")
    
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
        "neurenix.huggingface",
        "neurenix.automl",
        "neurenix.gnn",
        "neurenix.fuzzy",
        "neurenix.federated",
        "neurenix.data",
        "neurenix.cli",
        "neurenix.continual",
        "neurenix.async_train",
        "neurenix.docker",
        "neurenix.kubernetes",
        "neurenix.explainable",
        "neurenix.multiscale",
        "neurenix.zeroshot",
        "neurenix.hardware",
        "neurenix.neuroevolution",
        "neurenix.neuro_symbolic",
        "neurenix.mas",
        "neurenix.zkp",
        "neurenix.quantum",
    ]
    
    success_count = 0
    for module_name in modules:
        if test_import(module_name):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(modules)} modules imported successfully")
    
    if success_count == len(modules):
        print("All imports successful! Package configuration is working correctly.")
        return 0
    else:
        print("Some imports failed. Check package configuration in pyproject.toml.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
