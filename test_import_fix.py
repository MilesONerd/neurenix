"""
Test script to verify that all subpackages can be imported correctly.
"""

import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

if __name__ == "__main__":
    modules_to_test = [
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
        "neurenix.tensor",
        "neurenix.data",
        "neurenix.hardware",
        "neurenix.gnn",
        "neurenix.fuzzy",
        "neurenix.federated",
        "neurenix.automl",
        "neurenix.memory",
        "neurenix.continual",
        "neurenix.async_train",
        "neurenix.cli",
        "neurenix.docker",
        "neurenix.kubernetes",
        "neurenix.explainable",
        "neurenix.multiscale",
        "neurenix.zeroshot",
        "neurenix.neuroevolution",
        "neurenix.neuro_symbolic",
        "neurenix.mas",
        "neurenix.zkp",
        "neurenix.quantum",
    ]
    
    success_count = 0
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(modules_to_test)} modules imported successfully")
    
    if success_count == len(modules_to_test):
        print("All imports successful!")
        sys.exit(0)
    else:
        print("Some imports failed.")
        sys.exit(1)
