"""
Test script to verify that all Neurenix modules can be imported correctly.
This script tests the fix for the import errors in PR #62.
"""

import sys
import importlib
import traceback

def test_import(module_name):
    """Test importing a module and print the result."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False

def main():
    """Test importing all Neurenix modules."""
    modules_to_test = [
        "neurenix",
        "neurenix.nn",
        "neurenix.optim",
        "neurenix.tensor",
        "neurenix.device",
        "neurenix.core",
        "neurenix.binding",
        "neurenix.meta",
        "neurenix.transfer",
        "neurenix.unsupervised",
        "neurenix.distributed",
        "neurenix.wasm",
        "neurenix.hardware",
        "neurenix.data",
        "neurenix.automl",
        "neurenix.gnn",
        "neurenix.fuzzy",
        "neurenix.federated",
        "neurenix.cli",
        "neurenix.continual",
        "neurenix.async_train",
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
    failure_count = 0

    print("Testing Neurenix module imports...")
    print("-" * 50)

    for module_name in modules_to_test:
        if test_import(module_name):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 50)

    print(f"Import test results: {success_count} succeeded, {failure_count} failed")
    
    if failure_count > 0:
        sys.exit(1)
    else:
        print("All imports successful!")

if __name__ == "__main__":
    main()
