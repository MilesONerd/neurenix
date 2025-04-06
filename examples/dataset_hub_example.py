"""
Example demonstrating the use of DatasetHub for easy dataset loading and management.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from neurenix.data import DatasetHub, DatasetFormat

def main():
    hub = DatasetHub()
    
    print("Example 1: Loading a dataset from a URL")
    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris_dataset = hub.load_dataset(iris_url, format=DatasetFormat.CSV)
    
    print(f"Loaded Iris dataset with {len(iris_dataset)} samples")
    print(f"First 5 samples: {iris_dataset[:5]}")
    print()
    
    print("Example 2: Registering a dataset")
    hub.register_dataset(
        name="iris",
        url=iris_url,
        format=DatasetFormat.CSV,
        metadata={"description": "Iris flower dataset", "classes": 3}
    )
    
    iris_dataset = hub.load_dataset("iris")
    print(f"Loaded registered Iris dataset with metadata: {iris_dataset.metadata}")
    print()
    
    print("Example 3: Splitting dataset")
    train_dataset, val_dataset = iris_dataset.split(ratio=0.8, shuffle=True, seed=42)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print()
    
    print("Example 4: Converting to tensors")
    try:
        import torch
        iris_tensor = iris_dataset.to_tensor(framework="torch")
        print(f"PyTorch tensor shape: {iris_tensor.shape}")
    except (ImportError, TypeError):
        print("PyTorch not available or conversion failed")
    
    try:
        import tensorflow as tf
        iris_tensor = iris_dataset.to_tensor(framework="tensorflow")
        print(f"TensorFlow tensor shape: {iris_tensor.shape}")
    except (ImportError, TypeError):
        print("TensorFlow not available or conversion failed")
    
    iris_array = iris_dataset.to_tensor(framework="numpy")
    print(f"NumPy array shape: {iris_array.shape}")
    print()
    
    print("Example 5: Loading a local file")
    with open("sample_data.csv", "w") as f:
        f.write("x,y,z\n")
        for i in range(10):
            f.write(f"{i},{i*2},{i*3}\n")
    
    sample_dataset = hub.load_dataset("sample_data.csv")
    print(f"Loaded local CSV with {len(sample_dataset)} samples")
    print(f"Data: {sample_dataset[:]}")
    
    os.remove("sample_data.csv")
    print()
    
    print("Example 6: Applying transformations")
    
    def normalize_features(sample):
        features = [float(x) for x in sample[:4]]
        label = sample[4]
        
        min_val = min(features)
        max_val = max(features)
        if max_val > min_val:
            normalized = [(x - min_val) / (max_val - min_val) for x in features]
        else:
            normalized = features
            
        return normalized + [label]
    
    transformed_dataset = hub.load_dataset(
        "iris", 
        transform=normalize_features
    )
    
    print(f"Original sample: {iris_dataset[0]}")
    print(f"Transformed sample: {transformed_dataset[0]}")
    print()
    
    print("DatasetHub examples completed successfully!")

if __name__ == "__main__":
    main()
