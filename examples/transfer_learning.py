"""
Example of transfer learning with the Neurenix framework.

This example demonstrates how to:
1. Load a pre-trained model
2. Freeze its layers
3. Add new layers for a different task
4. Fine-tune the model on a new dataset
"""

import numpy as np

import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU, Softmax
from neurenix.optim import Adam
from neurenix.transfer import TransferModel, fine_tune, freeze_layers, unfreeze_layers

def main():
    # Initialize Neurenix
    nx.init({"device": "cpu", "log_level": "info"})
    
    print("Neurenix Transfer Learning Example")
    print("==================================")
    
    # Create a simple pre-trained model (in a real scenario, you would load this from a file)
    base_model = Sequential([
        Linear(784, 512),
        ReLU(),
        Linear(512, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])
    
    # Pretend we've trained this model on some task
    print("Simulating pre-trained model...")
    
    # Create new layers for our target task (e.g., classifying a different dataset)
    new_layers = Sequential([
        Linear(10, 64),
        ReLU(),
        Linear(64, 5),  # 5 classes in our new task
        Softmax(dim=1),
    ])
    
    # Create a transfer learning model
    transfer_model = TransferModel(
        base_model=base_model,
        new_layers=new_layers,
        freeze_base=True,  # Freeze the base model
        fine_tune_layers=["2", "3"],  # Fine-tune the last two layers of the base model
    )
    
    print("Transfer model created:")
    print(f"- Base model frozen: Yes (except layers 2 and 3)")
    print(f"- New layers: {new_layers}")
    
    # Create some dummy data for our new task
    # In a real scenario, you would load your actual dataset
    num_samples = 1000
    input_dim = 784  # e.g., 28x28 images flattened
    num_classes = 5
    
    # Generate random data
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_onehot[np.arange(num_samples), y] = 1.0
    
    # Split into train and validation sets
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y_onehot[:train_size], y_onehot[train_size:]
    
    # Convert to Neurenix tensors
    X_train_tensors = [nx.Tensor(x) for x in X_train]
    y_train_tensors = [nx.Tensor(y) for y in y_train]
    X_val_tensors = [nx.Tensor(x) for x in X_val]
    y_val_tensors = [nx.Tensor(y) for y in y_val]
    
    print(f"\nTraining data: {len(X_train_tensors)} samples")
    print(f"Validation data: {len(X_val_tensors)} samples")
    
    # Create an optimizer
    optimizer = Adam(transfer_model.parameters(), lr=0.001)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    history = fine_tune(
        model=transfer_model,
        optimizer=optimizer,
        train_data=X_train_tensors,
        train_labels=y_train_tensors,
        val_data=X_val_tensors,
        val_labels=y_val_tensors,
        epochs=5,
        batch_size=32,
        early_stopping=True,
        patience=2,
    )
    
    print("\nFine-tuning complete!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Unfreeze more layers for further fine-tuning
    print("\nUnfreezing more layers for further fine-tuning...")
    unfreeze_layers(base_model, ["0", "1", "2", "3"])
    
    # Fine-tune again with more layers unfrozen
    print("Fine-tuning with more layers unfrozen...")
    history = fine_tune(
        model=transfer_model,
        optimizer=optimizer,
        train_data=X_train_tensors,
        train_labels=y_train_tensors,
        val_data=X_val_tensors,
        val_labels=y_val_tensors,
        epochs=3,
        batch_size=32,
    )
    
    print("\nSecond fine-tuning complete!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    print("\nTransfer learning example completed successfully!")

if __name__ == "__main__":
    main()
