"""
Example of meta-learning with the Neurenix framework.

This example demonstrates how to:
1. Create a meta-learning model
2. Generate synthetic few-shot learning tasks
3. Train the model using MAML, Reptile, or Prototypical Networks
4. Evaluate the model on new tasks
"""

import numpy as np

import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.meta import MAML, Reptile, PrototypicalNetworks

def generate_sine_wave_tasks(num_tasks, num_samples_per_task, amplitude_range=(0.1, 5.0), phase_range=(0, np.pi)):
    """
    Generate synthetic sine wave regression tasks for meta-learning.
    
    Each task is to learn a sine wave with a different amplitude and phase.
    
    Args:
        num_tasks: Number of tasks to generate
        num_samples_per_task: Number of samples per task
        amplitude_range: Range of amplitudes
        phase_range: Range of phases
        
    Returns:
        List of (support_x, support_y, query_x, query_y) tuples
    """
    tasks = []
    
    for _ in range(num_tasks):
        # Sample amplitude and phase for this task
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(*phase_range)
        
        # Generate x values
        x = np.random.uniform(-5, 5, num_samples_per_task).astype(np.float32)
        
        # Generate y values
        y = amplitude * np.sin(x + phase).astype(np.float32)
        
        # Split into support and query sets
        num_support = num_samples_per_task // 2
        
        support_x = x[:num_support].reshape(-1, 1)
        support_y = y[:num_support].reshape(-1, 1)
        query_x = x[num_support:].reshape(-1, 1)
        query_y = y[num_support:].reshape(-1, 1)
        
        # Convert to Neurenix tensors
        support_x_tensor = nx.Tensor(support_x)
        support_y_tensor = nx.Tensor(support_y)
        query_x_tensor = nx.Tensor(query_x)
        query_y_tensor = nx.Tensor(query_y)
        
        tasks.append((support_x_tensor, support_y_tensor, query_x_tensor, query_y_tensor))
    
    return tasks

def generate_classification_tasks(num_tasks, num_classes=5, num_samples_per_class=10, input_dim=20):
    """
    Generate synthetic classification tasks for meta-learning.
    
    Each task is to classify points from different Gaussian distributions.
    
    Args:
        num_tasks: Number of tasks to generate
        num_classes: Number of classes per task
        num_samples_per_class: Number of samples per class
        input_dim: Dimensionality of the input space
        
    Returns:
        List of (support_x, support_y, query_x, query_y) tuples
    """
    tasks = []
    
    for _ in range(num_tasks):
        # Generate class means
        class_means = np.random.randn(num_classes, input_dim).astype(np.float32)
        
        # Generate samples for each class
        all_samples = []
        all_labels = []
        
        for c in range(num_classes):
            # Generate samples for this class
            samples = np.random.randn(num_samples_per_class, input_dim).astype(np.float32)
            samples = samples + class_means[c]  # Add class mean
            
            # Create one-hot labels
            labels = np.zeros((num_samples_per_class, num_classes), dtype=np.float32)
            labels[:, c] = 1.0
            
            all_samples.append(samples)
            all_labels.append(labels)
        
        # Concatenate samples and labels
        all_samples = np.vstack(all_samples)
        all_labels = np.vstack(all_labels)
        
        # Shuffle samples and labels
        indices = np.random.permutation(len(all_samples))
        all_samples = all_samples[indices]
        all_labels = all_labels[indices]
        
        # Split into support and query sets
        num_support = num_classes * (num_samples_per_class // 2)
        
        support_x = all_samples[:num_support]
        support_y = all_labels[:num_support]
        query_x = all_samples[num_support:]
        query_y = all_labels[num_support:]
        
        # Convert to Neurenix tensors
        support_x_tensor = nx.Tensor(support_x)
        support_y_tensor = nx.Tensor(support_y)
        query_x_tensor = nx.Tensor(query_x)
        query_y_tensor = nx.Tensor(query_y)
        
        tasks.append((support_x_tensor, support_y_tensor, query_x_tensor, query_y_tensor))
    
    return tasks

def main():
    # Initialize Neurenix
    nx.init({"device": "cpu", "log_level": "info"})
    
    print("Neurenix Meta-Learning Example")
    print("==============================")
    
    # Choose meta-learning algorithm
    algorithm = "maml"  # Options: "maml", "reptile", "prototypical"
    
    # Create a model for meta-learning
    model = Sequential([
        Linear(1, 40),
        ReLU(),
        Linear(40, 40),
        ReLU(),
        Linear(40, 1),
    ])
    
    if algorithm == "maml":
        print("Using Model-Agnostic Meta-Learning (MAML)")
        meta_model = MAML(
            model=model,
            inner_lr=0.01,
            meta_lr=0.001,
            first_order=False,
            inner_steps=5,
        )
    elif algorithm == "reptile":
        print("Using Reptile")
        meta_model = Reptile(
            model=model,
            inner_lr=0.02,
            meta_lr=0.001,
            inner_steps=8,
        )
    elif algorithm == "prototypical":
        print("Using Prototypical Networks")
        # For Prototypical Networks, we need a different model architecture
        # that outputs embeddings
        input_dim = 20  # Define input dimension for the embedding model
        embedding_model = Sequential([
            Linear(input_dim, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 32),  # Embedding dimension
        ])
        meta_model = PrototypicalNetworks(
            embedding_model=embedding_model,
            distance_metric="euclidean",
            meta_lr=0.001,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Generate tasks
    if algorithm == "prototypical":
        print("Generating classification tasks...")
        input_dim = 20
        tasks = generate_classification_tasks(
            num_tasks=100,
            num_classes=5,
            num_samples_per_class=10,
            input_dim=input_dim,
        )
    else:
        print("Generating sine wave tasks...")
        tasks = generate_sine_wave_tasks(
            num_tasks=100,
            num_samples_per_task=20,
        )
    
    # Split tasks into train and test
    num_train_tasks = 80
    train_tasks = tasks[:num_train_tasks]
    test_tasks = tasks[num_train_tasks:]
    
    print(f"Training on {len(train_tasks)} tasks")
    print(f"Testing on {len(test_tasks)} tasks")
    
    # Create optimizer
    meta_optimizer = Adam(meta_model.parameters(), lr=meta_model.meta_lr)
    
    # Meta-train the model
    print("\nMeta-training...")
    history = meta_model.meta_learn(
        tasks=train_tasks,
        meta_optimizer=meta_optimizer,
        epochs=10,
        tasks_per_batch=4,
    )
    
    # Evaluate on test tasks
    print("\nEvaluating on test tasks...")
    test_losses = []
    
    for support_x, support_y, query_x, query_y in test_tasks:
        # Adapt to the task
        adapted_model = meta_model.adapt_to_task(support_x, support_y)
        
        # Evaluate on query set
        predictions = adapted_model(query_x)
        
        # Compute loss
        if algorithm == "prototypical":
            # For classification, compute accuracy
            pred_classes = predictions.argmax(dim=1)
            true_classes = query_y.argmax(dim=1)
            accuracy = (pred_classes == true_classes).float().mean().item()
            test_losses.append(1.0 - accuracy)  # Convert accuracy to loss
        else:
            # For regression, compute MSE
            mse = ((predictions - query_y) ** 2).mean().item()
            test_losses.append(mse)
    
    # Print results
    avg_test_loss = sum(test_losses) / len(test_losses)
    if algorithm == "prototypical":
        print(f"Average test error rate: {avg_test_loss:.4f}")
        print(f"Average test accuracy: {1.0 - avg_test_loss:.4f}")
    else:
        print(f"Average test MSE: {avg_test_loss:.4f}")
    
    print("\nMeta-learning example completed successfully!")

if __name__ == "__main__":
    main()
