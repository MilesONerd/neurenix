# Explainable AI API Documentation

## Overview

The Explainable AI module provides tools and techniques for explaining and interpreting machine learning models, making AI systems more transparent, understandable, and trustworthy. This module enables developers to gain insights into how models make decisions, identify potential biases, and build more reliable AI applications.

Explainable AI (XAI) is becoming increasingly important as AI systems are deployed in critical domains such as healthcare, finance, and autonomous vehicles, where understanding model behavior is essential for ensuring safety, fairness, and regulatory compliance. Neurenix's Explainable AI module implements various state-of-the-art techniques for model interpretation, from feature importance methods to local explanations and visualization tools.

## Key Concepts

### Model Interpretability vs. Explainability

- **Interpretability** refers to the degree to which a human can understand the cause of a decision made by a model.
- **Explainability** refers to the ability to explain the internal workings of a model in human terms.

Neurenix provides tools for both interpretable models (which are transparent by design) and post-hoc explanations for complex models (which provide explanations after the model has been trained).

### Local vs. Global Explanations

- **Local explanations** focus on explaining individual predictions, helping to understand why a specific decision was made.
- **Global explanations** provide insights into the overall behavior of the model across all predictions.

Neurenix supports both local explanation methods (like LIME and SHAP) and global explanation methods (like feature importance and partial dependence plots).

### Model-Agnostic vs. Model-Specific Methods

- **Model-agnostic methods** can be applied to any machine learning model, regardless of its internal structure.
- **Model-specific methods** are designed for particular types of models, leveraging their specific properties.

Neurenix provides both model-agnostic methods (like LIME and Kernel SHAP) and model-specific methods (like Tree SHAP for tree-based models and Deep SHAP for neural networks).

## API Reference

### SHAP (SHapley Additive exPlanations)

```python
neurenix.explainable.ShapExplainer(model, data=None, link='identity')
```

Base class for SHAP explainers, which use Shapley values from game theory to explain model predictions.

**Parameters:**
- `model`: The model to explain
- `data`: Background data for integrating out features
- `link`: The link function used to map model outputs to predictions

**Methods:**
- `explain(X)`: Generate SHAP values for the given samples
- `plot_summary(X)`: Plot a summary of feature importance
- `plot_dependence(feature_idx)`: Plot the dependence of model output on a feature
- `plot_force(idx)`: Create a force plot for a single prediction

```python
neurenix.explainable.KernelShap(model, data, link='identity')
```

Model-agnostic SHAP implementation that uses a weighted linear regression to estimate Shapley values.

```python
neurenix.explainable.TreeShap(model, data=None, feature_perturbation='tree_path_dependent')
```

Optimized SHAP implementation for tree-based models (Random Forests, Gradient Boosting, etc.).

```python
neurenix.explainable.DeepShap(model, data)
```

SHAP implementation for deep learning models, using a connection to DeepLIFT.

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.explainable import DeepShap

# Create a simple neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)

# Generate some data
X = nx.Tensor.randn((100, 10))
y = model(X)

# Create a background dataset
background = nx.Tensor.randn((10, 10))

# Create a DeepShap explainer
explainer = DeepShap(model, background)

# Generate SHAP values for a sample
sample = X[0:1]
shap_values = explainer.explain(sample)

# Visualize the explanation
explainer.plot_force(0)
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
neurenix.explainable.LimeExplainer(kernel_width=0.75, verbose=False)
```

Base class for LIME explainers, which explain individual predictions by approximating the model locally with an interpretable model.

**Parameters:**
- `kernel_width`: Width of the exponential kernel used in the LIME algorithm
- `verbose`: Whether to print verbose output

**Methods:**
- `explain_instance(instance, predict_fn, num_features=10)`: Explain a prediction for an instance
- `plot_explanation(explanation)`: Visualize the explanation

```python
neurenix.explainable.LimeTabular(feature_names, categorical_features=None, kernel_width=0.75, verbose=False)
```

LIME implementation for tabular data.

**Additional Parameters:**
- `feature_names`: Names of the features
- `categorical_features`: Indices of categorical features

```python
neurenix.explainable.LimeText(kernel_width=0.75, verbose=False)
```

LIME implementation for text data.

```python
neurenix.explainable.LimeImage(kernel_width=0.75, verbose=False)
```

LIME implementation for image data.

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.explainable import LimeTabular

# Create a simple neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)

# Generate some data
X = nx.Tensor.randn((100, 10))
y = model(X)

# Define feature names
feature_names = [f'feature_{i}' for i in range(10)]

# Create a LIME explainer
explainer = LimeTabular(feature_names=feature_names)

# Define a prediction function
def predict_fn(x):
    return model(nx.Tensor(x)).numpy()

# Explain a prediction
instance = X[0].numpy()
explanation = explainer.explain_instance(instance, predict_fn, num_features=5)

# Visualize the explanation
explainer.plot_explanation(explanation)
```

### Feature Importance

```python
neurenix.explainable.FeatureImportance(model, feature_names=None)
```

Base class for feature importance methods, which quantify the contribution of each feature to model performance.

**Parameters:**
- `model`: The model to explain
- `feature_names`: Names of the features

**Methods:**
- `compute_importance(X, y)`: Compute feature importance
- `plot_importance(top_n=None)`: Plot feature importance

```python
neurenix.explainable.PermutationImportance(model, feature_names=None, n_repeats=5, random_state=None)
```

Computes feature importance by measuring the decrease in model performance when a feature is randomly permuted.

**Additional Parameters:**
- `n_repeats`: Number of times to permute each feature
- `random_state`: Random seed for reproducibility

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.explainable import PermutationImportance

# Create a simple neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)

# Generate some data
X = nx.Tensor.randn((100, 10))
y = model(X)

# Define feature names
feature_names = [f'feature_{i}' for i in range(10)]

# Create a permutation importance explainer
explainer = PermutationImportance(model, feature_names=feature_names)

# Compute feature importance
importance = explainer.compute_importance(X, y)

# Visualize feature importance
explainer.plot_importance(top_n=5)
```

### Partial Dependence

```python
neurenix.explainable.PartialDependence(model, feature_names=None)
```

Computes and visualizes partial dependence plots, which show the marginal effect of a feature on the model prediction.

**Parameters:**
- `model`: The model to explain
- `feature_names`: Names of the features

**Methods:**
- `compute_pd(X, feature_idx, grid_resolution=50)`: Compute partial dependence for a feature
- `plot_pd(feature_idx, X, grid_resolution=50)`: Plot partial dependence for a feature
- `plot_pd_interaction(feature_idx1, feature_idx2, X, grid_resolution=20)`: Plot partial dependence interaction between two features

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.explainable import PartialDependence

# Create a simple neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)

# Generate some data
X = nx.Tensor.randn((100, 10))
y = model(X)

# Define feature names
feature_names = [f'feature_{i}' for i in range(10)]

# Create a partial dependence explainer
explainer = PartialDependence(model, feature_names=feature_names)

# Plot partial dependence for a feature
explainer.plot_pd(feature_idx=0, X=X)

# Plot partial dependence interaction between two features
explainer.plot_pd_interaction(feature_idx1=0, feature_idx2=1, X=X)
```

### Counterfactual Explanations

```python
neurenix.explainable.Counterfactual(model, feature_names=None, categorical_features=None)
```

Generates counterfactual explanations, which show how to change the input to achieve a different prediction.

**Parameters:**
- `model`: The model to explain
- `feature_names`: Names of the features
- `categorical_features`: Indices of categorical features

**Methods:**
- `generate(instance, desired_class, max_iter=1000)`: Generate a counterfactual explanation
- `plot_counterfactual(instance, counterfactual)`: Visualize the counterfactual explanation

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.explainable import Counterfactual

# Create a simple neural network for binary classification
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 2)
)

# Generate some data
X = nx.Tensor.randn((100, 10))
y = model(X).argmax(dim=1)

# Define feature names
feature_names = [f'feature_{i}' for i in range(10)]

# Create a counterfactual explainer
explainer = Counterfactual(model, feature_names=feature_names)

# Generate a counterfactual explanation
instance = X[0].numpy()
current_class = model(X[0:1]).argmax(dim=1).item()
desired_class = 1 - current_class
counterfactual = explainer.generate(instance, desired_class)

# Visualize the counterfactual explanation
explainer.plot_counterfactual(instance, counterfactual)
```

### Activation Visualization

```python
neurenix.explainable.ActivationVisualization(model)
```

Visualizes activations of neural network layers to understand what the model is focusing on.

**Parameters:**
- `model`: The neural network model to visualize

**Methods:**
- `register_hooks()`: Register hooks to capture activations
- `remove_hooks()`: Remove activation hooks
- `get_activations(input_tensor)`: Get activations for an input
- `plot_activations(input_tensor, layer_name=None)`: Visualize activations for an input
- `plot_feature_maps(input_tensor, layer_name, n_features=9)`: Visualize feature maps for a convolutional layer

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear
from neurenix.explainable import ActivationVisualization

# Create a CNN
model = Sequential(
    Conv2d(1, 16, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Conv2d(16, 32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Flatten(),
    Linear(32 * 7 * 7, 10)
)

# Generate a sample image
image = nx.Tensor.randn((1, 1, 28, 28))

# Create an activation visualization
visualizer = ActivationVisualization(model)

# Register hooks to capture activations
visualizer.register_hooks()

# Visualize activations
visualizer.plot_activations(image)

# Visualize feature maps for a specific layer
visualizer.plot_feature_maps(image, layer_name='Conv2d_0')

# Remove hooks when done
visualizer.remove_hooks()
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Explainability Tools** | Comprehensive set of tools (SHAP, LIME, Feature Importance, etc.) | Limited native support, requires TF-Explain or other libraries |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries |
| **API Consistency** | Unified API for different explainability methods | Different APIs depending on the library used |
| **Edge Device Support** | Optimized for edge devices | Limited explainability support for edge devices |
| **Visualization Capabilities** | Rich visualization options | Basic visualization, requires additional libraries for advanced options |

Neurenix provides a more comprehensive and integrated explainable AI solution compared to TensorFlow. While TensorFlow requires additional libraries like TF-Explain or custom implementations for most explainability methods, Neurenix offers a unified API with native support for multiple state-of-the-art techniques. Additionally, Neurenix's optimization for edge devices makes it more suitable for explainable AI in resource-constrained environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Explainability Tools** | Comprehensive set of tools with unified API | Limited native support, requires libraries like Captum |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries |
| **API Consistency** | Unified API for different explainability methods | Different APIs depending on the library used |
| **Edge Device Support** | Optimized for edge devices | Limited explainability support for edge devices |
| **Visualization Capabilities** | Rich visualization options | Depends on the library used |

PyTorch has good support for explainable AI through libraries like Captum, but lacks native integration in the core framework. Neurenix's unified API and native optimization for edge devices provide advantages for deploying explainable AI systems in production environments, especially on resource-constrained hardware.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Explainability Tools** | Comprehensive set of tools for deep learning models | Basic tools for traditional ML models (permutation importance, partial dependence) |
| **Deep Learning Support** | Full deep learning support | Limited to shallow models |
| **API Consistency** | Unified API for different explainability methods | Consistent API but limited scope |
| **Edge Device Support** | Optimized for edge devices | No specific edge device support |
| **Visualization Capabilities** | Rich visualization options | Basic visualization capabilities |

Scikit-Learn provides some basic explainability tools for traditional machine learning models, but lacks support for deep learning models and advanced explainability techniques. Neurenix fills this gap with its comprehensive explainable AI module, which supports both traditional and deep learning models with a unified API.

## Best Practices

### Choosing the Right Explainability Method

Different explainability methods are suitable for different scenarios:

1. **For Global Understanding**: Use feature importance and partial dependence plots to understand the overall behavior of the model.
2. **For Local Explanations**: Use LIME or SHAP to explain individual predictions.
3. **For Deep Learning Models**: Use DeepShap or activation visualization to understand neural network behavior.
4. **For Tree-Based Models**: Use TreeShap for efficient and accurate explanations.

```python
import neurenix as nx
from neurenix.explainable import PermutationImportance, PartialDependence, LimeTabular, DeepShap

# For global understanding
if global_understanding_needed:
    # Feature importance for overall model behavior
    importance_explainer = PermutationImportance(model, feature_names=feature_names)
    importance = importance_explainer.compute_importance(X, y)
    importance_explainer.plot_importance()
    
    # Partial dependence for feature effects
    pd_explainer = PartialDependence(model, feature_names=feature_names)
    for feature_idx in range(X.shape[1]):
        pd_explainer.plot_pd(feature_idx, X)

# For local explanations
if local_explanation_needed:
    # LIME for individual predictions
    lime_explainer = LimeTabular(feature_names=feature_names)
    explanation = lime_explainer.explain_instance(instance, predict_fn)
    lime_explainer.plot_explanation(explanation)

# For deep learning models
if isinstance(model, nx.nn.Module):
    # DeepShap for neural networks
    background = X[:10]  # Use a subset of data as background
    deep_explainer = DeepShap(model, background)
    shap_values = deep_explainer.explain(instance)
    deep_explainer.plot_force(0)
```

### Interpreting Explanations

When interpreting explanations, consider the following:

1. **Correlation vs. Causation**: Feature importance doesn't necessarily imply causation.
2. **Local vs. Global**: Local explanations may not generalize to the entire dataset.
3. **Model Limitations**: Explanations are only as good as the model they explain.
4. **Domain Knowledge**: Combine explanations with domain knowledge for better insights.

```python
import neurenix as nx
from neurenix.explainable import ShapExplainer, PermutationImportance

# Combine multiple explanation methods for robust insights
shap_explainer = ShapExplainer(model, data=X)
shap_values = shap_explainer.explain(X_test)

importance_explainer = PermutationImportance(model, feature_names=feature_names)
importance = importance_explainer.compute_importance(X, y)

# Compare explanations from different methods
for feature_idx in range(X.shape[1]):
    feature_name = feature_names[feature_idx]
    shap_importance = abs(shap_values[:, feature_idx]).mean()
    perm_importance = importance[feature_idx]
    
    print(f"Feature: {feature_name}")
    print(f"SHAP Importance: {shap_importance:.4f}")
    print(f"Permutation Importance: {perm_importance:.4f}")
    
    # Check for consistency between methods
    if abs(shap_importance - perm_importance) > threshold:
        print("Warning: Inconsistent importance between methods")
```

### Visualization Best Practices

Effective visualization is key to communicating explanations:

1. **Simplicity**: Keep visualizations simple and focused on the key insights.
2. **Consistency**: Use consistent color schemes and scales across visualizations.
3. **Context**: Provide context for the explanations, such as feature distributions.
4. **Interactivity**: Use interactive visualizations when possible for exploration.

```python
import neurenix as nx
from neurenix.explainable import ShapExplainer
import matplotlib.pyplot as plt

# Create a SHAP explainer
explainer = ShapExplainer(model, data=X)
shap_values = explainer.explain(X_test)

# Create a multi-panel visualization
plt.figure(figsize=(15, 10))

# 1. Summary plot for global feature importance
plt.subplot(2, 2, 1)
explainer.plot_summary(X_test)
plt.title("Global Feature Importance")

# 2. Force plot for a specific prediction
plt.subplot(2, 2, 2)
explainer.plot_force(0)
plt.title("Explanation for Instance 0")

# 3. Dependence plot for the most important feature
most_important_feature = abs(shap_values).mean(axis=0).argmax()
plt.subplot(2, 2, 3)
explainer.plot_dependence(most_important_feature)
plt.title(f"Dependence Plot for Feature {feature_names[most_important_feature]}")

# 4. Actual vs. Predicted values
plt.subplot(2, 2, 4)
y_pred = model(X_test).numpy()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Model Performance")

plt.tight_layout()
plt.savefig("explanation_dashboard.png", dpi=300)
plt.show()
```

## Tutorials

### Explaining a Neural Network with SHAP

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.explainable import DeepShap
import matplotlib.pyplot as plt
import numpy as np

# Initialize Neurenix
nx.init()

# Load the Boston Housing dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to Neurenix tensors
X_train_tensor = nx.Tensor(X_train)
y_train_tensor = nx.Tensor(y_train).reshape(-1, 1)
X_test_tensor = nx.Tensor(X_test)
y_test_tensor = nx.Tensor(y_test).reshape(-1, 1)

# Create a neural network
model = Sequential(
    Linear(X_train.shape[1], 64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, 1)
)

# Create an optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nx.nn.MSELoss()

# Train the model
num_epochs = 100
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train_tensor[indices]
    y_shuffled = y_train_tensor[indices]
    
    for i in range(n_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    y_pred = model(X_test_tensor)
    test_loss = loss_fn(y_pred, y_test_tensor).item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/n_batches:.4f}, Test Loss: {test_loss:.4f}")

# Create a background dataset for SHAP
background = X_train_tensor[:100]  # Use a subset of training data as background

# Create a DeepShap explainer
explainer = DeepShap(model, background)

# Generate SHAP values for the test set
shap_values = explainer.explain(X_test_tensor)

# Plot a summary of feature importance
plt.figure(figsize=(10, 6))
explainer.plot_summary(X_test_tensor)
plt.title("Feature Importance Summary")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.show()

# Plot force plots for the first few test instances
plt.figure(figsize=(15, 10))
for i in range(min(4, len(X_test))):
    plt.subplot(2, 2, i+1)
    explainer.plot_force(i)
    plt.title(f"Explanation for Instance {i}")
plt.tight_layout()
plt.savefig("shap_force_plots.png", dpi=300)
plt.show()

# Plot dependence plots for the top 4 features
top_features = abs(shap_values).mean(axis=0).argsort()[-4:]
plt.figure(figsize=(15, 10))
for i, feature_idx in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    explainer.plot_dependence(feature_idx)
    plt.title(f"Dependence Plot for {feature_names[feature_idx]}")
plt.tight_layout()
plt.savefig("shap_dependence_plots.png", dpi=300)
plt.show()
```

### Explaining Image Classification with LIME

```python
import neurenix as nx
from neurenix.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear
from neurenix.optim import Adam
from neurenix.explainable import LimeImage
import matplotlib.pyplot as plt
import numpy as np

# Initialize Neurenix
nx.init()

# Load MNIST dataset
train_dataset = nx.data.MNIST(root='./data', train=True, download=True)
test_dataset = nx.data.MNIST(root='./data', train=False, download=True)

# Preprocess the data
X_train = train_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_train = train_dataset.targets
X_test = test_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_test = test_dataset.targets

# Create a CNN
model = Sequential(
    Conv2d(1, 16, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Conv2d(16, 32, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Linear(128, 10)
)

# Create an optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nx.nn.CrossEntropyLoss()

# Train the model (simplified for brevity)
num_epochs = 5
batch_size = 64
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    for i in range(n_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/n_batches:.4f}, Test Accuracy: {accuracy:.4f}")

# Create a LIME explainer for images
explainer = LimeImage()

# Define a prediction function for LIME
def predict_fn(images):
    # Convert images to Neurenix tensors
    batch = nx.Tensor(images).reshape(-1, 1, 28, 28)
    
    # Get model predictions
    model.eval()
    with nx.no_grad():
        outputs = model(batch)
        probs = nx.nn.Softmax(dim=1)(outputs)
    
    return probs.numpy()

# Select a test image to explain
image_idx = 10
image = X_test[image_idx].numpy()[0]  # Remove channel dimension for LIME
true_label = y_test[image_idx].item()

# Generate an explanation
explanation = explainer.explain_instance(
    image,
    predict_fn,
    top_labels=5,
    num_samples=1000,
    num_features=10
)

# Get the prediction
pred = predict_fn(image.reshape(1, 28, 28))[0]
pred_label = pred.argmax()

# Visualize the explanation
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title(f"Original Image\nTrue: {true_label}, Pred: {pred_label}")
plt.axis('off')

# Positive evidence
plt.subplot(1, 3, 2)
temp, mask = explanation.get_image_and_mask(
    pred_label,
    positive_only=True,
    num_features=5,
    hide_rest=False
)
plt.imshow(mark_boundaries(temp, mask), cmap='gray')
plt.title("Positive Evidence")
plt.axis('off')

# Negative evidence
plt.subplot(1, 3, 3)
temp, mask = explanation.get_image_and_mask(
    pred_label,
    positive_only=False,
    negative_only=True,
    num_features=5,
    hide_rest=False
)
plt.imshow(mark_boundaries(temp, mask), cmap='gray')
plt.title("Negative Evidence")
plt.axis('off')

plt.tight_layout()
plt.savefig("lime_explanation.png", dpi=300)
plt.show()

# Show explanation for multiple digits
plt.figure(figsize=(15, 10))
for i in range(5):
    image_idx = np.random.randint(0, len(X_test))
    image = X_test[image_idx].numpy()[0]
    true_label = y_test[image_idx].item()
    
    # Generate an explanation
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=1,
        num_samples=1000,
        num_features=5
    )
    
    # Get the prediction
    pred = predict_fn(image.reshape(1, 28, 28))[0]
    pred_label = pred.argmax()
    
    # Original image
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original\nTrue: {true_label}, Pred: {pred_label}")
    plt.axis('off')
    
    # Explanation
    plt.subplot(2, 5, i+6)
    temp, mask = explanation.get_image_and_mask(
        pred_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    plt.imshow(mark_boundaries(temp, mask), cmap='gray')
    plt.title("Explanation")
    plt.axis('off')

plt.tight_layout()
plt.savefig("lime_multiple_explanations.png", dpi=300)
plt.show()
```

### Feature Importance and Partial Dependence for Tabular Data

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.explainable import PermutationImportance, PartialDependence
import matplotlib.pyplot as plt
import numpy as np

# Initialize Neurenix
nx.init()

# Load the California Housing dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to Neurenix tensors
X_train_tensor = nx.Tensor(X_train)
y_train_tensor = nx.Tensor(y_train).reshape(-1, 1)
X_test_tensor = nx.Tensor(X_test)
y_test_tensor = nx.Tensor(y_test).reshape(-1, 1)

# Create a neural network
model = Sequential(
    Linear(X_train.shape[1], 64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, 1)
)

# Create an optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nx.nn.MSELoss()

# Train the model (simplified for brevity)
num_epochs = 50
batch_size = 64
n_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train_tensor[indices]
    y_shuffled = y_train_tensor[indices]
    
    for i in range(n_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        # Evaluate on test set
        model.eval()
        y_pred = model(X_test_tensor)
        test_loss = loss_fn(y_pred, y_test_tensor).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/n_batches:.4f}, Test Loss: {test_loss:.4f}")

# Create a permutation importance explainer
importance_explainer = PermutationImportance(model, feature_names=feature_names)

# Compute feature importance
importance = importance_explainer.compute_importance(X_test_tensor, y_test_tensor)

# Plot feature importance
plt.figure(figsize=(10, 6))
importance_explainer.plot_importance()
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=300)
plt.show()

# Create a partial dependence explainer
pd_explainer = PartialDependence(model, feature_names=feature_names)

# Plot partial dependence for the top 4 important features
top_features = importance.argsort()[-4:]
plt.figure(figsize=(15, 10))
for i, feature_idx in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    pd_explainer.plot_pd(feature_idx, X_test_tensor)
    plt.title(f"Partial Dependence for {feature_names[feature_idx]}")
plt.tight_layout()
plt.savefig("partial_dependence.png", dpi=300)
plt.show()

# Plot partial dependence interactions between the top 2 features
plt.figure(figsize=(10, 8))
pd_explainer.plot_pd_interaction(top_features[-1], top_features[-2], X_test_tensor)
plt.title(f"Interaction between {feature_names[top_features[-1]]} and {feature_names[top_features[-2]]}")
plt.tight_layout()
plt.savefig("partial_dependence_interaction.png", dpi=300)
plt.show()

# Create a comprehensive explanation dashboard
plt.figure(figsize=(15, 12))

# 1. Feature importance
plt.subplot(2, 2, 1)
importance_explainer.plot_importance(top_n=8)
plt.title("Feature Importance")

# 2. Partial dependence for the most important feature
plt.subplot(2, 2, 2)
pd_explainer.plot_pd(top_features[-1], X_test_tensor)
plt.title(f"Partial Dependence for {feature_names[top_features[-1]]}")

# 3. Partial dependence for the second most important feature
plt.subplot(2, 2, 3)
pd_explainer.plot_pd(top_features[-2], X_test_tensor)
plt.title(f"Partial Dependence for {feature_names[top_features[-2]]}")

# 4. Interaction between the top 2 features
plt.subplot(2, 2, 4)
pd_explainer.plot_pd_interaction(top_features[-1], top_features[-2], X_test_tensor)
plt.title(f"Interaction between Top Features")

plt.tight_layout()
plt.savefig("explanation_dashboard.png", dpi=300)
plt.show()
```

## Conclusion

The Explainable AI module in Neurenix provides a comprehensive set of tools for explaining and interpreting machine learning models, making AI systems more transparent, understandable, and trustworthy. With support for various explainability techniques, including SHAP, LIME, feature importance, partial dependence, counterfactuals, and activation visualization, the module enables developers to gain insights into how models make decisions, identify potential biases, and build more reliable AI applications.

Compared to other frameworks, Neurenix's Explainable AI module offers advantages in terms of API consistency, integration with the core framework, and optimization for edge devices. These features make Neurenix particularly well-suited for developing transparent and trustworthy AI systems in critical domains such as healthcare, finance, and autonomous vehicles.
