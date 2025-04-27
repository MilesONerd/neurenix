# AutoML API Documentation

## Overview

The AutoML (Automated Machine Learning) module provides tools and utilities for automating the machine learning workflow, from data preprocessing to model deployment. This module enables developers to automate repetitive tasks such as hyperparameter tuning, neural architecture search, model selection, and feature engineering, allowing them to focus on higher-level aspects of their AI applications.

Neurenix's AutoML capabilities are designed with a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python interface provides a user-friendly API for rapid development. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Hyperparameter Optimization

Hyperparameter optimization is the process of finding the optimal set of hyperparameters for a machine learning algorithm. Neurenix provides several strategies for hyperparameter optimization, including grid search, random search, Bayesian optimization, and evolutionary algorithms.

### Neural Architecture Search (NAS)

Neural Architecture Search is the process of automatically designing neural network architectures. Neurenix implements several state-of-the-art NAS algorithms, including Efficient Neural Architecture Search (ENAS), Differentiable Architecture Search (DARTS), and Progressive Neural Architecture Search (PNAS).

### Model Selection

Model selection involves choosing the best model from a set of candidate models. Neurenix provides tools for automated model selection, including cross-validation and nested cross-validation.

### Automated Pipeline

An automated pipeline combines data preprocessing, feature selection, model training, and evaluation into a single workflow. Neurenix's AutoPipeline simplifies the creation and optimization of end-to-end machine learning pipelines.

## API Reference

### Hyperparameter Search

```python
neurenix.automl.HyperparameterSearch(
    model_fn: Callable,
    param_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Base class for hyperparameter search algorithms.

**Parameters:**
- `model_fn`: Function that returns a model instance
- `param_space`: Dictionary of hyperparameter names and their possible values
- `scoring`: Scoring metric to evaluate models
- `n_trials`: Number of trials to run
- `cv`: Number of cross-validation folds
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Run the hyperparameter search
- `best_params()`: Get the best hyperparameters
- `best_score()`: Get the best score
- `best_model()`: Get the best model

```python
neurenix.automl.GridSearch(
    model_fn: Callable,
    param_space: Dict[str, List[Any]],
    scoring: Union[str, Callable],
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Exhaustive search over specified parameter values.

```python
neurenix.automl.RandomSearch(
    model_fn: Callable,
    param_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Random search over specified parameter distributions.

```python
neurenix.automl.BayesianOptimization(
    model_fn: Callable,
    param_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None,
    acquisition_function: str = 'ei'
)
```

Bayesian optimization for hyperparameter search.

**Additional Parameters:**
- `acquisition_function`: Acquisition function to use ('ei', 'pi', or 'ucb')

```python
neurenix.automl.EvolutionarySearch(
    model_fn: Callable,
    param_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None,
    population_size: int = 10,
    mutation_rate: float = 0.1
)
```

Evolutionary algorithm for hyperparameter search.

**Additional Parameters:**
- `population_size`: Size of the population
- `mutation_rate`: Probability of mutation

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import SGD
from neurenix.automl import BayesianOptimization

# Define a model function
def create_model(hidden_size=64, learning_rate=0.01):
    model = Sequential(
        Linear(10, hidden_size),
        ReLU(),
        Linear(hidden_size, 1)
    )
    optimizer = SGD(model.parameters(), lr=learning_rate)
    return model, optimizer

# Define parameter space
param_space = {
    'hidden_size': [32, 64, 128, 256],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Create a Bayesian optimization search
search = BayesianOptimization(
    model_fn=create_model,
    param_space=param_space,
    scoring='mse',
    n_trials=20,
    cv=5,
    random_state=42
)

# Run the search
X = nx.Tensor.randn((100, 10))
y = nx.Tensor.randn((100, 1))
search.fit(X, y)

# Get the best parameters and model
best_params = search.best_params()
best_model = search.best_model()
print(f"Best parameters: {best_params}")
print(f"Best score: {search.best_score()}")
```

### Neural Architecture Search

```python
neurenix.automl.NeuralArchitectureSearch(
    search_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Base class for neural architecture search algorithms.

**Parameters:**
- `search_space`: Dictionary defining the search space for architectures
- `scoring`: Scoring metric to evaluate architectures
- `n_trials`: Number of trials to run
- `cv`: Number of cross-validation folds
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Run the neural architecture search
- `best_architecture()`: Get the best architecture
- `best_score()`: Get the best score
- `best_model()`: Get the best model

```python
neurenix.automl.ENAS(
    search_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None,
    controller_type: str = 'lstm'
)
```

Efficient Neural Architecture Search (ENAS).

**Additional Parameters:**
- `controller_type`: Type of controller ('lstm' or 'mlp')

```python
neurenix.automl.DARTS(
    search_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None,
    unrolled: bool = False
)
```

Differentiable Architecture Search (DARTS).

**Additional Parameters:**
- `unrolled`: Whether to use the unrolled version of DARTS

```python
neurenix.automl.PNAS(
    search_space: Dict[str, Any],
    scoring: Union[str, Callable],
    n_trials: int = 10,
    cv: int = 5,
    random_state: Optional[int] = None,
    num_expansions: int = 5
)
```

Progressive Neural Architecture Search (PNAS).

**Additional Parameters:**
- `num_expansions`: Number of expansions per iteration

**Example:**
```python
import neurenix as nx
from neurenix.automl import DARTS

# Define search space
search_space = {
    'num_layers': [2, 3, 4, 5],
    'operations': ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3'],
    'channels': [16, 32, 64]
}

# Create a DARTS search
search = DARTS(
    search_space=search_space,
    scoring='accuracy',
    n_trials=10,
    cv=3,
    random_state=42
)

# Run the search
X = nx.Tensor.randn((100, 3, 32, 32))  # Example image data
y = nx.Tensor.randint(0, 10, (100,))   # Example labels
search.fit(X, y)

# Get the best architecture and model
best_architecture = search.best_architecture()
best_model = search.best_model()
print(f"Best architecture: {best_architecture}")
print(f"Best score: {search.best_score()}")
```

### Model Selection

```python
neurenix.automl.AutoModelSelection(
    models: List[Tuple[str, Any]],
    scoring: Union[str, Callable],
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Automated model selection from a list of candidate models.

**Parameters:**
- `models`: List of (name, model) tuples
- `scoring`: Scoring metric to evaluate models
- `cv`: Number of cross-validation folds
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Run the model selection
- `best_model_name()`: Get the name of the best model
- `best_model()`: Get the best model
- `best_score()`: Get the best score

```python
neurenix.automl.CrossValidation(
    model: Any,
    scoring: Union[str, Callable],
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Cross-validation for model evaluation.

```python
neurenix.automl.NestedCrossValidation(
    model_fn: Callable,
    param_space: Dict[str, Any],
    scoring: Union[str, Callable],
    outer_cv: int = 5,
    inner_cv: int = 3,
    random_state: Optional[int] = None
)
```

Nested cross-validation for model selection and evaluation.

**Parameters:**
- `model_fn`: Function that returns a model instance
- `param_space`: Dictionary of hyperparameter names and their possible values
- `scoring`: Scoring metric to evaluate models
- `outer_cv`: Number of outer cross-validation folds
- `inner_cv`: Number of inner cross-validation folds
- `random_state`: Random seed for reproducibility

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.automl import AutoModelSelection

# Create candidate models
models = [
    ('linear', Sequential(Linear(10, 1))),
    ('small_mlp', Sequential(Linear(10, 32), ReLU(), Linear(32, 1))),
    ('large_mlp', Sequential(Linear(10, 64), ReLU(), Linear(64, 32), ReLU(), Linear(32, 1)))
]

# Create an AutoModelSelection instance
model_selection = AutoModelSelection(
    models=models,
    scoring='mse',
    cv=5,
    random_state=42
)

# Run the model selection
X = nx.Tensor.randn((100, 10))
y = nx.Tensor.randn((100, 1))
model_selection.fit(X, y)

# Get the best model
best_model_name = model_selection.best_model_name()
best_model = model_selection.best_model()
print(f"Best model: {best_model_name}")
print(f"Best score: {model_selection.best_score()}")
```

### Automated Pipeline

```python
neurenix.automl.AutoPipeline(
    steps: List[Tuple[str, Any]],
    scoring: Union[str, Callable],
    cv: int = 5,
    random_state: Optional[int] = None
)
```

Automated pipeline for end-to-end machine learning.

**Parameters:**
- `steps`: List of (name, transformer/estimator) tuples
- `scoring`: Scoring metric to evaluate the pipeline
- `cv`: Number of cross-validation folds
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Fit the pipeline
- `transform(X)`: Transform data using the pipeline
- `predict(X)`: Make predictions using the pipeline
- `score(X, y)`: Score the pipeline on data

```python
neurenix.automl.FeatureSelection(
    estimator: Any,
    n_features_to_select: Optional[int] = None,
    scoring: Union[str, Callable] = 'accuracy',
    cv: int = 5
)
```

Feature selection using recursive feature elimination.

```python
neurenix.automl.DataPreprocessing(
    steps: List[Tuple[str, Any]],
    random_state: Optional[int] = None
)
```

Data preprocessing pipeline.

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.automl import AutoPipeline, DataPreprocessing, FeatureSelection

# Create preprocessing steps
preprocessing = DataPreprocessing([
    ('normalize', nx.preprocessing.StandardScaler()),
    ('pca', nx.preprocessing.PCA(n_components=5))
])

# Create feature selection
feature_selection = FeatureSelection(
    estimator=Sequential(Linear(5, 1)),
    n_features_to_select=3,
    scoring='mse',
    cv=3
)

# Create a model
model = Sequential(
    Linear(3, 16),
    ReLU(),
    Linear(16, 1)
)

# Create an AutoPipeline
pipeline = AutoPipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('feature_selection', feature_selection),
        ('model', model)
    ],
    scoring='mse',
    cv=5,
    random_state=42
)

# Fit the pipeline
X = nx.Tensor.randn((100, 10))
y = nx.Tensor.randn((100, 1))
pipeline.fit(X, y)

# Make predictions
X_new = nx.Tensor.randn((10, 10))
predictions = pipeline.predict(X_new)
print(f"Predictions: {predictions}")
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **AutoML Support** | Native implementation with unified API | TensorFlow Extended (TFX) and Keras Tuner |
| **Hyperparameter Optimization** | Multiple algorithms (Grid, Random, Bayesian, Evolutionary) | Keras Tuner with limited algorithms |
| **Neural Architecture Search** | ENAS, DARTS, PNAS with unified API | Limited support through separate libraries |
| **Edge Device Optimization** | Native optimization for edge devices | TensorFlow Lite with limited AutoML support |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries |
| **API Consistency** | Consistent API across all AutoML components | Different APIs for different components |

Neurenix provides a more comprehensive and integrated AutoML solution compared to TensorFlow. While TensorFlow offers AutoML capabilities through TFX and Keras Tuner, these are separate components with different APIs. Neurenix's unified API makes it easier to use different AutoML techniques together, and its native optimization for edge devices enables efficient AutoML on resource-constrained hardware.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **AutoML Support** | Native implementation with unified API | Third-party libraries (e.g., Ray Tune, Optuna) |
| **Hyperparameter Optimization** | Multiple algorithms with consistent API | Depends on third-party library |
| **Neural Architecture Search** | ENAS, DARTS, PNAS with unified API | Limited native support |
| **Edge Device Optimization** | Native optimization for edge devices | Limited AutoML support for edge devices |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries |
| **API Consistency** | Consistent API across all AutoML components | Different APIs for different libraries |

PyTorch lacks native AutoML support, relying instead on third-party libraries like Ray Tune and Optuna for hyperparameter optimization and NAS. This results in inconsistent APIs and additional dependencies. Neurenix's integrated AutoML module provides a more cohesive experience with better performance on edge devices.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **AutoML Support** | Comprehensive AutoML capabilities | Limited to GridSearchCV and RandomizedSearchCV |
| **Hyperparameter Optimization** | Multiple advanced algorithms | Basic grid and random search |
| **Neural Architecture Search** | ENAS, DARTS, PNAS | No support |
| **Deep Learning Integration** | Seamless integration with neural networks | Limited neural network support |
| **Edge Device Optimization** | Native optimization for edge devices | No specific edge device support |
| **Pipeline Automation** | Advanced pipeline automation | Basic pipeline support through Pipeline class |

Scikit-Learn provides basic hyperparameter optimization through GridSearchCV and RandomizedSearchCV, but lacks advanced algorithms like Bayesian optimization and evolutionary search. It also has no support for neural architecture search and limited integration with deep learning frameworks. Neurenix's AutoML module offers a more comprehensive solution with advanced algorithms and seamless integration with neural networks.

## Best Practices

### Hyperparameter Optimization

For effective hyperparameter optimization:

1. **Define a Meaningful Parameter Space**: Focus on parameters that significantly impact model performance.
2. **Choose the Right Algorithm**: Use grid search for small spaces, random search for larger spaces, and Bayesian optimization for expensive evaluations.
3. **Set Appropriate Evaluation Metrics**: Choose metrics that align with your problem's objectives.

```python
import neurenix as nx
from neurenix.automl import BayesianOptimization

# Define a focused parameter space
param_space = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128],
    'dropout_rate': [0.1, 0.3, 0.5]
}

# Choose Bayesian optimization for expensive evaluations
search = BayesianOptimization(
    model_fn=create_model,
    param_space=param_space,
    scoring='accuracy',  # Choose an appropriate metric
    n_trials=20,
    cv=5
)
```

### Neural Architecture Search

For effective neural architecture search:

1. **Define a Constrained Search Space**: Limit the search space to reduce computational cost.
2. **Use Progressive Search**: Start with simple architectures and progressively increase complexity.
3. **Consider Hardware Constraints**: Include hardware efficiency metrics in your evaluation.

```python
import neurenix as nx
from neurenix.automl import PNAS

# Define a constrained search space
search_space = {
    'num_layers': [2, 3, 4],  # Limit the number of layers
    'operations': ['conv3x3', 'maxpool3x3'],  # Limit operation types
    'channels': [16, 32, 64]  # Limit channel counts
}

# Use PNAS for progressive search
search = PNAS(
    search_space=search_space,
    scoring='accuracy',
    n_trials=10,
    cv=3,
    num_expansions=3  # Progressive expansion
)
```

### Model Selection

For effective model selection:

1. **Include Diverse Models**: Include models with different architectures and complexities.
2. **Use Nested Cross-Validation**: Separate model selection from evaluation to avoid overfitting.
3. **Consider Multiple Metrics**: Evaluate models on multiple metrics to get a comprehensive view.

```python
import neurenix as nx
from neurenix.automl import NestedCrossValidation

# Use nested cross-validation
ncv = NestedCrossValidation(
    model_fn=create_model,
    param_space=param_space,
    scoring=['accuracy', 'f1', 'roc_auc'],  # Multiple metrics
    outer_cv=5,
    inner_cv=3
)
```

### Pipeline Automation

For effective pipeline automation:

1. **Include All Steps**: Incorporate preprocessing, feature selection, and model training in your pipeline.
2. **Optimize the Entire Pipeline**: Tune hyperparameters for all steps together.
3. **Ensure Reproducibility**: Set random seeds for all components.

```python
import neurenix as nx
from neurenix.automl import AutoPipeline

# Create a complete pipeline
pipeline = AutoPipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('feature_selection', feature_selection),
        ('model', model)
    ],
    scoring='accuracy',
    cv=5,
    random_state=42  # Ensure reproducibility
)
```

## Tutorials

### Hyperparameter Optimization with Bayesian Optimization

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU, Dropout
from neurenix.optim import Adam
from neurenix.automl import BayesianOptimization

# Initialize Neurenix
nx.init()

# Define a model function with hyperparameters
def create_model(hidden_size=64, learning_rate=0.01, dropout_rate=0.2):
    model = Sequential(
        Linear(10, hidden_size),
        ReLU(),
        Dropout(dropout_rate),
        Linear(hidden_size, hidden_size // 2),
        ReLU(),
        Dropout(dropout_rate),
        Linear(hidden_size // 2, 1)
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# Define parameter space
param_space = {
    'hidden_size': [32, 64, 128, 256],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'dropout_rate': [0.1, 0.2, 0.3, 0.5]
}

# Create a Bayesian optimization search
search = BayesianOptimization(
    model_fn=create_model,
    param_space=param_space,
    scoring='mse',
    n_trials=30,
    cv=5,
    random_state=42,
    acquisition_function='ei'  # Expected Improvement
)

# Generate synthetic data
X = nx.Tensor.randn((500, 10))
y = nx.Tensor.randn((500, 1))

# Split data into train and test sets
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Run the search
search.fit(X_train, y_train)

# Get the best parameters and model
best_params = search.best_params()
best_model, best_optimizer = search.best_model()
print(f"Best parameters: {best_params}")
print(f"Best score: {search.best_score()}")

# Evaluate the best model on the test set
best_model.eval()
predictions = best_model(X_test)
mse = ((predictions - y_test) ** 2).mean().item()
print(f"Test MSE: {mse}")

# Visualize the optimization process
search.plot_optimization_history()
```

### Neural Architecture Search with DARTS

```python
import neurenix as nx
from neurenix.automl import DARTS
from neurenix.data import DataLoader

# Initialize Neurenix
nx.init()

# Define search space
search_space = {
    'num_cells': [4, 8],
    'num_blocks_per_cell': [3, 5],
    'operations': ['conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3', 'identity'],
    'channels': [16, 32, 64]
}

# Create a DARTS search
search = DARTS(
    search_space=search_space,
    scoring='accuracy',
    n_trials=10,
    cv=3,
    random_state=42,
    unrolled=True
)

# Load CIFAR-10 dataset
train_dataset = nx.data.CIFAR10(root='./data', train=True, download=True)
test_dataset = nx.data.CIFAR10(root='./data', train=False, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Run the search
search.fit(train_loader)

# Get the best architecture and model
best_architecture = search.best_architecture()
best_model = search.best_model()
print(f"Best architecture: {best_architecture}")
print(f"Best score: {search.best_score()}")

# Evaluate the best model on the test set
best_model.eval()
test_accuracy = search.evaluate(best_model, test_loader)
print(f"Test accuracy: {test_accuracy}")

# Visualize the best architecture
search.visualize_architecture(best_architecture)
```

### Automated Model Selection

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU, Conv2d, MaxPool2d, Flatten
from neurenix.automl import AutoModelSelection

# Initialize Neurenix
nx.init()

# Create candidate models
models = [
    ('linear', Sequential(
        Flatten(),
        Linear(784, 10)
    )),
    ('mlp', Sequential(
        Flatten(),
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )),
    ('cnn', Sequential(
        Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Flatten(),
        Linear(32 * 7 * 7, 10)
    ))
]

# Create an AutoModelSelection instance
model_selection = AutoModelSelection(
    models=models,
    scoring='accuracy',
    cv=5,
    random_state=42
)

# Load MNIST dataset
train_dataset = nx.data.MNIST(root='./data', train=True, download=True)
test_dataset = nx.data.MNIST(root='./data', train=False, download=True)

# Extract data
X_train = train_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_train = train_dataset.targets
X_test = test_dataset.data.reshape(-1, 1, 28, 28) / 255.0
y_test = test_dataset.targets

# Run the model selection
model_selection.fit(X_train, y_train)

# Get the best model
best_model_name = model_selection.best_model_name()
best_model = model_selection.best_model()
print(f"Best model: {best_model_name}")
print(f"Best score: {model_selection.best_score()}")

# Evaluate the best model on the test set
best_model.eval()
predictions = best_model(X_test).argmax(dim=1)
accuracy = (predictions == y_test).float().mean().item()
print(f"Test accuracy: {accuracy}")

# Compare model performances
model_selection.plot_model_comparison()
```

### End-to-End AutoML Pipeline

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.automl import AutoPipeline, DataPreprocessing, FeatureSelection, BayesianOptimization

# Initialize Neurenix
nx.init()

# Create preprocessing steps
preprocessing = DataPreprocessing([
    ('normalize', nx.preprocessing.StandardScaler()),
    ('pca', nx.preprocessing.PCA(n_components=10))
])

# Create feature selection
feature_selection = FeatureSelection(
    estimator=Sequential(Linear(10, 1)),
    n_features_to_select=5,
    scoring='mse',
    cv=3
)

# Define model function with hyperparameters
def create_model(hidden_size=32, learning_rate=0.01):
    model = Sequential(
        Linear(5, hidden_size),
        ReLU(),
        Linear(hidden_size, 1)
    )
    return model

# Define parameter space
param_space = {
    'hidden_size': [16, 32, 64, 128],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}

# Create hyperparameter optimization
hpo = BayesianOptimization(
    model_fn=create_model,
    param_space=param_space,
    scoring='mse',
    n_trials=20,
    cv=5,
    random_state=42
)

# Create an AutoPipeline
pipeline = AutoPipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('feature_selection', feature_selection),
        ('hpo', hpo)
    ],
    scoring='mse',
    cv=5,
    random_state=42
)

# Generate synthetic data
X = nx.Tensor.randn((500, 20))
y = nx.Tensor.randn((500, 1))

# Split data into train and test sets
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
mse = ((predictions - y_test) ** 2).mean().item()
print(f"Test MSE: {mse}")

# Get the best model and parameters
best_model = pipeline.get_step('hpo').best_model()
best_params = pipeline.get_step('hpo').best_params()
print(f"Best parameters: {best_params}")

# Visualize the pipeline
pipeline.visualize()
```

## Conclusion

The AutoML module in Neurenix provides a comprehensive set of tools for automating the machine learning workflow, from data preprocessing to model deployment. With support for hyperparameter optimization, neural architecture search, model selection, and pipeline automation, the module enables developers to create high-performance machine learning solutions with minimal manual intervention.

Compared to other frameworks, Neurenix's AutoML module offers advantages in terms of API consistency, integration with the core framework, and optimization for edge devices. These features make Neurenix particularly well-suited for automated machine learning in resource-constrained environments and for applications requiring end-to-end automation.
