# Neuro-Symbolic AI API Documentation

## Overview

The Neuro-Symbolic AI module provides implementations of hybrid neuro-symbolic models that combine neural networks with symbolic reasoning systems. This integration leverages the strengths of both approaches: the pattern recognition and learning capabilities of neural networks, and the interpretability, reasoning, and generalization capabilities of symbolic systems.

Neuro-symbolic AI addresses key limitations of pure neural approaches, including interpretability, data efficiency, reasoning capabilities, and knowledge incorporation. By combining neural and symbolic components, these hybrid models can achieve better performance on tasks requiring both pattern recognition and logical reasoning, while providing more transparent and explainable results.

## Key Concepts

### Symbolic Reasoning

Symbolic reasoning uses formal logic and symbolic representations to perform inference:

- **Logic Programs**: Collections of rules and facts expressed in formal logic
- **Rule Sets**: Sets of logical rules for inference
- **Knowledge Bases**: Structured repositories of facts and rules
- **Symbolic Reasoners**: Systems that perform inference using symbolic representations

### Neural-Symbolic Integration

Neural-symbolic integration combines neural networks with symbolic reasoning:

- **Neural-Symbolic Models**: Hybrid models that integrate neural and symbolic components
- **Neural-Symbolic Loss**: Loss functions that incorporate symbolic knowledge
- **Neural-Symbolic Training**: Training procedures for hybrid models
- **Neural-Symbolic Inference**: Inference procedures that combine neural and symbolic reasoning

### Differentiable Logic

Differentiable logic enables the integration of logical reasoning with gradient-based learning:

- **Fuzzy Logic**: Logic that handles degrees of truth rather than binary truth values
- **Probabilistic Logic**: Logic that incorporates uncertainty through probabilities
- **Logic Tensors**: Tensor representations of logical expressions
- **Differentiable Logical Operations**: Logical operations that support gradient propagation

### Knowledge Distillation

Knowledge distillation transfers knowledge between models:

- **Symbolic Distillation**: Extracting symbolic knowledge from neural networks
- **Rule Extraction**: Deriving logical rules from trained neural networks
- **Symbolic Teachers**: Symbolic systems that guide neural network training

### Reasoning Types

Different types of reasoning are supported:

- **Deductive Reasoning**: Drawing conclusions from premises using logical rules
- **Inductive Reasoning**: Generalizing from specific instances to general principles
- **Abductive Reasoning**: Inferring the most likely explanation for observations
- **Constraint Satisfaction**: Finding solutions that satisfy a set of constraints

## API Reference

### Symbolic Reasoning

```python
neurenix.neuro_symbolic.SymbolicReasoner(
    engine: str = "prolog",
    config: Optional[Dict[str, Any]] = None
)
```

Creates a symbolic reasoning engine.

**Parameters:**
- `engine`: Type of reasoning engine ("prolog", "datalog", "answer_set")
- `config`: Configuration options for the reasoning engine

**Methods:**
- `add_rule(rule)`: Add a rule to the reasoner
- `add_fact(fact)`: Add a fact to the reasoner
- `query(query)`: Perform a query and get results
- `explain(result)`: Get an explanation for a result

**Example:**
```python
from neurenix.neuro_symbolic import SymbolicReasoner

# Create a symbolic reasoner
reasoner = SymbolicReasoner(engine="prolog")

# Add rules and facts
reasoner.add_rule("parent(X, Y) :- father(X, Y)")
reasoner.add_rule("parent(X, Y) :- mother(X, Y)")
reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
reasoner.add_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)")

reasoner.add_fact("father(john, mary)")
reasoner.add_fact("mother(mary, bob)")

# Perform a query
results = reasoner.query("ancestor(john, bob)")
print(f"Is John an ancestor of Bob? {results}")

# Get an explanation
explanation = reasoner.explain(results[0])
print(f"Explanation: {explanation}")
```

```python
neurenix.neuro_symbolic.LogicProgram(
    rules: List[str] = None,
    facts: List[str] = None
)
```

Represents a logic program with rules and facts.

**Parameters:**
- `rules`: List of logical rules
- `facts`: List of facts

**Methods:**
- `add_rule(rule)`: Add a rule to the program
- `add_fact(fact)`: Add a fact to the program
- `to_tensor()`: Convert the program to a tensor representation
- `from_tensor(tensor)`: Create a program from a tensor representation

```python
neurenix.neuro_symbolic.RuleSet(
    rules: List[str] = None
)
```

Represents a set of logical rules.

**Parameters:**
- `rules`: List of logical rules

**Methods:**
- `add_rule(rule)`: Add a rule to the set
- `remove_rule(rule)`: Remove a rule from the set
- `apply(facts)`: Apply the rules to a set of facts
- `to_neural(embedding_size)`: Convert the rules to a neural representation

```python
neurenix.neuro_symbolic.SymbolicKnowledgeBase(
    facts: List[str] = None,
    rules: List[str] = None
)
```

Represents a symbolic knowledge base with facts and rules.

**Parameters:**
- `facts`: List of facts
- `rules`: List of rules

**Methods:**
- `add_fact(fact)`: Add a fact to the knowledge base
- `add_rule(rule)`: Add a rule to the knowledge base
- `query(query)`: Query the knowledge base
- `to_neural(embedding_size)`: Convert the knowledge base to a neural representation

### Neural-Symbolic Integration

```python
neurenix.neuro_symbolic.NeuralSymbolicModel(
    neural_component: neurenix.nn.Module,
    symbolic_component: Union[SymbolicReasoner, LogicProgram, RuleSet],
    integration_method: str = "sequential"
)
```

Creates a neural-symbolic model that integrates neural and symbolic components.

**Parameters:**
- `neural_component`: Neural network component
- `symbolic_component`: Symbolic reasoning component
- `integration_method`: Method for integrating the components ("sequential", "parallel", "iterative")

**Methods:**
- `forward(x)`: Process input through the model
- `reason(x)`: Apply symbolic reasoning to the input
- `learn(x, y)`: Update the model based on input-output pairs
- `explain(x)`: Generate an explanation for the model's output

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.neuro_symbolic import NeuralSymbolicModel, SymbolicReasoner

# Create a neural component
neural_component = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Create a symbolic component
symbolic_component = SymbolicReasoner(engine="prolog")
symbolic_component.add_rule("category(X, animal) :- has_fur(X)")
symbolic_component.add_rule("category(X, bird) :- has_wings(X), has_feathers(X)")

# Create a neural-symbolic model
model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="sequential"
)

# Process input
input_tensor = nx.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
output = model(input_tensor)

# Get explanation
explanation = model.explain(input_tensor)
print(f"Explanation: {explanation}")
```

```python
neurenix.neuro_symbolic.NeuralSymbolicLoss(
    neural_loss: Callable,
    symbolic_loss: Callable,
    alpha: float = 0.5
)
```

Creates a loss function that combines neural and symbolic losses.

**Parameters:**
- `neural_loss`: Loss function for the neural component
- `symbolic_loss`: Loss function for the symbolic component
- `alpha`: Weight for the neural loss (1-alpha is the weight for the symbolic loss)

**Methods:**
- `forward(y_pred, y_true, symbolic_constraints)`: Compute the combined loss

```python
neurenix.neuro_symbolic.NeuralSymbolicTrainer(
    model: NeuralSymbolicModel,
    optimizer: neurenix.optim.Optimizer,
    loss_fn: Union[NeuralSymbolicLoss, Callable],
    symbolic_constraints: List[str] = None
)
```

Creates a trainer for neural-symbolic models.

**Parameters:**
- `model`: Neural-symbolic model to train
- `optimizer`: Optimizer for updating model parameters
- `loss_fn`: Loss function for training
- `symbolic_constraints`: Symbolic constraints to enforce during training

**Methods:**
- `train(data_loader, epochs)`: Train the model
- `evaluate(data_loader)`: Evaluate the model
- `save(path)`: Save the model
- `load(path)`: Load the model

```python
neurenix.neuro_symbolic.NeuralSymbolicInference(
    model: NeuralSymbolicModel,
    inference_method: str = "joint"
)
```

Creates an inference engine for neural-symbolic models.

**Parameters:**
- `model`: Neural-symbolic model to perform inference with
- `inference_method`: Method for inference ("neural", "symbolic", "joint")

**Methods:**
- `infer(x)`: Perform inference on the input
- `explain(x)`: Generate an explanation for the inference
- `confidence(x)`: Get confidence scores for the inference

### Differentiable Logic

```python
neurenix.neuro_symbolic.DifferentiableLogic(
    logic_type: str = "fuzzy"
)
```

Creates a differentiable logic system.

**Parameters:**
- `logic_type`: Type of logic to use ("fuzzy", "probabilistic")

**Methods:**
- `and_op(x, y)`: Differentiable AND operation
- `or_op(x, y)`: Differentiable OR operation
- `not_op(x)`: Differentiable NOT operation
- `implies_op(x, y)`: Differentiable implication operation

**Example:**
```python
import neurenix as nx
from neurenix.neuro_symbolic import DifferentiableLogic

# Create a differentiable logic system
logic = DifferentiableLogic(logic_type="fuzzy")

# Create tensors representing truth values
a = nx.Tensor([0.7])  # 70% true
b = nx.Tensor([0.4])  # 40% true

# Apply logical operations
c = logic.and_op(a, b)  # Fuzzy AND
d = logic.or_op(a, b)   # Fuzzy OR
e = logic.not_op(a)     # Fuzzy NOT
f = logic.implies_op(a, b)  # Fuzzy implication

print(f"a AND b = {c.item()}")
print(f"a OR b = {d.item()}")
print(f"NOT a = {e.item()}")
print(f"a IMPLIES b = {f.item()}")
```

```python
neurenix.neuro_symbolic.FuzzyLogic()
```

Creates a fuzzy logic system with differentiable operations.

**Methods:**
- `and_op(x, y)`: Fuzzy AND operation (typically min or product)
- `or_op(x, y)`: Fuzzy OR operation (typically max or probabilistic sum)
- `not_op(x)`: Fuzzy NOT operation (typically 1-x)
- `implies_op(x, y)`: Fuzzy implication operation

```python
neurenix.neuro_symbolic.ProbabilisticLogic()
```

Creates a probabilistic logic system with differentiable operations.

**Methods:**
- `and_op(x, y)`: Probabilistic AND operation (typically product)
- `or_op(x, y)`: Probabilistic OR operation (typically noisy-or)
- `not_op(x)`: Probabilistic NOT operation (typically 1-x)
- `implies_op(x, y)`: Probabilistic implication operation

```python
neurenix.neuro_symbolic.LogicTensor(
    data: Union[neurenix.Tensor, List, np.ndarray],
    logic_system: Union[DifferentiableLogic, FuzzyLogic, ProbabilisticLogic] = None
)
```

Creates a tensor with logical operations.

**Parameters:**
- `data`: Data for the tensor
- `logic_system`: Logic system to use for operations

**Methods:**
- `and_(other)`: Logical AND with another tensor
- `or_(other)`: Logical OR with another tensor
- `not_()`: Logical NOT
- `implies(other)`: Logical implication

### Knowledge Distillation

```python
neurenix.neuro_symbolic.KnowledgeDistillation(
    teacher_model: neurenix.nn.Module,
    student_model: neurenix.nn.Module,
    temperature: float = 1.0
)
```

Creates a knowledge distillation system.

**Parameters:**
- `teacher_model`: Model to distill knowledge from
- `student_model`: Model to distill knowledge to
- `temperature`: Temperature for softening probability distributions

**Methods:**
- `distill(data_loader, epochs)`: Distill knowledge from teacher to student
- `evaluate(data_loader)`: Evaluate the student model
- `save_student(path)`: Save the student model

```python
neurenix.neuro_symbolic.SymbolicDistillation(
    neural_model: neurenix.nn.Module,
    symbolic_model: Union[SymbolicReasoner, LogicProgram, RuleSet],
    distillation_method: str = "rule_extraction"
)
```

Creates a system for distilling knowledge between neural and symbolic models.

**Parameters:**
- `neural_model`: Neural model
- `symbolic_model`: Symbolic model
- `distillation_method`: Method for distillation ("rule_extraction", "knowledge_transfer")

**Methods:**
- `neural_to_symbolic(data_loader)`: Distill knowledge from neural to symbolic model
- `symbolic_to_neural(data_loader)`: Distill knowledge from symbolic to neural model
- `evaluate(data_loader)`: Evaluate the models

```python
neurenix.neuro_symbolic.RuleExtraction(
    neural_model: neurenix.nn.Module,
    extraction_method: str = "decision_tree"
)
```

Creates a system for extracting rules from neural networks.

**Parameters:**
- `neural_model`: Neural model to extract rules from
- `extraction_method`: Method for rule extraction ("decision_tree", "m_of_n", "eclectic")

**Methods:**
- `extract_rules(data_loader)`: Extract rules from the neural model
- `get_ruleset()`: Get the extracted rules as a RuleSet
- `evaluate(data_loader)`: Evaluate the extracted rules

```python
neurenix.neuro_symbolic.SymbolicTeacher(
    symbolic_model: Union[SymbolicReasoner, LogicProgram, RuleSet],
    teaching_method: str = "data_generation"
)
```

Creates a symbolic teacher for guiding neural network training.

**Parameters:**
- `symbolic_model`: Symbolic model to use as a teacher
- `teaching_method`: Method for teaching ("data_generation", "loss_guidance")

**Methods:**
- `generate_data(num_samples)`: Generate training data from the symbolic model
- `guide_training(neural_model, data_loader, epochs)`: Guide the training of a neural model
- `evaluate(neural_model, data_loader)`: Evaluate the neural model

### Reasoning

```python
neurenix.neuro_symbolic.ConstraintSatisfaction(
    constraints: List[str],
    variables: List[str],
    domains: Dict[str, List]
)
```

Creates a constraint satisfaction problem solver.

**Parameters:**
- `constraints`: List of constraints
- `variables`: List of variables
- `domains`: Dictionary mapping variables to their domains

**Methods:**
- `solve()`: Solve the constraint satisfaction problem
- `add_constraint(constraint)`: Add a constraint
- `add_variable(variable, domain)`: Add a variable with its domain

```python
neurenix.neuro_symbolic.LogicalInference(
    knowledge_base: Union[SymbolicKnowledgeBase, LogicProgram, RuleSet],
    inference_method: str = "backward_chaining"
)
```

Creates a logical inference engine.

**Parameters:**
- `knowledge_base`: Knowledge base to perform inference on
- `inference_method`: Method for inference ("forward_chaining", "backward_chaining", "resolution")

**Methods:**
- `infer(query)`: Perform inference on the query
- `explain(result)`: Generate an explanation for the result
- `add_knowledge(knowledge)`: Add knowledge to the knowledge base

```python
neurenix.neuro_symbolic.AbductiveReasoning(
    knowledge_base: Union[SymbolicKnowledgeBase, LogicProgram, RuleSet],
    cost_function: Callable = None
)
```

Creates an abductive reasoning engine.

**Parameters:**
- `knowledge_base`: Knowledge base to perform reasoning on
- `cost_function`: Function for evaluating the cost of explanations

**Methods:**
- `explain(observation)`: Generate the best explanation for an observation
- `rank_explanations(observation)`: Rank possible explanations for an observation
- `add_knowledge(knowledge)`: Add knowledge to the knowledge base

```python
neurenix.neuro_symbolic.DeductiveReasoning(
    knowledge_base: Union[SymbolicKnowledgeBase, LogicProgram, RuleSet]
)
```

Creates a deductive reasoning engine.

**Parameters:**
- `knowledge_base`: Knowledge base to perform reasoning on

**Methods:**
- `infer(premises)`: Draw conclusions from premises
- `verify(conclusion, premises)`: Verify if a conclusion follows from premises
- `add_knowledge(knowledge)`: Add knowledge to the knowledge base

```python
neurenix.neuro_symbolic.InductiveReasoning(
    knowledge_base: Union[SymbolicKnowledgeBase, LogicProgram, RuleSet],
    induction_method: str = "foil"
)
```

Creates an inductive reasoning engine.

**Parameters:**
- `knowledge_base`: Knowledge base to perform reasoning on
- `induction_method`: Method for induction ("foil", "progol", "aleph")

**Methods:**
- `learn_rules(examples)`: Learn rules from examples
- `evaluate(test_examples)`: Evaluate learned rules on test examples
- `add_knowledge(knowledge)`: Add knowledge to the knowledge base

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Neuro-Symbolic Integration** | Native support | Limited support through custom implementations |
| **Differentiable Logic** | Built-in implementations | Requires custom implementation |
| **Symbolic Reasoning** | Multiple reasoning engines | No built-in support |
| **Knowledge Distillation** | Comprehensive API | Basic implementation in TF-Keras |
| **Rule Extraction** | Multiple methods | No built-in support |
| **Logical Inference** | Multiple inference methods | No built-in support |

Neurenix provides comprehensive support for neuro-symbolic AI, with built-in implementations of various integration methods, differentiable logic systems, and reasoning engines. TensorFlow lacks native support for most neuro-symbolic features, requiring custom implementations for these capabilities.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Neuro-Symbolic Integration** | Native support | Limited support through third-party libraries |
| **Differentiable Logic** | Built-in implementations | Requires custom implementation |
| **Symbolic Reasoning** | Multiple reasoning engines | No built-in support |
| **Knowledge Distillation** | Comprehensive API | Basic implementation in third-party libraries |
| **Rule Extraction** | Multiple methods | No built-in support |
| **Logical Inference** | Multiple inference methods | No built-in support |

While PyTorch provides flexibility for implementing neuro-symbolic models, it lacks native support for most neuro-symbolic features. Neurenix offers a more comprehensive and integrated approach to neuro-symbolic AI, with built-in support for various integration methods, differentiable logic, and reasoning capabilities.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Neuro-Symbolic Integration** | Comprehensive support | No support |
| **Differentiable Logic** | Built-in implementations | No support |
| **Symbolic Reasoning** | Multiple reasoning engines | Limited rule-based models |
| **Knowledge Distillation** | Comprehensive API | No support |
| **Rule Extraction** | Multiple methods | Decision tree extraction only |
| **Logical Inference** | Multiple inference methods | No support |

Scikit-Learn provides some rule-based models like decision trees but lacks support for neuro-symbolic integration, differentiable logic, and most reasoning capabilities. Neurenix fills this gap with its comprehensive neuro-symbolic module, enabling the development of hybrid models that combine neural and symbolic approaches.

## Best Practices

### Integration Method Selection

Choose the appropriate integration method based on your task:

```python
from neurenix.neuro_symbolic import NeuralSymbolicModel

# For tasks where neural processing should precede symbolic reasoning
sequential_model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="sequential"
)

# For tasks requiring iterative refinement between neural and symbolic components
iterative_model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="iterative"
)

# For tasks where neural and symbolic processing should happen in parallel
parallel_model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="parallel"
)
```

### Balancing Neural and Symbolic Components

Balance the contributions of neural and symbolic components:

```python
from neurenix.neuro_symbolic import NeuralSymbolicLoss
from neurenix.nn import MSELoss

# Create a loss function that balances neural and symbolic components
# More weight on neural component (0.7) for data-driven tasks
neural_dominant_loss = NeuralSymbolicLoss(
    neural_loss=MSELoss(),
    symbolic_loss=symbolic_loss_fn,
    alpha=0.7
)

# Equal weight for balanced tasks
balanced_loss = NeuralSymbolicLoss(
    neural_loss=MSELoss(),
    symbolic_loss=symbolic_loss_fn,
    alpha=0.5
)

# More weight on symbolic component for knowledge-rich tasks
symbolic_dominant_loss = NeuralSymbolicLoss(
    neural_loss=MSELoss(),
    symbolic_loss=symbolic_loss_fn,
    alpha=0.3
)
```

### Knowledge Incorporation

Incorporate domain knowledge effectively:

```python
from neurenix.neuro_symbolic import SymbolicReasoner, NeuralSymbolicModel

# Create a symbolic reasoner with domain knowledge
reasoner = SymbolicReasoner(engine="prolog")

# Add domain-specific rules
reasoner.add_rule("is_valid_transaction(T) :- amount(T, A), A > 0, A < 10000")
reasoner.add_rule("is_suspicious(T) :- amount(T, A), A > 9000")
reasoner.add_rule("requires_review(T) :- is_suspicious(T)")
reasoner.add_rule("requires_review(T) :- frequency(T, F), F > 10")

# Create a neural-symbolic model that leverages this knowledge
model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=reasoner,
    integration_method="sequential"
)
```

### Explainability

Leverage the explainability of neuro-symbolic models:

```python
from neurenix.neuro_symbolic import NeuralSymbolicModel, RuleExtraction

# Create a neural-symbolic model
model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="sequential"
)

# Extract rules from the model for explainability
rule_extractor = RuleExtraction(
    neural_model=model.neural_component,
    extraction_method="decision_tree"
)

# Extract rules based on model behavior
rules = rule_extractor.extract_rules(data_loader)

# Get explanations for specific predictions
explanation = model.explain(input_data)
```

## Tutorials

### Hybrid Classification with Neural-Symbolic Integration

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from neurenix.optim import Adam
from neurenix.data import DataLoader
from neurenix.neuro_symbolic import (
    SymbolicReasoner,
    NeuralSymbolicModel,
    NeuralSymbolicLoss,
    NeuralSymbolicTrainer
)

# Create a neural component for feature extraction
neural_component = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Create a symbolic component with domain knowledge
symbolic_component = SymbolicReasoner(engine="prolog")
symbolic_component.add_rule("class(X, 0) :- feature1(X, F), F < 0.3")
symbolic_component.add_rule("class(X, 1) :- feature2(X, F), F > 0.7")
symbolic_component.add_rule("class(X, 2) :- feature3(X, F), feature4(X, G), F > 0.5, G > 0.5")
symbolic_component.add_rule("class(X, 3) :- feature5(X, F), F > 0.8")
symbolic_component.add_rule("class(X, 4) :- not(class(X, 0)), not(class(X, 1)), not(class(X, 2)), not(class(X, 3))")

# Define a symbolic loss function
def symbolic_loss(y_pred, y_true, symbolic_constraints):
    # Calculate loss based on violation of symbolic constraints
    # This is a simplified example
    return nx.mean(symbolic_constraints)

# Create a neural-symbolic model
model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=symbolic_component,
    integration_method="sequential"
)

# Create a combined loss function
loss_fn = NeuralSymbolicLoss(
    neural_loss=CrossEntropyLoss(),
    symbolic_loss=symbolic_loss,
    alpha=0.7  # 70% weight on neural loss, 30% on symbolic loss
)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a trainer
trainer = NeuralSymbolicTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    symbolic_constraints=["class(X, Y) :- neural_output(X, Y)"]
)

# Assume we have a DataLoader for training data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train the model
trainer.train(train_loader, epochs=10)

# Evaluate the model
accuracy = trainer.evaluate(test_loader)
print(f"Test accuracy: {accuracy:.4f}")

# Get explanations for predictions
for inputs, targets in test_loader:
    outputs = model(inputs)
    explanations = model.explain(inputs)
    
    for i in range(len(inputs)):
        print(f"Input: {inputs[i]}")
        print(f"Prediction: {outputs[i].argmax().item()}")
        print(f"Explanation: {explanations[i]}")
        print()
```

### Rule Extraction from Neural Networks

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.data import DataLoader
from neurenix.neuro_symbolic import RuleExtraction

# Create and train a neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 20),
    ReLU(),
    Linear(20, 5)
)

# Assume the model is already trained
# ...

# Create a rule extractor
rule_extractor = RuleExtraction(
    neural_model=model,
    extraction_method="decision_tree"
)

# Assume we have a DataLoader for the data
data_loader = DataLoader(dataset, batch_size=32)

# Extract rules from the neural network
rules = rule_extractor.extract_rules(data_loader)

# Get the extracted ruleset
ruleset = rule_extractor.get_ruleset()

# Print the extracted rules
print("Extracted Rules:")
for rule in ruleset.rules:
    print(rule)

# Evaluate the extracted rules
accuracy = rule_extractor.evaluate(data_loader)
print(f"Rule accuracy: {accuracy:.4f}")

# Use the extracted rules for inference
for inputs, targets in data_loader:
    # Get predictions from the original neural network
    neural_predictions = model(inputs).argmax(dim=1)
    
    # Get predictions from the extracted rules
    rule_predictions = []
    for input_tensor in inputs:
        # Convert tensor to a format suitable for rule application
        input_dict = {f"feature{i}": input_tensor[i].item() for i in range(len(input_tensor))}
        
        # Apply rules
        prediction = ruleset.apply(input_dict)
        rule_predictions.append(prediction)
    
    # Compare predictions
    for i in range(len(inputs)):
        print(f"Input: {inputs[i]}")
        print(f"Neural prediction: {neural_predictions[i].item()}")
        print(f"Rule prediction: {rule_predictions[i]}")
        print(f"Target: {targets[i].item()}")
        print()
```

### Constraint Satisfaction with Neural Guidance

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.neuro_symbolic import ConstraintSatisfaction, NeuralSymbolicModel

# Create a neural network for predicting variable assignments
neural_component = Sequential(
    Linear(20, 40),
    ReLU(),
    Linear(40, 10)
)

# Define a constraint satisfaction problem
csp = ConstraintSatisfaction(
    constraints=[
        "X + Y + Z <= 10",
        "X >= 2",
        "Y >= 3",
        "Z >= 1",
        "X + Y >= 6"
    ],
    variables=["X", "Y", "Z"],
    domains={
        "X": list(range(10)),
        "Y": list(range(10)),
        "Z": list(range(10))
    }
)

# Create a neural-symbolic model
model = NeuralSymbolicModel(
    neural_component=neural_component,
    symbolic_component=csp,
    integration_method="iterative"
)

# Define a function to encode the CSP state as input to the neural network
def encode_csp_state(csp):
    # This is a simplified example
    # In practice, you would use a more sophisticated encoding
    state = []
    for var in csp.variables:
        for val in csp.domains[var]:
            state.append(1.0 if val in csp.current_assignment.get(var, []) else 0.0)
    return nx.Tensor(state)

# Define a function to decode neural network output to variable assignments
def decode_neural_output(output, csp):
    # This is a simplified example
    assignments = {}
    idx = 0
    for var in csp.variables:
        var_values = []
        for val in csp.domains[var]:
            if output[idx] > 0.5:  # Threshold for assignment
                var_values.append(val)
            idx += 1
        if var_values:
            assignments[var] = var_values[0]  # Take the first value above threshold
    return assignments

# Solve the CSP with neural guidance
def solve_with_neural_guidance(model, csp, max_iterations=100):
    for i in range(max_iterations):
        # Encode current CSP state
        state = encode_csp_state(csp)
        
        # Get neural prediction
        neural_output = model.neural_component(state)
        
        # Decode neural output to assignments
        suggested_assignments = decode_neural_output(neural_output, csp)
        
        # Update CSP with neural suggestions
        for var, val in suggested_assignments.items():
            csp.assign(var, val)
        
        # Check if CSP is solved
        if csp.is_solved():
            return csp.current_assignment
        
        # If not solved, backtrack and continue
        csp.backtrack()
    
    # If max iterations reached, solve without neural guidance
    return csp.solve()

# Solve the CSP
solution = solve_with_neural_guidance(model, csp)
print("Solution:")
for var, val in solution.items():
    print(f"{var} = {val}")
```

This documentation provides a comprehensive overview of the Neuro-Symbolic AI module in Neurenix, including key concepts, API reference, framework comparisons, best practices, and tutorials for developing hybrid neuro-symbolic models.
