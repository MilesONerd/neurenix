# Fuzzy Logic API Documentation

## Overview

The Fuzzy Logic module provides implementations of fuzzy logic systems, including fuzzy sets, membership functions, fuzzy rules, and fuzzy inference systems. Fuzzy logic extends classical boolean logic to handle the concept of partial truth, where the truth value can range between completely true and completely false, making it particularly useful for modeling uncertainty and imprecision in decision-making systems.

Unlike classical logic, which requires precise inputs and produces discrete outputs, fuzzy logic accepts imprecise inputs and follows a more human-like reasoning process to produce outputs that can be precise or imprecise. This makes fuzzy logic especially valuable in control systems, decision support systems, pattern recognition, and other applications where traditional binary logic may be too rigid.

## Key Concepts

### Fuzzy Sets

A fuzzy set is a set whose elements have degrees of membership, as opposed to classical sets where an element either belongs to the set or does not. The degree of membership is represented by a value between 0 and 1, where 0 means the element is not in the set, 1 means it is fully in the set, and values in between represent partial membership.

Neurenix supports various types of fuzzy sets:
- **Triangular Sets**: Defined by three points forming a triangle
- **Trapezoidal Sets**: Defined by four points forming a trapezoid
- **Gaussian Sets**: Based on the Gaussian distribution
- **Bell Sets**: Based on the generalized bell function
- **Sigmoid Sets**: Based on the sigmoid function

### Fuzzy Variables

Fuzzy variables are variables whose values are fuzzy sets. They are used to represent linguistic concepts like "temperature" with linguistic values like "cold," "warm," and "hot," each represented by a fuzzy set.

- **Fuzzy Variable**: A variable that can take fuzzy set values
- **Linguistic Variable**: A fuzzy variable with linguistic terms as its values

### Fuzzy Rules

Fuzzy rules define the relationship between input and output fuzzy variables using if-then statements. For example, "If temperature is hot, then fan speed is high."

- **Fuzzy Rule**: A single if-then rule
- **Fuzzy Rule Set**: A collection of fuzzy rules

### Fuzzy Inference Systems

A fuzzy inference system (FIS) is a system that uses fuzzy logic and fuzzy rules to map inputs to outputs. The process typically involves:

1. **Fuzzification**: Converting crisp input values to fuzzy values
2. **Rule Evaluation**: Applying fuzzy rules to determine output fuzzy sets
3. **Aggregation**: Combining the output fuzzy sets
4. **Defuzzification**: Converting the aggregated fuzzy set to a crisp output value

Neurenix supports several types of fuzzy inference systems:
- **Mamdani System**: Uses fuzzy sets for both inputs and outputs
- **Sugeno System**: Uses fuzzy sets for inputs but crisp functions for outputs
- **Tsukamoto System**: Uses monotonic membership functions for outputs

## API Reference

### Fuzzy Sets

```python
neurenix.fuzzy.FuzzySet(universe, membership_function)
```

Base class for all fuzzy sets.

**Parameters:**
- `universe`: The universe of discourse (range of possible values)
- `membership_function`: Function that maps values to membership degrees

**Methods:**
- `membership(x)`: Returns the membership degree of value x
- `complement()`: Returns the complement of the fuzzy set
- `intersection(other)`: Returns the intersection with another fuzzy set
- `union(other)`: Returns the union with another fuzzy set

```python
neurenix.fuzzy.TriangularSet(universe, a, b, c)
```

Creates a triangular fuzzy set.

**Parameters:**
- `universe`: The universe of discourse
- `a`: Left point of the triangle
- `b`: Peak point of the triangle
- `c`: Right point of the triangle

```python
neurenix.fuzzy.TrapezoidalSet(universe, a, b, c, d)
```

Creates a trapezoidal fuzzy set.

**Parameters:**
- `universe`: The universe of discourse
- `a`: Left point of the trapezoid
- `b`: Left shoulder point of the trapezoid
- `c`: Right shoulder point of the trapezoid
- `d`: Right point of the trapezoid

```python
neurenix.fuzzy.GaussianSet(universe, mean, sigma)
```

Creates a Gaussian fuzzy set.

**Parameters:**
- `universe`: The universe of discourse
- `mean`: Mean of the Gaussian distribution
- `sigma`: Standard deviation of the Gaussian distribution

### Fuzzy Variables

```python
neurenix.fuzzy.FuzzyVariable(name, universe)
```

Represents a fuzzy variable.

**Parameters:**
- `name`: Name of the variable
- `universe`: Universe of discourse for the variable

**Methods:**
- `add_set(name, fuzzy_set)`: Adds a fuzzy set to the variable
- `fuzzify(value)`: Returns the membership degrees for a crisp value
- `get_set(name)`: Returns a fuzzy set by name

```python
neurenix.fuzzy.LinguisticVariable(name, universe, terms=None)
```

Represents a linguistic variable with linguistic terms.

**Parameters:**
- `name`: Name of the variable
- `universe`: Universe of discourse for the variable
- `terms`: Dictionary mapping term names to fuzzy sets

### Fuzzy Rules

```python
neurenix.fuzzy.FuzzyRule(antecedent, consequent)
```

Represents a fuzzy rule.

**Parameters:**
- `antecedent`: Dictionary mapping input variables to terms
- `consequent`: Dictionary mapping output variables to terms

```python
neurenix.fuzzy.FuzzyRuleSet()
```

Represents a set of fuzzy rules.

**Methods:**
- `add_rule(rule)`: Adds a rule to the set
- `evaluate(inputs)`: Evaluates all rules for given inputs

### Fuzzy Inference Systems

```python
neurenix.fuzzy.FuzzyInferenceSystem(name)
```

Base class for fuzzy inference systems.

**Parameters:**
- `name`: Name of the inference system

**Methods:**
- `add_input_variable(variable)`: Adds an input variable
- `add_output_variable(variable)`: Adds an output variable
- `add_rule(rule)`: Adds a rule
- `compute(inputs)`: Computes outputs for given inputs

```python
neurenix.fuzzy.MamdaniSystem(name, and_operator='min', or_operator='max', implication='min', aggregation='max', defuzzification='centroid')
```

Implements the Mamdani fuzzy inference system.

```python
neurenix.fuzzy.SugenoSystem(name, and_operator='min', or_operator='max', defuzzification='weighted_average')
```

Implements the Sugeno fuzzy inference system.

```python
neurenix.fuzzy.TsukamotoSystem(name, and_operator='min', or_operator='max', defuzzification='weighted_average')
```

Implements the Tsukamoto fuzzy inference system.

### Defuzzification Methods

```python
neurenix.fuzzy.centroid(fuzzy_set)
```

Computes the centroid (center of gravity) of a fuzzy set.

```python
neurenix.fuzzy.bisector(fuzzy_set)
```

Computes the bisector of a fuzzy set.

```python
neurenix.fuzzy.mean_of_maximum(fuzzy_set)
```

Computes the mean of maximum of a fuzzy set.

```python
neurenix.fuzzy.smallest_of_maximum(fuzzy_set)
```

Computes the smallest of maximum of a fuzzy set.

```python
neurenix.fuzzy.largest_of_maximum(fuzzy_set)
```

Computes the largest of maximum of a fuzzy set.

```python
neurenix.fuzzy.weighted_average(fuzzy_sets, weights)
```

Computes the weighted average of fuzzy sets (used in Sugeno systems).

## Framework Comparison

### Neurenix vs. scikit-fuzzy

| Feature | Neurenix | scikit-fuzzy |
|---------|----------|--------------|
| **API Design** | Unified, object-oriented API | Object-oriented but less cohesive API |
| **Fuzzy Set Types** | Comprehensive (Triangular, Trapezoidal, Gaussian, Bell, Sigmoid) | Basic (Triangular, Trapezoidal, Gaussian) |
| **Inference Systems** | Multiple (Mamdani, Sugeno, Tsukamoto) | Primarily Mamdani |
| **Defuzzification Methods** | Comprehensive | Limited |
| **Integration with Core Framework** | Seamless integration with Neurenix | Standalone library |
| **Performance** | Optimized for large-scale applications | Good for small to medium applications |

Neurenix provides a more comprehensive fuzzy logic solution compared to scikit-fuzzy, with support for more fuzzy set types, inference systems, and defuzzification methods.

### Neurenix vs. PyFuzzy

| Feature | Neurenix | PyFuzzy |
|---------|----------|---------|
| **API Design** | Modern, object-oriented API | Older, less intuitive API |
| **Fuzzy Set Types** | Comprehensive | Basic |
| **Inference Systems** | Multiple | Primarily Mamdani |
| **Defuzzification Methods** | Comprehensive | Limited |
| **Integration with Core Framework** | Seamless integration with Neurenix | Standalone library |
| **Performance** | Optimized for large-scale applications | Suitable for small applications |

Neurenix offers a more modern and comprehensive fuzzy logic implementation compared to PyFuzzy, with better performance, more features, and better integration with other machine learning techniques.

### Neurenix vs. TensorFlow Fuzzy Logic

| Feature | Neurenix | TensorFlow Fuzzy Logic |
|---------|----------|------------------------|
| **API Design** | Unified, object-oriented API | Low-level, tensor-based API |
| **Fuzzy Set Types** | Comprehensive | Basic |
| **Inference Systems** | Multiple | Primarily Mamdani |
| **Defuzzification Methods** | Comprehensive | Limited |
| **Integration with Core Framework** | Seamless integration with Neurenix | Integrated with TensorFlow |
| **Performance** | Optimized for general applications | Optimized for GPU acceleration |

While TensorFlow's fuzzy logic implementation benefits from GPU acceleration, Neurenix provides a more comprehensive and user-friendly fuzzy logic solution.

## Best Practices

### Designing Fuzzy Sets

When designing fuzzy sets, consider the following:

1. **Coverage**: Ensure that the universe of discourse is adequately covered by fuzzy sets
2. **Overlap**: Ensure sufficient overlap between adjacent fuzzy sets
3. **Symmetry**: Use symmetric fuzzy sets when appropriate
4. **Normalization**: Ensure that each fuzzy set has at least one element with membership degree 1

### Rule Design

When designing fuzzy rules, consider the following:

1. **Completeness**: Ensure that rules cover all possible input combinations
2. **Consistency**: Avoid contradictory rules
3. **Simplicity**: Keep rules simple and intuitive
4. **Modularity**: Group related rules together

### Choosing the Right Inference System

Different inference systems are suitable for different applications:

1. **Mamdani**: Intuitive, good for human-readable systems, but computationally intensive
2. **Sugeno**: Computationally efficient, good for optimization and control, but less intuitive
3. **Tsukamoto**: Good for monotonic systems, computationally efficient

## Tutorials

### Temperature Control System

```python
import neurenix as nx
import numpy as np

# Create universes of discourse
temp_universe = np.linspace(0, 100, 1000)
humidity_universe = np.linspace(0, 100, 1000)
fan_universe = np.linspace(0, 100, 1000)

# Create linguistic variables
temperature = nx.fuzzy.LinguisticVariable("Temperature", temp_universe)
temperature.add_term("Cold", nx.fuzzy.TriangularSet(temp_universe, 0, 0, 50))
temperature.add_term("Warm", nx.fuzzy.TriangularSet(temp_universe, 0, 50, 100))
temperature.add_term("Hot", nx.fuzzy.TriangularSet(temp_universe, 50, 100, 100))

humidity = nx.fuzzy.LinguisticVariable("Humidity", humidity_universe)
humidity.add_term("Dry", nx.fuzzy.TriangularSet(humidity_universe, 0, 0, 50))
humidity.add_term("Normal", nx.fuzzy.TriangularSet(humidity_universe, 0, 50, 100))
humidity.add_term("Humid", nx.fuzzy.TriangularSet(humidity_universe, 50, 100, 100))

fan_speed = nx.fuzzy.LinguisticVariable("FanSpeed", fan_universe)
fan_speed.add_term("Low", nx.fuzzy.TriangularSet(fan_universe, 0, 0, 50))
fan_speed.add_term("Medium", nx.fuzzy.TriangularSet(fan_universe, 0, 50, 100))
fan_speed.add_term("High", nx.fuzzy.TriangularSet(fan_universe, 50, 100, 100))

# Create a Mamdani fuzzy inference system
fis = nx.fuzzy.MamdaniSystem("FanController")

# Add variables
fis.add_input_variable(temperature)
fis.add_input_variable(humidity)
fis.add_output_variable(fan_speed)

# Add rules
fis.add_rule(nx.fuzzy.FuzzyRule(
    antecedent={"Temperature": "Cold", "Humidity": "Dry"},
    consequent={"FanSpeed": "Low"}
))

fis.add_rule(nx.fuzzy.FuzzyRule(
    antecedent={"Temperature": "Hot", "Humidity": "Humid"},
    consequent={"FanSpeed": "High"}
))

# Compute output for specific inputs
inputs = {"Temperature": 75, "Humidity": 65}
output = fis.compute(inputs)

print(f"Fan Speed: {output['FanSpeed']:.2f}%")
```

This tutorial demonstrates how to create a simple temperature control system using fuzzy logic. The system takes temperature and humidity as inputs and produces fan speed as output.
