# Contributing to Neurenix

Thank you for your interest in contributing to Neurenix! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the Neurenix Code of Conduct. Please report any unacceptable behavior to the project maintainers.

## How to Contribute

There are many ways to contribute to Neurenix:

1. **Reporting Bugs**: If you find a bug, please create an issue in the GitHub repository with a detailed description of the bug, steps to reproduce it, and your environment.

2. **Suggesting Enhancements**: If you have ideas for new features or improvements, please create an issue in the GitHub repository with a detailed description of your suggestion.

3. **Contributing Code**: If you want to contribute code, please follow the guidelines below.

4. **Improving Documentation**: If you find errors or gaps in the documentation, please submit a pull request with your improvements.

5. **Reviewing Pull Requests**: Reviewing pull requests from other contributors is a valuable way to contribute to the project.

## Development Setup

To set up your development environment:

1. **Fork the Repository**: Fork the Neurenix repository on GitHub.

2. **Clone Your Fork**: Clone your fork to your local machine.

   ```bash
   git clone https://github.com/your-username/neurenix.git
   cd framework
   ```

3. **Install Dependencies**: Install the required dependencies.

   ```bash
   # For Python components
   pip install -e ".[dev]"
   
   # For Rust components
   cd src/phynexus/rust
   cargo build
   
   # For C++ components
   cd src/phynexus/cpp
   mkdir build && cd build
   cmake .. && make
   
   # For Go components
   cd src/distributed/go
   go build
   ```

4. **Create a Branch**: Create a branch for your changes.

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Coding Guidelines

### General Guidelines

- Write clean, readable, and maintainable code.
- Follow the existing code style and conventions.
- Write comprehensive tests for your code.
- Document your code with docstrings and comments.

### Python Guidelines

- Follow PEP 8 style guide.
- Use type hints where appropriate.
- Write docstrings for all public functions, classes, and methods.
- Use meaningful variable and function names.

### Rust Guidelines

- Follow the Rust API Guidelines.
- Use Rust's type system to prevent errors.
- Write comprehensive documentation for all public items.
- Use Rust's error handling mechanisms appropriately.

### C++ Guidelines

- Follow the C++ Core Guidelines.
- Use modern C++ features where appropriate.
- Write comprehensive documentation for all public items.
- Use C++'s error handling mechanisms appropriately.

### Go Guidelines

- Follow the Go Code Review Comments.
- Use Go's error handling mechanisms appropriately.
- Write comprehensive documentation for all public items.
- Use meaningful variable and function names.

## Pull Request Process

1. **Create a Pull Request**: Create a pull request from your branch to the main repository.

2. **Describe Your Changes**: Provide a detailed description of your changes, including the motivation for the changes and any relevant issues.

3. **Pass CI Checks**: Ensure that your changes pass all CI checks, including tests, linting, and type checking.

4. **Review Process**: Your pull request will be reviewed by the project maintainers. Address any feedback or comments from the reviewers.

5. **Merge**: Once your pull request is approved, it will be merged into the main repository.

## License

By contributing to Neurenix, you agree that your contributions will be licensed under the Apache License 2.0.
