# micrograd

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Graphviz-5C3EE8?style=for-the-badge&logo=graphviz&logoColor=white" alt="Graphviz" />
    <img src="https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  </p>
</div>

micrograd is a tiny, scalar-valued autograd engine with a clean, educational PyTorch-like API. It implements backpropagation (reverse-mode autodiff) over a dynamically constructed directed acyclic graph (DAG), supporting modular neural network building blocks like Neurons, Layers, and Multi-Layer Perceptrons (MLPs).

## Features

- **Scalar Autograd Core**: Implements a custom `Value` class that tracks computational history (using a set of child nodes, operation markers, and backpropagation closures) and computes gradients dynamically.
- **Topological DAG Sorting**: Computes exact reverse-mode derivatives by automatically sorting the mathematical DAG in topological order during the `.backward()` trigger.
- **Extended Math Operators**: Supports comprehensive scalar arithmetic (`+`, `-`, `*`, `/`, custom powers `**`, unary negation) and activation functions (`relu()`, `tanh()`, `exp()`) with localized gradient rules.
- **Neural Network Library**:
  - **`Neuron`**: Models a single node with randomized weights and biases, supporting `.tanh()` activations.
  - **`Layer`**: Organizes a collection of parallel neurons, returning flat scalar outputs or lists of outputs.
  - **`MLP`**: Constructs sequential layers of parameterized dimensions with quick parameter access and zero-gradient resets.
- **Visual Graph Generation**: Employs Graphviz bindings to render the computational flow, displaying node values, operations, and exact backpropagated gradients in real time.
- **Exhaustive Testing**: Over 40 unit and integration tests written in `pytest` verifying boundary math conditions, edge cases, gradient accumulations, and network integration pipelines.

## Tech Stack

- **Core Engine:** Python (Standard Library, Math, Random)
- **Visualization:** Graphviz (`graphviz` library)
- **Unit Testing:** Pytest

## Project Structure

```text
├── micrograd/
│   ├── engine/
│   │   ├── value.py        # Core scalar tracking and operator gradients
│   │   ├── neuron.py       # Neuron model class with weights, bias, and activations
│   │   ├── layer.py        # Layer organization representing multiple Neurons
│   │   └── mlp.py          # Sequential Layer orchestrator with helper utilities
│   └── utils/
│       └── graph.py        # Graphviz DAG generator and renderer
├── tests/
│   ├── test_value.py       # Core scalar, backward, and operations assertions
│   ├── test_neuron.py      # Forward, backward, and boundary tests for Neurons
│   └── test_layer.py       # Layer chaining, reproduction, and integration tests
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python (v3.10+)
- Graphviz binaries installed on your system path (necessary for generating DAG graphs)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/abderrahmenex86/micrograd.git
cd micrograd
```

2. **Set up a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

To run the complete test suite and assert mathematical accuracy across the engine:
```bash
pytest tests/
```
