ML-By-Hand Documentation
========================

Welcome to the ML-By-Hand project documentation. This project is organized into several core modules, each providing specific functionality:

- **autograd**: Contains the core automatic differentiation engine, including the `Tensor` class and differentiable tensor-level operations.
- **tensor**: This module implements an automatic differentiation engine with a custom Tensor class, featuring forward and backward operations, mathematical functions, and computation graph management for gradient-based optimization.
- **nn**: Implements neural network modules and layers (e.g., Linear, Convolutional, Recurrent blocks).
- **optim**: Provides optimizers (e.g., SGD, Adam) for training neural network models.
- **functional**: This module provides differentiable activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU) and loss functions (Binary Cross-Entropy, Cross-Entropy, Hinge Loss, Mean Squared Error) for an automatic differentiation framework
- **tools**: This module provides utilities for dataset loading and general training workflows with checkpoint handling with serialization of model states and hyperparameters.
- **text**: Contains text processing utilities, including vocabulary creation, tokenization (such as Byte Pair Encoding), one-hot encoding, padding and causal masks, batch validation, and inference functions for language models

By the way, we leverage CuPy as the drop-in replacement for Numpy for GPU acceleration. For CPU-only machines, we just use Numpy. No code change needed. It just works out-of-the-box.

.. toctree::
   :maxdepth: 3
   :caption: Module Overview:

   modules
