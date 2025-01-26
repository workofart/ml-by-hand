# ML By Hand

<div align="center">
<img src="https://github.com/user-attachments/assets/0655f743-6bb0-46c8-9cdf-ec3a8c84058a" width="400" height="400">
</div>



We are creating a deep learning library from scratch (that evolved from a simple autograd engine). It was designed to demystify the inner workings of building deep learning models by exposing every mathematical detail and stripping down the abstractions shiny ML libraries (e.g. PyTorch/TensorFlow) have. **This project tries to provide an opportunity to learn deep learning from first-principles.**


> “What I cannot create, I do not understand.”
> — Richard Feynman

**Key Principles**
  - **Learn By Doing:** All formulas and calculations are derived in code, so you see exactly how gradients (or derivatives) are computed—no hidden black boxes!
  - **Learning Over Optimization:** Focus on understanding the underlying mathematics and algorithms, rather than optimizing for speed or memory usage (though we can still train GPT models on a single CPU)
  - **PyTorch-Like API:** API interface closely mirrors [PyTorch](https://github.com/pytorch/pytorch/tree/main) for low adoption overhead
  - **Minimal Dependencies:** Only uses `numpy` (and `pytorch` for gradient correctness checks in unit tests).

<details>
  <summary><strong>Why build a deep learning library from scratch?</strong></summary>

  This project initially took inspiration from [Micrograd](https://github.com/karpathy/micrograd/tree/master), which was trying to build an **Autograd** ([Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)) engine from scratch for educational purposes. An autograd engine computes exact derivatives by tracking computations and applying the chain rule systematically. It enables neural networks to learn from errors and adjust parameters automatically. That's the core of deep learning. Then I started to add more features since everything seemed very straightforward after I had the initial building blocks (i.e. Tensor-level operations) implemented.

  The primary motivation is to learn about neural networks from scratch and from first principles. There are many good ML libraries out there (e.g. Tensorflow, PyTorch, Scikit-learn, etc.) that are well-optimized and have a lot of features. But they often introduce lots of abstractions, which hide the underlying concepts and make it difficult to understand how they work. I believe, to better utilize those abstractions/libraries, we must first understand how everything works from the ground up. This is the guiding principle for this project. All mathematical and calculus operations are explicitly derived in the code without abstraction. Also, debugging a neural network, especially the `backward()` implementations of various functions (e.g. loss, and activation), offers a rewarding learning experience.

  The goal is to keep the API interface as close as possible to PyTorch to reduce extra onboarding overhead and utilize it to validate correctness.


</details>

## **Demo/Examples**

Explore the [`examples/`](https://github.com/workofart/ml-by-hand/tree/main/examples) directory for real-world demonstrations of how this engine can power neural network training on various tasks:

📌 **Transformers & GPT (Newly added):**
  - Original Transformers [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/transformers.py)
  - Byte Pair Encoder (BPE) Tokenizer [(Code)](https://github.com/workofart/ml-by-hand/blob/main/autograd/text/tokenizer.py)
  - GPT-1 [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/gpt-1.py)
  - GPT-2 [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/gpt-2.py)

<details>
  <summary><strong>Click to see all other examples</strong></summary>

- **Regression** [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/test/autograd/test_train.py#L31)

- **Binary Classification:**
  - MNIST (One vs Rest) [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/mnist.py#L100)
  - Breast Cancer [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/test/autograd/test_train.py#L17)

- **Multi-Class Classification:**
  - MNIST [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/mnist.py#L35)
  - CIFAR-10/CIFAR-100 [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/cifar.py#L14)

- **Convolutional Neural Networks:**
  - MNIST [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/mnist.py#L55)
  - CIFAR-10/CIFAR-100 [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/cifar.py#L54)

- **Residual Neural Networks:**
  - MNIST [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/mnist.py#L17)
  - CIFAR-10/CIFAR-100 [(Code)](https://github.com/workofart/ml-by-hand/blob/c19a4a18349a4eec9084793cbdfca02195e594b6/examples/cifar.py#L36)

- **RNN + LSTM:**
  - Movie Sentiment Analysis [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/movie_sentiment.py)

- **Neural Turing Machine (LSTM Controller):**
  - Copy Tasks [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/neural_turing_machine.py)

- **Sequence-to-Sequence:**
  - WikiSum [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/seq2seq.py)
</details>


## Toy Example
<details>
  <summary><strong>Click to expand</strong></summary>

```python
from autograd.tensor import Tensor
from autograd.nn import Linear, Module
from autograd.optim import SGD
import numpy as np

class SimpleNN(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # A single linear layer (input_dim -> output_dim).
        # Mathematically: fc(x) = xW^T + b
        # where W is weight and b is bias.
        self.fc = Linear(input_dim, output_dim)

    def forward(self, x):
        # Simply compute xW^T + b without any additional activation.
        return self.fc(x)

# Create a sample input tensor x with shape (1, 3).
# 'requires_grad=True' means we want to track gradients for x.
x = Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)

# We want the output to get close to 1.0 over time.
y_true = 1.0

# Initialize the simple neural network.
# This layer has a weight matrix W of shape (3, 1) and a bias of shape (1,).
model = SimpleNN(input_dim=3, output_dim=1)

# Use SGD with a learning rate of 0.03
optimizer = SGD(model.parameters, lr=0.03)

for epoch in range(20):
    # Reset (zero out) all accumulated gradients before each update.
    optimizer.zero_grad()

    # --- Forward pass ---
    # prediction = xW^T + b
    y_pred = model(x)
    print(f"Epoch {epoch}: {y_pred}")

    # Define a simple mean squared error function
    loss = ((y_pred - y_true) ** 2).mean()

    # --- Backward pass ---
    # Ultimately we need to compute the gradient of the loss with respect to the weights
    # Specifically, if Loss = (pred - 1)^2, then:
    #   dL/d(pred) = 2 * (pred - 1)
    #   d(pred)/dW = d(xW^T + b) / dW = x^T
    # By chain rule, dL/dW = dL/d(pred) * d(pred)/dW = [2 * (pred - 1)] * x^T
    loss.backward()

    # --- Update weights ---
    optimizer.step()

# See the computed gradients for the linear layer’s weight matrix:
weights = model.fc.parameters["weight"].data
bias = model.fc.parameters["bias"].data
gradient = model.fc.parameters["weight"].grad
print("[After Training] Gradients for fc weights:", gradient)
print("[After Training] layer weights:", weights)
print("[After Training] layer bias:", bias)
assert np.isclose(x.data @ weights + bias, y_true)
```
</details>

## **Technical Overview**

Here’s a brief look at the major modules in this project. This not exhaustive. API doc coming out soon.
- **`tensor`**
  - Supports scalar, vector, and N-dimensional data
  - Arithmetic ops: add, mul, matmul, pow, sub, division, neg, max, mean
  - Core methods: forward, backward, reshape

- **`nn`**
  - `Module` (base class)
  - `Linear`, `BatchNorm`, `Conv2d`, `MaxPool2d`, `Dropout`
  - Recurrent Layers: `RNN`, `LSTM`

- **`functional`**
  - Activation functions: relu, sigmoid, softmax
  - Losses: binary_cross_entropy, sparse_cross_entropy, hinge_loss

- **`optim`**
  - Optimizer (base)
  - SGD, Adam

- **`tools`**
  - `trainer.py` for end-to-end training
  - `data.py` for data loading/splitting
  - `metrics.py` for accuracy, etc.

- **`test/`**
  - Unit tests & integration tests for all modules
  - Validation with PyTorch for gradient correctness

## **Environment Setup**

Run the bootstrap script to install dependencies:
```bash
./bootstrap.sh
source .venv/bin/activate
```
This sets up your virtual environment.

## Tests
Comprehensive unit tests and integration tests available in `test/autograd`

```bash
python -m pytest
```

## Future Work

- Expanding the autograd engine to power cutting-edge neural architectures
- Further performance tuning while maintaining clarity and educational value
- Interactive tutorials for newcomers to ML and advanced topics alike

## Contributing
Contributions are welcome! If you find bugs, want to request features, or add examples, feel free to open an issue or submit a pull request.

## License
MIT
