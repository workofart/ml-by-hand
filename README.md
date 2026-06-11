# ML By Hand

<div align="center">
<img src="https://github.com/user-attachments/assets/0655f743-6bb0-46c8-9cdf-ec3a8c84058a" width="400" height="400">

[![Unit Tests](https://github.com/workofart/ml-by-hand/actions/workflows/test.yml/badge.svg)](https://github.com/workofart/ml-by-hand/actions/workflows/test.yml) |
📝 [Blog Post 1](https://www.henrypan.com/blog/2025-02-06-ml-by-hand/) |
📝 [Blog Post 2](https://www.henrypan.com/blog/2026-03-14-how-deep-learning-library-enables-learning/)

</div>



We are creating a deep learning library from scratch (that evolved from a simple autograd engine). It is designed to demystify the inner workings of building deep learning models by exposing every mathematical detail and stripping down the abstractions shiny ML libraries (e.g. PyTorch/TensorFlow) have. **This project tries to provide an opportunity to learn deep learning from first-principles. And use the hand-built library to create and train state-of-art models (such as [GPT-2](https://github.com/workofart/ml-by-hand/blob/main/examples/gpt_2.py))).**



> “What I cannot create, I do not understand.”
> — Richard Feynman

**Key Principles**
  - **Learn By Doing:** All formulas and calculations are derived in code, so you see exactly how gradients (or derivatives) are computed—no hidden black boxes!
  - **Learning Over Optimization:** Focus on understanding the underlying mathematics and algorithms, rather than optimizing for speed or memory usage (though we can still train GPT models on a single CPU)
  - **PyTorch-Like API:** API interface closely mirrors [PyTorch](https://github.com/pytorch/pytorch/tree/main) for low adoption overhead
  - **Minimal Dependencies:** User code and examples should go through `autograd.backend.xp`; that alias binds to `numpy` by default, `mlx` on macOS when available, or `cupy` on CUDA Linux hosts. `pytorch` is used for gradient correctness checks in unit tests.

<details>
  <summary><strong>Why build a deep learning library from scratch?</strong></summary>

  This project initially took inspiration from [Micrograd](https://github.com/karpathy/micrograd/tree/master), which was trying to build an **Autograd** ([Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)) engine from scratch for educational purposes. An autograd engine computes exact derivatives by tracking computations and applying the chain rule systematically. It enables neural networks to learn from errors and adjust parameters automatically. That's the core of deep learning. Then I started to add more features since everything seemed very straightforward after I had the initial building blocks (i.e. Tensor-level operations) implemented.

  The primary motivation is to learn about neural networks from scratch and from first principles. There are many good ML libraries out there (e.g. Tensorflow, PyTorch, Scikit-learn, etc.) that are well-optimized and have a lot of features. But they often introduce lots of abstractions, which hide the underlying concepts and make it difficult to understand how they work. I believe, to better utilize those abstractions/libraries, we must first understand how everything works from the ground up. This is the guiding principle for this project. All mathematical and calculus operations are explicitly derived in the code without abstraction. Also, debugging a neural network, especially the `backward()` implementations of various functions (e.g. loss, and activation), offers a rewarding learning experience.

  The goal is to keep the API interface as close as possible to PyTorch to reduce extra onboarding overhead and utilize it to validate correctness.


</details>

<table>
<tr>
<td width="45%" valign="top">

## Demo

<img src="https://www.henrypan.com/blog/assets/images/ml/ml-by-hand/ml-by-hand.gif" width="480" />

</td>
<td width="50%" valign="top">

## Try inference with the pre-trained checkpoint

Using the library, we're able to pre-train GPT-2 124M model from scratch on OpenWebText (56k steps * 1024 context length * 480 global batch size = 27 billion tokens, bfloat16) at [GitHub release](https://github.com/workofart/ml-by-hand/releases/tag/gpt2-124m-openwebtext-56000), including its BPE tokenizer vocabulary.

Download the weights (~240 MB) and run inference locally:

```bash
git clone https://github.com/workofart/ml-by-hand.git
cd ml-by-hand
./bootstrap.sh
source .venv/bin/activate

RELEASE=https://github.com/workofart/ml-by-hand/releases/download/gpt2-124m-openwebtext-56000
CKPT=gpt2_124m_openwebtext_56000_inference
VOCAB=openwebtext_vocab_49990.pkl
curl -L --create-dirs -o "checkpoints/$CKPT.json" "$RELEASE/$CKPT.json"
curl -L --create-dirs -o "checkpoints/$CKPT.npz" "$RELEASE/$CKPT.npz"
curl -L --create-dirs -o "training_data/$VOCAB" "$RELEASE/$VOCAB"
```

Then, from the repo root (after the [Environment Setup](#environment-setup) below):

```python
from autograd.text.tokenizer import BytePairEncoder
from autograd.text.utils import generate_text
from autograd.tools.model import load_checkpoint
from examples.gpt_2 import GPT2, GPT2ForwardFn

ckpt = load_checkpoint(
    "checkpoints/gpt2_124m_openwebtext_56000_inference.json",
    "checkpoints/gpt2_124m_openwebtext_56000_inference.npz",
)
model = GPT2(**ckpt["model_init_kwargs"])
model.load_state_dict(ckpt["model_state_dict"])

bpe = BytePairEncoder(
    num_merges=49990,
    vocab_file_path="training_data/openwebtext_vocab_49990.pkl",
)

generate_text(
    model=model,
    prediction_func=GPT2ForwardFn(),
    bpe=bpe,
    start_tokens="The meaning of life is",
    max_length=100,
    temperature=0.8,
    top_k=50,
)

> Inference: 100%|██████████| 95/95 [00:02<00:00, 31.90it/s]
> [prompt 5 tokens + 95 new tokens in 2.98s, 31.9 tok/s]
> 'The meaning of life is based on the ability to feel and remember the things that ...'
```

</td>
</tr>
</table>

<table>
<tr>
<td width="50%" valign="top">

## Examples

Explore the [`examples/`](https://github.com/workofart/ml-by-hand/tree/main/examples) directory for real-world demonstrations of how this engine can power neural network training on various tasks:

📌 **LLMs:**
  - Original Transformers [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/transformers.py)
  - Byte Pair Encoder (BPE) Tokenizer [(Code)](https://github.com/workofart/ml-by-hand/blob/main/autograd/text/tokenizer.py)
  - GPT-1 [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/gpt-1.py)
  - GPT-2 [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/gpt_2.py)
  - (Newly added) Supervised Fine-Tuning (SFT) [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/sft_gpt_2.py) — fine-tunes a pretrained GPT-2 on chat-formatted data ([no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots))
  - (Newly added) Group Relative Policy Optimization (GRPO) [(Code)](https://github.com/workofart/ml-by-hand/blob/main/examples/grpo.py) — reinforcement learning on GSM8K math problems (the technique behind DeepSeek-R1)

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

</td>
<td width="50%" valign="top">

## Toy Example of Using the Library

<details>
  <summary><strong>Click to expand</strong></summary>

```python
from autograd.tensor import Tensor
from autograd.nn import Linear, Module
from autograd.optim import SGD
from autograd.backend import xp

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
assert xp.to_scalar(xp.allclose(x.data @ weights + bias, y_true))
```
</details>

</td>
</tr>
</table>

## Environment Setup

This repo uses `uv.lock` as the source of truth for dependency installation:
```bash
./bootstrap.sh
source .venv/bin/activate
```

## **Documentation**

Check out the modules in this project in the [docs website](https://ml-by-hand.readthedocs.io/en/latest/) built from the docs/ directory.

## Hardware Backends

There are couple of hardware backends that are supported for acceleration, thanks to CuPy and MLX for the seemless NumPy-like primitive API.

Backend selection happens automatically. In user code, import and use `autograd.backend.xp`; the alias is then bound to one of these backends:
- `mlx` is preferred when available on macOS
- `cupy` is preferred on Linux when a CUDA device is detected
  - `bootstrap.sh` auto-detects CUDA on Linux and syncs one of the pinned extras: `cuda11`, `cuda12`, or `cuda13`.
  - Manual installs are also available through `pyproject.toml` extras:
  ```bash
  uv sync --extra dev --extra cuda12
  ```
  - Pick exactly one CUDA extra that matches your installed CUDA major version.
- `numpy` is the fallback, which doesn't have any hardware acceleration

You can also force a backend explicitly:
```bash
AUTOGRAD_BACKEND=numpy uv run pytest
AUTOGRAD_BACKEND=mlx uv run pytest
AUTOGRAD_BACKEND=cupy uv run pytest
```

## Tests
Comprehensive unit tests and integration tests available in `test/autograd`

```bash
uv run pytest
```

CI exercises both backend paths:
- MLX on `macos-latest`
- NumPy on `ubuntu-latest`
- CuPy auto-detection is available on CUDA Linux hosts, but is not covered by the current GitHub Actions matrix.

## Future Work

- Expanding the autograd engine to power cutting-edge neural architectures
- Further performance tuning while maintaining clarity and educational value
- Interactive tutorials for newcomers to ML and advanced topics alike

## Contributing
Contributions are welcome! If you find bugs, want to request features, or add examples, feel free to open an issue or submit a pull request.

## License
MIT
