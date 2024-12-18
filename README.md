# ML By Hand

We are creating an autograd engine from scratch and use it to build/train more complex neural networks to learn from first principles.

- Focus on learning and transparency over optimization
- API interface closely mirrors [Pytorch](https://github.com/pytorch/pytorch/tree/main) for validation and low usage overhead
- Minimal third-party dependencies (e.g. only uses Pytorch to compare gradient correctness)
- All mathematical operations explicitly derived for comprehensive understanding

> What I cannot create, I do not understand
>
> -- Richard Feynman

<details>
  <summary>Long version</summary>

  **Autograd** ([wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)) computes exact derivatives by tracking computations and applying the chain rule systematically. It enables efficient backpropagation in neural networks, allowing them to learn from errors and adjust parameters automatically.

  The primary motivation is to learn about neural networks from scratch and from first principles. There are many good ML libraries out there (e.g. Tensorflow, Pytorch, Scikit-learn, etc.) that are well-optimized and have a lot of features. But they often introduce lots of abstractions, which hides the underlying concepts and make it difficult to understand how they work. I believe, to better utilize those abstractions/libraries, we must first understand how everything works from ground up. This is the guiding princple for this project. All mathematical and calculus operations are explicitly derived in the code without abstraction. Also, debugging a neural network, especially the `backward()` implementations of various functions (e.g. loss, and activation), offers a rewarding learning experience.

  This project took inspiration from [Micrograd](https://github.com/karpathy/micrograd/tree/master), and kept the API interface as close as possible to [Pytorch](https://github.com/pytorch/pytorch/tree/main) to reduce extra usage overhead and utilize it to validate correctness.
</details>

## Demo/Examples

`examples/` contains various examples of neural networks built using the library that tackle some classical problems
- MNIST classifier: [`examples/mnist.py`](https://github.com/workofart/ml-by-hand/blob/main/examples/mnist.py)
- CIFAR-10/CIFAR-100 classifier: [`examples/cifar.py`](https://github.com/workofart/ml-by-hand/blob/main/examples/cifar.py)
- End-to-end training implementation for a binary classifier on the [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) can be found in [`tests/autograd/test_train.py`](https://github.com/workofart/ml-by-hand/blob/c1156ee0c7a252484df1cd5234316a299e008b8b/test/autograd/test_train.py#L7-L66).

## Technical Overview
- `tensor` (base class)
  - support scalar, vector, N-dimensional matrices
  - arithmetic: add, mul, matmul, pow, sub, division, neg, max, mean
  - core: forward, backward, reshape
- `nn` (neural network components)
  - Module (base class)
  - Linear (basic building block for a perceptron/hidden layer)
  - BatchNorm (batch normalization)
  - Dropout (regularization)
- `functional` (including backprop derivation defined in `backward()`)
  - Activation functions: relu, sigmoid, softmax
  - Loss: binary_cross_entropy, sparse_cross_entropy, hinge_loss
- `optim`
  - Optimizer (base class)
  - SGD (stochastic gradient descent)
  - Adam
- `test/`
  - All the unit tests for the above functionality
  - End-to-end training run by combining all the components on a real dataset

## Environment Setup & Dependencies
`./bootstrap.sh` will install all the necessary dependencies.

Then you can activate the installed virtual environment by `source .venv/bin/activate`

`numpy` is the main dependencies. `pytorch` is only used for validating gradient calculation correctness in the tests.

## Tests
Comprehensive unit tests and integration tests available in `test/autograd`

Run `python -m pytest`

## Performance Testing

`memray` is used to track memory usage.

Run:
```
rm -rf memray_output.bin && memray run -o memray_output.bin -m pytest test/autograd/performance_test.py && memray tree memray_output.bin
```

## Future Work
Potential use autograd engine to create and train state-of-the-art neural network architectures, prioritizing educational value over training efficiency.

## License
MIT
