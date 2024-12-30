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

[`examples/`](https://github.com/workofart/ml-by-hand/tree/main/examples) contains various examples of neural networks built using the library that tackle some classical problems

- [x] Regression using Deep Neural Network
  - [White Wine](https://github.com/workofart/ml-by-hand/blob/a2d55fdd9dc969f3848e0b15c3ac01a47736e655/test/autograd/test_train.py#L30)
- [x] Binary Classification using Deep Logistic Regression
  - [MNIST (One vs Rest)](https://github.com/workofart/ml-by-hand/blob/f4d3ab9e7903e2e675bdcd781695ab3e23908472/examples/mnist.py#L82)
  - [Breast Cancer](https://github.com/workofart/ml-by-hand/blob/f4d3ab9e7903e2e675bdcd781695ab3e23908472/test/autograd/test_train.py#L12)
- [x] Multi-class Classification using Deep Logistic Regression
  - [MNIST](https://github.com/workofart/ml-by-hand/blob/f4d3ab9e7903e2e675bdcd781695ab3e23908472/examples/mnist.py#L14)
  - [CIFAR-10/CIFAR-100](https://github.com/workofart/ml-by-hand/blob/f4d3ab9e7903e2e675bdcd781695ab3e23908472/examples/cifar.py#L13)
- [x] Convolutional Neural Network
  - [MNIST](https://github.com/workofart/ml-by-hand/blob/a2d55fdd9dc969f3848e0b15c3ac01a47736e655/examples/mnist.py#L37)
  - [CIFAR-10/CIFAR-100](https://github.com/workofart/ml-by-hand/blob/f8dbe2454fa2b04fe2fe2a9bca02c584c9c7b54a/examples/cifar.py#L35)
- [x] Residual Neural Network
  - [MNIST](https://github.com/workofart/ml-by-hand/blob/09c680f9864c842f5e4d543f4cc837fd15dd5269/examples/mnist.py#L15)
  - [CIFAR-10/CIFAR-100](https://github.com/workofart/ml-by-hand/blob/09c680f9864c842f5e4d543f4cc837fd15dd5269/examples/cifar.py#L36)
- [x] Recurrent Neural Network (RNN) + Long Short-Term Memory Network (LSTM)
  - [Movie Sentiment Analysis](https://github.com/workofart/ml-by-hand/blob/cedd9ef72a0b7d2c04958e5a7819e530efc87916/examples/movie_sentiment.py#L76)
- [x] [Neural Turing Machine (with LSTM controller)](https://github.com/workofart/ml-by-hand/blob/main/examples/neural_turing_machine.py)
## Technical Overview
- `tensor` (base class)
  - support scalar, vector, N-dimensional matrices
  - arithmetic: add, mul, matmul, pow, sub, division, neg, max, mean
  - core: forward, backward, reshape
- `nn` (neural network components)
  - Module (base class)
  - Linear (basic building block for a perceptron/hidden layer)
  - BatchNorm (batch normalization)
  - Conv2d (convolutional layer)
  - MaxPool2d (max pooling layer)
  - Dropout (regularization)
  - RNN (recurrent block for building RNNs)
  - LSTM (long short-term memory block for building LSTM networks)
- `functional` (including backprop derivation defined in `backward()`)
  - Activation functions: relu, sigmoid, softmax
  - Loss: binary_cross_entropy, sparse_cross_entropy, hinge_loss
- `optim`
  - Optimizer (base class)
  - SGD (stochastic gradient descent)
  - Adam
- `tools`
  - `trainer.py` (end-to-end training runner to train a model on a given dataset)
  - `data.py` (data loading, splitting, etc.)
  - `metrics.py` (accuracy etc.)
- `test/`
  - All the unit tests for the above functionality
  - End-to-end training run by combining all the components on a real dataset

## Environment Setup & Dependencies
`./bootstrap.sh` will install all the necessary dependencies.

Then you can activate the installed virtual environment by `source .venv/bin/activate`

`numpy` is the main dependency. `pytorch` is only used for validating gradient calculation correctness in the tests.

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
