# ML By Hand

We are creating an autograd engine from scratch for learning purposes.

There are many good ML libraries out there (e.g. Tensorflow, Pytorch, Scikit-learn, etc.) that are well-optimized and have a lot of features. But they often introduce lots of abstractions, which hides the underlying concepts and make it difficult to understand how they work. I believe, to better utilize those abstractions/libraries, we must first understand how everything works from ground up.

> What I cannot create, I do not understand
> 
> -- Richard Feynman


**Autograd** ([wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)) stands for automatic differentation, which efficiently computes the derivatives of complex functions by systematically applying the chain rule of calculus. Unlike numerical differentiation (which approximates derivatives) or symbolic differentiation (which derives analytical expressions into a single expression), Autograd tracks function computations and automatically generates exact derivatives. This computation is crucial for deep learning because it allows neural networks to learn from their errors and adjust their parameters accordingly (backpropagation), making the training process both possible and efficient.

As for the scope, we will build both the Autograd engine and use it to build/train more complex neural networks.

## Demo
End-to-end training run in [`tests/autograd/test_train.py`](https://github.com/workofart/ml-algorithms/blob/9f10bf33d2ea406a1f2895c18f1b3deb6d932246/test/autograd/test_train.py#L36-L55) for a binary classifier on the [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

## Motivation & Inspiration
The primary motivation is to learn about neural networks from scratch and from first principles. All mathematical and calculus operations are explicitly derived in the code without abstraction, providing a comprehensive understanding. Additionally, debugging a neural network, especially the `backward()` implementations of various functions (unary, arithmetic, loss, and activation), offers a rewarding learning experience.

This project took inspiration from [Micrograd](https://github.com/karpathy/micrograd/tree/master), and kept the API interface as close as possible to [Pytorch](https://github.com/pytorch/pytorch/tree/main) to reduce extra usage overhead and utilize it to validate correctness.

## Assumptions
- Library effiency is not the top priority, learning is. Therefore, productionization is not the plan.
- Only use 3rd party libraries for visualization and testing (e.g. comparing gradients with Pytorch.)

## Details
More specifically, this includes:
- tensor
  - add, mul, matmul, pow, sub, division, neg ops
  - forward, backward, reshape
- nn
  - Module (base class)
  - Linear (basic building block for a perceptron/hidden layer)
- functional (including backprop derivation defined in `backward()`)
  - relu (activation function)
  - sigmoid (activation function)
  - binary_cross_entropy (loss)
- optim
  - Optimizer (base class)
  - SGD (basic stochastic gradient descent optimizer)
- test/
  - All the unit tests for the above functionality
  - End-to-end training run by combining all the components on a real dataset

## Dependencies
- Numpy
- Pytorch (only used for validating gradient calculation correctness in the tests)
- sklearn.datasets (for providing datasets to validate end-to-end training)

## Test Cases

In `test/autograd`

## Next Steps
Consider expanding the engine's functionality to eventually incorporate all state-of-the-art neural network architectures. While efficient training is not guaranteed, the primary purpose is for learning and experimentation.

## License
MIT