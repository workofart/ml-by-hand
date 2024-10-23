import numpy as np

class Tensor:
    def __init__(self, data, prev=(), requires_grad=False):
        # Handle different input types
        if np.isscalar(data):
            data = np.array(data)  # Creates 0-dim array
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
        
        self.data = data.astype(np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self.prev = set(prev)  # all the operations before this Tensor
        self.requires_grad = requires_grad

    # For each of these primitive operations, we need to adjust the backward gradient computation accordingly
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            data=self.data + other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        def _backward():
            """
            d(loss) / dx = d(loss) / d(x + y) * d(x + y) / dx
            d(loss) / d(x + y) = result.grad
            d(x + y) / dx = 1
            d(x + y) / dy = 1
            We need to multiply by result.grad because of the chain rule
            """
            self.grad += 1 * result.grad
            other.grad += 1 * result.grad
        result._backward = _backward
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            data=self.data * other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        def _backward():
            """
            d(loss) / dx = d(loss) / d(xy) * d(xy) / dx
            d(loss) / d(xy) = result.grad
            d(xy) / dx = y
            d(xy) / dy = x
            """
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result._backward = _backward
        return result
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(
            data=np.matmul(self.data, other.data),
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        def _backward():
            """
            d(loss) / dx
            = self.grad
            = d(loss) / d(x·y) * d(x·y) / dx
            = result.grad * d(x·y) / dx
            = result.grad * y.T
            
            d(loss) / dy
            = other.grad
            = d(loss) / d(x·y) * d(x·y) / dy
            = result.grad * d(x·y) / dy
            = x.T * result.grad
            Note:
                need to move x.T to the left because:
                1) Each element in result is a dot product of a row from x with a column from y
                2) When we backprop, we need x.T on the left to match dimensions:
                   x = (num_samples, num_features)
                   y = (num_features, num_classes)
                   x.T = (num_features, num_samples)
                   result.grad = (num_samples, num_classes)
                   x.T * result.grad = (num_features, num_classes)  # same shape as y
            """

            # When result.grad is scalar (like in our test case)
            if isinstance(result.grad, (int, float)) or result.grad.ndim == 0:
                
                self.grad += result.grad * other.data.T  # Scalar multiplication
                other.grad += result.grad * self.data.T
            # When result.grad is a matrix
            else:
                self.grad += np.matmul(result.grad, other.data.T)  # Matrix multiplication
                other.grad += np.matmul(self.data.T, result.grad)
        result._backward = _backward
        return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            data=self.data**other.data,
            prev=(self, other),
            requires_grad=self.requires_grad or other.requires_grad,
        )
        def _backward():
            """
            d(loss) / dx = d(loss) / d(x**y) * d(x**y) / dx
            d(loss) / d(x**y) = result.grad
            d(x**y) / dx = y*x^(y-1)
            d(x**y) / dy = x**y * ln(x)
            where x is self
            y is other
            """
            self.grad += other.data * (self.data ** (other.data - 1)) * result.grad
            if self.data > 0:
                other.grad += (self.data ** other.data) * np.log(self.data) * result.grad
        result._backward = _backward
        return result

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def forward(self, data):
        pass

    def backward(self):
        topological_sorted_tensors = []
        visited = set()
        def dfs(node: Tensor):
            if node not in visited:
                visited.add(node)
                for prev in node.prev:
                    dfs(prev)
                # the order in which we append to the list is in reverse order
                # because we always move backwards looking at the previous nodes
                # that point to the current node
                topological_sorted_tensors.append(node)
        dfs(self)
        self.grad = 1
        for tensor in reversed(topological_sorted_tensors):
            tensor._backward()
            
            
        
