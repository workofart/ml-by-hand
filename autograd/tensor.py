import numpy as np

class Tensor:
    def __init__(self, data, prev=(), requires_grad=False):
        self.data = data
        self.grad = 0
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
            
            
        
