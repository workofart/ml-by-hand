import numpy as np

class Tensor:
    def __init__(self, data, prev=(), requires_grad=True):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        else:
            data = data
        
        self.data = data
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
            def reverse_broadcast(grad_to_add, target_shape):
                # Calculate the number of dimensions to add to target_shape to match grad_to_add
                num_extra_dims = len(grad_to_add.shape) - len(target_shape)
                
                # Create a new shape for target_shape with ones in the extra dimensions
                expanded_target_shape = (1,) * num_extra_dims + target_shape
                
                # Identify the axes to sum over by comparing shapes
                axes_to_sum = tuple(i for i, (g_dim, t_dim) in enumerate(zip(grad_to_add.shape, expanded_target_shape)) if g_dim != t_dim)
                
                # Sum over the identified axes
                return np.sum(grad_to_add, axis=axes_to_sum).reshape(target_shape)
            
            self.grad += result.grad
            other.grad += reverse_broadcast(result.grad, other.data.shape)
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
            grad = result.grad
            # Handle broadcasting: sum along broadcasted dimensions
            if np.isscalar(self.data) or (isinstance(other.data, np.ndarray) and self.data.shape != other.data.shape):
                self.grad += np.sum(other.data * grad)
            else:
                self.grad += other.data * grad
                
            # Handle broadcasting: sum along broadcasted dimensions
            if np.isscalar(other.data) or (isinstance(self.data, np.ndarray) and self.data.shape != other.data.shape):
                other.grad += np.sum(self.data * grad)
            else:
                other.grad += self.data * grad
        result._backward = _backward
        return result
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        # Raise error if either input is scalar (0D) - Same as Pytorch assumption
        if np.isscalar(self.data) or np.isscalar(other.data):
            raise RuntimeError("both arguments to matmul need to be at least 1D")
        
        # Handle matrix multiplication shapes:
        # - If input is 1D vector, reshape it for matrix multiplication:
        #   - First operand (x): reshape to (1, n) row vector
        #   - Second operand (y): reshape to (n, 1) column vector
        # - If input is 2D matrix, keep original shape
        x = self.data.reshape((1, -1)) if self.data.ndim == 1 else self.data
        y = other.data.reshape((-1, 1)) if other.data.ndim == 1 else other.data
        
        result = Tensor(
            data=np.matmul(x, y).squeeze(),
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
            # print("In matmul backward:")
            # print(f"self.data shape: {self.data.shape}")
            
            # Vector @ Vector case (result is scalar)
            if self.data.ndim == 1 and other.data.ndim == 1:
                self.grad += result.grad * other.data
                other.grad += result.grad * self.data
                    
            # Matrix @ Vector case (result is vector)
            elif self.data.ndim == 2 and other.data.ndim == 1:
                self.grad += np.outer(result.grad, other.data)
                other.grad += np.matmul(self.data.T, result.grad)
                    
            # Matrix @ Matrix case (result is matrix)
            else:
                # Ensure result.grad is 2D
                if result.grad.ndim == 1:
                    result_grad = result.grad.reshape(1, -1)
                else:
                    result_grad = result.grad
                    
                # Ensure result_grad has the same shape as the forward pass output
                if result_grad.shape != (self.data.shape[0], other.data.shape[1]):
                    result_grad = result_grad.reshape(self.data.shape[0], other.data.shape[1])
                    
                # print(f"self.grad shape before: {self.grad.shape}")
                # print(f"other.grad shape before: {other.grad.shape}")
                
                # print(f"other.data shape: {other.data.shape}")
                # print(f"result_grad shape: {result_grad.shape}")
                self.grad += np.matmul(result_grad, other.data.T)
                other.grad += np.matmul(self.data.T, result_grad)
                    
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
        
        # Note that this is important to ensure our gradients shape is not a scalar
        # to ensure we follow the same matmul assumption as Pytorch.
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topological_sorted_tensors):
            tensor._backward()
            
    def reshape(self, *shape):
        self.data = self.data.reshape(*shape)
        return self
