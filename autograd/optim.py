

class Optimizer:
    """
    Base Optimizer Class
    
    Below is a sample API usage for this class:
    optimizer = optim.Optimizer(model.parameters(), lr=0.01)
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    """
    def __init__(self, model_parameters, lr, **kwargs) -> None:
        self.model_parameters = model_parameters
        self.lr = lr
    
    def zero_grad(self):
        """
        Set the gradients of all optimized tensors to zero.
        """
        for param in self.model_parameters:
            param.grad = 0
    
    def step(self):
        """
        Performs a single optimization step.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer
    """
    def __init__(self, model_parameters, lr, **kwargs) -> None:
        super(SGD, self).__init__(model_parameters, lr, **kwargs)
        
    def step(self):
        for param in self.model_parameters:
            param.data -= self.lr * param.grad

    