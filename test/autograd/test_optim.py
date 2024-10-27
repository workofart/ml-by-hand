from autograd.nn import Tensor, Linear
from autograd.optim import SGD
from unittest import TestCase

class TestSGD(TestCase):
    def setUp(self) -> None:
        self.param1 = Tensor(1.0)
        self.param2 = Tensor(2.0)
        self.params = [self.param1, self.param2]
        self.optimizer = SGD(
            model_parameters={
                "sample_module": {
                    "weight": self.param1,
                    "bias": self.param2
                },
            },
        lr=0.01)
        
    def test_zero_grad(self):
        self.optimizer.zero_grad()
        self.assertEqual(self.param1.grad, 0)
        self.assertEqual(self.param2.grad, 0)
        
    def test_step(self):
        self.param1.grad = 0.1
        self.param2.grad = 0.2
        
        # data - grad * lr
        expected_param1 = [
            1 - (0.1 * 0.01) * 1, 
            1 - (0.1 * 0.01) * 2,
            1 - (0.1 * 0.01) * 3,
        ]
        expected_param2 = [
            2 - (0.2 * 0.01) * 1, 
            2 - (0.2 * 0.01) * 2,
            2 - (0.2 * 0.01) * 3,
        ] 
        
        for _ in range(3):
            self.optimizer.step()
            self.assertAlmostEqual(self.param1.data, expected_param1[_])
            self.assertAlmostEqual(self.param2.data, expected_param2[_])
        