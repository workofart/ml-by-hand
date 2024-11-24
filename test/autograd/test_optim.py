from autograd.nn import Tensor
from autograd.optim import SGD, Adam
from unittest import TestCase
import numpy as np
import torch  # for test validation


class TestSGD(TestCase):
    def setUp(self) -> None:
        self.param1 = Tensor(1.0)
        self.param2 = Tensor(2.0)
        self.params = [self.param1, self.param2]
        self.optimizer = SGD(
            model_parameters={
                "sample_module": {"weight": self.param1, "bias": self.param2},
            },
            lr=0.01,
        )

    def test_zero_grad(self):
        self.optimizer.zero_grad()
        self.assertEqual(self.param1.grad, None)
        self.assertEqual(self.param2.grad, None)

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
            assert np.allclose(self.param1.data, expected_param1[_])
            assert np.allclose(self.param2.data, expected_param2[_])


class TestAdam(TestCase):
    def setUp(self) -> None:
        # Set same gradients for both implementations
        self.grad1_val = 0.1
        self.grad2_val = 0.2
        self.param1 = Tensor(1.0)
        self.param2 = Tensor(2.0)
        self.params = [self.param1, self.param2]
        self.torch_params = [
            torch.nn.Parameter(torch.tensor(1.0)),
            torch.nn.Parameter(torch.tensor(2.0)),
        ]

        # Init optimizers
        self.optim = Adam(
            model_parameters={
                "sample_module": {"weight": self.param1, "bias": self.param2},
            },
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        self.torch_optim = torch.optim.Adam(
            self.torch_params, lr=0.01, betas=(0.9, 0.999), eps=1e-8
        )

    def test_zero_grad(self):
        self.optim.zero_grad()
        self.optim.step()
        self.torch_optim.zero_grad()
        self.torch_optim.step()

        assert np.allclose(
            self.param1.data,
            self.torch_params[0].detach().numpy(),
            atol=1e-6,
        )
        assert np.allclose(
            self.param2.data,
            self.torch_params[1].detach().numpy(),
            atol=1e-6,
        )

    def test_step(self):
        # Perform multiple steps and compare results
        for _ in range(5):
            # Step both optimizers
            self.optim.step()
            self.torch_optim.step()

            # Zero out gradients (this is similar to our training step)
            self.optim.zero_grad()
            self.torch_optim.zero_grad()

            # Compare parameters
            assert np.allclose(
                self.param1.data,
                self.torch_params[0].detach().numpy(),
                atol=1e-6,
            )
            assert np.allclose(
                self.param2.data,
                self.torch_params[1].detach().numpy(),
                atol=1e-6,
            )

            # Set new gradients for next step
            self.setUp()

    def test_different_gradients(self):
        # Test with different gradient values
        gradient_pairs = [
            (0.7, -0.4),
            (0.2, 0.3),
            (-0.3, 0.5),
            (0.05, -0.02),
            (2.0, -1.5),
        ]

        for grad1, grad2 in gradient_pairs:
            # Reset optimizers (this is needed because the state of the optimizers is changed during the step() method)
            self.setUp()

            # Set gradients
            self.param1.grad = grad1
            self.param2.grad = grad2
            self.torch_params[0].grad = torch.tensor(grad1)
            self.torch_params[1].grad = torch.tensor(grad2)

            # Step both optimizers
            self.optim.step()
            self.torch_optim.step()

            # Compare results
            assert np.allclose(
                self.param1.data,
                self.torch_params[0].detach().numpy(),
                atol=1e-6,
            )
            assert np.allclose(
                self.param2.data,
                self.torch_params[1].detach().numpy(),
                atol=1e-6,
            )
