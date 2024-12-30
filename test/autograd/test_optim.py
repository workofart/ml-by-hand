from copy import deepcopy
from autograd.nn import Tensor
from autograd.optim import Optimizer, SGD, Adam
from unittest import TestCase
import numpy as np
import torch  # for test validation


class TestOptimizer(TestCase):
    def setUp(self):
        # Create a small set of mock parameters
        self.params = {
            "param1": Tensor([1.0, 2.0, 3.0]),
            "param2": Tensor([4.0, 5.0, 6.0]),
        }

    def test_base_optimizer_state_dict(self):
        """
        This test checks that the base Optimizer can properly
        save and load minimal state (hyperparams + _states).
        """
        # Instantiate the base Optimizer
        optimizer = Optimizer(self.params, lr=0.01)

        # If we want to store a custom "global" piece of data, let's put it in _hyperparams:
        optimizer._hyperparams["some_state"] = {"extra_info": 123}

        # Save state
        saved_state = optimizer.state_dict()
        # The new structure might look like:
        # {
        #   "hyperparams": { "lr": 0.01, "some_state": {"extra_info": 123} },
        #   "states": {}  # No per-parameter states if we haven't done a step
        # }

        # Create a new instance and load state
        new_optimizer = Optimizer(
            self.params, lr=999.0
        )  # use a different LR to confirm overwrite
        new_optimizer.load_state_dict(saved_state)

        # Check that LR was restored from hyperparams
        self.assertEqual(new_optimizer._hyperparams["lr"], 0.01)

        # Check that the custom "some_state" was also restored
        self.assertIn("some_state", new_optimizer._hyperparams)
        self.assertEqual(new_optimizer._hyperparams["some_state"]["extra_info"], 123)

    def test_adam_state_dict(self):
        """
        This test checks that Adam's internal momentum buffers (m, v) and
        hyperparams (timestep, etc.) are properly saved and loaded.
        """
        adam = Adam(self.params, lr=0.001, beta1=0.8, beta2=0.9, epsilon=1e-5)

        # Simulate one step to populate internal momentum buffers
        for p in self.params.values():
            p.grad = Tensor([0.1, 0.2, 0.3]).data
        adam.step()

        # 1) Check that Adam's states have something in them
        self.assertIn("m", adam._states)
        self.assertIn("v", adam._states)
        self.assertIn("timestep", adam._states)

        # 2) Save the state
        saved_state_dict = deepcopy(adam.state_dict())
        # Should look like:
        # {
        #   "hyperparams": {"lr": 0.001, "beta1": 0.8, "beta2": 0.9, "epsilon": 1e-5, ...},
        #   "states": {
        #       "m": { param_id -> np.array(...) },
        #       "v": { param_id -> np.array(...) },
        #       "timestep": 1
        #   }
        # }

        # 3) Create a new Adam instance with different hyperparams
        new_adam = Adam(self.params, lr=0.999, beta1=0.5, beta2=0.5, epsilon=1e-1)

        # 4) Load the saved state
        new_adam.load_state_dict(saved_state_dict)

        # 5) Verify hyperparameters are restored
        self.assertAlmostEqual(new_adam._hyperparams["lr"], 0.001, places=7)
        self.assertAlmostEqual(new_adam._hyperparams["beta1"], 0.8, places=7)
        self.assertAlmostEqual(new_adam._hyperparams["beta2"], 0.9, places=7)
        self.assertAlmostEqual(new_adam._hyperparams["epsilon"], 1e-5, places=9)

        # 7) Verify the actual momentum buffers are the same
        for pid in self.params:
            np.testing.assert_allclose(
                new_adam._states["m"][pid], adam._states["m"][pid]
            )
            np.testing.assert_allclose(
                new_adam._states["v"][pid], adam._states["v"][pid]
            )
            np.testing.assert_allclose(
                new_adam._states["timestep"][pid], adam._states["timestep"][pid]
            )

        # 8) Optionally, verify if the entire dict matches
        old_sd = adam.state_dict()
        new_sd = new_adam.state_dict()

        # Check hyperparams match
        self.assertEqual(old_sd["hyperparams"].keys(), new_sd["hyperparams"].keys())
        for k in old_sd["hyperparams"]:
            self.assertEqual(old_sd["hyperparams"][k], new_sd["hyperparams"][k])

        # Check top-level states
        self.assertEqual(old_sd["states"].keys(), new_sd["states"].keys())
        # Check that the param_id keys match in m & v
        self.assertEqual(old_sd["states"]["m"].keys(), new_sd["states"]["m"].keys())
        self.assertEqual(old_sd["states"]["v"].keys(), new_sd["states"]["v"].keys())

        # Finally check values inside m, v
        for pid in old_sd["states"]["m"].keys():
            np.testing.assert_allclose(
                old_sd["states"]["m"][pid], new_sd["states"]["m"][pid]
            )
            np.testing.assert_allclose(
                old_sd["states"]["v"][pid], new_sd["states"]["v"][pid]
            )

        # Check timestep
        self.assertEqual(old_sd["states"]["timestep"], new_sd["states"]["timestep"])


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
                "sample_module.weight": self.param1,
                "sample_module.bias": self.param2,
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
