import os
from copy import deepcopy
from unittest import TestCase
import numpy as np

from autograd.nn import Module
from autograd.init import xavier_uniform
from autograd.tensor import Tensor
from autograd.tools.model import load_model, save_model


class MockModule(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parameters["weight"] = xavier_uniform(Tensor(np.zeros((4, 5))))
        self._parameters["bias"] = xavier_uniform(Tensor(np.zeros((1, 1))))

        self.stateful_states = np.array([1, 1, 1])
        self.arg0 = args[0]
        self.kwarg0 = kwargs["kwarg0"]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self._parameters["weight"] + self._parameters["bias"]


class TestModel(TestCase):
    def setUp(self):
        # Create a test model with some initial arguments
        self.model = MockModule(999, kwarg0="testing_kwarg0")
        self.json_path = "test_model.json"
        self.npz_path = "test_model.npz"

    def tearDown(self):
        # Clean up any generated files
        for f in [self.json_path, self.npz_path]:
            if os.path.exists(f):
                os.remove(f)

    def test_save_load_model(self):
        # 1. Save the original parameters
        original_params = deepcopy(self.model.parameters)

        save_model(
            self.model.state_dict(), json_path=self.json_path, npz_path=self.npz_path
        )

        # 2. Perform a forward/backward pass to change the parameters
        x = Tensor(np.ones((5, 4)) + 1.234)
        out = self.model.forward(x)
        out.backward()

        # Example gradient descent update
        for p in self.model.parameters.values():
            # Suppose learning rate = 0.01
            p.data -= 0.01 * p.grad.data

        # Check that parameters have indeed changed
        for name, p in self.model.parameters.items():
            self.assertFalse(
                np.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not change after update.",
            )

        # But states have not changed by this update
        self.assertTrue(
            np.allclose(self.model.stateful_states, np.array([1, 1, 1])),
            "States should not change with parameter updates.",
        )

        # 3. Load the previously saved parameters (and states) from checkpoint
        loaded_sd = load_model(json_path=self.json_path, npz_path=self.npz_path)
        self.model.load_state_dict(loaded_sd)

        # Check that parameters are back to original
        for name, p in self.model.parameters.items():
            self.assertTrue(
                np.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not restore to original.",
            )
        # Check states are also back to original
        self.assertTrue(
            np.allclose(self.model.stateful_states, np.array([1, 1, 1])),
            "Stateful array did not restore to original.",
        )

    def test_load_weights_only(self):
        # 1. Instantiate a new model and demonstrate weights-only loading
        new_model = MockModule(999, kwarg0="testing_kwarg0")
        original_params = deepcopy(new_model.parameters)
        # Save the model including the states
        save_model(
            self.model.state_dict(), json_path=self.json_path, npz_path=self.npz_path
        )

        # Manually change the new model's states from the default
        new_model.stateful_states = np.array([999, 999, 999])

        # 2. Load only weights (parameters) from the checkpoint, ignoring states
        weights_only_data = load_model(
            json_path=self.json_path, npz_path=self.npz_path, weights_only=True
        )
        new_model.load_state_dict(weights_only_data)

        # 3. Parameters should match original
        for name, p in new_model.parameters.items():
            self.assertTrue(
                np.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not match original in new model with weights-only load.",
            )

        # But states should remain the custom value we set (999,999,999)
        self.assertTrue(
            np.allclose(new_model.stateful_states, np.array([999, 999, 999])),
            "States should NOT have been overwritten in weights-only scenario.",
        )
