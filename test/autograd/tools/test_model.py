import json
import os
import tempfile
from copy import deepcopy
from unittest import TestCase

from autograd.backend import xp
from autograd.init import xavier_uniform
from autograd.nn import Module
from autograd.optim import CosineScheduler
from autograd.tensor import Tensor
from autograd.tools.model import load_checkpoint, save_checkpoint


class MockModule(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parameters["weight"] = xavier_uniform(Tensor(xp.zeros((4, 5))))
        self._parameters["bias"] = xavier_uniform(Tensor(xp.zeros((1, 1))))

        self.stateful_states = xp.array([1, 1, 1])
        self.arg0 = args[0]
        self.kwarg0 = kwargs["kwarg0"]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self._parameters["weight"] + self._parameters["bias"]


class TestModel(TestCase):
    def setUp(self):
        # Create a test model with some initial arguments
        self.model = MockModule(999, kwarg0="testing_kwarg0")
        self._tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self.checkpoint_dir = self._tmp_dir.name
        self.checkpoint_name = "test_model"
        self.json_path = os.path.join(
            self.checkpoint_dir, f"{self.checkpoint_name}.json"
        )
        self.npz_path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_name}.npz")

    def tearDown(self):
        # Clean up any generated files
        for f in [self.json_path, self.npz_path]:
            if os.path.exists(f):
                os.remove(f)
        self._tmp_dir.cleanup()

    def _meta_types(self, node):
        types = []
        if isinstance(node, dict):
            node_type = node.get("_type")
            if node_type is not None:
                types.append(node_type)
            items = node.get("items")
            if isinstance(items, dict):
                for value in items.values():
                    types.extend(self._meta_types(value))
            elif isinstance(items, list):
                for value in items:
                    types.extend(self._meta_types(value))
        return types

    def test_save_load_model(self):
        # 1. Save the original parameters
        original_params = deepcopy(self.model.parameters)

        save_checkpoint(
            self.model.state_dict(),
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name=self.checkpoint_name,
        )

        # 2. Perform a forward/backward pass to change the parameters
        x = Tensor(xp.ones((5, 4)) + 1.234)
        out = self.model.forward(x)
        out.backward()

        # Example gradient descent update
        for p in self.model.parameters.values():
            # Suppose learning rate = 0.01
            p.data -= 0.01 * p.grad.data

        # Check that parameters have indeed changed
        for name, p in self.model.parameters.items():
            self.assertFalse(
                xp.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not change after update.",
            )

        # But states have not changed by this update
        self.assertTrue(
            xp.allclose(self.model.stateful_states, xp.array([1, 1, 1])),
            "States should not change with parameter updates.",
        )

        # 3. Load the previously saved parameters (and states) from checkpoint
        loaded_sd = load_checkpoint(json_path=self.json_path, npz_path=self.npz_path)
        self.model.load_state_dict(loaded_sd)

        # Check that parameters are back to original
        for name, p in self.model.parameters.items():
            self.assertTrue(
                xp.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not restore to original.",
            )
        # Check states are also back to original
        self.assertTrue(
            xp.allclose(self.model.stateful_states, xp.array([1, 1, 1])),
            "Stateful array did not restore to original.",
        )

    def test_load_weights_only(self):
        # 1. Instantiate a new model and demonstrate weights-only loading
        new_model = MockModule(999, kwarg0="testing_kwarg0")
        original_params = deepcopy(new_model.parameters)
        # Save the model including the states
        save_checkpoint(
            self.model.state_dict(),
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name=self.checkpoint_name,
        )

        # Manually change the new model's states from the default
        new_model.stateful_states = xp.array([999, 999, 999])

        # 2. Load only weights (parameters) from the checkpoint, ignoring states
        weights_only_data = load_checkpoint(
            json_path=self.json_path, npz_path=self.npz_path, weights_only=True
        )
        new_model.load_state_dict(weights_only_data)

        # 3. Parameters should match original
        for name, p in new_model.parameters.items():
            self.assertTrue(
                xp.allclose(original_params[name].data, p.data),
                f"Parameter {name} did not match original in new model with weights-only load.",
            )

        # But states should remain the custom value we set (999,999,999)
        self.assertTrue(
            xp.allclose(new_model.stateful_states, xp.array([999, 999, 999])),
            "States should NOT have been overwritten in weights-only scenario.",
        )

    def test_save_checkpoint_uses_backend_neutral_array_metadata(self):
        save_checkpoint(
            self.model.state_dict(),
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name=self.checkpoint_name,
        )

        with open(self.json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        meta_types = self._meta_types(meta)
        self.assertIn("array", meta_types)
        self.assertNotIn("np.ndarray", meta_types)

    def test_save_checkpoint_builds_paths_from_checkpoint_name(self):
        checkpoint_dir = os.path.join(self._tmp_dir.name, "test_checkpoints")
        checkpoint_name = "mock_run_MockModule_7"
        json_path, npz_path = save_checkpoint(
            self.model.state_dict(),
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
        )

        self.assertEqual(json_path, f"{checkpoint_dir}/{checkpoint_name}.json")
        self.assertEqual(npz_path, f"{checkpoint_dir}/{checkpoint_name}.npz")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(npz_path))

        for path in (json_path, npz_path):
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)

    def test_load_checkpoint_accepts_legacy_np_ndarray_metadata(self):
        save_checkpoint(
            self.model.state_dict(),
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name=self.checkpoint_name,
        )

        with open(self.json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        def replace_array_with_legacy_label(node):
            if not isinstance(node, dict):
                return
            if node.get("_type") == "array":
                node["_type"] = "np.ndarray"
            items = node.get("items")
            if isinstance(items, dict):
                for value in items.values():
                    replace_array_with_legacy_label(value)
            elif isinstance(items, list):
                for value in items:
                    replace_array_with_legacy_label(value)

        replace_array_with_legacy_label(meta)

        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)

        loaded_sd = load_checkpoint(json_path=self.json_path, npz_path=self.npz_path)
        self.model.load_state_dict(loaded_sd)
        self.assertTrue(
            xp.allclose(self.model.stateful_states, xp.array([1, 1, 1])),
            "Legacy array metadata should still deserialize correctly.",
        )

    def test_save_load_preserves_class_objects(self):
        save_checkpoint(
            {"lr_scheduler_cls": CosineScheduler},
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_name=self.checkpoint_name,
        )

        loaded = load_checkpoint(json_path=self.json_path, npz_path=self.npz_path)

        self.assertIs(loaded["lr_scheduler_cls"], CosineScheduler)
