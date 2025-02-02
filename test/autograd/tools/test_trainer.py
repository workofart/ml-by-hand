import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from autograd.nn import Module
from autograd.optim import Optimizer
from autograd.tensor import Tensor
from autograd.tools.config_schema import (
    GenericTrainingConfig,
    TransformerTrainingConfig,
)
from autograd.tools.trainer import LLMTrainer, SimpleTrainer


class MockDataLoader:
    """
    A simple loader that stores a list of batches in memory.
    Each time __iter__ is called, it yields those same batches again.
    """

    def __init__(self, data, pad_idx=0, seq_len=1):
        self.data = data
        self.pad_idx = pad_idx
        self.seq_len = seq_len
        self.bpe = MagicMock()

    def on_epoch_start(self):
        # If you want to shuffle or do something each epoch, do it here
        pass

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # A fresh iterator every time => each epoch can re-iterate from the start
        print(f"MockDataLoader: yielding {len(self.data)} batch(es)")
        return iter(self.data)


class MockModelClass(Module):
    """
    A minimal mock model class conforming to nn.Module that returns
    a constant prediction and tracks calls to num_parameters().
    """

    def __init__(self, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self._num_params = 1e6  # pretend we have 1 million params

    def num_parameters(self):
        return self._num_params

    def __call__(self, x):
        # Return a consistent shape for classification (e.g. batch_size x num_classes)
        # Or adapt to your needs.
        batch_size = x.shape[0]
        return Tensor(np.zeros((batch_size, 3), dtype=np.float32))

    def train(self):
        pass

    def eval(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        # Return something that can be saved as checkpoint
        return {}


class MockOptimizerClass(Optimizer):
    """
    A minimal mock optimizer that tracks step calls and has a .lr property.
    """

    def __init__(self, parameters, lr=1e-3, **kwargs):
        super().__init__(parameters, lr=lr)
        self._lr = lr
        self.step_call_count = 0

    @property
    def lr(self):
        return self._lr

    def step(self):
        self.step_call_count += 1

    def zero_grad(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}


class BaseTrainerTest(unittest.TestCase):
    """
    A base test class that provides common setUp logic for any trainer.
    The specialized test classes will subclass this and then run super().setUp().
    """

    def setUp(self):
        # A mock loss function returning a constant scalar
        self.loss_fn = MagicMock(return_value=Tensor(1.23))

    def tearDown(self) -> None:
        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        if os.path.exists(metrics_path):
            os.remove(metrics_path)


class TestSimpleTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()

        # Build a GenericTrainingConfig
        self.config = GenericTrainingConfig(
            total_epochs=2,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
        )

        # Make the model return a fake_pred
        self.fake_pred = np.zeros((2, 3), dtype=np.float32)

        # Build training data for 2 batches, validation data for 1 batch
        self.train_data = [
            (np.ones((2, 5), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
            (np.ones((2, 5), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
        ]
        self.val_data = [
            (np.ones((2, 5), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
        ]

        # Now use our real in-memory loader
        self.train_loader = MockDataLoader(self.train_data)
        self.val_loader = MockDataLoader(self.val_data)

        # Create the trainer
        self.trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=self.config,
            output_type="logits",
        )

    def test_fit_calls_optimizer_step(self):
        """Check that fit() results in the correct number of optimizer steps."""
        self.trainer.fit(self.train_loader, self.val_loader)
        # 2 epochs * 2 train batches = 4 steps
        self.assertEqual(self.trainer.optimizer.step_call_count, 4)

    def test_train_step_returns_loss(self):
        """Check that the train_step returns the expected scalar loss."""
        batch = next(iter(self.train_data))
        loss_val = self.trainer.train_step(batch)
        self.assertAlmostEqual(loss_val, 1.23, places=5)
        self.assertEqual(self.trainer.optimizer.step_call_count, 1)


class TestLLMTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()
        seq_len = 10

        # For LLM, define a shape like (batch_size, seq_len, vocab_size).
        self.fake_pred = np.zeros((2, seq_len, 10), dtype=np.float32)

        # LLM data (just 2 batches for training, 1 batch for validation)
        self.train_data = [
            (
                np.zeros((2, seq_len), dtype=np.int32),
                np.ones((2, seq_len), dtype=np.int32),
                np.full((2, seq_len), 2, dtype=np.int32),
                None,
                None,
                None,
            )
            for _ in range(2)
        ]
        self.val_data = [
            (
                np.zeros((2, seq_len), dtype=np.int32),
                np.ones((2, seq_len), dtype=np.int32),
                np.full((2, seq_len), 2, dtype=np.int32),
                None,
                None,
                None,
            )
        ]

        # Replace MagicMock with a real in-memory loader
        self.train_loader = MockDataLoader(self.train_data, pad_idx=0, seq_len=seq_len)
        self.val_loader = MockDataLoader(self.val_data, pad_idx=0, seq_len=seq_len)

        # We'll mock forward_fn to skip real model logic
        self.forward_fn = MagicMock()
        # Suppose it returns (logits, y) => shape: (2,10,10) for logits, plus (2,10) for labels
        self.forward_fn.return_value = (
            Tensor(self.fake_pred),
            np.zeros((2, seq_len), dtype=np.int32),
        )

        self.config = TransformerTrainingConfig(
            total_epochs=2,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 128},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
            label_smoothing=0.0,
            teacher_enforcing=False,
            include_decoder_input=True,
            create_padding_masks=False,
        )

        # Construct the trainer
        self.trainer = LLMTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            forward_fn=self.forward_fn,
            config=self.config,
        )

    def test_fit_calls_optimizer_step(self):
        """Check that fit() calls optimizer.step the correct number of times."""
        self.trainer.fit(self.train_loader, self.val_loader)
        # 2 epochs * 2 train batches = 4 steps
        self.assertEqual(self.trainer.optimizer.step_call_count, 4)

    def test_train_step_returns_loss(self):
        """Check that a single train step returns the constant scalar loss."""
        batch = next(iter(self.train_data))
        loss_val = self.trainer.train_step(batch, self.train_loader)
        self.assertAlmostEqual(loss_val, 1.23, places=5)
        self.assertEqual(self.trainer.optimizer.step_call_count, 1)

    @patch("autograd.text.utils.inference")
    def test_perform_inference_called(self, mock_inference):
        # epoch=0 => calls _perform_inference => at least one inference function is called
        self.trainer.fit(self.train_loader, self.val_loader)
        self.assertTrue(mock_inference.called)
