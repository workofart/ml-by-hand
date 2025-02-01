import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from autograd.nn import Module
from autograd.tensor import Tensor
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


class BaseTrainerTest(unittest.TestCase):
    """
    A base test class that provides common setUp logic for any trainer.
    The specialized test classes will subclass this and then run super().setUp().
    """

    def setUp(self):
        # Mock model with numeric num_parameters() and an override for __call__.
        self.model = Module()
        # model.num_parameters() must return a float so that format strings work (:.2f).
        self.model.num_parameters = MagicMock()
        self.model.num_parameters.return_value = 1e6
        self.model.state_dict = MagicMock()
        self.model.state_dict.return_value = {}
        # We'll leave self.model.__call__ to be defined in child classes (because shapes differ).

        # Mock optimizer with a numeric .lr property
        self.optimizer = MagicMock()
        self.optimizer.lr = 0.01
        self.optimizer.zero_grad = MagicMock()
        self.optimizer.step = MagicMock()

        # A mock loss function returning a constant scalar
        self.loss_fn = MagicMock(return_value=Tensor(1.23))

    def tearDown(self) -> None:
        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{self.model.__class__.__name__}_default.npz"
        )
        if os.path.exists(metrics_path):
            os.remove(metrics_path)


class TestSimpleTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()
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
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=2,
            output_type="logits",
        )

    def test_fit_calls_optimizer_step(self):
        with patch.object(self.model, "forward", return_value=Tensor(self.fake_pred)):
            self.trainer.fit(self.train_loader, self.val_loader)
            # We have 2 epochs * 2 train batches => 4 calls to optimizer.step
            self.assertEqual(self.optimizer.step.call_count, 4)

    def test_train_step_returns_loss(self):
        with patch.object(self.model, "forward", return_value=Tensor(self.fake_pred)):
            batch = next(iter(self.train_data))
            loss_val = self.trainer.train_step(batch)
            self.assertAlmostEqual(loss_val, 1.23, places=5)
            self.optimizer.step.assert_called_once()


class TestLLMTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()
        seq_len = 10

        # For LLM, define a shape like (batch_size, seq_len, vocab_size).
        self.fake_pred = np.zeros((2, seq_len, 10), dtype=np.float32)
        self.model.hidden_size = 128  # for the trainer's warmup logic

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

        # Construct the trainer
        self.trainer = LLMTrainer(
            model=self.model,
            optimizer=self.optimizer,
            forward_fn=self.forward_fn,
            loss_fn=self.loss_fn,
            warmup_steps=100,
            tokenizer=MagicMock(),
            epochs=2,
        )

    def test_fit_calls_optimizer_step(self):
        with patch.object(self.model, "forward", return_value=Tensor(self.fake_pred)):
            self.trainer.fit(self.train_loader, self.val_loader)
            # 2 epochs * 2 train batches => 4 calls to optimizer.step
            self.assertEqual(self.optimizer.step.call_count, 4)

    def test_train_step_returns_loss(self):
        with patch.object(self.model, "forward", return_value=Tensor(self.fake_pred)):
            batch = next(iter(self.train_data))
            loss_val = self.trainer.train_step(batch)
            self.assertAlmostEqual(loss_val, 1.23, places=5)
            self.optimizer.step.assert_called_once()

    @patch("autograd.text.utils.inference")
    def test_perform_inference_called(self, mock_inference):
        # epoch=0 => calls _perform_inference => at least one inference function is called
        self.trainer.fit(self.train_loader, self.val_loader)
        self.assertTrue(mock_inference.called)
