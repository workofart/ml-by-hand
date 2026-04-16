import math
import os
import unittest
from unittest.mock import MagicMock, patch

from autograd.backend import xp
from autograd.functional import cross_entropy
from autograd.nn import AbstractLLMForwardFn, Module
from autograd.optim import Optimizer
from autograd.tensor import Tensor
from autograd.tools.callback import (
    run_sampling_inference,
    run_teacher_forcing_inference,
)
from autograd.tools.config_schema import (
    GenericTrainingConfig,
    TransformerTrainingConfig,
)
from autograd.tools.data import (
    DataLoader,
    LanguageModelingCollator,
    TokenSequenceDataset,
    openai_chat_to_prompt_completion,
    tokenize_prompt_completion,
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
        return Tensor(xp.zeros((batch_size, 3), dtype=xp.float32))

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


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            if token == "<SOS>":
                return [1]
        return [ord(char) for char in token]


class PromptMaskAwareForwardFn(AbstractLLMForwardFn):
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size

    def _build_logits(self, y):
        logits = xp.full(
            (y.shape[0], y.shape[1], self.vocab_size),
            -6.0,
            dtype=xp.float32,
        )
        for batch_idx in range(y.shape[0]):
            for token_idx in range(y.shape[1]):
                target = int(y[batch_idx, token_idx])
                predicted_class = target if target != 0 else 3
                logits[batch_idx, token_idx, predicted_class] = 6.0
        return Tensor(logits, requires_grad=True)

    def train(self, model, batch_data):
        _, _, y, _, _, _ = batch_data
        return self._build_logits(y), y

    def sample(self, model, batch_data):
        _, _, y, _, _, _ = batch_data
        return self._build_logits(y), None


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
        self.fake_pred = xp.zeros((2, 3), dtype=xp.float32)

        # Build training data for 2 batches, validation data for 1 batch
        self.train_data = [
            (xp.ones((2, 5), dtype=xp.float32), xp.zeros((2,), dtype=xp.float32)),
            (xp.ones((2, 5), dtype=xp.float32), xp.zeros((2,), dtype=xp.float32)),
        ]
        self.val_data = [
            (xp.ones((2, 5), dtype=xp.float32), xp.zeros((2,), dtype=xp.float32)),
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

    def test_save_metrics_persists_none_as_nan(self):
        self.trainer.metrics["epoch"] = [0]
        self.trainer.metrics["train_loss"] = [1.23]
        self.trainer.metrics["val_loss"] = [None]

        self.trainer._save_metrics()

        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        archive = xp.load(metrics_path)
        val_loss = xp.to_numpy(archive["val_loss"])

        self.assertTrue(math.isnan(float(val_loss[0])))

    def test_fit_with_single_epoch_and_sample_predictions_does_not_crash(self):
        self.config.total_epochs = 1
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=self.config,
            output_type="logits",
            sample_predictions=True,
        )

        trainer.fit(self.train_loader, self.val_loader)
        self.assertEqual(trainer.optimizer.step_call_count, len(self.train_data))

    def test_fit_respects_steps_per_epoch_from_config(self):
        self.config.total_epochs = 1
        self.config.steps_per_epoch = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(self.trainer.optimizer.step_call_count, 1)

    def test_evaluate_respects_eval_iters_from_config(self):
        self.config.eval_iters = 1
        val_loader = MockDataLoader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 1)


class TestLLMTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()
        seq_len = 10

        # For LLM, define a shape like (batch_size, seq_len, vocab_size).
        self.fake_pred = xp.zeros((2, seq_len, 10), dtype=xp.float32)

        # LLM data (just 2 batches for training, 1 batch for validation)
        self.train_data = [
            (
                xp.zeros((2, seq_len), dtype=xp.int32),
                xp.ones((2, seq_len), dtype=xp.int32),
                xp.full((2, seq_len), 2, dtype=xp.int32),
                None,
                None,
                None,
            )
            for _ in range(2)
        ]
        self.val_data = [
            (
                xp.zeros((2, seq_len), dtype=xp.int32),
                xp.ones((2, seq_len), dtype=xp.int32),
                xp.full((2, seq_len), 2, dtype=xp.int32),
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
            xp.zeros((2, seq_len), dtype=xp.int32),
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

    def test_evaluate_does_not_require_loader_metadata(self):
        class MinimalLoader:
            pad_idx = 0

            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        avg_val_loss = self.trainer.evaluate(MinimalLoader(self.val_data))

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)

    def test_evaluate_runs_eval_callbacks_and_returns_val_loss(self):
        callback = MagicMock()
        trainer = LLMTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            forward_fn=self.forward_fn,
            config=self.config,
            eval_callbacks=[callback],
        )

        avg_val_loss = trainer.evaluate(self.val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        callback.assert_called_once_with(
            trainer.model, trainer.forward_fn, self.val_loader, trainer.config
        )

    @patch("autograd.tools.callback.text_utils.inference")
    def test_manual_qualitative_inference_helpers_do_not_depend_on_loader(
        self, mock_inference
    ):
        bpe = MockBPE()
        groundtruth = xp.arange(10, dtype=xp.int32)

        teacher_forcing_text = run_teacher_forcing_inference(
            model=self.trainer.model,
            forward_fn=PromptMaskAwareForwardFn(),
            bpe=bpe,
            groundtruth_data=groundtruth,
            max_length=3,
        )
        sampled_text = run_sampling_inference(
            model=self.trainer.model,
            forward_fn=PromptMaskAwareForwardFn(),
            bpe=bpe,
            start_tokens="ABC",
            max_length=4,
            top_k=5,
        )

        self.assertIs(mock_inference.return_value, teacher_forcing_text)
        self.assertIs(mock_inference.return_value, sampled_text)
        self.assertEqual(mock_inference.call_count, 2)

    def test_llm_evaluate_respects_eval_iters_from_config(self):
        self.config.eval_iters = 1
        val_loader = MockDataLoader(self.val_data * 2, pad_idx=0, seq_len=10)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 1)

    def test_train_step_supports_sft_batches_from_generic_data_loader(self):
        bpe = MockBPE()
        pad_idx = bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        sos_idx = bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        tokenized_example = tokenize_prompt_completion(
            openai_chat_to_prompt_completion(
                {
                    "messages": [
                        {"role": "user", "content": "ABC"},
                        {"role": "assistant", "content": "CB"},
                    ]
                }
            ),
            bpe,
        )
        loader = DataLoader(
            dataset=TokenSequenceDataset(
                token_sequences=[tokenized_example["tokens"]],
                loss_masks=[tokenized_example["loss_mask"]],
                shuffle=False,
            ),
            batch_size=1,
            collate_fn=LanguageModelingCollator(
                max_tokens=6,
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )
        trainer = LLMTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=cross_entropy,
            forward_fn=PromptMaskAwareForwardFn(),
            config=TransformerTrainingConfig(
                total_epochs=1,
                checkpoint_freq=1,
                model_kwargs={"hidden_size": 128},
                optimizer_kwargs={"lr": 0.01},
                resume_epoch=None,
                label_smoothing=0.0,
                teacher_enforcing=False,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )

        batch = next(iter(loader))
        logits, y = trainer.forward_fn(trainer.model, batch, mode="train")
        expected_loss = cross_entropy(logits, y, pad_idx=loader.pad_idx)

        train_loss = trainer.train_step(batch, loader)
        val_loss = trainer.evaluate(loader)

        self.assertAlmostEqual(train_loss, float(expected_loss.item()), places=6)
        self.assertAlmostEqual(val_loss, float(expected_loss.item()), places=6)

    def test_fit_supports_unsized_streaming_data_loaders(self):
        bpe = MockBPE()
        pad_idx = bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        sos_idx = bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        self.config.total_epochs = 1
        self.config.steps_per_epoch = 1
        self.config.eval_iters = 1

        stream_loader = DataLoader(
            dataset=TokenSequenceDataset(
                data=xp.arange(100, dtype=xp.int32),
                seq_len=4,
                shuffle=False,
                random_window=True,
            ),
            batch_size=2,
            collate_fn=LanguageModelingCollator(
                max_tokens=5,
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )
        trainer = LLMTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=cross_entropy,
            forward_fn=PromptMaskAwareForwardFn(),
            config=self.config,
        )

        trainer.fit(stream_loader, stream_loader)

        self.assertEqual(trainer.optimizer.step_call_count, 1)
