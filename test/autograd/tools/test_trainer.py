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
from autograd.tools.trainer import LLMTrainer, SimpleTrainer, grad_l2_norm


class MockDataLoader(DataLoader):
    """
    A simple loader that stores a list of batches in memory.
    Each time __iter__ is called, it yields those same batches again.
    """

    def __init__(self, data, pad_idx=0, seq_len=1, batch_token_count_fn=None):
        self.data = data
        self.pad_idx = pad_idx
        self.seq_len = seq_len
        self.bpe = MagicMock()
        self.batch_token_count_fn = batch_token_count_fn

    def on_epoch_start(self):
        # If you want to shuffle or do something each epoch, do it here
        pass

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # A fresh iterator every time => each epoch can re-iterate from the start
        print(f"MockDataLoader: yielding {len(self.data)} batch(es)")
        return iter(self.data)

    def batch_token_count(self, batch):
        if self.batch_token_count_fn is None:
            return None
        return self.batch_token_count_fn(batch)


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


class RecordingTqdm:
    def __init__(self, iterable, desc=None, leave=False, elapsed_s=2.0):
        self.iterable = iterable
        self.desc = desc
        self.leave = leave
        self.format_dict = {"elapsed": elapsed_s}
        self.postfixes = []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        postfix = dict(ordered_dict or {})
        postfix.update(kwargs)
        self.postfixes.append(postfix)


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
        _, _, y, _, _ = batch_data
        return self._build_logits(y), y

    def sample(self, model, batch_data):
        _, _, y, _, _ = batch_data
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
            max_epochs=2,
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
        """Check that train_step returns the expected loss tensor."""
        batch = next(iter(self.train_data))
        loss_val = self.trainer.train_step(batch)
        self.assertAlmostEqual(float(loss_val.item()), 1.23, places=5)

    def test_save_metrics_persists_none_as_nan(self):
        self.trainer.metrics["epoch"] = [xp.array(0.0, dtype=xp.float32)]
        self.trainer.metrics["train_loss"] = [xp.array(1.23, dtype=xp.float32)]
        self.trainer.metrics["val_loss"] = [None]

        self.trainer._save_metrics()

        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        archive = xp.load(metrics_path)
        val_loss = xp.to_numpy(archive["val_loss"])

        self.assertTrue(math.isnan(float(val_loss[0])))

    def test_fit_with_single_epoch_and_sample_predictions_does_not_crash(self):
        self.config.max_epochs = 1
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

    def test_fit_respects_max_steps_from_config(self):
        self.config.max_epochs = 3
        self.config.max_steps = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(self.trainer.optimizer.step_call_count, 1)

    def test_fit_supports_max_steps_without_max_epochs(self):
        step_only_config = GenericTrainingConfig(
            max_steps=3,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=step_only_config,
            output_type="logits",
        )

        trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(trainer.optimizer.step_call_count, 3)

    def test_fit_accumulates_using_global_and_micro_batch_sizes(self):
        config = GenericTrainingConfig(
            max_epochs=2,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            global_batch_size=4,
            micro_batch_size=2,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )

        trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(trainer.optimizer.step_call_count, 2)

    def test_config_requires_at_least_one_training_limit(self):
        with self.assertRaisesRegex(
            ValueError, "At least one of max_epochs or max_steps must be set"
        ):
            GenericTrainingConfig(
                max_epochs=None,
                max_steps=None,
                checkpoint_freq=1,
                model_kwargs={"hidden_size": 256},
                optimizer_kwargs={"lr": 0.01},
            )

    def test_config_requires_global_batch_size_to_be_divisible_by_micro_batch_size(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "global_batch_size must be divisible by micro_batch_size",
        ):
            GenericTrainingConfig(
                max_epochs=1,
                checkpoint_freq=1,
                model_kwargs={"hidden_size": 256},
                optimizer_kwargs={"lr": 0.01},
                global_batch_size=3,
                micro_batch_size=2,
            )

    def test_evaluate_uses_full_loader_when_max_eval_steps_is_none(self):
        val_loader = MockDataLoader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 2)

    def test_evaluate_respects_max_eval_steps_from_config(self):
        self.config.max_eval_steps = 1
        val_loader = MockDataLoader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 1)

    def test_evaluate_runs_without_grad_tracking(self):
        class LinearModel(Module):
            def __init__(self, input_size=5, output_size=3, **kwargs):
                super().__init__()
                self.weight = Tensor(
                    xp.ones((input_size, output_size), dtype=xp.float32),
                    requires_grad=True,
                )

            def forward(self, x):
                return Tensor(x, requires_grad=False) @ self.parameters["weight"]

        class RecordingLoss:
            def __call__(self, y_pred, y_true, **kwargs):
                self.last_pred = y_pred
                self.last_loss = y_pred.sum()
                return self.last_loss

        trainer = SimpleTrainer(
            model_cls=LinearModel,
            optimizer_cls=MockOptimizerClass,
            loss_fn=RecordingLoss(),
            config=GenericTrainingConfig(
                max_epochs=1,
                checkpoint_freq=1,
                model_kwargs={},
                optimizer_kwargs={"lr": 0.01},
                resume_epoch=None,
            ),
            output_type="logits",
        )

        avg_val_loss = trainer.evaluate(MockDataLoader(self.val_data))

        self.assertIsNotNone(trainer.loss_fn.last_pred)
        self.assertFalse(trainer.loss_fn.last_pred.requires_grad)
        self.assertIsNone(trainer.loss_fn.last_pred.creator)
        self.assertFalse(trainer.loss_fn.last_loss.requires_grad)
        self.assertIsNone(trainer.loss_fn.last_loss.creator)
        self.assertGreaterEqual(avg_val_loss, 0.0)

    def test_grad_l2_norm_returns_backend_scalar(self):
        param = Tensor(xp.array([1.0, 2.0], dtype=xp.float32), requires_grad=True)
        param.grad = Tensor(xp.array([3.0, 4.0], dtype=xp.float32), requires_grad=False)

        grad_norm = grad_l2_norm({"weight": param})

        self.assertTrue(xp.isclose(grad_norm, xp.array(5.0, dtype=xp.float32)))

    def test_on_epoch_end_logs_backend_scalar_metrics(self):
        self.trainer.metrics["grad_l2_norm"].append(xp.array(2.5, dtype=xp.float32))

        self.trainer._on_epoch_end(epoch=0, train_loss=1.23, val_loss=None)

        self.assertTrue(
            xp.isclose(
                self.trainer.metrics["epoch"][0],
                xp.array(0.0, dtype=xp.float32),
            )
        )

    def test_save_metrics_persists_backend_scalar_metrics(self):
        self.trainer.metrics["epoch"] = [xp.array(0.0, dtype=xp.float32)]
        self.trainer.metrics["train_loss"] = [xp.array(1.23, dtype=xp.float32)]
        self.trainer.metrics["val_loss"] = [None]
        self.trainer.metrics["grad_l2_norm"] = [xp.array(2.5, dtype=xp.float32)]

        self.trainer._save_metrics()

        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        archive = xp.load(metrics_path)
        grad_norm = xp.to_numpy(archive["grad_l2_norm"])

        self.assertEqual(float(grad_norm[0]), 2.5)

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_save_checkpoint_uses_stored_val_loss_history(self, mock_save_checkpoint):
        self.trainer.metrics["val_loss"] = [xp.array(0.5, dtype=xp.float32)]

        self.trainer._save_checkpoint(epoch=0)

        mock_save_checkpoint.assert_called_once()

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_save_checkpoint_skips_when_val_loss_does_not_improve(
        self, mock_save_checkpoint
    ):
        self.trainer.metrics["val_loss"] = [
            xp.array(0.5, dtype=xp.float32),
            xp.array(0.7, dtype=xp.float32),
        ]

        self.trainer._save_checkpoint(epoch=1)

        mock_save_checkpoint.assert_not_called()


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
        def batch_token_count_fn(batch):
            return int(batch[0].shape[0] * batch[0].shape[1])

        self.train_loader = MockDataLoader(
            self.train_data,
            pad_idx=0,
            seq_len=seq_len,
            batch_token_count_fn=batch_token_count_fn,
        )
        self.val_loader = MockDataLoader(
            self.val_data,
            pad_idx=0,
            seq_len=seq_len,
            batch_token_count_fn=batch_token_count_fn,
        )

        # We'll mock forward_fn to skip real model logic
        self.forward_fn = MagicMock()
        # Suppose it returns (logits, y) => shape: (2,10,10) for logits, plus (2,10) for labels
        self.forward_fn.return_value = (
            Tensor(self.fake_pred),
            xp.zeros((2, seq_len), dtype=xp.int32),
        )

        self.config = TransformerTrainingConfig(
            max_epochs=2,
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

    @patch("autograd.tools.trainer.tqdm")
    def test_fit_updates_tqdm_with_cumulative_tokens_and_token_throughput(
        self, mock_tqdm
    ):
        progress_bars = []

        def make_progress_bar(iterable, desc=None, leave=False):
            progress_bar = RecordingTqdm(iterable, desc=desc, leave=leave)
            progress_bars.append(progress_bar)
            return progress_bar

        mock_tqdm.side_effect = make_progress_bar
        self.config.max_epochs = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        training_bar = next(
            bar for bar in progress_bars if bar.desc == "Training Batches"
        )
        self.assertEqual(
            training_bar.postfixes,
            [
                {"tokens": "20", "tok/s": "10.0"},
                {"tokens": "40", "tok/s": "20.0"},
            ],
        )

    def test_train_step_returns_loss(self):
        """Check that a single train step returns the constant loss tensor."""
        batch = next(iter(self.train_data))
        loss_val = self.trainer.train_step(batch, self.train_loader)
        self.assertAlmostEqual(float(loss_val.item()), 1.23, places=5)

    def test_evaluate_supports_data_loader_without_token_count(self):
        avg_val_loss = self.trainer.evaluate(MockDataLoader(self.val_data))

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)

    @patch("autograd.tools.trainer.tqdm")
    def test_evaluate_updates_tqdm_with_cumulative_tokens_and_token_throughput(
        self, mock_tqdm
    ):
        progress_bars = []

        def make_progress_bar(iterable, desc=None, leave=False):
            progress_bar = RecordingTqdm(iterable, desc=desc, leave=leave)
            progress_bars.append(progress_bar)
            return progress_bar

        mock_tqdm.side_effect = make_progress_bar

        avg_val_loss = self.trainer.evaluate(self.val_loader)

        self.assertAlmostEqual(avg_val_loss, 1.23, places=5)
        self.assertEqual(len(progress_bars), 1)
        self.assertEqual(progress_bars[0].desc, "Evaluation")
        self.assertEqual(
            progress_bars[0].postfixes,
            [{"tokens": "20", "tok/s": "10.0"}],
        )

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

    def test_llm_evaluate_runs_without_grad_tracking(self):
        class TinyLM(Module):
            def __init__(self, vocab_size=10, **kwargs):
                super().__init__()
                self.logit_bias = Tensor(
                    xp.arange(vocab_size, dtype=xp.float32), requires_grad=True
                )

            def forward(self, tokens, mask=None):
                batch_size, seq_len = tokens.shape
                logit_bias = self.parameters["logit_bias"]
                return logit_bias.expand(batch_size, seq_len, logit_bias.shape[0])

        class TinyForwardFn(AbstractLLMForwardFn):
            def train(self, model, batch_data):
                X, _, y, _, _, causal_mask = batch_data
                return model(X, causal_mask), y

            def sample(self, model, batch_data):
                return model(batch_data, None), None

        class RecordingLoss:
            def __call__(self, y_pred, y_true, **kwargs):
                self.last_pred = y_pred
                self.last_loss = y_pred.sum()
                return self.last_loss

        trainer = LLMTrainer(
            model_cls=TinyLM,
            optimizer_cls=MockOptimizerClass,
            loss_fn=RecordingLoss(),
            forward_fn=TinyForwardFn(),
            config=TransformerTrainingConfig(
                max_epochs=1,
                checkpoint_freq=1,
                model_kwargs={},
                optimizer_kwargs={"lr": 0.01},
                resume_epoch=None,
                label_smoothing=0.0,
                teacher_enforcing=False,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )

        avg_val_loss = trainer.evaluate(self.val_loader)

        self.assertIsNotNone(trainer.loss_fn.last_pred)
        self.assertFalse(trainer.loss_fn.last_pred.requires_grad)
        self.assertIsNone(trainer.loss_fn.last_pred.creator)
        self.assertFalse(trainer.loss_fn.last_loss.requires_grad)
        self.assertIsNone(trainer.loss_fn.last_loss.creator)
        self.assertGreaterEqual(avg_val_loss, 0.0)

    def test_llm_sample_mode_runs_without_grad_tracking(self):
        class TinyLM(Module):
            def __init__(self, vocab_size=8, **kwargs):
                super().__init__()
                self.logit_bias = Tensor(
                    xp.arange(vocab_size, dtype=xp.float32), requires_grad=True
                )

            def forward(self, tokens, mask=None):
                batch_size, seq_len = tokens.shape
                logit_bias = self.parameters["logit_bias"]
                return logit_bias.expand(batch_size, seq_len, logit_bias.shape[0])

        class TinyForwardFn(AbstractLLMForwardFn):
            def train(self, model, batch_data):
                return model(batch_data, None), batch_data

            def sample(self, model, batch_data):
                return model(batch_data, None), None

        model = TinyLM()
        forward_fn = TinyForwardFn()
        tokens = xp.zeros((2, 4), dtype=xp.int32)

        logits, _ = forward_fn(model, tokens, mode="sample")

        self.assertFalse(logits.requires_grad)
        self.assertIsNone(logits.creator)

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

    def test_llm_evaluate_respects_max_eval_steps_from_config(self):
        self.config.max_eval_steps = 1
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
                max_epochs=1,
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

        self.assertAlmostEqual(
            float(train_loss.item()), float(expected_loss.item()), places=6
        )
        self.assertAlmostEqual(val_loss, float(expected_loss.item()), places=6)

    def test_fit_supports_unsized_streaming_data_loaders(self):
        bpe = MockBPE()
        pad_idx = bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        sos_idx = bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        self.config.max_epochs = 10
        self.config.max_steps = 1
        self.config.max_eval_steps = 1

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
