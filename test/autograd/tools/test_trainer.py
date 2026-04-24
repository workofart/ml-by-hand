import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from autograd.backend import xp
from autograd.data.collator import CausalLMCollator, CausalLMWindowCollator
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import (
    IterableDataset,
    TokenSequenceDataset,
    TokenWindowDataset,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch
from autograd.data.utils import (
    openai_chat_to_prompt_completion,
    tokenize_prompt_completion,
)
from autograd.functional import IGNORE_INDEX, cross_entropy
from autograd.nn import AbstractLLMForwardFn, Module
from autograd.optim import SGD, Optimizer
from autograd.tensor import Tensor, is_grad_enabled
from autograd.tools.callback import (
    run_sampling_inference,
    run_teacher_forcing_inference,
)
from autograd.tools.config_schema import (
    GenericTrainingConfig,
    TransformerTrainingConfig,
)
from autograd.tools.trainer import (
    LLMTrainer,
    SimpleTrainer,
    TrainingPlan,
    TrainingState,
)


def identity_collate(batch_items):
    return batch_items[0]


class InMemoryBatchDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
        self.epoch_start_calls = 0

    def on_epoch_start(self):
        self.epoch_start_calls += 1

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class UnsizedInfiniteDataset(IterableDataset):
    def __len__(self):
        raise TypeError("infinite")

    def __iter__(self):
        raise RuntimeError(
            "fit should reject unsized infinite loaders before iterating"
        )


class UnsizedEmptyPassDataset(IterableDataset):
    def __iter__(self):
        return iter(())


def make_data_loader(data):
    return DataLoader(
        dataset=InMemoryBatchDataset(data),
        batch_size=1,
        collate_fn=identity_collate,
    )


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


class TrackingModeModel(MockModelClass):
    def __init__(self, hidden_size=128, **kwargs):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.mode_log = []

    def train(self):
        self.mode_log.append("train")

    def eval(self):
        self.mode_log.append("eval")


class ScalarWeightModel(Module):
    def __init__(self, initial_weight=0.0, **kwargs):
        super().__init__()
        self.weight = Tensor(
            xp.array([[initial_weight]], dtype=xp.float32), requires_grad=True
        )

    def num_parameters(self):
        return 1

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x @ self._parameters["weight"]


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


class FakeTqdm:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.updates = []
        self.postfixes = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        self.updates.append(n)

    def set_postfix(self, *args, **kwargs):
        self.postfixes.append((args, kwargs))


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
                predicted_class = target if target != IGNORE_INDEX else 3
                logits[batch_idx, token_idx, predicted_class] = 6.0
        return Tensor(logits, requires_grad=True)

    def train(self, model, batch_data):
        y = batch_data.labels
        return self._build_logits(y)

    def sample(self, model, batch_data):
        if isinstance(batch_data, CausalLMBatch):
            y = batch_data.labels
        else:
            y = xp.array(batch_data, dtype=xp.int32)
            if y.ndim == 1:
                y = y[None, :]
        return self._build_logits(y), None


def test_abstract_llm_forward_fn_call_forwards_raw_mode_outputs():
    forward_fn = PromptMaskAwareForwardFn()
    batch = CausalLMBatch(
        input_ids=xp.zeros((1, 2), dtype=xp.int32),
        labels=xp.array([[1, 2]], dtype=xp.int32),
    )

    train_out = forward_fn(None, batch, mode="train")
    sample_out = forward_fn(None, batch, mode="sample")

    assert isinstance(train_out, Tensor)
    assert isinstance(sample_out, tuple)
    assert sample_out[1] is None


def test_transformer_forward_fn_train_returns_logits_only():
    from examples.transformers import TransformerForwardFn

    logits = Tensor(xp.zeros((2, 3, 5), dtype=xp.float32))

    class DummyTransformer:
        def __call__(self, input_ids, decoder_input_ids, src_mask, tgt_mask):
            return logits

    batch_data = Seq2SeqBatch(
        input_ids=xp.zeros((2, 3), dtype=xp.int32),
        decoder_input_ids=xp.zeros((2, 3), dtype=xp.int32),
        labels=xp.zeros((2, 3), dtype=xp.int32),
        src_mask=xp.zeros((2, 1, 1, 3), dtype=xp.float32),
        tgt_mask=xp.zeros((2, 1, 3, 3), dtype=xp.float32),
    )

    assert TransformerForwardFn().train(DummyTransformer(), batch_data) is logits


class BaseTrainerTest(unittest.TestCase):
    """
    A base test class that provides common setUp logic for any trainer.
    The specialized test classes will subclass this and then run super().setUp().
    """

    def setUp(self):
        # A mock loss function returning a constant scalar
        self.loss_fn = MagicMock(return_value=Tensor(1.23))
        self._tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self._trainer_dirs = {}
        self.checkpoint_dir = os.path.join(self._tmp_dir.name, "checkpoints")
        self.metrics_dir = os.path.join(self._tmp_dir.name, "training_runs")
        for trainer_cls in (SimpleTrainer, LLMTrainer):
            self._trainer_dirs[trainer_cls] = (
                trainer_cls.CHECKPOINT_DIR,
                trainer_cls.METRICS_DIR,
            )
            trainer_cls.CHECKPOINT_DIR = self.checkpoint_dir
            trainer_cls.METRICS_DIR = self.metrics_dir

    def tearDown(self) -> None:
        for trainer_cls, (checkpoint_dir, metrics_dir) in self._trainer_dirs.items():
            trainer_cls.CHECKPOINT_DIR = checkpoint_dir
            trainer_cls.METRICS_DIR = metrics_dir
        self._tmp_dir.cleanup()


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
        self.train_loader = make_data_loader(self.train_data)
        self.val_loader = make_data_loader(self.val_data)

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

    def test_compute_loss_returns_tensor(self):
        """Check that compute_loss returns the expected scalar Tensor."""
        batch = next(iter(self.train_data))
        loss = self.trainer._compute_loss(batch)
        self.assertAlmostEqual(float(loss.item()), 1.23, places=5)

    def test_save_metrics_persists_none_as_nan(self):
        self.trainer.metric_rows = [
            {"epoch": 0, "train_loss": 1.23, "val_loss": None},
        ]

        self.trainer._save_metrics()

        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        archive = xp.load(metrics_path)
        val_loss = xp.to_numpy(archive["val_loss"])

        self.assertTrue(math.isnan(float(val_loss[0])))

    def test_metrics_backfills_missing_values_for_late_metric_keys(self):
        self.trainer.metric_rows = [
            {"epoch": 1, "train_loss": 1.23},
            {"epoch": 2, "train_loss": 0.98, "val_accuracy": 0.75},
        ]

        self.trainer._save_metrics()
        metrics_path = (
            f"{SimpleTrainer.METRICS_DIR}/{MockModelClass.__name__}_default.npz"
        )
        archive = xp.load(metrics_path)

        self.assertEqual(xp.to_numpy(archive["epoch"]).tolist(), [1, 2])
        train_loss = xp.to_numpy(archive["train_loss"])
        self.assertAlmostEqual(float(train_loss[0]), 1.23, places=5)
        self.assertAlmostEqual(float(train_loss[1]), 0.98, places=5)
        val_accuracy = xp.to_numpy(archive["val_accuracy"])
        self.assertTrue(math.isnan(float(val_accuracy[0])))
        self.assertAlmostEqual(float(val_accuracy[1]), 0.75, places=5)

    def test_training_state_to_metrics_row_includes_eval_metrics(self):
        state = TrainingState()
        state.record_loss(1.23)
        eval_state = TrainingState()
        eval_state.record_eval_loss(0.5)
        eval_state.record_eval_metric("accuracy", numerator=3, denominator=4)

        row = state.to_metrics_row(eval_state=eval_state)

        self.assertEqual(
            row,
            {
                "train_loss": 1.23,
                "val_loss": 0.5,
                "val_accuracy": 0.75,
            },
        )

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_save_checkpoint_saves_first_validation_loss(self, mock_save_checkpoint):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_0.json",
            "checkpoints/default_MockModelClass_0.npz",
        )
        plan = TrainingPlan.for_epochs(
            max_epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            checkpoint_every=self.config.checkpoint_freq,
        )
        eval_state = TrainingState()
        eval_state.record_eval_loss(1.23)
        self.trainer._maybe_save_checkpoint(
            plan=plan,
            eval_state=eval_state,
        )

        mock_save_checkpoint.assert_called_once()
        checkpoint = mock_save_checkpoint.call_args.args[0]
        self.assertEqual(checkpoint["config_repr"], repr(self.config))
        self.assertNotIn("config", checkpoint)

    def test_config_requires_integer_checkpoint_frequency(self):
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_freq must be an int",
        ):
            GenericTrainingConfig(
                max_epochs=1,
                checkpoint_freq=1.5,
                model_kwargs={"hidden_size": 256},
                optimizer_kwargs={"lr": 0.01},
            )

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
        self.config.max_epochs = None
        self.config.max_steps = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(self.trainer.optimizer.step_call_count, 1)

    @patch("autograd.tools.trainer.tqdm")
    def test_fit_wraps_training_steps_in_tqdm(self, mock_tqdm):
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 10
        train_loader = make_data_loader(self.train_data * 3)
        progress_bar = FakeTqdm()
        mock_tqdm.return_value = progress_bar

        self.trainer.fit(train_loader)

        mock_tqdm.assert_called_once()
        self.assertEqual(mock_tqdm.call_args.kwargs["total"], 3)
        self.assertEqual(mock_tqdm.call_args.kwargs["desc"], "Training")
        self.assertEqual(sum(progress_bar.updates), 3)
        self.assertEqual(progress_bar.postfixes, [])

    def test_fit_does_not_call_tensor_item_for_train_loss_each_batch(self):
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 10
        train_loader = make_data_loader(self.train_data * 3)
        item_call_count = 0
        original_item = Tensor.item

        def counting_item(tensor):
            nonlocal item_call_count
            item_call_count += 1
            return original_item(tensor)

        with patch.object(Tensor, "item", new=counting_item):
            self.trainer.fit(train_loader)

        self.assertEqual(item_call_count, 0)

    @patch("autograd.tools.trainer.tqdm")
    def test_fit_logs_fit_plan_before_first_compute_loss(self, mock_tqdm):
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 10
        mock_tqdm.return_value = FakeTqdm()
        self.trainer._compute_loss = MagicMock(side_effect=RuntimeError("boom"))

        with self.assertLogs("autograd.tools.trainer", level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self.trainer.fit(self.train_loader)

        joined_logs = "\n".join(captured.output)
        self.assertIn("Fit plan:", joined_logs)
        self.assertIn("'mode': 'steps'", joined_logs)
        self.assertIn("'target_step': 3", joined_logs)
        self.assertNotIn("Starting first training batch", joined_logs)

    def test_fit_with_max_steps_runs_evaluation_on_checkpoint_steps_and_target_step(
        self,
    ):
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 2
        train_loader = make_data_loader(self.train_data * 3)
        eval_steps = []

        def record_eval(_):
            eval_steps.append(self.trainer.global_step)
            eval_state = TrainingState()
            eval_state.record_eval_loss(1.23)
            return eval_state

        self.trainer._evaluate = MagicMock(side_effect=record_eval)

        self.trainer.fit(train_loader, self.val_loader)

        self.assertEqual(eval_steps, [2, 3])

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_with_max_steps_can_report_before_checkpoint_steps(
        self, mock_save_checkpoint
    ):
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 10
        self.config.report_every_steps = 2
        train_loader = make_data_loader(self.train_data * 3)
        eval_steps = []

        def record_eval(_):
            eval_steps.append(self.trainer.global_step)
            eval_state = TrainingState()
            eval_state.record_eval_loss(1.23)
            return eval_state

        self.trainer._evaluate = MagicMock(side_effect=record_eval)

        self.trainer.fit(train_loader, self.val_loader)

        self.assertEqual(eval_steps, [2, 3])
        mock_save_checkpoint.assert_not_called()

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_with_max_steps_saves_checkpoint_on_checkpoint_steps(
        self, mock_save_checkpoint
    ):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_2.json",
            "checkpoints/default_MockModelClass_2.npz",
        )
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 2
        train_loader = make_data_loader(self.train_data * 3)

        self.trainer.fit(train_loader, self.val_loader)

        mock_save_checkpoint.assert_called_once()
        checkpoint = mock_save_checkpoint.call_args.args[0]
        self.assertEqual(checkpoint["step_count"], 2)

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_without_val_loader_saves_latest_checkpoint_on_checkpoint_steps(
        self, mock_save_checkpoint
    ):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_2.json",
            "checkpoints/default_MockModelClass_2.npz",
        )
        self.config.max_epochs = None
        self.config.max_steps = 3
        self.config.checkpoint_freq = 2
        train_loader = make_data_loader(self.train_data * 3)

        self.trainer.fit(train_loader, None)

        mock_save_checkpoint.assert_called_once()
        checkpoint = mock_save_checkpoint.call_args.args[0]
        self.assertEqual(checkpoint["step_count"], 2)
        self.assertIsNone(checkpoint["best_val_loss"])

    def test_fit_records_completed_epochs_from_global_step(self):
        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(
            [row["epoch"] for row in self.trainer.metric_rows],
            [1, 2],
        )

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_uses_completed_epoch_for_checkpoint_index(self, mock_save_checkpoint):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_1.json",
            "checkpoints/default_MockModelClass_1.npz",
        )
        self.config.max_epochs = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertTrue(
            mock_save_checkpoint.call_args.kwargs["checkpoint_name"].endswith("_1")
        )

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_saves_steps_per_epoch_in_epoch_mode_checkpoints(
        self, mock_save_checkpoint
    ):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_1.json",
            "checkpoints/default_MockModelClass_1.npz",
        )
        self.config.max_epochs = 1

        self.trainer.fit(self.train_loader, self.val_loader)

        checkpoint = mock_save_checkpoint.call_args.args[0]
        self.assertEqual(checkpoint["steps_per_epoch"], len(self.train_loader))
        self.assertAlmostEqual(checkpoint["best_val_loss"], 1.23, places=5)

    @patch("autograd.tools.trainer.save_checkpoint")
    def test_fit_in_epoch_mode_checkpoints_only_on_checkpoint_epochs(
        self, mock_save_checkpoint
    ):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_2.json",
            "checkpoints/default_MockModelClass_2.npz",
        )
        self.config.max_epochs = 2
        self.config.checkpoint_freq = 2

        self.trainer.fit(self.train_loader, self.val_loader)

        mock_save_checkpoint.assert_called_once()
        self.assertTrue(
            mock_save_checkpoint.call_args.kwargs["checkpoint_name"].endswith("_2")
        )

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_fit_rejects_epoch_mode_mid_epoch_resume(self, mock_load_checkpoint):
        mock_load_checkpoint.return_value = {
            "epoch": 0,
            "step_count": 1,
            "steps_per_epoch": len(self.train_loader),
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": self.config.model_kwargs,
            "optimizer_init_kwargs": self.config.optimizer_kwargs,
        }
        with self.assertRaisesRegex(
            ValueError,
            "Epoch-mode resume requires an epoch-boundary checkpoint",
        ):
            SimpleTrainer(
                model_cls=MockModelClass,
                optimizer_cls=MockOptimizerClass,
                loss_fn=self.loss_fn,
                config=self.config,
                output_type="logits",
                checkpoint_path="dummy",
            )

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_fit_rejects_epoch_mode_resume_with_changed_steps_per_epoch(
        self, mock_load_checkpoint
    ):
        mock_load_checkpoint.return_value = {
            "epoch": 1,
            "step_count": len(self.train_loader) + 1,
            "steps_per_epoch": len(self.train_loader) + 1,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": self.config.model_kwargs,
            "optimizer_init_kwargs": self.config.optimizer_kwargs,
        }
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=self.config,
            output_type="logits",
            checkpoint_path="dummy",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Cannot resume epoch-mode checkpoint with changed steps_per_epoch",
        ):
            trainer.fit(self.train_loader, self.val_loader)

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_resume_restores_best_val_loss_from_checkpoint(self, mock_load_checkpoint):
        mock_load_checkpoint.return_value = {
            "epoch": 1,
            "step_count": len(self.train_loader),
            "steps_per_epoch": len(self.train_loader),
            "best_val_loss": 0.5,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": self.config.model_kwargs,
            "optimizer_init_kwargs": self.config.optimizer_kwargs,
        }
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=self.config,
            output_type="logits",
            checkpoint_path="dummy",
        )

        self.assertEqual(trainer.best_val_loss, 0.5)

    @patch("autograd.tools.trainer.save_checkpoint")
    @patch("autograd.tools.trainer.load_checkpoint")
    def test_save_checkpoint_reuses_loaded_init_kwargs(
        self, mock_load_checkpoint, mock_save_checkpoint
    ):
        mock_save_checkpoint.return_value = (
            "checkpoints/default_MockModelClass_1.json",
            "checkpoints/default_MockModelClass_1.npz",
        )
        loaded_model_kwargs = {"hidden_size": 128}
        loaded_optimizer_kwargs = {"lr": 0.02}
        mock_load_checkpoint.return_value = {
            "epoch": 1,
            "step_count": len(self.train_loader),
            "steps_per_epoch": len(self.train_loader),
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": loaded_model_kwargs,
            "optimizer_init_kwargs": loaded_optimizer_kwargs,
        }
        config = GenericTrainingConfig(
            max_epochs=2,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
            checkpoint_path="dummy",
        )
        trainer.global_step = len(self.train_loader)
        plan = TrainingPlan.for_steps(
            max_steps=config.max_steps,
            report_every_steps=config.report_every_steps or config.checkpoint_freq,
            checkpoint_every=config.checkpoint_freq,
        )
        eval_state = TrainingState()
        eval_state.record_eval_loss(0.4)

        trainer._maybe_save_checkpoint(
            plan=plan,
            eval_state=eval_state,
        )

        checkpoint = mock_save_checkpoint.call_args.args[0]
        self.assertEqual(checkpoint["model_init_kwargs"], loaded_model_kwargs)
        self.assertEqual(checkpoint["optimizer_init_kwargs"], loaded_optimizer_kwargs)

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_pretrained_checkpoint_path_loads_weights_but_resets_training_state(
        self, mock_load_checkpoint
    ):
        class TrackingModel(MockModelClass):
            def __init__(self, hidden_size=128, **kwargs):
                super().__init__(hidden_size=hidden_size, **kwargs)
                self.loaded_state_dict = None

            def load_state_dict(self, state_dict):
                self.loaded_state_dict = state_dict

        class TrackingOptimizer(MockOptimizerClass):
            def __init__(self, parameters, lr=1e-3, **kwargs):
                super().__init__(parameters, lr=lr, **kwargs)
                self.loaded_state_dict = None

            def load_state_dict(self, state_dict):
                self.loaded_state_dict = state_dict

        model_state_dict = {"weights": "from-pretrain"}
        optimizer_state_dict = {"step": 123}
        mock_load_checkpoint.return_value = {
            "epoch": 9,
            "step_count": len(self.train_loader),
            "steps_per_epoch": len(self.train_loader),
            "best_val_loss": 0.25,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "model_init_kwargs": {"hidden_size": 512},
            "optimizer_init_kwargs": {"lr": 0.02},
        }
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
            pretrained_checkpoint_path="dummy_pretrained",
        )

        trainer = SimpleTrainer(
            model_cls=TrackingModel,
            optimizer_cls=TrackingOptimizer,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )

        self.assertEqual(trainer.model.hidden_size, 256)
        self.assertEqual(trainer.model.loaded_state_dict, model_state_dict)
        self.assertIsNone(trainer.optimizer.loaded_state_dict)
        self.assertEqual(trainer.optimizer.lr, 0.01)
        self.assertEqual(trainer.global_step, 0)
        self.assertIsNone(trainer.best_val_loss)
        self.assertEqual(trainer.checkpoint, {})
        mock_load_checkpoint.assert_called_once_with(
            "dummy_pretrained.json", "dummy_pretrained.npz"
        )

    def test_pretrained_checkpoint_path_conflicts_with_resume_epoch(self):
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=1,
            pretrained_checkpoint_path="dummy_pretrained",
        )

        with self.assertRaisesRegex(
            ValueError,
            "pretrained_checkpoint_path cannot be combined with resume_epoch or checkpoint_path",
        ):
            SimpleTrainer(
                model_cls=MockModelClass,
                optimizer_cls=MockOptimizerClass,
                loss_fn=self.loss_fn,
                config=config,
                output_type="logits",
            )

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_checkpoint_path_resume_uses_checkpoint_progress_not_config_hint(
        self, mock_load_checkpoint
    ):
        mock_load_checkpoint.return_value = {
            "epoch": 99,
            "step_count": len(self.train_loader),
            "steps_per_epoch": len(self.train_loader),
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": self.config.model_kwargs,
            "optimizer_init_kwargs": self.config.optimizer_kwargs,
        }
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
            checkpoint_path="dummy",
        )

        trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(trainer.global_step, len(self.train_loader))
        self.assertEqual(trainer.optimizer.step_call_count, 0)

    @patch("autograd.tools.trainer.load_checkpoint")
    def test_fit_rejects_checkpoint_step_beyond_max_steps(self, mock_load_checkpoint):
        mock_load_checkpoint.return_value = {
            "epoch": 0,
            "step_count": 3,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "model_init_kwargs": self.config.model_kwargs,
            "optimizer_init_kwargs": self.config.optimizer_kwargs,
        }
        config = GenericTrainingConfig(
            max_steps=2,
            checkpoint_freq=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            resume_epoch=None,
        )
        with self.assertRaisesRegex(ValueError, "Checkpoint step exceeds max_steps"):
            SimpleTrainer(
                model_cls=MockModelClass,
                optimizer_cls=MockOptimizerClass,
                loss_fn=self.loss_fn,
                config=config,
                output_type="logits",
                checkpoint_path="dummy",
            )

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

    def test_fit_flushes_pending_gradients_before_report_boundary(self):
        config = GenericTrainingConfig(
            max_steps=2,
            checkpoint_freq=2,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            global_batch_size=2,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )

        trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(trainer.optimizer.step_call_count, 1)

    def test_fit_restarts_epoch_lifecycle_when_step_budget_exhausts_loader(self):
        self.config.max_epochs = None
        self.config.max_steps = 3

        self.trainer.fit(self.train_loader, self.val_loader)

        self.assertEqual(self.train_loader.dataset.epoch_start_calls, 2)

    def test_fit_flushes_partial_accumulation_at_finite_loader_boundary_in_step_mode(
        self,
    ):
        config = GenericTrainingConfig(
            max_steps=3,
            checkpoint_freq=10,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            global_batch_size=8,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )
        train_loader = make_data_loader(self.train_data)

        trainer.fit(train_loader)

        self.assertEqual(trainer.global_step, 3)
        self.assertEqual(trainer.optimizer.step_call_count, 2)
        self.assertEqual(train_loader.dataset.epoch_start_calls, 2)

    def test_fit_requires_max_steps_for_unsized_infinite_train_loader(self):
        self.config.max_epochs = 1
        self.config.max_steps = None

        with self.assertRaisesRegex(
            ValueError,
            "Infinite training loaders require max_steps",
        ):
            self.trainer.fit(
                DataLoader(
                    dataset=UnsizedInfiniteDataset(),
                    batch_size=1,
                    collate_fn=identity_collate,
                ),
                self.val_loader,
            )

    def test_fit_step_mode_rejects_loader_that_yields_no_batches(self):
        self.config.max_epochs = None
        self.config.max_steps = 1
        empty_loader = DataLoader(
            dataset=UnsizedEmptyPassDataset(),
            batch_size=1,
            collate_fn=identity_collate,
        )

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            self.trainer.fit(empty_loader)

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

    def test_fit_reporting_does_not_force_partial_optimizer_steps(self):
        config = GenericTrainingConfig(
            max_steps=40,
            checkpoint_freq=40,
            report_every_steps=10,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            global_batch_size=4,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )
        train_loader = make_data_loader(self.train_data * 20)

        trainer.fit(train_loader)

        self.assertEqual(trainer.global_step, 40)
        self.assertEqual(trainer.optimizer.step_call_count, 10)

    def test_fit_flushes_leftover_gradients_at_epoch_end(self):
        config = GenericTrainingConfig(
            max_epochs=1,
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
        train_loader = make_data_loader(self.train_data + self.val_data)

        trainer.fit(train_loader, self.val_loader)

        self.assertEqual(trainer.optimizer.step_call_count, 2)

    def test_optimizer_step_is_no_op_when_no_batches_accumulated(self):
        state = TrainingState()
        self.trainer.last_grad_l2_norm = 7.0

        self.trainer.optimizer_step(state)

        self.assertEqual(self.trainer.optimizer.step_call_count, 0)
        self.assertEqual(self.trainer.last_grad_l2_norm, 7.0)
        self.assertEqual(state.accumulated_batches, 0)

    def test_fit_does_not_report_grad_norm_without_clipping(self):
        config = GenericTrainingConfig(
            max_steps=4,
            checkpoint_freq=4,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
            global_batch_size=2,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=MockModelClass,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=config,
            output_type="logits",
        )
        train_loader = make_data_loader(self.train_data * 2)

        with patch.object(
            trainer.optimizer,
            "grad_l2_norm",
            side_effect=AssertionError("trainer should not call grad_l2_norm"),
        ):
            trainer.fit(train_loader)

        self.assertNotIn("grad_l2_norm", trainer.metric_rows[0])

    def _fit_scalar_weight_model(
        self, train_data, *, global_batch_size, micro_batch_size
    ):
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"initial_weight": 0.0},
            optimizer_kwargs={"lr": 0.1},
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        )
        trainer = SimpleTrainer(
            model_cls=ScalarWeightModel,
            optimizer_cls=SGD,
            loss_fn=lambda pred, target: (
                (pred - Tensor(target, requires_grad=False)) ** 2
            ).mean(),
            config=config,
        )

        trainer.fit(make_data_loader(train_data))

        return float(xp.to_scalar(trainer.model.parameters["weight"].data[0, 0]))

    def test_gradient_accumulation_matches_full_batch_update(self):
        full_batch_data = [
            (
                xp.array([[1.0], [3.0]], dtype=xp.float32),
                xp.array([[1.0], [3.0]], dtype=xp.float32),
            )
        ]
        micro_batch_data = [
            (
                xp.array([[1.0]], dtype=xp.float32),
                xp.array([[1.0]], dtype=xp.float32),
            ),
            (
                xp.array([[3.0]], dtype=xp.float32),
                xp.array([[3.0]], dtype=xp.float32),
            ),
        ]

        full_batch_weight = self._fit_scalar_weight_model(
            full_batch_data,
            global_batch_size=2,
            micro_batch_size=2,
        )
        accumulated_weight = self._fit_scalar_weight_model(
            micro_batch_data,
            global_batch_size=2,
            micro_batch_size=1,
        )

        self.assertAlmostEqual(accumulated_weight, full_batch_weight, places=6)

    def test_grad_l2_norm_records_pre_clip_gradient_norm(self):
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"initial_weight": 0.0},
            optimizer_kwargs={"lr": 0.0, "max_grad_norm": 1.0},
            global_batch_size=1,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=ScalarWeightModel,
            optimizer_cls=SGD,
            loss_fn=lambda pred, target: (
                (pred - Tensor(target, requires_grad=False)) ** 2
            ).mean(),
            config=config,
        )
        train_loader = make_data_loader(
            [
                (
                    xp.array([[1.0]], dtype=xp.float32),
                    xp.array([[10.0]], dtype=xp.float32),
                )
            ]
        )

        trainer.fit(train_loader)

        self.assertAlmostEqual(trainer.metric_rows[0]["grad_l2_norm"], 20.0, places=5)

    def test_clip_enabled_uses_clip_grad_norm_for_reporting(self):
        config = GenericTrainingConfig(
            max_epochs=1,
            checkpoint_freq=1,
            model_kwargs={"initial_weight": 0.0},
            optimizer_kwargs={"lr": 0.0, "max_grad_norm": 1.0},
            global_batch_size=1,
            micro_batch_size=1,
        )
        trainer = SimpleTrainer(
            model_cls=ScalarWeightModel,
            optimizer_cls=SGD,
            loss_fn=lambda pred, target: (
                (pred - Tensor(target, requires_grad=False)) ** 2
            ).mean(),
            config=config,
        )
        train_loader = make_data_loader(
            [
                (
                    xp.array([[1.0]], dtype=xp.float32),
                    xp.array([[10.0]], dtype=xp.float32),
                )
            ]
        )

        with (
            patch.object(
                trainer.optimizer,
                "grad_l2_norm",
                side_effect=AssertionError("trainer should not call grad_l2_norm"),
            ),
            patch.object(
                trainer.optimizer,
                "_clip_grad_norm",
                wraps=trainer.optimizer._clip_grad_norm,
            ) as mock_clip_grad_norm,
        ):
            trainer.fit(train_loader)

        self.assertEqual(mock_clip_grad_norm.call_count, 1)
        self.assertAlmostEqual(trainer.metric_rows[0]["grad_l2_norm"], 20.0, places=5)

    def test_leftover_accumulation_matches_single_batch_update(self):
        single_batch_data = [
            (
                xp.array([[2.0]], dtype=xp.float32),
                xp.array([[1.0]], dtype=xp.float32),
            )
        ]

        single_batch_weight = self._fit_scalar_weight_model(
            single_batch_data,
            global_batch_size=1,
            micro_batch_size=1,
        )
        leftover_accumulated_weight = self._fit_scalar_weight_model(
            single_batch_data,
            global_batch_size=2,
            micro_batch_size=1,
        )

        self.assertAlmostEqual(
            leftover_accumulated_weight,
            single_batch_weight,
            places=6,
        )

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

    def test_config_rejects_both_training_limits(self):
        with self.assertRaisesRegex(
            ValueError, "max_epochs and max_steps are mutually exclusive"
        ):
            GenericTrainingConfig(
                max_epochs=1,
                max_steps=1,
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
        val_loader = make_data_loader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss.val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 2)

    def test_evaluate_respects_max_eval_steps_from_config(self):
        self.config.max_eval_steps = 1
        val_loader = make_data_loader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss.val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 1)

    def test_evaluate_rejects_empty_validation_loader(self):
        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches.",
        ):
            self.trainer.evaluate(make_data_loader([]))

    def test_evaluate_owns_eval_and_train_mode_switching(self):
        trainer = SimpleTrainer(
            model_cls=TrackingModeModel,
            optimizer_cls=MockOptimizerClass,
            loss_fn=self.loss_fn,
            config=self.config,
            output_type="logits",
        )
        eval_state = TrainingState()
        eval_state.record_eval_loss(1.23)
        trainer._evaluate = MagicMock(return_value=eval_state)

        trainer.evaluate(self.val_loader)

        self.assertEqual(trainer.model.mode_log, ["eval", "train"])

    def test_evaluate_runs_under_no_grad(self):
        def assert_no_grad(_):
            self.assertFalse(is_grad_enabled())
            eval_state = TrainingState()
            eval_state.record_eval_loss(1.23)
            return eval_state

        self.trainer._evaluate = MagicMock(side_effect=assert_no_grad)

        eval_result = self.trainer.evaluate(self.val_loader)

        self.assertAlmostEqual(float(eval_result.val_loss), 1.23, places=5)

    def test_report_prefixes_eval_metrics_with_val(self):
        self.trainer.global_step = len(self.train_loader)
        plan = TrainingPlan.for_epochs(
            max_epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            checkpoint_every=self.config.checkpoint_freq,
        )
        state = TrainingState()
        state.record_loss(1.23)
        self.trainer.last_grad_l2_norm = 7.0

        eval_state = TrainingState()
        eval_state.record_eval_loss(0.5)
        eval_state.record_eval_metric("accuracy", numerator=3, denominator=4)
        self.trainer.report(
            state,
            plan,
            eval_state,
        )

        row = self.trainer.metric_rows[0]
        self.assertEqual(row["train_loss"], 1.23)
        self.assertEqual(row["val_loss"], 0.5)
        self.assertEqual(row["val_accuracy"], 0.75)

    def test_report_logs_pretty_printed_metrics_dict(self):
        self.trainer.global_step = len(self.train_loader)
        plan = TrainingPlan.for_epochs(
            max_epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            checkpoint_every=self.config.checkpoint_freq,
        )
        state = TrainingState()
        state.record_loss(1.23)
        self.trainer.last_grad_l2_norm = 7.0

        eval_state = TrainingState()
        eval_state.record_eval_loss(0.5)
        eval_state.record_eval_metric("accuracy", numerator=3, denominator=4)

        with self.assertLogs("autograd.tools.trainer", level="INFO") as captured:
            self.trainer.report(state, plan, eval_state)

        joined_logs = "\n".join(captured.output)
        self.assertIn("[Epoch 1]", joined_logs)
        self.assertIn("'epoch': 1", joined_logs)
        self.assertIn("'step': 2", joined_logs)
        self.assertIn("'train_loss': '1.2300'", joined_logs)
        self.assertIn("'val_loss': '0.5000'", joined_logs)
        self.assertIn("'val_accuracy': '0.7500'", joined_logs)
        self.assertIn("'grad_l2_norm': '7.0000'", joined_logs)
        self.assertIn("'lr': '0.010000'", joined_logs)

    def test_fit_plan_derives_step_mode_fields_from_config(self):
        config = GenericTrainingConfig(
            max_steps=3,
            checkpoint_freq=2,
            report_every_steps=1,
            model_kwargs={"hidden_size": 256},
            optimizer_kwargs={"lr": 0.01},
        )

        plan = TrainingPlan.for_steps(
            max_steps=config.max_steps,
            report_every_steps=config.report_every_steps,
            checkpoint_every=config.checkpoint_freq,
        )

        self.assertFalse(plan.by_epoch)
        self.assertEqual(plan.target_step, 3)
        self.assertEqual(plan.report_every_steps, 1)
        self.assertEqual(plan.checkpoint_every, 2)
        self.assertIsNone(plan.steps_per_epoch)
        self.assertEqual(plan.metrics_interval, 1)

    def test_fit_plan_derives_epoch_mode_fields_from_config(self):
        plan = TrainingPlan.for_epochs(
            max_epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            checkpoint_every=self.config.checkpoint_freq,
        )

        self.assertTrue(plan.by_epoch)
        self.assertEqual(
            plan.target_step,
            self.config.max_epochs * len(self.train_loader),
        )
        self.assertEqual(plan.report_every_steps, len(self.train_loader))
        self.assertEqual(plan.checkpoint_every, self.config.checkpoint_freq)
        self.assertEqual(plan.steps_per_epoch, len(self.train_loader))
        self.assertEqual(plan.metrics_interval, 1)

    def test_evaluate_does_not_mutate_trainer_metrics(self):
        eval_result = self.trainer.evaluate(self.val_loader)

        self.assertAlmostEqual(eval_result.val_loss, 1.23, places=5)
        self.assertEqual(eval_result.eval_metrics["accuracy"], 1.0)
        self.assertEqual(self.trainer.metric_rows, [])

    def test_evaluate_returns_raw_eval_aggregates(self):
        eval_state = self.trainer.evaluate(self.val_loader)

        self.assertAlmostEqual(eval_state.eval_loss_sum, 1.23, places=5)
        self.assertEqual(eval_state.eval_loss_batches, 1)
        self.assertEqual(eval_state.eval_metric_totals["accuracy"]["numerator"], 2.0)
        self.assertEqual(
            eval_state.eval_metric_totals["accuracy"]["denominator"],
            2.0,
        )


class TestLLMTrainer(BaseTrainerTest):
    def setUp(self):
        super().setUp()
        seq_len = 10

        # For LLM, define a shape like (batch_size, seq_len, vocab_size).
        self.fake_pred = xp.zeros((2, seq_len, 10), dtype=xp.float32)

        # LLM data (just 2 batches for training, 1 batch for validation)
        self.train_data = [
            CausalLMBatch(
                input_ids=xp.zeros((2, seq_len), dtype=xp.int32),
                labels=xp.full((2, seq_len), 2, dtype=xp.int32),
            )
            for _ in range(2)
        ]
        self.val_data = [
            CausalLMBatch(
                input_ids=xp.zeros((2, seq_len), dtype=xp.int32),
                labels=xp.full((2, seq_len), 2, dtype=xp.int32),
            )
        ]

        # Replace MagicMock with a real in-memory loader
        self.train_loader = make_data_loader(self.train_data)
        self.val_loader = make_data_loader(self.val_data)

        # We'll mock forward_fn to skip real model logic
        self.forward_fn = MagicMock()
        self.forward_fn.train.return_value = Tensor(self.fake_pred)
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
            teacher_forcing=False,
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

    def test_compute_loss_returns_tensor(self):
        """Check that compute_loss returns the constant scalar Tensor."""
        batch = next(iter(self.train_data))
        loss = self.trainer._compute_loss(batch)
        self.assertAlmostEqual(float(loss.item()), 1.23, places=5)

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

        self.assertAlmostEqual(avg_val_loss.val_loss, 1.23, places=5)
        self.assertEqual(avg_val_loss.eval_metrics, {})
        callback.assert_called_once_with(
            trainer.model, trainer.forward_fn, self.val_loader, trainer.config
        )

    def test_llm_evaluate_rejects_empty_val_loader(self):
        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches.",
        ):
            self.trainer.evaluate(make_data_loader([]))

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
        val_loader = make_data_loader(self.val_data * 2)

        avg_val_loss = self.trainer.evaluate(val_loader)

        self.assertAlmostEqual(avg_val_loss.val_loss, 1.23, places=5)
        self.assertEqual(self.loss_fn.call_count, 1)

    def test_compute_loss_supports_sft_batches_from_generic_data_loader(self):
        bpe = MockBPE()
        pad_idx = bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
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
            collate_fn=CausalLMCollator(
                max_tokens=6,
                pad_idx=pad_idx,
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
                teacher_forcing=False,
            ),
        )

        batch = next(iter(loader))
        logits = trainer.forward_fn.train(trainer.model, batch)
        expected_loss = cross_entropy(
            logits,
            batch.labels,
            label_smoothing=0.0,
        )

        train_loss = trainer._compute_loss(batch)
        val_loss = trainer.evaluate(loader)

        self.assertAlmostEqual(
            float(train_loss.item()), float(expected_loss.item()), places=6
        )
        self.assertAlmostEqual(val_loss.val_loss, float(expected_loss.item()), places=6)

    def test_fit_supports_unsized_streaming_data_loaders(self):
        self.config.max_epochs = None
        self.config.max_steps = 1
        self.config.max_eval_steps = 1

        stream_loader = DataLoader(
            dataset=TokenWindowDataset(
                xp.arange(100, dtype=xp.int32),
                window_len=5,
                sampling="sequential",
            ),
            batch_size=2,
            collate_fn=CausalLMWindowCollator(),
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
