from unittest import TestCase

from autograd.tools.config_schema import TransformerTrainingConfig


class TestTransformerTrainingConfig(TestCase):
    def test_rejects_non_positive_max_grad_norm(self):
        with self.assertRaisesRegex(ValueError, "max_grad_norm must be > 0"):
            TransformerTrainingConfig(
                training_run_name="test",
                dataset_name="dataset",
                max_steps=5,
                checkpoint_freq=5,
                global_batch_size=1,
                micro_batch_size=1,
                model_kwargs={},
                optimizer_kwargs={"lr": 1e-3},
                max_grad_norm=0.0,
                label_smoothing=0.0,
                teacher_forcing=False,
            )

    def test_rejects_checkpoint_before_gradient_accumulation_boundary(self):
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_freq must be divisible by gradient_accumulation_steps",
        ):
            TransformerTrainingConfig(
                training_run_name="test",
                dataset_name="dataset",
                max_steps=5,
                checkpoint_freq=5,
                global_batch_size=32,
                micro_batch_size=4,
                model_kwargs={},
                optimizer_kwargs={"lr": 1e-3},
                label_smoothing=0.0,
                teacher_forcing=False,
            )
