from unittest import TestCase

from autograd import distributed as dist
from autograd.tools.config_schema import CustomBpeConfig, TransformerTrainingConfig
from test.distributed.mock import MockBackend, MockComm


class TestTransformerTrainingConfig(TestCase):
    def test_custom_bpe_rejects_non_positive_parquet_shards_per_batch(self):
        with self.assertRaisesRegex(
            ValueError, "parquet_shards_per_batch must be >= 1"
        ):
            CustomBpeConfig(
                num_merges=10,
                encoded_data_path="encoded.npz",
                vocab_path="vocab.pkl",
                overwrite_encoded_data=False,
                overwrite_vocabulary_file=False,
                start_token="<SOS>",
                split_token="<|endoftext|>",
                parquet_shards_per_batch=0,
            )

    def test_custom_bpe_rejects_non_positive_n_workers(self):
        with self.assertRaisesRegex(ValueError, "n_workers must be >= 1"):
            CustomBpeConfig(
                num_merges=10,
                encoded_data_path="encoded.npz",
                vocab_path="vocab.pkl",
                overwrite_encoded_data=False,
                overwrite_vocabulary_file=False,
                start_token="<SOS>",
                split_token="<|endoftext|>",
                n_workers=0,
            )

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

    def test_allows_checkpoint_frequency_before_accumulation_boundary(self):
        config = TransformerTrainingConfig(
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

        self.assertEqual(config.gradient_accumulation_steps, 8)

    def test_rejects_global_batch_size_not_divisible_by_distributed_world_size(self):
        dist._set_thread_local_rank(
            rank_=0,
            world_size_=3,
            local_rank_=0,
            backend=MockBackend(MockComm(3), 0),
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                r"global_batch_size \(144\) must be divisible by "
                r"micro_batch_size \* world_size \(9 \* 3 = 27\)",
            ):
                TransformerTrainingConfig(
                    training_run_name="test",
                    dataset_name="dataset",
                    max_steps=5,
                    checkpoint_freq=5,
                    global_batch_size=144,
                    micro_batch_size=9,
                    model_kwargs={},
                    optimizer_kwargs={"lr": 1e-3},
                    label_smoothing=0.0,
                    teacher_forcing=False,
                )
        finally:
            dist._clear_thread_local()

    def test_allows_global_batch_size_divisible_by_distributed_world_size(self):
        dist._set_thread_local_rank(
            rank_=0,
            world_size_=4,
            local_rank_=0,
            backend=MockBackend(MockComm(4), 0),
        )
        try:
            config = TransformerTrainingConfig(
                training_run_name="test",
                dataset_name="dataset",
                max_steps=5,
                checkpoint_freq=5,
                global_batch_size=144,
                micro_batch_size=9,
                model_kwargs={},
                optimizer_kwargs={"lr": 1e-3},
                label_smoothing=0.0,
                teacher_forcing=False,
            )
            self.assertEqual(config.gradient_accumulation_steps, 4)
        finally:
            dist._clear_thread_local()
