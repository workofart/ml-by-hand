import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from autograd.backend import xp
from autograd.tools.data import (
    DataLoader,
    LanguageModelingCollator,
    OneHotCollator,
    PairedCollator,
    PairedIterableDataset,
    TokenSequenceDataset,
    TransformDataset,
    load_data,
    openai_chat_to_prompt_completion,
    pack_tokens,
    tokenize_prompt_completion,
)


def mock_padding_mask(X_chunk, pad_idx):
    # For testing, assume no actual padding occurs.
    # Return a zero mask of shape (batch_size, 1, 1, seq_len)
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            elif token == "<SOS>":
                return [1]
        return [ord(char) for char in token]


class TestDataLoaders(unittest.TestCase):
    def setUp(self):
        self.X = xp.arange(20).reshape(10, 2)
        self.y = xp.arange(10)
        self.data = xp.arange(200)
        self.chat_examples = [
            {
                "messages": [
                    {"role": "system", "content": "ABC"},
                    {"role": "assistant", "content": "DE"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "RS"},
                ]
            },
        ]
        self.seq_len = 10
        self.batch_size_simple = 3
        self.batch_size_llm = 4
        self.steps = 5
        self.bpe = MockBPE()

    def make_pretraining_loader(
        self,
        data=None,
        *,
        batch_size=None,
        seq_len=None,
        shuffle=True,
        include_decoder_input=True,
        create_padding_masks=True,
    ):
        data = self.data if data is None else data
        batch_size = self.batch_size_llm if batch_size is None else batch_size
        seq_len = self.seq_len if seq_len is None else seq_len
        pad_idx = self.bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        sos_idx = self.bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        dataset = TokenSequenceDataset(
            data=xp.array(data),
            seq_len=seq_len,
            shuffle=shuffle,
            random_window=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=LanguageModelingCollator(
                max_tokens=seq_len + 1,
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                include_decoder_input=include_decoder_input,
                create_padding_masks=create_padding_masks,
            ),
        )

    def make_sft_loader(
        self,
        chat_examples=None,
        *,
        batch_size=1,
        seq_len=5,
        shuffle=False,
    ):
        chat_examples = (
            chat_examples if chat_examples is not None else self.chat_examples
        )
        pad_idx = self.bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        sos_idx = self.bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        tokenized_examples = [
            tokenize_prompt_completion(
                openai_chat_to_prompt_completion(example), self.bpe
            )
            for example in chat_examples
        ]
        return DataLoader(
            dataset=TokenSequenceDataset(
                token_sequences=[example["tokens"] for example in tokenized_examples],
                loss_masks=[example["loss_mask"] for example in tokenized_examples],
                shuffle=shuffle,
            ),
            batch_size=batch_size,
            collate_fn=LanguageModelingCollator(
                max_tokens=seq_len + 1,
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                include_decoder_input=False,
                create_padding_masks=False,
            ),
        )

    def test_data_loader_no_shuffle(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_simple,
            collate_fn=PairedCollator(),
        )
        self.assertEqual(list(dataset.indices), list(range(len(self.X))))
        batches = list(loader)
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(batches), expected_batches)
        reconstructed_y = xp.concatenate([batch[1] for batch in batches])
        self.assertTrue(xp.array_equal(reconstructed_y, self.y))

    def test_paired_iterable_dataset_shuffle(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=True)
        dataset.on_epoch_start()
        self.assertIsInstance(dataset.indices, list)
        self.assertEqual(sorted(dataset.indices), list(range(len(self.X))))

    def test_data_loader_length(self):
        loader = DataLoader(
            PairedIterableDataset(self.X, self.y, shuffle=False),
            batch_size=self.batch_size_simple,
            collate_fn=PairedCollator(),
        )
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple
        self.assertEqual(len(loader), expected_batches)

    def test_pretraining_data_loader_on_epoch_start_reseeds_without_crashing(self):
        loader = self.make_pretraining_loader(shuffle=True)

        loader.on_epoch_start()

    def test_transform_dataset_applies_transform_and_target_transform(self):
        dataset = TransformDataset(
            PairedIterableDataset(
                xp.array([[1, 2], [3, 4]]),
                xp.array([10, 20]),
                shuffle=False,
            ),
            transform=lambda x: x * 2,
            target_transform=lambda y: y * 3,
        )

        examples = list(dataset)

        self.assertTrue(xp.array_equal(examples[0]["inputs"], xp.array([2, 4])))
        self.assertEqual(int(examples[0]["targets"]), 30)
        self.assertTrue(xp.array_equal(examples[1]["inputs"], xp.array([6, 8])))
        self.assertEqual(int(examples[1]["targets"]), 60)

    def test_pack_tokens_left_truncates_then_pads(self):
        packed = pack_tokens(
            tokens=xp.array([10, 11, 12, 13, 20, 21, 22], dtype=xp.int32),
            max_tokens=5,
            pad_idx=0,
        )

        self.assertTrue(
            xp.array_equal(packed, xp.array([12, 13, 20, 21, 22], dtype=xp.int32))
        )

    def test_paired_iterable_dataset_yields_single_example(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("inputs", "targets"))
        self.assertTrue(xp.array_equal(example["inputs"], self.X[0]))
        self.assertEqual(int(example["targets"]), int(self.y[0]))

    def test_transform_dataset_applies_target_transform(self):
        dataset = TransformDataset(
            PairedIterableDataset(self.X, self.y, shuffle=False),
            target_transform=lambda y: xp.array(int(y == 3), dtype=xp.int32),
        )

        examples = list(dataset)

        self.assertTrue(xp.array_equal(examples[0]["inputs"], self.X[0]))
        self.assertEqual(int(examples[2]["targets"]), 0)
        self.assertEqual(int(examples[3]["targets"]), 1)

    def test_paired_collator_batches_examples(self):
        collator = PairedCollator()

        batch_X, batch_y = collator(
            [
                {
                    "inputs": xp.array([1, 2], dtype=xp.int32),
                    "targets": xp.array(3, dtype=xp.int32),
                },
                {
                    "inputs": xp.array([4, 5], dtype=xp.int32),
                    "targets": xp.array(6, dtype=xp.int32),
                },
            ]
        )

        self.assertTrue(
            xp.array_equal(
                batch_X,
                xp.array([[1, 2], [4, 5]], dtype=xp.int32),
            )
        )
        self.assertTrue(xp.array_equal(batch_y, xp.array([3, 6], dtype=xp.int32)))

    def test_paired_collator_accepts_numpy_scalar_targets(self):
        collator = PairedCollator()

        _, batch_y = collator(
            [
                {
                    "inputs": xp.array([1, 2], dtype=xp.float32),
                    "targets": np.int64(3),
                },
                {
                    "inputs": xp.array([4, 5], dtype=xp.float32),
                    "targets": np.int64(6),
                },
            ]
        )

        self.assertTrue(xp.array_equal(batch_y, xp.array([3, 6], dtype=xp.int64)))

    def test_data_loader_batches_supervised_examples(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=PairedCollator(),
        )

        batches = list(loader)

        self.assertEqual(len(batches), 3)
        first_X, first_y = batches[0]
        self.assertTrue(xp.array_equal(first_X, self.X[:4]))
        self.assertTrue(xp.array_equal(first_y, self.y[:4]))

    def test_data_loader_batch_token_count_defaults_to_none_for_paired_batches(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=PairedCollator(),
        )

        batch = next(iter(loader))

        self.assertIsNone(loader.batch_token_count(batch))

    def test_data_loader_batch_token_count_delegates_to_lm_collator(self):
        loader = self.make_pretraining_loader(batch_size=2, seq_len=4, shuffle=False)

        batch = next(iter(loader))

        self.assertEqual(loader.batch_token_count(batch), 8)

    def test_one_hot_collator_materializes_one_hot_inputs(self):
        collator = OneHotCollator(num_classes=4)

        batch_X, batch_y = collator(
            [
                {
                    "inputs": xp.array([0, 2], dtype=xp.int32),
                    "targets": xp.array(1, dtype=xp.int32),
                },
                {
                    "inputs": xp.array([1, 3], dtype=xp.int32),
                    "targets": xp.array(0, dtype=xp.int32),
                },
            ]
        )

        expected_X = xp.array(
            [
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 0, 1]],
            ],
            dtype=xp.float32,
        )
        self.assertTrue(xp.array_equal(batch_X, expected_X))
        self.assertTrue(xp.array_equal(batch_y, xp.array([1, 0], dtype=xp.int32)))

    def test_load_data_reads_parquet_without_pandas(self):
        rows = [
            {
                "url": "u1",
                "title": "t1",
                "summary": "s1",
                "article": "a1",
                "step_headers": "h1",
            },
            {
                "url": "u2",
                "title": "t2",
                "summary": "s2",
                "article": "a2",
                "step_headers": "h2",
            },
        ]
        table = pa.Table.from_pylist(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "sample.parquet")
            pq.write_table(table, parquet_path)

            data = load_data("unused", parquet_path, max_rows=1)

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["summary"], "s1")
        self.assertEqual(data[0]["article"], "a1")

    def test_load_data_reads_csv_without_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("review,sentiment\n")
                handle.write('"great movie, would watch again",positive\n')
                handle.write('"bad ending",negative\n')

            rows = load_data(csv_path, csv_path)

        self.assertEqual(
            rows,
            [
                ["great movie, would watch again", "positive"],
                ["bad ending", "negative"],
            ],
        )

    # Use decorators to patch the text_utils functions for LM collator tests.
    @patch(
        "autograd.tools.data.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_pretraining_data_loader_length(self, mock_padding):
        loader_infinite = self.make_pretraining_loader()
        with self.assertRaises(TypeError):
            _ = len(loader_infinite)

    def test_random_window_token_sequence_dataset_yields_single_example(self):
        dataset = TokenSequenceDataset(
            data=self.data,
            seq_len=4,
            shuffle=False,
            random_window=True,
        )

        example = next(iter(dataset))

        self.assertEqual(tuple(example.keys()), ("tokens", "loss_mask"))
        self.assertEqual(example["tokens"].shape, (5,))
        self.assertTrue(
            xp.array_equal(example["loss_mask"], xp.ones((5,), dtype=xp.int32))
        )

    def test_openai_chat_to_prompt_completion_extracts_text(self):
        example = openai_chat_to_prompt_completion(
            {
                "messages": [
                    {"role": "system", "content": "ABC"},
                    {"role": "assistant", "content": "DE"},
                ]
            }
        )

        self.assertEqual(example["prompt_text"], "ABC")
        self.assertEqual(example["completion_text"], "DE")

    def test_tokenize_prompt_completion_builds_tokens_and_loss_mask(self):
        example = tokenize_prompt_completion(
            {"prompt_text": "ABC", "completion_text": "DE"},
            self.bpe,
        )

        self.assertTrue(
            xp.array_equal(
                example["tokens"], xp.array([65, 66, 67, 68, 69], dtype=xp.int32)
            )
        )
        self.assertTrue(
            xp.array_equal(
                example["loss_mask"], xp.array([0, 0, 0, 1, 1], dtype=xp.int32)
            )
        )

    def test_token_sequence_dataset_holds_tokenized_prompt_completion_examples(self):
        tokenized_example = tokenize_prompt_completion(
            openai_chat_to_prompt_completion(
                {
                    "messages": [
                        {"role": "system", "content": "ABC"},
                        {"role": "assistant", "content": "DE"},
                    ]
                }
            ),
            self.bpe,
        )
        dataset = TokenSequenceDataset(
            token_sequences=[tokenized_example["tokens"]],
            loss_masks=[tokenized_example["loss_mask"]],
            shuffle=False,
        )

        example = next(iter(dataset))

        self.assertTrue(
            xp.array_equal(
                example["tokens"], xp.array([65, 66, 67, 68, 69], dtype=xp.int32)
            )
        )
        self.assertTrue(
            xp.array_equal(
                example["loss_mask"], xp.array([0, 0, 0, 1, 1], dtype=xp.int32)
            )
        )

    @patch(
        "autograd.tools.data.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_pretraining_data_loader_small_data(self, mock_padding):
        data_small = xp.arange(5)
        loader = self.make_pretraining_loader(
            data=data_small,
            batch_size=2,
        )
        it = iter(loader)
        with self.assertRaises(ValueError):
            next(it)

    @patch(
        "autograd.tools.data.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_pretraining_data_loader_output(self, mock_padding):
        loader = self.make_pretraining_loader()
        batch = next(iter(loader))
        self.assertEqual(len(batch), 5)
        X_chunk, dec_inp, Y_chunk, smask, tmask = batch
        self.assertEqual(X_chunk.shape, (self.batch_size_llm, self.seq_len))
        self.assertEqual(Y_chunk.shape, (self.batch_size_llm, self.seq_len))
        if loader.collate_fn.include_decoder_input:
            self.assertEqual(dec_inp.shape, (self.batch_size_llm, self.seq_len))
            self.assertTrue(
                xp.all(xp.asarray(dec_inp[:, 0] == loader.collate_fn.sos_idx))
            )
        else:
            self.assertIsNone(dec_inp)
        if loader.collate_fn.create_padding_masks:
            self.assertEqual(smask.shape, (self.batch_size_llm, 1, 1, self.seq_len))
            self.assertEqual(tmask.shape, (self.batch_size_llm, 1, 1, self.seq_len))
        else:
            self.assertIsNone(smask)
            self.assertIsNone(tmask)

    @patch(
        "autograd.tools.data.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_pretraining_data_loader_no_decoder_input(self, mock_padding):
        loader = self.make_pretraining_loader(
            include_decoder_input=False,
        )
        batch = next(iter(loader))
        self.assertIsNone(batch[1])

    def test_sft_dataloader_length(self):
        loader = self.make_sft_loader(batch_size=1, seq_len=4, shuffle=False)
        self.assertEqual(len(loader), 2)

    def test_sft_dataloader_masks_prompt_targets(self):
        loader = self.make_sft_loader(
            chat_examples=[
                {
                    "messages": [
                        {"role": "user", "content": "ABC"},
                        {"role": "assistant", "content": "DE"},
                    ]
                }
            ],
            batch_size=1,
            seq_len=5,
            shuffle=False,
        )

        X_chunk, dec_inp, Y_chunk, smask, tmask = next(iter(loader))

        self.assertIsNone(dec_inp)
        self.assertIsNone(smask)
        self.assertIsNone(tmask)
        self.assertEqual(X_chunk.shape, (1, 5))
        self.assertEqual(Y_chunk.shape, (1, 5))
        self.assertTrue(
            xp.array_equal(
                Y_chunk[0],
                xp.array([0, 0, 68, 69, 0], dtype=xp.int32),
            )
        )

    def test_data_loader_can_use_sft_batch_and_label_primitives(self):
        loader = self.make_sft_loader(
            chat_examples=[
                {
                    "messages": [
                        {"role": "user", "content": "ABC"},
                        {"role": "assistant", "content": "DE"},
                    ]
                }
            ],
            batch_size=1,
            seq_len=5,
            shuffle=False,
        )

        X_chunk, dec_inp, Y_chunk, smask, tmask = next(iter(loader))

        self.assertIsNone(dec_inp)
        self.assertIsNone(smask)
        self.assertIsNone(tmask)
        self.assertEqual(X_chunk.shape, (1, 5))
        self.assertEqual(Y_chunk.shape, (1, 5))
        self.assertTrue(
            xp.array_equal(
                X_chunk[0],
                xp.array([65, 66, 67, 68, 69], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(
                Y_chunk[0],
                xp.array([0, 0, 68, 69, 0], dtype=xp.int32),
            )
        )

    def test_sft_dataloader_left_truncates_to_keep_response_tokens(self):
        loader = self.make_sft_loader(
            chat_examples=[
                {
                    "messages": [
                        {"role": "user", "content": "ABCD"},
                        {"role": "assistant", "content": "EFG"},
                    ]
                }
            ],
            batch_size=1,
            seq_len=4,
            shuffle=False,
        )

        X_chunk, _, Y_chunk, _, _ = next(iter(loader))

        self.assertTrue(
            xp.array_equal(X_chunk[0], xp.array([67, 68, 69, 70], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(Y_chunk[0], xp.array([0, 69, 70, 71], dtype=xp.int32))
        )
