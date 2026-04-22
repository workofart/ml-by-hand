import unittest
from unittest.mock import patch

from autograd.backend import xp
from autograd.data.collator import (
    CausalLMCollator,
    CausalLMWindowCollator,
    PairedCollator,
    Seq2SeqCollator,
)
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import (
    IterableDataset,
    PairedIterableDataset,
    Seq2SeqDataset,
    TokenSequenceDataset,
    TokenWindowDataset,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch
from autograd.data.utils import (
    openai_chat_to_prompt_completion,
    tokenize_prompt_completion,
)
from autograd.functional import IGNORE_INDEX


def mock_padding_mask(X_chunk, pad_idx):
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            if token == "<SOS>":
                return [1]
        return [ord(char) for char in token]


class EmptyDataset(IterableDataset):
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class BuggyEmptyPassDataset(IterableDataset):
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 3


class TestDataLoader(unittest.TestCase):
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
        self.bpe = MockBPE()

    def make_pretraining_loader(
        self,
        data=None,
        *,
        batch_size=None,
        seq_len=None,
        shuffle=True,
        encoder_decoder=False,
    ):
        data = self.data if data is None else data
        batch_size = self.batch_size_llm if batch_size is None else batch_size
        seq_len = self.seq_len if seq_len is None else seq_len
        pad_idx = self.bpe.encode("<PAD>", allowed_special={"<PAD>"})[0]
        if encoder_decoder:
            data = xp.array(data, dtype=xp.int32)
            window_count = len(data) - (2 * seq_len) + 1
            if window_count < 1:
                raise ValueError("data too small for encoder-decoder examples")
            sources = [
                data[offset : offset + seq_len] for offset in range(window_count)
            ]
            targets = [
                data[offset + seq_len : offset + (2 * seq_len)]
                for offset in range(window_count)
            ]
            dataset = Seq2SeqDataset(
                input_sequences=sources,
                label_sequences=targets,
                shuffle=shuffle,
            )
            collator = Seq2SeqCollator(
                max_tokens=seq_len,
                pad_idx=pad_idx,
                sos_idx=self.bpe.encode("<SOS>", allowed_special={"<SOS>"})[0],
            )
        else:
            dataset = TokenWindowDataset(
                xp.array(data),
                window_len=seq_len + 1,
                sampling="random" if shuffle else "sequential",
            )
            collator = CausalLMWindowCollator()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
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
            collate_fn=CausalLMCollator(
                max_tokens=seq_len + 1,
                pad_idx=pad_idx,
            ),
        )

    def test_data_loader_no_shuffle(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_simple,
            collate_fn=PairedCollator(),
        )

        expected_indices = xp.arange(len(self.X))
        batches = list(loader)
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple

        self.assertTrue(xp.array_equal(dataset.indices, expected_indices))
        self.assertEqual(len(batches), expected_batches)
        reconstructed_y = xp.concatenate([batch[1] for batch in batches])
        self.assertTrue(xp.array_equal(reconstructed_y, self.y))

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

    def test_data_loader_rejects_empty_dataset_iteration(self):
        loader = DataLoader(EmptyDataset(), batch_size=1)

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            next(iter(loader))

    def test_data_loader_rejects_dataset_that_yields_nothing_for_pass(self):
        loader = DataLoader(BuggyEmptyPassDataset(), batch_size=2)

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            next(iter(loader))

    def test_data_loader_rejects_drop_last_when_only_partial_batch_exists(self):
        loader = DataLoader(
            PairedIterableDataset(self.X[:1], self.y[:1], shuffle=False),
            batch_size=2,
            collate_fn=PairedCollator(),
            drop_last=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            next(iter(loader))

    def test_pretraining_data_loader_on_epoch_start_reseeds_without_crashing(self):
        loader = self.make_pretraining_loader(shuffle=True)

        loader.on_epoch_start()

    def test_pretraining_data_loader_length(self):
        loader_infinite = self.make_pretraining_loader()

        with self.assertRaises(TypeError):
            _ = len(loader_infinite)

    def test_data_loader_batches_supervised_examples(self):
        dataset = PairedIterableDataset(self.X, self.y, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=PairedCollator(),
        )

        batches = list(loader)
        first_X, first_y = batches[0]

        self.assertEqual(len(batches), 3)
        self.assertTrue(xp.array_equal(first_X, self.X[:4]))
        self.assertTrue(xp.array_equal(first_y, self.y[:4]))

    def test_causal_lm_window_loader_yields_expected_next_token_pairs(self):
        loader = DataLoader(
            dataset=TokenWindowDataset(
                xp.arange(20, dtype=xp.int32),
                window_len=5,
                sampling="sequential",
            ),
            batch_size=2,
            collate_fn=CausalLMWindowCollator(),
        )

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertTrue(
            xp.array_equal(
                batch.input_ids,
                xp.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(
                batch.labels,
                xp.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=xp.int32),
            )
        )

    def test_causal_lm_window_loader_allows_minimal_valid_stream(self):
        loader = DataLoader(
            dataset=TokenWindowDataset(
                xp.arange(5, dtype=xp.int32),
                window_len=5,
                sampling="sequential",
            ),
            batch_size=1,
            collate_fn=CausalLMWindowCollator(),
        )

        batch = next(iter(loader))

        self.assertTrue(
            xp.array_equal(batch.input_ids, xp.array([[0, 1, 2, 3]], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(batch.labels, xp.array([[1, 2, 3, 4]], dtype=xp.int32))
        )

    def test_data_loader_returns_causal_lm_batch_with_window_collator(self):
        loader = DataLoader(
            dataset=TokenWindowDataset(
                xp.arange(20, dtype=xp.int32),
                window_len=5,
                sampling="sequential",
            ),
            batch_size=2,
            collate_fn=CausalLMWindowCollator(),
        )

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertTrue(xp.array_equal(batch.input_ids[0], xp.array([0, 1, 2, 3])))

    def test_data_loader_without_collator_yields_raw_window_examples(self):
        loader = DataLoader(
            dataset=TokenWindowDataset(
                xp.arange(20, dtype=xp.int32),
                window_len=5,
                sampling="sequential",
            ),
            batch_size=2,
        )

        batch = next(iter(loader))

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].offset, 0)
        self.assertEqual(batch[1].offset, 1)

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

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertEqual(batch.input_ids.shape, (1, 5))
        self.assertEqual(batch.labels.shape, (1, 5))
        self.assertTrue(
            xp.array_equal(
                batch.labels[0],
                xp.array(
                    [IGNORE_INDEX, IGNORE_INDEX, 68, 69, IGNORE_INDEX],
                    dtype=xp.int32,
                ),
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

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertEqual(batch.input_ids.shape, (1, 5))
        self.assertEqual(batch.labels.shape, (1, 5))
        self.assertTrue(
            xp.array_equal(
                batch.input_ids[0],
                xp.array([65, 66, 67, 68, 69], dtype=xp.int32),
            )
        )
        self.assertTrue(
            xp.array_equal(
                batch.labels[0],
                xp.array(
                    [IGNORE_INDEX, IGNORE_INDEX, 68, 69, IGNORE_INDEX],
                    dtype=xp.int32,
                ),
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

        batch = next(iter(loader))

        self.assertTrue(
            xp.array_equal(
                batch.input_ids[0], xp.array([67, 68, 69, 70], dtype=xp.int32)
            )
        )
        self.assertTrue(
            xp.array_equal(
                batch.labels[0],
                xp.array([IGNORE_INDEX, 69, 70, 71], dtype=xp.int32),
            )
        )

    @patch(
        "autograd.data.collator.text_utils.create_padding_mask",
        side_effect=mock_padding_mask,
    )
    def test_encoder_decoder_pretraining_loader_output(self, mock_padding):
        loader = self.make_pretraining_loader(encoder_decoder=True)

        batch = next(iter(loader))

        self.assertIsInstance(batch, Seq2SeqBatch)
        self.assertEqual(batch.input_ids.shape, (self.batch_size_llm, self.seq_len))
        self.assertEqual(batch.labels.shape, (self.batch_size_llm, self.seq_len))
        self.assertEqual(
            batch.decoder_input_ids.shape, (self.batch_size_llm, self.seq_len)
        )
        self.assertTrue(
            xp.all(
                xp.asarray(batch.decoder_input_ids[:, 0] == loader.collate_fn.sos_idx)
            )
        )
        self.assertFalse(hasattr(batch, "loss_mask"))
        self.assertEqual(
            batch.src_mask.shape, (self.batch_size_llm, 1, 1, self.seq_len)
        )
        self.assertEqual(
            batch.tgt_mask.shape, (self.batch_size_llm, 1, 1, self.seq_len)
        )

    def test_causal_pretraining_loader_output(self):
        loader = self.make_pretraining_loader()

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertEqual(batch.input_ids.shape, (self.batch_size_llm, self.seq_len))
        self.assertEqual(batch.labels.shape, (self.batch_size_llm, self.seq_len))
