import unittest
from unittest.mock import patch

from autograd.backend import xp
from autograd.data.collator import (
    CausalLMWindowCollator,
    PairedCollator,
    Seq2SeqCollator,
)
from autograd.data.data_loader import DataLoader
from autograd.data.dataset import (
    MapDataset,
    PairedMapDataset,
    TokenWindowMapDataset,
)
from autograd.data.sampler import (
    RandomSampler,
    Sampler,
    SequentialSampler,
    TokenLengthGroupedRandomSampler,
)
from autograd.data.types import CausalLMBatch, Seq2SeqBatch


def mock_padding_mask(X_chunk, pad_idx, dims=None):
    if dims is not None:
        return xp.zeros(dims)
    batch_size, seq_len = X_chunk.shape
    return xp.zeros((batch_size, 1, 1, seq_len))


def make_token_dataset(token_sequences, loss_masks=None):
    if loss_masks is None:
        loss_masks = [
            xp.ones((len(tokens),), dtype=xp.int32) for tokens in token_sequences
        ]
    return PairedMapDataset(
        token_sequences,
        loss_masks,
        input_key="tokens",
        target_key="loss_mask",
        dtype=xp.int32,
    )


class MockBPE:
    def encode(self, token, allowed_special=set()):
        if token in allowed_special:
            if token == "<PAD>":
                return [0]
            if token == "<SOS>":
                return [1]
            if token == "<|endoftext|>":
                return [2]
        special_token = "<|endoftext|>"
        if token == special_token:
            return [2]

        encoded = []
        start = 0
        while True:
            special_index = token.find(special_token, start)
            if special_index == -1:
                encoded.extend(ord(char) for char in token[start:])
                break
            encoded.extend(ord(char) for char in token[start:special_index])
            encoded.append(2)
            start = special_index + len(special_token)
        return encoded


class StaticSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.X = xp.arange(20).reshape(10, 2)
        self.y = xp.arange(10)
        self.data = xp.arange(200)
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
            dataset = PairedMapDataset(
                sources,
                targets,
                input_key="input_ids",
                target_key="labels",
                dtype=xp.int32,
            )
            sampler = RandomSampler(dataset) if shuffle else None
            collator = Seq2SeqCollator(
                max_tokens=seq_len,
                pad_idx=pad_idx,
                sos_idx=self.bpe.encode("<SOS>", allowed_special={"<SOS>"})[0],
            )
        else:
            dataset = TokenWindowMapDataset(
                xp.array(data),
                window_len=seq_len + 1,
            )
            sampler = (
                RandomSampler(dataset, replacement=True, num_samples=len(dataset))
                if shuffle
                else SequentialSampler(dataset)
            )
            collator = CausalLMWindowCollator()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collator=collator,
            sampler=sampler,
        )

    def test_data_loader_no_shuffle(self):
        dataset = PairedMapDataset(self.X, self.y)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_simple,
            collator=PairedCollator(),
        )

        batches = list(loader)
        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple

        self.assertEqual(len(batches), expected_batches)
        reconstructed_y = xp.concatenate([batch[1] for batch in batches])
        self.assertTrue(xp.array_equal(reconstructed_y, self.y))

    def test_data_loader_length(self):
        loader = DataLoader(
            PairedMapDataset(self.X, self.y),
            batch_size=self.batch_size_simple,
            collator=PairedCollator(),
        )

        expected_batches = (
            len(self.X) + self.batch_size_simple - 1
        ) // self.batch_size_simple

        self.assertEqual(len(loader), expected_batches)

    def test_data_loader_rejects_empty_dataset_iteration(self):
        loader = DataLoader(MapDataset([]), batch_size=1)

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            next(iter(loader))

    def test_data_loader_rejects_empty_dataset_length(self):
        loader = DataLoader(MapDataset([]), batch_size=1)

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            len(loader)

    def test_data_loader_rejects_drop_last_when_only_partial_batch_exists(self):
        loader = DataLoader(
            PairedMapDataset(self.X[:1], self.y[:1]),
            batch_size=2,
            collator=PairedCollator(),
            drop_last=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            next(iter(loader))

    def test_data_loader_rejects_drop_last_zero_batch_length(self):
        loader = DataLoader(
            PairedMapDataset(self.X[:1], self.y[:1]),
            batch_size=2,
            collator=PairedCollator(),
            drop_last=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "DataLoader yielded no batches",
        ):
            len(loader)

    def test_data_loader_rejects_non_integer_sampler_index(self):
        dataset = make_token_dataset([xp.arange(2, dtype=xp.int32)])
        loader = DataLoader(dataset, batch_size=1, sampler=StaticSampler(["0"]))

        with self.assertRaisesRegex(TypeError, "sampler yielded non-integer index"):
            next(iter(loader))

    def test_data_loader_rejects_out_of_range_sampler_index(self):
        dataset = make_token_dataset([xp.arange(2, dtype=xp.int32)])
        loader = DataLoader(dataset, batch_size=1, sampler=StaticSampler([1]))

        with self.assertRaisesRegex(
            IndexError,
            "sampler yielded index 1 outside dataset length 1",
        ):
            next(iter(loader))

    def test_pretraining_data_loader_on_epoch_start_reseeds_without_crashing(self):
        loader = self.make_pretraining_loader(shuffle=True)

        loader.on_epoch_start()

    def test_pretraining_data_loader_length(self):
        loader = self.make_pretraining_loader()
        expected_windows = len(self.data) - self.seq_len
        expected_batches = (
            expected_windows + self.batch_size_llm - 1
        ) // self.batch_size_llm

        self.assertEqual(len(loader), expected_batches)

    def test_data_loader_batches_supervised_examples(self):
        dataset = PairedMapDataset(self.X, self.y)
        loader = DataLoader(
            dataset,
            batch_size=4,
            collator=PairedCollator(),
        )

        batches = list(loader)
        first_X, first_y = batches[0]

        self.assertEqual(len(batches), 3)
        self.assertTrue(xp.array_equal(first_X, self.X[:4]))
        self.assertTrue(xp.array_equal(first_y, self.y[:4]))

    def test_causal_lm_window_loader_yields_expected_next_token_pairs(self):
        dataset = TokenWindowMapDataset(xp.arange(20, dtype=xp.int32), window_len=5)
        loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            collator=CausalLMWindowCollator(),
            sampler=SequentialSampler(dataset),
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
        dataset = TokenWindowMapDataset(xp.arange(5, dtype=xp.int32), window_len=5)
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collator=CausalLMWindowCollator(),
            sampler=SequentialSampler(dataset),
        )

        batch = next(iter(loader))

        self.assertTrue(
            xp.array_equal(batch.input_ids, xp.array([[0, 1, 2, 3]], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(batch.labels, xp.array([[1, 2, 3, 4]], dtype=xp.int32))
        )

    def test_data_loader_returns_causal_lm_batch_with_window_collator(self):
        dataset = TokenWindowMapDataset(xp.arange(20, dtype=xp.int32), window_len=5)
        loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            collator=CausalLMWindowCollator(),
            sampler=SequentialSampler(dataset),
        )

        batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertTrue(xp.array_equal(batch.input_ids[0], xp.array([0, 1, 2, 3])))

    def test_data_loader_without_collator_yields_raw_window_examples(self):
        dataset = TokenWindowMapDataset(xp.arange(20, dtype=xp.int32), window_len=5)
        loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            sampler=SequentialSampler(dataset),
        )

        batch = next(iter(loader))

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].offset, 0)
        self.assertEqual(batch[1].offset, 1)

    def test_data_loader_can_use_length_grouped_sampler(self):
        dataset = make_token_dataset(
            [
                xp.arange(2, dtype=xp.int32),
                xp.arange(8, dtype=xp.int32),
                xp.arange(3, dtype=xp.int32),
                xp.arange(7, dtype=xp.int32),
            ],
            [
                xp.ones((2,), dtype=xp.int32),
                xp.ones((8,), dtype=xp.int32),
                xp.ones((3,), dtype=xp.int32),
                xp.ones((7,), dtype=xp.int32),
            ],
        )

        batches = list(
            DataLoader(
                dataset,
                batch_size=2,
                sampler=TokenLengthGroupedRandomSampler(
                    dataset,
                    sort_buffer_size=4,
                ),
            )
        )
        batch_lengths = [
            [len(example["tokens"]) for example in batch] for batch in batches
        ]

        self.assertEqual(batch_lengths, [[2, 3], [7, 8]])

    @patch("autograd.data.dataset.xp.random.permutation")
    def test_length_grouped_sampler_shuffle_stays_on_cpu(
        self,
        backend_permutation,
    ):
        backend_permutation.side_effect = AssertionError(
            "backend permutation should not be used"
        )

        dataset = make_token_dataset(
            [
                xp.arange(2, dtype=xp.int32),
                xp.arange(8, dtype=xp.int32),
                xp.arange(3, dtype=xp.int32),
                xp.arange(7, dtype=xp.int32),
            ],
            [
                xp.ones((2,), dtype=xp.int32),
                xp.ones((8,), dtype=xp.int32),
                xp.ones((3,), dtype=xp.int32),
                xp.ones((7,), dtype=xp.int32),
            ],
        )
        sampler = TokenLengthGroupedRandomSampler(
            dataset,
            sort_buffer_size=4,
        )

        sampler.on_epoch_start()

        self.assertIsInstance(sampler.indices, list)
        backend_permutation.assert_not_called()

    @patch(
        "autograd.data.collator.create_padding_mask",
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
        sos_idx = self.bpe.encode("<SOS>", allowed_special={"<SOS>"})[0]
        self.assertTrue(xp.all(xp.asarray(batch.decoder_input_ids[:, 0] == sos_idx)))
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
