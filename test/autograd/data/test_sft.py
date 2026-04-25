import os
import shutil
from json import dumps, loads
from unittest import TestCase
from unittest.mock import MagicMock, patch

from autograd.backend import xp
from autograd.data.sft import (
    load_no_robots_sft,
    load_sft,
    prepare_sft_token_sequences,
    tokenize_sft_messages,
)
from autograd.text.tokenizer import BytePairEncoder
from test.helpers import array_equal


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


class TestSFTData(TestCase):
    def setUp(self):
        self.bpe = BytePairEncoder(
            num_merges=50,
            vocab_file_path="test_sft_vocab.pkl",
            encoded_data_path="test_sft_encoded_data.npz",
        )

    def tearDown(self) -> None:
        for path in [self.bpe.vocab_file_path, self.bpe.encoded_data_path]:
            if os.path.exists(path):
                os.remove(path)

    @patch("autograd.data.sft.load_parquet_rows")
    @patch("autograd.data.sft.load_data")
    def test_load_sft_loads_full_dataset_from_parquet_manifest(
        self, mock_load_data, mock_load_parquet_rows
    ):
        mock_load_data.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "train_sft",
                        "filename": "0000.parquet",
                        "url": "https://example.test/ultrachat/0000.parquet",
                    }
                ]
            }
        )
        mock_load_parquet_rows.return_value = [
            {
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Completion"},
                ]
            }
        ]

        examples = load_sft()

        self.assertIsInstance(examples, list)
        self.assertEqual(examples[0]["messages"][-1]["role"], "assistant")
        mock_load_parquet_rows.assert_called_once_with(
            "https://example.test/ultrachat/0000.parquet",
            "training_data/ultrachat_2k_train_sft_0000.parquet",
            max_rows=None,
        )

    @patch("autograd.data.utils.urlopen")
    @patch("autograd.data.utils.pq.read_table")
    @patch("builtins.open")
    @patch("os.path.exists")
    def test_load_sft_skips_download_when_manifest_and_parquet_are_cached(
        self,
        mock_exists,
        mock_open,
        mock_read_table,
        mock_urlopen,
    ):
        mock_exists.return_value = True

        manifest_handle = MagicMock()
        manifest_handle.__enter__.return_value.read.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "train_sft",
                        "filename": "0000.parquet",
                        "url": "https://example.test/ultrachat/0000.parquet",
                    }
                ]
            }
        )
        mock_open.return_value = manifest_handle

        mock_table = MagicMock()
        mock_table.to_pylist.return_value = [
            {
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Completion"},
                ]
            }
        ]
        mock_read_table.return_value = mock_table

        examples = load_sft()

        self.assertEqual(len(examples), 1)
        mock_urlopen.assert_not_called()
        mock_read_table.assert_called_once_with(
            "training_data/ultrachat_2k_train_sft_0000.parquet"
        )

    @patch("autograd.data.sft.load_parquet_rows")
    @patch("autograd.data.sft.load_data")
    def test_load_no_robots_sft_loads_requested_split(
        self, mock_load_data, mock_load_parquet_rows
    ):
        mock_load_data.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "test",
                        "filename": "0000.parquet",
                        "url": "https://example.test/no_robots/test.parquet",
                    },
                    {
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/no_robots/train.parquet",
                    },
                ]
            }
        )
        mock_load_parquet_rows.return_value = [
            {
                "messages": (
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Completion"},
                )
            }
        ]

        examples = load_no_robots_sft(split="train")

        self.assertEqual(
            examples,
            [
                {
                    "messages": [
                        {"role": "user", "content": "Prompt"},
                        {"role": "assistant", "content": "Completion"},
                    ]
                }
            ],
        )
        mock_load_parquet_rows.assert_called_once_with(
            "https://example.test/no_robots/train.parquet",
            "training_data/no_robots_train_0000.parquet",
            max_rows=None,
        )

    @patch("autograd.data.sft.load_data")
    def test_load_no_robots_sft_rejects_missing_split(self, mock_load_data):
        mock_load_data.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/no_robots/train.parquet",
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "Available splits"):
            load_no_robots_sft(split="validation")

    def test_tokenize_sft_messages_supervises_all_assistant_turns(self):
        example = tokenize_sft_messages(
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                    {"role": "user", "content": "C"},
                    {"role": "assistant", "content": "DE"},
                ]
            },
            MockBPE(),
        )

        expected_tokens = xp.array(
            [
                *[ord(char) for char in "User: "],
                65,
                2,
                *[ord(char) for char in "Assistant: "],
                66,
                2,
                *[ord(char) for char in "User: "],
                67,
                2,
                *[ord(char) for char in "Assistant: "],
                68,
                69,
                2,
            ],
            dtype=xp.int32,
        )
        expected_loss_mask = xp.array(
            [
                *([0] * len("User: ")),
                0,
                0,
                *([0] * len("Assistant: ")),
                1,
                1,
                *([0] * len("User: ")),
                0,
                0,
                *([0] * len("Assistant: ")),
                1,
                1,
                1,
            ],
            dtype=xp.int32,
        )

        self.assertTrue(xp.array_equal(example["tokens"], expected_tokens))
        self.assertTrue(xp.array_equal(example["loss_mask"], expected_loss_mask))

    def test_prepare_sft_token_sequences_reuses_cached_encoded_data(self):
        chat_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Completion"},
                ]
            }
        ]

        first_tokens, first_loss_masks = prepare_sft_token_sequences(
            chat_examples,
            self.bpe,
            overwrite_encoded_data=True,
            desc="Tokenizing test SFT examples",
        )
        second_tokens, second_loss_masks = prepare_sft_token_sequences(
            chat_examples,
            self.bpe,
            overwrite_encoded_data=False,
            desc="Tokenizing test SFT examples",
        )

        self.assertTrue(os.path.exists(self.bpe.encoded_data_path))
        self.assertEqual(len(first_tokens), 1)
        self.assertEqual(len(first_loss_masks), 1)
        self.assertTrue(array_equal(first_tokens[0], second_tokens[0]))
        self.assertTrue(array_equal(first_loss_masks[0], second_loss_masks[0]))

    def test_prepare_sft_token_sequences_cache_stores_offsets_and_metadata(self):
        chat_examples = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "C"},
                    {"role": "assistant", "content": "DE"},
                ]
            },
        ]

        token_sequences, _ = prepare_sft_token_sequences(
            chat_examples,
            self.bpe,
            overwrite_encoded_data=True,
            desc="Tokenizing test SFT examples",
        )
        encoded_archive = xp.load(self.bpe.encoded_data_path)

        self.assertEqual(
            set(encoded_archive.keys()),
            {"tokens", "loss_mask", "example_offsets", "metadata"},
        )
        expected_offsets = [0]
        total = 0
        for tokens in token_sequences:
            total += len(tokens)
            expected_offsets.append(total)
        self.assertTrue(
            array_equal(encoded_archive["example_offsets"], expected_offsets)
        )
        metadata = loads(bytes(int(x) for x in encoded_archive["metadata"]).decode())
        self.assertEqual(
            set(metadata.keys()),
            {"chat_examples_sha256", "tokenizer_sha256"},
        )

    def test_prepare_sft_token_sequences_rejects_stale_cached_encoded_data(self):
        first_examples = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            }
        ]
        changed_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Changed"},
                    {"role": "assistant", "content": "B"},
                ]
            }
        ]

        prepare_sft_token_sequences(
            first_examples,
            self.bpe,
            overwrite_encoded_data=True,
            desc="Tokenizing test SFT examples",
        )

        with self.assertRaisesRegex(ValueError, "stale SFT tokenized data"):
            prepare_sft_token_sequences(
                changed_examples,
                self.bpe,
                overwrite_encoded_data=False,
                desc="Tokenizing test SFT examples",
            )

    def test_prepare_sft_token_sequences_rejects_legacy_positional_cache(self):
        xp.savez_compressed(
            self.bpe.encoded_data_path,
            xp.array([1], dtype=xp.int32),
            xp.array([1], dtype=xp.int32),
            xp.array([0, 1], dtype=xp.int32),
        )
        chat_examples = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ]
            }
        ]

        with self.assertRaisesRegex(ValueError, "rebuild legacy caches"):
            prepare_sft_token_sequences(
                chat_examples,
                self.bpe,
                overwrite_encoded_data=False,
                desc="Tokenizing test SFT examples",
            )

    def test_prepare_sft_token_sequences_creates_missing_parent_directory(self):
        chat_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "Completion"},
                ]
            }
        ]
        encoded_path = "test_tmp/sft/test_encoded_data.npz"
        self.addCleanup(lambda: os.path.isdir("test_tmp") and shutil.rmtree("test_tmp"))

        tokens, loss_masks = prepare_sft_token_sequences(
            chat_examples,
            self.bpe,
            encoded_data_path=encoded_path,
            overwrite_encoded_data=True,
            desc="Tokenizing test SFT examples",
        )

        self.assertTrue(os.path.exists(encoded_path))
        self.assertEqual(len(tokens), 1)
        self.assertEqual(len(loss_masks), 1)
