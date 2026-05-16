import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from autograd.text.tokenizer import BytePairEncoder
from test.helpers import array_equal


class TestTokenizer(TestCase):
    def setUp(self):
        self.bpe = BytePairEncoder(
            num_merges=50,
            vocab_file_path="test_vocab.pkl",
            encoded_data_path="test_encoded_data.npz",
        )
        with open("test/autograd/text/test_text.txt", "r", encoding="utf-8") as f:
            self.original_text = f.read()

    def tearDown(self) -> None:
        for path in [self.bpe.vocab_file_path, self.bpe.mmap_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_construct_unicode_to_int_vocab(self):
        vocab = self.bpe._construct_unicode_to_int_vocab()
        # 256 + number of special tokens
        self.assertEqual(len(vocab), 256 + len(self.bpe.SPECIAL_TOKENS))

    def test_legacy_special_token_ids_are_stable(self):
        legacy_tokens = [
            "<|endoftext|>",
            "<PAD>",
            "<SOS>",
            "<UNK>",
            "<|USER|>",
            "<|ASSISTANT|>",
        ]

        self.assertEqual(self.bpe.SPECIAL_TOKENS[: len(legacy_tokens)], legacy_tokens)
        for offset, token in enumerate(legacy_tokens):
            self.assertEqual(
                self.bpe._unicode_to_int_vocab[token.encode("utf-8")],
                256 + offset,
            )

    def test_encode_decode(self):
        input_text = self.original_text + "<|endoftext|>" + self.original_text
        encoded = self.bpe.encode(input_text)
        decoded = self.bpe.decode(encoded)
        self.assertEqual(input_text, decoded)

    def test_encode_roundtrip_matches_original(self):
        self.bpe.train_vocabulary([self.original_text], overwrite_saved_file=True)

        for text in [
            self.original_text,
            self.original_text + "<|endoftext|>" + self.original_text,
            "low lower newest widest",
        ]:
            encoded = self.bpe.encode(text)
            decoded = self.bpe.decode(encoded)
            self.assertEqual(decoded, text)
            # Encoding must produce fewer tokens than raw bytes (merges compress)
            self.assertLessEqual(len(encoded), len(text.encode("utf-8")))

    def test_special_tokens_encoded_as_single_id(self):
        for special_token in self.bpe.SPECIAL_TOKENS:
            expected_id = self.bpe._unicode_to_int_vocab[special_token.encode("utf-8")]

            encoded = self.bpe.encode(special_token)

            self.assertEqual(encoded, [expected_id])
            self.assertEqual(self.bpe.decode(encoded), special_token)

    def test_encode_reuses_cached_non_special_chunks(self):
        text = "repeat repeat repeat repeat"

        self.bpe._encoded_chunk_cache.clear()
        first = self.bpe.encode(text)
        # Distinct pretokenized chunks: "repeat" and " repeat".
        cached_after_first = dict(self.bpe._encoded_chunk_cache)
        second = self.bpe.encode(text)

        self.assertEqual(first, second)
        self.assertEqual(self.bpe.decode(second), text)
        self.assertEqual(len(cached_after_first), 2)
        self.assertEqual(dict(self.bpe._encoded_chunk_cache), cached_after_first)

    def test_load_dictionary_fallback(self):
        with open(self.bpe.vocab_file_path, "wb") as f:
            f.write(b"\x80\x03}q\x00.")  # incomplete or invalid pickle data
        # Now re-initialize the BytePairEncoder; it should catch the error and rebuild
        bpe_new = BytePairEncoder(
            num_merges=50, vocab_file_path=self.bpe.vocab_file_path
        )
        self.assertGreater(len(bpe_new._unicode_to_int_vocab), 0)

    def test_train_vocabulary_skip_if_loaded_and_no_overwrite(self):
        # First train
        self.bpe.train_vocabulary([self.original_text], overwrite_saved_file=True)
        # Now call again with different text but overwrite_saved_file=False
        old_size = self.bpe.n_vocab
        self.bpe.train_vocabulary(["some different text"], overwrite_saved_file=False)
        # Check that nothing changed
        self.assertEqual(self.bpe.n_vocab, old_size)

    def test_decode_unknown_token(self):
        decoded = self.bpe.decode([999999])  # a token ID that doesn't exist
        self.assertIn("<UNK>", decoded)

    def test_prepare_data_reuses_cached_encoded_data(self):
        first = self.bpe.prepare_data(
            [self.original_text],
            overwrite_vocabulary_file=True,
            overwrite_encoded_data=True,
        )
        second = self.bpe.prepare_data(
            [self.original_text],
            overwrite_vocabulary_file=False,
            overwrite_encoded_data=False,
        )

        self.assertTrue(os.path.exists(self.bpe.mmap_path))
        self.assertTrue(array_equal(first, second))

    def test_streaming_encode_matches_full_text(self):
        docs = [
            "abab cdcd<|endoftext|>",
            "xyxy abab<|endoftext|>",
            "cdcd xyxy<|endoftext|>",
        ]
        full_text = "".join(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            full_bpe = BytePairEncoder(
                num_merges=20,
                vocab_file_path=os.path.join(tmpdir, "full_vocab.pkl"),
                encoded_data_path=os.path.join(tmpdir, "full_encoded.npy"),
                n_workers=1,
            )
            stream_bpe = BytePairEncoder(
                num_merges=20,
                vocab_file_path=os.path.join(tmpdir, "stream_vocab.pkl"),
                encoded_data_path=os.path.join(tmpdir, "stream_encoded.npy"),
                n_workers=1,
            )

            full_bpe.train_vocabulary([full_text], overwrite_saved_file=True)
            stream_bpe.train_vocabulary(iter(docs), overwrite_saved_file=True)
            mmap_path = stream_bpe._encode_to_mmap(
                docs,
                overwrite_encoded_data=True,
                text_batch_size=2,
            )

            stream_encoded = np.load(mmap_path, mmap_mode="r")

            self.assertEqual(full_bpe.learned_merges, stream_bpe.learned_merges)
            self.assertEqual(
                full_bpe._unicode_to_int_vocab,
                stream_bpe._unicode_to_int_vocab,
            )
            self.assertEqual(full_bpe.encode(full_text), list(stream_encoded))

    def test_prepare_data_writes_direct_npy_memmap(self):
        docs = ["hello<|endoftext|>", "world<|endoftext|>"]

        with tempfile.TemporaryDirectory() as tmpdir:
            bpe = BytePairEncoder(
                num_merges=5,
                vocab_file_path=os.path.join(tmpdir, "vocab.pkl"),
                encoded_data_path=os.path.join(tmpdir, "encoded.npz"),
                n_workers=1,
            )
            bpe.train_vocabulary(docs, overwrite_saved_file=True)
            stale_raw_tmp = f"{bpe.mmap_path}.raw.tmp"
            with open(stale_raw_tmp, "wb") as f:
                f.write(b"stale temp file")

            with patch(
                "autograd.text.tokenizer.np.lib.format.write_array",
                side_effect=AssertionError("prepare_data should not copy raw data"),
            ):
                encoded = bpe.prepare_data(
                    docs,
                    overwrite_vocabulary_file=False,
                    overwrite_encoded_data=True,
                )

            self.assertEqual(bpe.encode("".join(docs)), list(encoded))
            self.assertFalse(os.path.exists(f"{bpe.mmap_path}.raw.tmp"))
            self.assertFalse(os.path.exists(f"{bpe.mmap_path}.tmp"))

    def test_prepare_data_checks_disk_space_before_writing(self):
        docs = ["hello<|endoftext|>"]

        with tempfile.TemporaryDirectory() as tmpdir:
            bpe = BytePairEncoder(
                num_merges=5,
                vocab_file_path=os.path.join(tmpdir, "vocab.pkl"),
                encoded_data_path=os.path.join(tmpdir, "encoded.npz"),
                n_workers=1,
            )
            bpe.train_vocabulary(docs, overwrite_saved_file=True)
            disk_usage = shutil._ntuple_diskusage(total=10, used=9, free=1)

            with patch("autograd.text.tokenizer.shutil.disk_usage") as mock_disk_usage:
                mock_disk_usage.return_value = disk_usage
                with self.assertRaisesRegex(OSError, "Insufficient disk space"):
                    bpe.prepare_data(
                        docs,
                        overwrite_vocabulary_file=False,
                        overwrite_encoded_data=True,
                    )

            self.assertFalse(os.path.exists(bpe.mmap_path))
            self.assertFalse(os.path.exists(f"{bpe.mmap_path}.tmp"))

    def test_count_encoded_tokens_uses_unordered_worker_results(self):
        class RecordingPool:
            calls = []

            def __init__(self, _n_workers):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def imap(self, fn, batches):
                self.calls.append("imap")
                return map(fn, batches)

            def imap_unordered(self, fn, batches):
                self.calls.append("imap_unordered")
                return map(fn, batches)

        docs = ["aa", "bbb", "c"]
        self.bpe.n_workers = 2

        with patch("autograd.text.tokenizer.Pool", RecordingPool):
            token_count = self.bpe._count_encoded_tokens(docs, text_batch_size=1)
            list(
                self.bpe._iter_encoded_batches(
                    docs,
                    text_batch_size=1,
                    desc="ordered",
                    preserve_order=True,
                )
            )

        self.assertEqual(token_count, sum(len(self.bpe.encode(doc)) for doc in docs))
        self.assertEqual(RecordingPool.calls, ["imap_unordered", "imap"])

    def test_encode_roundtrips(self):
        text = "hello world"
        encoded = self.bpe.encode(text)
        decoded = self.bpe.decode(encoded)
        self.assertEqual(decoded, text)

    def test_train_vocabulary_rejects_bare_str(self):
        with self.assertRaises(TypeError):
            self.bpe.train_vocabulary("bare string is not allowed")
