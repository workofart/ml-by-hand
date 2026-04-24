import os
from collections import Counter
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from autograd.text.tokenizer import BytePairEncoder
from test.helpers import array_equal

ORIGINAL_NP_CONCATENATE = np.concatenate


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
        for path in [self.bpe.vocab_file_path, self.bpe.encoded_data_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_construct_unicode_to_int_vocab(self):
        vocab = self.bpe._construct_unicode_to_int_vocab()
        # 256 + number of special tokens
        self.assertEqual(len(vocab), 256 + len(self.bpe.SPECIAL_TOKENS))

    def test_pair_counting(self):
        # Instead of test_get_bigrams_to_count, we test _get_initial_pair_counts directly
        # We'll create a small word_freq manually
        word_freq = Counter(
            {
                (10, 11, 12): 3,
                (11, 12, 12): 2,
            }
        )
        pair_counts = self.bpe._get_initial_pair_counts(word_freq)

        # (10,11) occurs in the first tuple, 3 times
        # (11,12) occurs in the first tuple (3 times) and second tuple (2 times) -> total 5
        # (12,12) occurs in second tuple (2 times)
        expected = {
            (10, 11): 3,
            (11, 12): 5,
            (12, 12): 2,
        }
        self.assertEqual(pair_counts, expected)

    @patch(
        "autograd.text.tokenizer.Pool", side_effect=AssertionError("pool not expected")
    )
    def test_pair_counting_skips_pool_for_small_corpus(self, _mock_pool):
        bpe = BytePairEncoder(num_merges=50, n_workers=4)
        word_freq = Counter(
            {
                (10, 11, 12): 3,
                (11, 12, 12): 2,
            }
        )

        pair_counts = bpe._get_initial_pair_counts(word_freq)

        self.assertEqual(
            pair_counts,
            {
                (10, 11): 3,
                (11, 12): 5,
                (12, 12): 2,
            },
        )

    def test_apply_merges_to_corpus(self):
        # Instead of test_merge_pairs, we test _apply_merges_to_corpus
        word_freq = Counter(
            {
                (10, 11, 11, 12): 2,
            }
        )
        pair_counts = {
            (10, 11): 2,
            (11, 11): 2,
            (11, 12): 2,
        }
        pair = (10, 11)
        new_id = 256

        self.bpe._apply_merges_to_corpus(pair, new_id, word_freq, pair_counts)

        # The old tuple (10, 11, 11, 12) should be removed
        # The new tuple is (256, 11, 12)
        # And word_freq should have updated counts
        self.assertFalse((10, 11, 11, 12) in word_freq)
        self.assertEqual(word_freq[(256, 11, 12)], 2)

        # Also check that pair_counts updated
        # old bigrams: (10,11), (11,11), (11,12)
        # new bigrams: (256,11), (11,12)
        self.assertFalse((10, 11) in pair_counts)
        self.assertFalse((11, 11) in pair_counts)
        self.assertTrue((11, 12) in pair_counts)  # still remain in the pair_counts
        self.assertIn((256, 11), pair_counts)
        self.assertIn((11, 12), pair_counts)

    @patch(
        "autograd.text.tokenizer.Pool", side_effect=AssertionError("pool not expected")
    )
    def test_apply_merges_to_corpus_skips_pool_for_small_update_set(self, _mock_pool):
        bpe = BytePairEncoder(num_merges=50, n_workers=4)
        word_freq = Counter({(10, 11, 11, 12): 2})
        pair_counts = {(10, 11): 2, (11, 11): 2, (11, 12): 2}

        bpe._apply_merges_to_corpus((10, 11), 256, word_freq, pair_counts)

        self.assertEqual(word_freq, Counter({(256, 11, 12): 2}))
        self.assertEqual(pair_counts, {(11, 12): 2, (256, 11): 2})

    def test_encode_decode(self):
        input_text = self.original_text + "<|endoftext|>" + self.original_text
        encoded = self.bpe.encode(input_text)
        decoded = self.bpe.decode(encoded)
        self.assertEqual(input_text, decoded)

    def test_encode_matches_learned_merge_replay(self):
        self.bpe.train_vocabulary(self.original_text, overwrite_saved_file=True)

        def replay_learned_merges(text):
            tokens = []
            for chunk in self.bpe._pretokenize(text):
                if chunk in self.bpe.SPECIAL_TOKENS:
                    tokens.append(self.bpe._unicode_to_int_vocab[chunk.encode("utf-8")])
                else:
                    tokens.extend(
                        self.bpe._unicode_to_int_vocab[bytes([b])]
                        for b in chunk.encode("utf-8")
                    )

            pairs = set(zip(tokens, tokens[1:]))
            for pair, new_id in self.bpe.learned_merges:
                if pair in pairs:
                    tokens = list(BytePairEncoder._merge_pairs(pair, new_id, tokens))
                    pairs = set(zip(tokens, tokens[1:]))
            return tokens

        for text in [
            self.original_text,
            self.original_text + "<|endoftext|>" + self.original_text,
            "low lower newest widest",
        ]:
            self.assertEqual(self.bpe.encode(text), replay_learned_merges(text))

    def test_special_tokens_encoded_as_single_id(self):
        # This checks special tokens remain single ID
        st_id = self.bpe._unicode_to_int_vocab[
            self.bpe.SPECIAL_TOKENS[0].encode("utf-8")
        ]
        self.assertIsNotNone(st_id)

        input_text = self.bpe.SPECIAL_TOKENS[0]
        encoded = self.bpe.encode(input_text)
        self.assertIn(st_id, encoded)
        self.assertEqual(self.bpe.decode(encoded), input_text)

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
        self.bpe.train_vocabulary(self.original_text, overwrite_saved_file=True)
        # Now call again with different text but overwrite_saved_file=False
        old_size = self.bpe.n_vocab
        self.bpe.train_vocabulary("some different text", overwrite_saved_file=False)
        # Check that nothing changed
        self.assertEqual(self.bpe.n_vocab, old_size)

    def test_decode_unknown_token(self):
        decoded = self.bpe.decode([999999])  # a token ID that doesn't exist
        self.assertIn("<UNK>", decoded)

    def test_prepare_data_reuses_cached_encoded_data(self):
        first = self.bpe.prepare_data(
            self.original_text,
            overwrite_vocabulary_file=True,
            overwrite_encoded_data=True,
        )
        second = self.bpe.prepare_data(
            self.original_text,
            overwrite_vocabulary_file=False,
            overwrite_encoded_data=False,
        )

        self.assertTrue(os.path.exists(self.bpe.encoded_data_path))
        self.assertTrue(array_equal(first, second))

    @patch(
        "autograd.text.tokenizer.Pool", side_effect=AssertionError("pool not expected")
    )
    def test_prepare_data_skips_pool_for_small_text(self, _mock_pool):
        bpe = BytePairEncoder(
            num_merges=2,
            vocab_file_path="small_vocab.pkl",
            encoded_data_path="small_encoded_data.npz",
            n_workers=4,
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.vocab_file_path) and os.remove(bpe.vocab_file_path)
            )
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.encoded_data_path)
                and os.remove(bpe.encoded_data_path)
            )
        )

        encoded = bpe.prepare_data(
            "hello hello",
            overwrite_vocabulary_file=True,
            overwrite_encoded_data=True,
        )

        self.assertGreater(len(encoded), 0)

    @patch("autograd.text.tokenizer.Pool")
    def test_prepare_data_disables_per_chunk_encode_progress_in_parallel(
        self, mock_pool
    ):
        class RecordingBPE(BytePairEncoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.show_progress_calls = []

            def encode(self, input_text: str, *, show_progress: bool = True, **kwargs):
                self.show_progress_calls.append(show_progress)
                return super().encode(input_text, show_progress=show_progress, **kwargs)

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                for item in iterable:
                    yield func(item)

        bpe = RecordingBPE(
            num_merges=0,
            vocab_file_path="parallel_vocab.pkl",
            encoded_data_path="parallel_encoded_data.npz",
            n_workers=2,
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.vocab_file_path) and os.remove(bpe.vocab_file_path)
            )
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.encoded_data_path)
                and os.remove(bpe.encoded_data_path)
            )
        )
        mock_pool.return_value = FakePool()

        encoded = bpe.prepare_data(
            "x" * 20000,
            overwrite_vocabulary_file=True,
            overwrite_encoded_data=True,
        )

        self.assertGreater(len(encoded), 0)
        self.assertEqual(bpe.show_progress_calls, [False, False])

    def test_encode_disables_progress_by_default(self):
        with patch("autograd.text.tokenizer.tqdm") as mock_tqdm:
            self.bpe.encode("hello")

        self.assertTrue(mock_tqdm.called)
        self.assertTrue(mock_tqdm.call_args.kwargs["disable"])

    @patch("autograd.text.tokenizer.Pool")
    @patch("autograd.text.tokenizer.xp.concatenate")
    @patch("autograd.text.tokenizer.xp.savez_compressed")
    def test_prepare_data_concatenates_parallel_chunks_once(
        self, _mock_savez, mock_concatenate, mock_pool
    ):
        class RecordingBPE(BytePairEncoder):
            def train_vocabulary(
                self, input_text: str, overwrite_saved_file: bool = False
            ):
                return self._unicode_to_int_vocab, self._int_to_unicode_vocab

            def encode(self, input_text: str, *, show_progress: bool = True, **kwargs):
                return [len(input_text)]

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                for item in iterable:
                    yield func(item)

        def counted_concatenate(arrays, axis=0):
            return ORIGINAL_NP_CONCATENATE(arrays, axis=axis)

        bpe = RecordingBPE(
            num_merges=0,
            vocab_file_path="concat_vocab.pkl",
            encoded_data_path="concat_encoded_data.npz",
            n_workers=2,
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.encoded_data_path)
                and os.remove(bpe.encoded_data_path)
            )
        )
        mock_pool.return_value = FakePool()
        mock_concatenate.side_effect = counted_concatenate

        encoded = bpe.prepare_data(
            "x" * 20000,
            overwrite_vocabulary_file=False,
            overwrite_encoded_data=True,
        )

        self.assertEqual(mock_concatenate.call_count, 1)
        self.assertEqual(encoded.tolist(), [10000, 10000])

    @patch(
        "autograd.text.tokenizer.Pool", side_effect=AssertionError("pool not expected")
    )
    def test_prepare_data_enables_progress_for_single_text_encode(self, _mock_pool):
        class RecordingBPE(BytePairEncoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.show_progress_calls = []

            def encode(self, input_text: str, *, show_progress: bool = False, **kwargs):
                self.show_progress_calls.append(show_progress)
                return super().encode(input_text, show_progress=show_progress, **kwargs)

        bpe = RecordingBPE(
            num_merges=2,
            vocab_file_path="single_progress_vocab.pkl",
            encoded_data_path="single_progress_encoded_data.npz",
            n_workers=4,
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.vocab_file_path) and os.remove(bpe.vocab_file_path)
            )
        )
        self.addCleanup(
            lambda: (
                os.path.exists(bpe.encoded_data_path)
                and os.remove(bpe.encoded_data_path)
            )
        )

        bpe.prepare_data(
            "hello hello",
            overwrite_vocabulary_file=True,
            overwrite_encoded_data=True,
        )

        self.assertEqual(bpe.show_progress_calls, [True])
