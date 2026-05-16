import json
import os
import re
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from autograd.backend import xp
from autograd.text.utils import (
    _download_url,
    clean_and_tokenize,
    create_causal_mask,
    create_vocabulary,
    generate,
    generate_text,
    load_openwebtext,
    teacher_force,
    text_to_one_hot_and_sparse,
)


class MockedBPE:
    def encode(self, text: str):
        if text == "<|endoftext|>":
            return [9]
        if text == "<SOS>":
            return [0]
        # For simplicity, convert each uppercase letter to an integer (A=0, B=1, …)
        return [ord(ch) - 65 for ch in text if ch.isupper()]

    def decode(self, tokens: list) -> str:
        return "".join("<|endoftext|>" if t == 9 else chr(t + 65) for t in tokens)


class BytesResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, length: int = -1):
        if length is None or length < 0:
            length = len(self.content) - self.offset
        chunk = self.content[self.offset : self.offset + length]
        self.offset += len(chunk)
        return chunk


def _make_urlopen_responder(routes: dict):
    """Build a urlopen.side_effect that serves bytes per URL, honoring Range."""

    def responder(req, timeout):
        if isinstance(req, str):
            url, byte_range = req, None
        else:
            url, byte_range = req.full_url, req.get_header("Range")
        content = routes[url]
        if byte_range is None:
            return BytesResponse(content)
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        return BytesResponse(content[start : end + 1])

    return responder


class TestTextUtils(TestCase):
    def setUp(self) -> None:
        self.bpe = MockedBPE()

    def test_create_vocabulary_basic(self):
        texts = ["I love apples", "I love to eat apples every day", "Apples are great"]
        vocab = create_vocabulary(texts, max_features=5)

        # Check that we have <PAD> and <UNK>
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)

        # Because max_features=5, we expect exactly 5 tokens in the vocab
        self.assertEqual(len(vocab), 5)

        # Check that "apples" is definitely in the vocabulary
        self.assertIn("apples", vocab)

    def test_create_vocabulary_custom_tokenizer(self):
        # Custom tokenizer that splits on punctuation
        def custom_tok(text):
            return re.split(r"[,\s]+", text.lower())

        texts = ["Hello,world", "Hello, Universe!"]
        vocab = create_vocabulary(texts, max_features=6, custom_tokenizer=custom_tok)

        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("universe!", vocab)
        self.assertIn("<PAD>", vocab)
        self.assertIn("<UNK>", vocab)

    @patch("autograd.text.utils.urlopen", create=True)
    def test_load_openwebtext_uses_public_parquet_without_datasets(self, mock_urlopen):
        first_shard = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([{"text": "alpha"}]), first_shard)
        second_shard = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([{"text": "beta"}]), second_shard)
        first_bytes = first_shard.getvalue().to_pybytes()
        second_bytes = second_shard.getvalue().to_pybytes()
        manifest = {
            "parquet_files": [
                {
                    "split": "train",
                    "url": "https://example.test/openwebtext/0000.parquet",
                    "filename": "0000.parquet",
                    "size": len(first_bytes),
                },
                {
                    "split": "train",
                    "url": "https://example.test/openwebtext/0001.parquet",
                    "filename": "0001.parquet",
                    "size": len(second_bytes),
                },
            ]
        }
        mock_urlopen.side_effect = _make_urlopen_responder(
            {
                "https://datasets-server.huggingface.co/parquet?dataset=Skylion007%2Fopenwebtext": json.dumps(
                    manifest
                ).encode("utf-8"),
                "https://example.test/openwebtext/0000.parquet": first_bytes,
                "https://example.test/openwebtext/0001.parquet": second_bytes,
            }
        )

        real_import = __import__

        def import_without_datasets(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("datasets is intentionally unavailable")
            return real_import(name, *args, **kwargs)

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                with patch("builtins.__import__", side_effect=import_without_datasets):
                    source = load_openwebtext(
                        parquet_shards_per_batch=2,
                        start_token="<SOS>",
                        split_token="<|endoftext|>",
                    )
                    data = list(source)
            finally:
                os.chdir(cwd)

        self.assertEqual(
            data,
            ["<SOS>alpha<|endoftext|>", "<SOS>beta<|endoftext|>"],
        )

    @patch("autograd.text.utils.urlopen", create=True)
    def test_load_openwebtext_can_wrap_docs_with_start_and_split_tokens(
        self, mock_urlopen
    ):
        shard = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist([{"text": "alpha"}]), shard)
        shard_bytes = shard.getvalue().to_pybytes()
        manifest = {
            "parquet_files": [
                {
                    "split": "train",
                    "url": "https://example.test/openwebtext/0000.parquet",
                    "filename": "0000.parquet",
                    "size": len(shard_bytes),
                },
            ]
        }
        mock_urlopen.side_effect = _make_urlopen_responder(
            {
                "https://datasets-server.huggingface.co/parquet?dataset=Skylion007%2Fopenwebtext": json.dumps(
                    manifest
                ).encode("utf-8"),
                "https://example.test/openwebtext/0000.parquet": shard_bytes,
            }
        )

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                source = load_openwebtext(
                    parquet_shards_per_batch=1,
                    start_token="<SOS>",
                    split_token="<|endoftext|>",
                )
                data = list(source)
            finally:
                os.chdir(cwd)

        self.assertEqual(data, ["<SOS>alpha<|endoftext|>"])

    @patch("autograd.text.utils.urlopen", create=True)
    def test_download_url_issues_parallel_range_requests(self, mock_urlopen):
        content = bytes(range(251)) * 100  # 25100 bytes

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            byte_range = request.get_header("Range")
            self.assertIsNotNone(byte_range)
            start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
            return BytesResponse(content[start : end + 1])

        mock_urlopen.side_effect = fake_urlopen

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "blob.bin")
            _download_url(
                "https://example.test/blob.bin", path, expected_size=len(content)
            )

            with open(path, "rb") as f:
                self.assertEqual(f.read(), content)

        self.assertGreater(mock_urlopen.call_count, 1)

    def test_text_to_one_hot_and_sparse(self):
        texts = ["I love apples", "I love apples too"]
        vocab = create_vocabulary(texts, max_features=10)  # large enough
        max_seq_len = 4

        one_hot, matrix = text_to_one_hot_and_sparse(
            texts, vocab, max_seq_len, pad_str="<PAD>"
        )

        # Check shape
        self.assertEqual(one_hot.shape, (2, max_seq_len, len(vocab)))
        self.assertEqual(matrix.shape, (2, max_seq_len))

        # Check that the same positions in matrix have 1 in one_hot
        for i in range(2):
            for j in range(max_seq_len):
                idx = matrix[i, j]
                self.assertTrue(one_hot[i, j, idx] == 1.0)

        pad_idx = vocab["<PAD>"]
        # "I love apples" => 3 words => last index should be pad_idx
        self.assertEqual(matrix[0, 3], pad_idx)

    def test_create_causal_mask_lookforward(self):
        seq_len = 4
        batch_size = 2
        mask = create_causal_mask(
            seq_len, batch_size, lookback=False, mask_diagonal=True
        )

        # shape => (batch_size, 1, seq_len, seq_len)
        self.assertEqual(mask.shape, (2, 1, 4, 4))
        # For lookforward, upper triangle is masked.
        # Because mask_diagonal=True, diagonal is included in the mask.
        expected_single = xp.array(
            [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=xp.float32
        )

        for b in range(batch_size):
            assert xp.array_equal(mask[b, 0], expected_single)

        mask = create_causal_mask(
            seq_len, batch_size, lookback=False, mask_diagonal=False
        )

        # shape => (batch_size, 1, seq_len, seq_len)
        self.assertEqual(mask.shape, (2, 1, 4, 4))
        # For lookforward, upper triangle is not masked.
        # Because mask_diagonal=False, diagonal is not included in the mask.
        expected_single = xp.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=xp.float32
        )

        for b in range(batch_size):
            assert xp.array_equal(mask[b, 0], expected_single)

    def test_create_causal_mask_lookback(self):
        seq_len = 3
        batch_size = 1
        mask = create_causal_mask(
            seq_len, batch_size, lookback=True, mask_diagonal=False
        )
        # shape => (1, 1, 3, 3)
        # For lookback, we mask the lower triangle.  mask_diagonal=False => diagonal is not masked.
        # So the result should mask strictly below diagonal
        expected = xp.array([[[[0, 0, 0], [1, 0, 0], [1, 1, 0]]]], dtype=xp.float32)
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        assert xp.array_equal(mask, expected)

        mask = create_causal_mask(
            seq_len, batch_size, lookback=True, mask_diagonal=True
        )
        # For lookback, we mask the lower triangle.  mask_diagonal=True => diagonal is masked.
        # So the result should mask including diagonal
        expected = xp.array([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=xp.float32)
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        assert xp.array_equal(mask, expected)

    def test_clean_and_tokenize_default(self):
        text = "Hello, World! (Testing) \n new-lines?"
        tokens = clean_and_tokenize(text)
        # default pattern => r"\w+|[^\w\s]|[\n\s]"
        # expect punctuation kept, lowercased => "hello", ",", "world", "!", "(", "testing", ")", "new", "-", "lines", "?"
        self.assertTrue("hello" in tokens)
        self.assertTrue("," in tokens)
        self.assertTrue("world" in tokens)
        self.assertTrue("!" in tokens)
        self.assertTrue("(" in tokens)
        self.assertTrue("testing" in tokens)
        self.assertTrue(")" in tokens)
        self.assertTrue("-" in tokens)
        self.assertTrue("lines" in tokens)
        self.assertTrue("?" in tokens)

    def test_clean_and_tokenize_custom_pattern(self):
        text = "Hello, world! This is 2025"
        # let's split only on spaces and punctuation ignoring digits
        pattern = r"[a-zA-Z]+|[^\w\s]"  # just alpha words + punctuation
        tokens = clean_and_tokenize(text, pattern=pattern, lowercase=False)
        # Because lowercase=False, "Hello" should remain "Hello"
        self.assertIn("Hello", tokens)
        # "2025" should be excluded by pattern
        self.assertNotIn("2025", tokens)
        # punctuation like "," and "!" is included
        self.assertIn(",", tokens)
        self.assertIn("!", tokens)

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_returns_logprob_from_sampling_distribution(self, mock_choice):
        def fake_prediction(model, batch_data, mode):
            dummy_obj = MagicMock()
            dummy_obj.data = xp.array([[[0.0, 1.0, 2.0]]], dtype=xp.float32)
            return dummy_obj

        mock_choice.return_value = 1

        result = generate(
            model=MagicMock(),
            prediction_func=MagicMock(side_effect=fake_prediction),
            prompt_tokens=[0],
            max_new_tokens=1,
            temperature=1.0,
            top_k=2,
            eos_token_id=9,
            num_generations=1,
        )[0]

        expected = 1.0 - xp.log(xp.exp(xp.array(1.0)) + xp.exp(xp.array(2.0)))
        self.assertEqual(result.completion_tokens, [1])
        self.assertAlmostEqual(
            result.logprobs[0], float(xp.to_scalar(expected)), places=6
        )
        self.assertEqual(result.stop_reason, "max_new_tokens")

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_batches_parallel_completions(self, mock_choice):
        batch_shapes = []

        def fake_prediction(model, batch_data, mode):
            batch_shapes.append(tuple(batch_data.shape))
            batch_size, seq_len = batch_data.shape
            dummy_obj = MagicMock()
            dummy_obj.data = xp.zeros((batch_size, seq_len, 4), dtype=xp.float32)
            return dummy_obj

        mock_choice.side_effect = [1, 2, 1, 2]

        results = generate(
            model=MagicMock(),
            prediction_func=MagicMock(side_effect=fake_prediction),
            prompt_tokens=[0],
            max_new_tokens=2,
            temperature=1.0,
            top_k=None,
            eos_token_id=9,
            show_progress=False,
            num_generations=2,
        )

        self.assertEqual(batch_shapes, [(2, 1), (2, 2)])
        self.assertEqual(
            [result.completion_tokens for result in results], [[1, 1], [2, 2]]
        )

    @patch("autograd.text.utils.tqdm")
    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_can_disable_progress_bar(self, mock_choice, mock_tqdm):
        def fake_prediction(model, batch_data, mode):
            dummy_obj = MagicMock()
            dummy_obj.data = xp.array([[[0.0, 1.0]]], dtype=xp.float32)
            return dummy_obj

        mock_choice.return_value = 1
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        generate(
            model=MagicMock(),
            prediction_func=MagicMock(side_effect=fake_prediction),
            prompt_tokens=[0],
            max_new_tokens=1,
            temperature=1.0,
            top_k=None,
            eos_token_id=9,
            show_progress=False,
            num_generations=1,
        )

        self.assertTrue(mock_tqdm.call_args.kwargs["disable"])

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_text(self, mock_choice):
        """
        In normal (auto-regressive) mode, we expect generate_text to use sampling.
        We patch xp.random.categorical to always return 1. Also, we simulate the prediction_func
        with a MagicMock that returns a dummy object with a proper 'data' attribute.
        """

        def fake_prediction(model, batch_data, mode):
            # Determine the current sequence length.
            seq_len = batch_data.shape[1]
            # Build a dummy array of shape (1, seq_len, vocab_size); the values don't matter
            # because xp.random.categorical is patched.
            dummy_arr = xp.zeros((1, seq_len, 10))
            dummy_obj = MagicMock()
            dummy_obj.data = dummy_arr
            return dummy_obj

        # Always return 1 when sampling (so our token will be 1 → 'B').
        mock_choice.return_value = 1
        prediction_func = MagicMock(side_effect=fake_prediction)

        # For predictability, use a small max_length.
        max_length = 3
        result = generate_text(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            start_tokens="<SOS>",
            max_length=max_length + 1,  # +1 for the start_token
            temperature=1.0,
            top_k=5,
        )
        # bpe.encode("<SOS>") returns [0] → 'A'
        # Then we append token 1 for each iteration → 'B'
        # Final output: [0] + [1, 1, 1] decodes to "A" followed by "B" repeated 3 times: "ABBB"
        expected = self.bpe.decode([0] + [1] * max_length)
        self.assertEqual(result, expected)
        # Ensure prediction_func was called exactly max_length times.
        self.assertEqual(prediction_func.call_count, max_length)

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_text_respects_max_length_above_100(self, mock_choice):
        def fake_prediction(model, batch_data, mode):
            seq_len = batch_data.shape[1]
            dummy_obj = MagicMock()
            dummy_obj.data = xp.zeros((1, seq_len, 10))
            return dummy_obj

        mock_choice.return_value = 1
        prediction_func = MagicMock(side_effect=fake_prediction)
        generated_tokens = 120

        result = generate_text(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            start_tokens="<SOS>",
            max_length=generated_tokens + 1,
            temperature=1.0,
            top_k=5,
        )

        self.assertEqual(result, self.bpe.decode([0] + [1] * generated_tokens))
        self.assertEqual(prediction_func.call_count, generated_tokens)

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_text_stops_at_endoftext(self, mock_choice):
        def fake_prediction(model, batch_data, mode):
            seq_len = batch_data.shape[1]
            dummy_obj = MagicMock()
            dummy_obj.data = xp.zeros((1, seq_len, 10))
            return dummy_obj

        mock_choice.side_effect = [1, 9, 1]
        prediction_func = MagicMock(side_effect=fake_prediction)

        result = generate_text(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            start_tokens="<SOS>",
            max_length=5,
            temperature=1.0,
            top_k=5,
        )

        self.assertEqual(result, self.bpe.decode([0, 1, 9]))
        self.assertEqual(prediction_func.call_count, 2)

    @patch("autograd.text.utils.xp.sample_categorical")
    def test_generate_text_runs_in_eval_mode_and_restores_model_mode(self, mock_choice):
        class FakeModel:
            def __init__(self):
                self._is_training = True

            def eval(self):
                self._is_training = False

            def train(self):
                self._is_training = True

        observed_modes = []

        def fake_prediction(model, batch_data, mode):
            observed_modes.append(model._is_training)
            seq_len = batch_data.shape[1]
            dummy_obj = MagicMock()
            dummy_obj.data = xp.zeros((1, seq_len, 10))
            return dummy_obj

        mock_choice.return_value = 9
        model = FakeModel()

        result = generate_text(
            model=model,  # type: ignore
            prediction_func=MagicMock(side_effect=fake_prediction),
            bpe=self.bpe,  # type: ignore
            start_tokens="<SOS>",
            max_length=2,
            temperature=1.0,
            top_k=5,
        )

        self.assertEqual(result, self.bpe.decode([0, 9]))
        self.assertEqual(observed_modes, [False])
        self.assertTrue(model._is_training)

    def test_teacher_force(self):
        """
        teacher_force should use argmax at each ground-truth prefix.
        We simulate a prediction function that returns dummy logits such that xp.argmax
        returns the next groundtruth token.
        For groundtruth [0, 1, 2, 3], we expect the output tokens to be 0 then 1, 2, and 3.
        """
        groundtruth = xp.array([0, 1, 2, 3])

        def fake_prediction_teacher(model, batch_data, mode):
            # x is an array of shape (1, seq_len). The current seq_len tells us which token to predict.
            seq_len = batch_data.shape[1]
            # In teacher forcing, for iteration i (where i = seq_len - 1), we want to predict groundtruth[i+1].
            token = groundtruth[
                seq_len
            ]  # e.g., for seq_len=1, predict groundtruth[1] which is 1.
            # Create logits with a high value at the desired token.
            logits = xp.full(10, -100.0)
            logits[token] = 100.0
            # Build an array of shape (1, seq_len, 10) and place logits at the last position.
            pred_arr = xp.zeros((1, seq_len, 10))
            pred_arr[0, -1] = logits
            mock_tensor = MagicMock()
            mock_tensor.data = pred_arr
            return mock_tensor, None

        prediction_func = MagicMock(side_effect=fake_prediction_teacher)

        # Even if max_length is larger, teacher forcing uses groundtruth length.
        result = teacher_force(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            groundtruth_data=groundtruth,
            max_length=10,
        )
        # The output tokens should be: [groundtruth[0], groundtruth[1], groundtruth[2], groundtruth[3]]
        expected = self.bpe.decode([0, 1, 2, 3])
        self.assertEqual(result, expected)
        # Teacher forcing should have made 3 calls (len(groundtruth) - 1)
        self.assertEqual(prediction_func.call_count, 3)

    def test_teacher_force_runs_in_eval_mode_and_restores_model_mode(self):
        class FakeModel:
            def __init__(self):
                self._is_training = True

            def eval(self):
                self._is_training = False

            def train(self):
                self._is_training = True

        observed_modes = []
        groundtruth = xp.array([0, 1])

        def fake_prediction(model, batch_data, mode):
            observed_modes.append(model._is_training)
            pred_arr = xp.full((1, 1, 10), -100.0)
            pred_arr[0, -1, 1] = 100.0
            mock_tensor = MagicMock()
            mock_tensor.data = pred_arr
            return mock_tensor

        model = FakeModel()
        result = teacher_force(
            model=model,  # type: ignore
            prediction_func=MagicMock(side_effect=fake_prediction),
            bpe=self.bpe,  # type: ignore
            groundtruth_data=groundtruth,
            max_length=2,
        )

        self.assertEqual(result, self.bpe.decode([0, 1]))
        self.assertEqual(observed_modes, [False])
        self.assertTrue(model._is_training)

    def test_teacher_forcing_single_token(self):
        """
        If groundtruth_data contains only one token, the teacher-forcing loop should not be entered.
        The output should exactly match the decoded single token, and the prediction function should not be called.
        """
        groundtruth = xp.array([65])
        prediction_func = MagicMock()
        result = teacher_force(
            model=MagicMock(),
            prediction_func=prediction_func,
            bpe=self.bpe,  # type: ignore
            groundtruth_data=groundtruth,
            max_length=10,
        )
        expected = self.bpe.decode([65])
        self.assertEqual(result, expected)
        # Verify that the prediction function was never called.
        prediction_func.assert_not_called()
