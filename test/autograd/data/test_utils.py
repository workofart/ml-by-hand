import os
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from autograd.backend import xp
from autograd.data.dataset import Seq2SeqDataset
from autograd.data.utils import (
    build_seq2seq_dataset_from_text_pairs,
    load_data,
    load_parquet_rows,
    openai_chat_to_prompt_completion,
    tokenize_prompt_completion,
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


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.bpe = MockBPE()

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

    def test_load_parquet_rows_returns_dict_rows(self):
        rows = [
            {"summary": "s1", "article": "a1"},
            {"summary": "s2", "article": "a2"},
        ]
        table = pa.Table.from_pylist(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "sample.parquet")
            pq.write_table(table, parquet_path)

            data = load_parquet_rows("unused", parquet_path, max_rows=1)

        self.assertEqual(data, [{"summary": "s1", "article": "a1"}])

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

    def test_build_seq2seq_dataset_from_text_pairs_encodes_source_and_target(self):
        dataset = build_seq2seq_dataset_from_text_pairs(
            [("ABC", "DE")],
            self.bpe,
            shuffle=False,
            target_suffix="<|endoftext|>",
        )

        self.assertIsInstance(dataset, Seq2SeqDataset)
        example = next(iter(dataset))
        self.assertTrue(
            xp.array_equal(example["input_ids"], xp.array([65, 66, 67], dtype=xp.int32))
        )
        self.assertTrue(
            xp.array_equal(example["labels"], xp.array([68, 69, 2], dtype=xp.int32))
        )
