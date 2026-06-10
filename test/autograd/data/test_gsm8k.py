from json import dumps
from unittest import TestCase
from unittest.mock import patch

from autograd.data.gsm8k import load_gsm8k_rows, split_gsm8k_answer


class TestGSM8KData(TestCase):
    @patch("autograd.data.gsm8k.load_parquet_rows")
    @patch("autograd.data.gsm8k.load_data")
    def test_load_gsm8k_rows_loads_requested_split(
        self,
        mock_load_data,
        mock_load_parquet_rows,
    ):
        mock_load_data.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "test",
                        "filename": "0000.parquet",
                        "url": "https://example.test/gsm8k/test.parquet",
                    },
                    {
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/gsm8k/train.parquet",
                    },
                ]
            }
        )
        mock_load_parquet_rows.return_value = [
            {
                "question": "Jan has 2 apples. Tom gives her 3. How many?",
                "answer": "Jan has 2 + 3 = 5 apples. #### 5",
            }
        ]

        rows = load_gsm8k_rows(split="train")

        self.assertEqual(
            rows,
            [
                {
                    "question": "Jan has 2 apples. Tom gives her 3. How many?",
                    "answer": "Jan has 2 + 3 = 5 apples. #### 5",
                }
            ],
        )
        mock_load_parquet_rows.assert_called_once_with(
            "https://example.test/gsm8k/train.parquet",
            "training_data/gsm8k_train_0000.parquet",
            max_rows=None,
        )

    @patch("autograd.data.gsm8k.load_data")
    def test_load_gsm8k_rows_rejects_missing_split(self, mock_load_data):
        mock_load_data.return_value = dumps(
            {
                "parquet_files": [
                    {
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/gsm8k/train.parquet",
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "Available splits"):
            load_gsm8k_rows(split="validation")

    def test_split_gsm8k_answer_extracts_reasoning_and_final_answer(self):
        self.assertEqual(
            split_gsm8k_answer("scratch work #### 22,500"),
            ("scratch work", "22,500"),
        )

    def test_split_gsm8k_answer_rejects_missing_marker(self):
        with self.assertRaisesRegex(ValueError, "after '####'"):
            split_gsm8k_answer("scratch work only")
