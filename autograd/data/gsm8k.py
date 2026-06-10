import json
import os
from typing import Optional, cast

from autograd.data.utils import load_data, load_parquet_rows

# TODO: move this to a dedicated class/module organized with other dataset-specific logic for better maintainability

GSM8K_PARQUET_MANIFEST_URL = (
    "https://datasets-server.huggingface.co/parquet?dataset=openai%2Fgsm8k&config=main"
)


def split_gsm8k_answer(answer: str) -> tuple[str, str]:
    reasoning, separator, final_answer = answer.partition("####")
    if separator == "":
        raise ValueError("GSM8K answer must contain a final answer after '####'")
    return reasoning.strip(), final_answer.strip()


def load_gsm8k_rows(
    split: str,
    max_rows: Optional[int] = None,
) -> list[dict[str, str]]:
    payload = json.loads(
        cast(
            str,
            load_data(
                GSM8K_PARQUET_MANIFEST_URL,
                "training_data/gsm8k_parquet_manifest.json",
            ),
        )
    )
    parquet_files = [
        parquet_file
        for parquet_file in payload["parquet_files"]
        if parquet_file["split"] == split
    ]
    if not parquet_files:
        available_splits = sorted(
            {parquet_file["split"] for parquet_file in payload["parquet_files"]}
        )
        raise ValueError(
            f"GSM8K split {split!r} not found. Available splits: {available_splits}"
        )

    rows: list[dict[str, str]] = []
    for parquet_file in parquet_files:
        if max_rows is not None and len(rows) >= max_rows:
            break
        remaining_rows = None if max_rows is None else max_rows - len(rows)
        raw_rows = load_parquet_rows(
            parquet_file["url"],
            os.path.join(
                "training_data",
                f"gsm8k_{parquet_file['split']}_{parquet_file['filename']}",
            ),
            max_rows=remaining_rows,
        )
        for row in raw_rows:
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(
                    "GSM8K rows must contain string question and answer fields"
                )
            rows.append({"question": question, "answer": answer})

    return rows
