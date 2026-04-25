import hashlib
import json
import logging
import os
import pickle
from multiprocessing import Pool
from typing import Any, Sequence, cast

from tqdm import tqdm

from autograd.backend import Array, xp
from autograd.data.utils import load_data, load_parquet_rows
from autograd.text.tokenizer import BytePairEncoder

logger = logging.getLogger(__name__)
_WORKER_BPE: "BytePairEncoder | None" = None
_WORKER_ROLE_TOKENS: "dict[str, list[int]] | None" = None
SFT_TURN_SEPARATOR = "<|endoftext|>"
SFT_ROLE_MARKERS = {
    "system": "System: ",
    "user": "User: ",
    "assistant": "Assistant: ",
}


def _normalize_sft_messages(messages: Any) -> list[dict[str, str]]:
    if messages is None or len(messages) == 0:
        raise ValueError("chat example must contain at least one message")

    normalized_messages = []
    for message in list(messages):
        message_dict = dict(message)
        role = message_dict.get("role")
        content = message_dict.get("content")
        if role not in SFT_ROLE_MARKERS:
            raise ValueError(f"unsupported chat role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def _normalize_sft_chat_example(chat_example: dict[str, Any]) -> dict[str, Any]:
    return {"messages": _normalize_sft_messages(chat_example.get("messages"))}


def load_sft() -> list[dict[str, Any]]:
    metadata_path = "training_data/ultrachat_2k_parquet_manifest.json"
    if not os.path.exists(metadata_path):
        logger.info("Downloading data")

    payload = json.loads(
        cast(
            str,
            load_data(
                "https://datasets-server.huggingface.co/parquet?dataset=neuralmagic%2Fultrachat_2k",
                metadata_path,
            ),
        )
    )
    parquet_files = payload["parquet_files"]

    chat_examples = []
    for parquet_file in parquet_files:
        local_parquet_path = os.path.join(
            "training_data",
            f"ultrachat_2k_{parquet_file['split']}_{parquet_file['filename']}",
        )
        chat_examples.extend(
            load_parquet_rows(
                parquet_file["url"],
                local_parquet_path,
                max_rows=None,
            )
        )

    logger.info(
        "%s chat examples in the SFT dataset. Sample: %s",
        len(chat_examples),
        chat_examples[0],
    )
    return chat_examples


def load_no_robots_sft(split: str = "train") -> list[dict[str, Any]]:
    metadata_path = "training_data/no_robots_parquet_manifest.json"
    if not os.path.exists(metadata_path):
        logger.info("Downloading No Robots parquet manifest")

    payload = json.loads(
        cast(
            str,
            load_data(
                "https://datasets-server.huggingface.co/parquet?dataset=HuggingFaceH4%2Fno_robots",
                metadata_path,
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
            f"No Robots split {split!r} not found. Available splits: {available_splits}"
        )

    chat_examples = []
    for parquet_file in parquet_files:
        local_parquet_path = os.path.join(
            "training_data",
            f"no_robots_{parquet_file['split']}_{parquet_file['filename']}",
        )
        rows = load_parquet_rows(
            parquet_file["url"],
            local_parquet_path,
            max_rows=None,
        )
        chat_examples.extend(_normalize_sft_chat_example(row) for row in rows)

    logger.info(
        "%s No Robots %s chat examples. Sample: %s",
        len(chat_examples),
        split,
        chat_examples[0],
    )
    return chat_examples


def _encode_role_markers(bpe) -> dict[str, list[int]]:
    return {role: bpe.encode(marker) for role, marker in SFT_ROLE_MARKERS.items()}


def _tokenize_sft_messages_to_lists(
    chat_example: dict[str, Any],
    bpe,
    *,
    role_tokens: dict[str, list[int]] | None = None,
    turn_separator: str = SFT_TURN_SEPARATOR,
) -> tuple[list[int], list[int]]:
    messages = _normalize_sft_messages(chat_example.get("messages"))

    role_tokens = role_tokens or _encode_role_markers(bpe)
    tokens = []
    loss_mask = []
    assistant_token_count = 0

    for message in messages:
        role = message["role"]
        content = message["content"]

        marker_tokens = role_tokens[role]
        content_tokens = bpe.encode(content + turn_separator)
        tokens.extend(marker_tokens)
        tokens.extend(content_tokens)
        loss_mask.extend([0] * len(marker_tokens))
        if role == "assistant":
            assistant_token_count += len(content_tokens)
            loss_mask.extend([1] * len(content_tokens))
        else:
            loss_mask.extend([0] * len(content_tokens))

    if assistant_token_count == 0:
        raise ValueError("chat example must contain at least one assistant token")
    return tokens, loss_mask


def tokenize_sft_messages(
    chat_example: dict[str, Any],
    bpe,
    *,
    turn_separator: str = SFT_TURN_SEPARATOR,
) -> dict[str, Array]:
    """
    Tokenize one multi-turn chat row for SFT.

    Contract:
    - `tokens`: one flat causal-LM token sequence.
    - `loss_mask`: same length as `tokens`; 1 means this token should be predicted.

    Example, conceptually:
        User: A <|endoftext|> Assistant: B <|endoftext|>
        mask: 0...0              0...0       1...

    The textual role markers avoid tokenizer/checkpoint migration for now.
    TODO: replace them with dedicated special tokens after vocab resizing and
    checkpoint loading support explicit embedding/output-weight growth.
    """
    tokens, loss_mask = _tokenize_sft_messages_to_lists(
        chat_example,
        bpe,
        turn_separator=turn_separator,
    )
    return {
        "tokens": xp.array(tokens, dtype=xp.int32),
        "loss_mask": xp.array(loss_mask, dtype=xp.int32),
    }


def _init_sft_tokenizer_worker(
    num_merges: int,
    vocab_file_path: str,
    encoded_data_path: str,
) -> None:
    global _WORKER_BPE, _WORKER_ROLE_TOKENS
    _WORKER_BPE = BytePairEncoder(
        num_merges=num_merges,
        vocab_file_path=vocab_file_path,
        encoded_data_path=encoded_data_path,
        n_workers=1,
    )
    _WORKER_ROLE_TOKENS = _encode_role_markers(_WORKER_BPE)


def _tokenize_sft_chat_example(chat_example: dict[str, Any]) -> tuple[Any, Any]:
    if _WORKER_BPE is None or _WORKER_ROLE_TOKENS is None:
        raise RuntimeError("SFT tokenizer worker was not initialized")
    return _tokenize_sft_messages_to_lists(
        chat_example,
        _WORKER_BPE,
        role_tokens=_WORKER_ROLE_TOKENS,
    )


def _slice_tokenized_examples(flat_tokens, flat_loss_masks, example_offsets):
    spans = [
        (int(start), int(end))
        for start, end in zip(example_offsets[:-1], example_offsets[1:])
    ]
    return (
        [flat_tokens[start:end] for start, end in spans],
        [flat_loss_masks[start:end] for start, end in spans],
    )


def prepare_sft_token_sequences(
    chat_examples: Sequence[dict[str, Any]],
    bpe: BytePairEncoder,
    *,
    overwrite_encoded_data: bool = False,
    encoded_data_path: str | None = None,
    desc: str = "Tokenizing SFT examples",
) -> tuple[list[Any], list[Any]]:
    """
    Tokenize OpenAI-style chat rows into SFT token/loss-mask sequences.

    This lives in the data package because chat formatting, loss masks, cache
    schema, and worker setup are SFT data-preparation concerns. `BytePairEncoder`
    stays focused on generic text tokenization.
    """
    cache_path = encoded_data_path or bpe.encoded_data_path

    chat_digest = hashlib.sha256()
    for example in chat_examples:
        messages = _normalize_sft_messages(example.get("messages"))
        for message in messages:
            role = message["role"]
            content = message["content"]
            chat_digest.update(role.encode("utf-8"))
            chat_digest.update(b"\0")
            chat_digest.update(content.encode("utf-8"))
            chat_digest.update(b"\0")
        chat_digest.update(b"\xff")

    tokenizer_payload = (
        bpe.num_merges,
        bpe._unicode_to_int_vocab,
        bpe.learned_merges,
        SFT_ROLE_MARKERS,
        SFT_TURN_SEPARATOR,
    )
    expected_metadata = {
        "chat_examples_sha256": chat_digest.hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            pickle.dumps(tokenizer_payload, protocol=4)
        ).hexdigest(),
    }

    if overwrite_encoded_data:
        logger.info("Encoding SFT examples because overwrite_encoded_data=True.")
    elif os.path.exists(cache_path):
        logger.info(
            "Found SFT cache at '%s'; validating metadata before loading.",
            cache_path,
        )
        encoded_archive: Any = xp.load(cache_path)
        required_keys = {"tokens", "loss_mask", "example_offsets", "metadata"}
        if not required_keys.issubset(encoded_archive.keys()):
            raise ValueError(
                "encoded_data_path does not contain SFT tokenized data. "
                "Expected tokens, loss_mask, example_offsets, and metadata. "
                "Set overwrite_encoded_data=True to rebuild legacy caches."
            )
        actual_metadata = json.loads(
            bytes(int(x) for x in encoded_archive["metadata"]).decode("utf-8")
        )
        if actual_metadata != expected_metadata:
            raise ValueError(
                "encoded_data_path contains stale SFT tokenized data. "
                "Set overwrite_encoded_data=True to rebuild it. "
                f"Expected metadata: {expected_metadata}. "
                f"Found metadata: {actual_metadata}."
            )
        logger.info("SFT cache metadata matched; loading cached tokenized data.")
        return _slice_tokenized_examples(
            encoded_archive["tokens"],
            encoded_archive["loss_mask"],
            encoded_archive["example_offsets"],
        )
    else:
        logger.info(
            "Encoding SFT examples because no cache exists at '%s'.", cache_path
        )

    flat_token_ids: list[int] = []
    flat_loss_mask_ids: list[int] = []
    example_offsets = [0]

    def append_tokenized(token_ids, loss_mask):
        if len(token_ids) != len(loss_mask):
            raise ValueError("token sequence and loss mask must have the same length")
        flat_token_ids.extend(token_ids)
        flat_loss_mask_ids.extend(loss_mask)
        example_offsets.append(len(flat_token_ids))

    if bpe._should_parallelize(
        work_items=len(chat_examples),
        min_items_per_worker=64,
    ):
        with Pool(
            bpe.n_workers,
            initializer=_init_sft_tokenizer_worker,
            initargs=(
                bpe.num_merges,
                bpe.vocab_file_path,
                cache_path,
            ),
        ) as pool:
            chunksize = max(1, len(chat_examples) // (bpe.n_workers * 4))
            for token_ids, loss_mask in tqdm(
                pool.imap(_tokenize_sft_chat_example, chat_examples, chunksize),
                total=len(chat_examples),
                desc=desc,
            ):
                append_tokenized(token_ids, loss_mask)
    else:
        role_tokens = _encode_role_markers(bpe)
        for example in tqdm(chat_examples, desc=desc):
            append_tokenized(
                *_tokenize_sft_messages_to_lists(
                    example,
                    bpe,
                    role_tokens=role_tokens,
                )
            )

    flat_tokens = xp.array(flat_token_ids, dtype=xp.int32)
    flat_loss_masks = xp.array(flat_loss_mask_ids, dtype=xp.int32)
    offsets = xp.array(example_offsets, dtype=xp.int32)

    parent_dir = os.path.dirname(cache_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    xp.savez_compressed(
        cache_path,
        tokens=flat_tokens,
        loss_mask=flat_loss_masks,
        example_offsets=offsets,
        metadata=xp.array(
            list(json.dumps(expected_metadata, sort_keys=True).encode("utf-8"))
        ),
    )
    return _slice_tokenized_examples(flat_tokens, flat_loss_masks, offsets)
