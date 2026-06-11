from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.request import Request, urlopen

from pyarrow import parquet as pq  # pyright: ignore[reportMissingImports]
from tqdm import tqdm

from autograd.backend import Array, ArrayLike, xp
from autograd.tensor import Tensor
from autograd.text.tokenizer import BytePairEncoder

if TYPE_CHECKING:
    from autograd import nn

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Token-level output from autoregressive generation."""

    completion_tokens: list[int]
    logprobs: list[float]
    stop_reason: str


@dataclass
class OpenWebTextSource:
    parquet_files: list[dict[str, Any]]
    parquet_dir: str
    start_token: str
    split_token: str
    parquet_shards_per_batch: int

    def __iter__(self) -> Iterator[str]:
        for shard_batch in _iter_batches(
            self.parquet_files,
            self.parquet_shards_per_batch,
        ):
            parquet_paths = []
            for parquet_file in shard_batch:
                parquet_paths.append(
                    _ensure_openwebtext_shard(parquet_file, self.parquet_dir)
                )

            for parquet_path in parquet_paths:
                for batch in pq.ParquetFile(parquet_path).iter_batches(
                    columns=["text"],
                    batch_size=2048,
                ):
                    for doc in batch.column("text").to_pylist():
                        yield format_document_for_causal_lm(
                            doc,
                            start_token=self.start_token,
                            split_token=self.split_token,
                        )


def format_document_for_causal_lm(
    doc: str,
    *,
    start_token: str,
    split_token: str,
) -> str:
    return f"{start_token}{doc}{split_token}"


def generate(
    model: nn.Module,
    prediction_func: nn.AbstractLLMForwardFn,
    prompt_tokens: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    eos_token_id: int,
    *,
    show_progress: bool = True,
    num_generations: int,
    compute_logprobs: bool = True,
) -> list[GenerationResult]:
    """Generate token ids autoregressively and record sampled-token logprobs.

    This is the structured generation primitive: callers pass already-tokenized
    prompt ids, and the function owns the forward/sample loop. It returns only
    completion tokens, so callers can keep prompt and completion boundaries
    exact without decoding and re-encoding text.

    Args:
        model: Language model used for the forward pass.
        prediction_func: Forward function called with `mode="sample"`.
        prompt_tokens: Token ids that seed generation.
        max_new_tokens: Maximum number of completion tokens to generate.
        temperature: Sampling temperature. Values <= 0 use argmax.
        top_k: Optional top-k filter applied before sampling.
        eos_token_id: Token id that stops generation when sampled.
        show_progress: Whether to show token-level inference progress.
        num_generations: Number of independent completions to generate in
            parallel for the same prompt.
        compute_logprobs: Whether to compute sampled-token logprobs. Text-only
            generation can disable this to skip the sampling-distribution math.

    Returns:
        One result per generated completion.
    """
    if num_generations < 1:
        raise ValueError(f"num_generations must be >= 1, got {num_generations}")

    prompt_token_list = [int(token) for token in prompt_tokens]
    output_ids = [prompt_token_list.copy() for _ in range(num_generations)]
    completion_tokens: list[list[int]] = [[] for _ in range(num_generations)]
    logprobs: list[list[float]] = [[] for _ in range(num_generations)]
    stop_reasons = ["max_new_tokens" for _ in range(num_generations)]
    active = [True for _ in range(num_generations)]

    for _ in tqdm(range(max_new_tokens), desc="Inference", disable=not show_progress):
        if not any(active):
            break

        # Auto-regressive generation feeds the full prompt plus all tokens sampled
        # so far back into the model, then samples from the final position.
        # Shape: (num_generations, current_seq_len). Each row is one completion
        # being advanced in parallel for this decoding step.
        batch_data = xp.array(output_ids, dtype=xp.int32)
        prediction = prediction_func(model=model, batch_data=batch_data, mode="sample")
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        next_token_logits = prediction.data[:, -1]
        token_ids, step_logprobs = _sample_next_tokens(
            next_token_logits=next_token_logits,
            active=active,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            compute_logprobs=compute_logprobs,
        )

        # One device-to-host transfer per step for the whole batch, instead of
        # per-row scalar syncs.
        token_list = [int(token) for token in xp.to_numpy(token_ids).tolist()]
        logprob_list = None
        if step_logprobs is not None:
            logprob_list = [
                float(logprob) for logprob in xp.to_numpy(step_logprobs).tolist()
            ]

        for row_idx, token_id in enumerate(token_list):
            output_ids[row_idx].append(token_id)
            if not active[row_idx]:
                continue
            completion_tokens[row_idx].append(token_id)
            logprobs[row_idx].append(
                0.0 if logprob_list is None else logprob_list[row_idx]
            )
            if token_id == eos_token_id:
                stop_reasons[row_idx] = "eos"
                active[row_idx] = False

    return [
        GenerationResult(
            completion_tokens=tokens,
            logprobs=logprobs[row_idx],
            stop_reason=stop_reasons[row_idx],
        )
        for row_idx, tokens in enumerate(completion_tokens)
    ]


def _sample_next_tokens(
    next_token_logits: Array,
    active: list[bool],
    temperature: float,
    top_k: Optional[int],
    eos_token_id: int,
    compute_logprobs: bool,
) -> tuple[Array, Optional[Array]]:
    """Sample one token per row from next-token logits.

    This owns the sampling contract for the generation loop: greedy vs sampled
    decoding, top-k masking, EOS padding for finished rows, and logprobs. The
    math is batched across rows so each step costs one set of array ops rather
    than one per completion.
    """
    num_generations = len(active)
    if temperature <= 0:
        sampled_ids = []
        for row_idx, is_active in enumerate(active):
            if is_active:
                sampled_ids.append(
                    xp.argmax(next_token_logits[row_idx]).astype(xp.int32)
                )
            else:
                sampled_ids.append(xp.array(eos_token_id, dtype=xp.int32))
        token_ids = xp.stack(sampled_ids, axis=0)
        step_logprobs = (
            xp.zeros((num_generations,), dtype=xp.float32) if compute_logprobs else None
        )
        return token_ids, step_logprobs

    # Temperature rescales the distribution before sampling. Larger values
    # flatten it; smaller values make it sharper.
    behavior_logits = xp.array(next_token_logits, dtype=xp.float32) / temperature
    if top_k is not None and top_k < behavior_logits.shape[-1]:
        # Top-k keeps only the k most likely tokens in each row and masks the
        # rest to -inf so softmax gives them zero probability.
        threshold = xp.sort(behavior_logits, axis=-1)[:, -top_k]
        behavior_logits = xp.where(
            behavior_logits >= threshold[:, None],
            behavior_logits,
            xp.full(
                behavior_logits.shape,
                -float("inf"),
                dtype=behavior_logits.dtype,
            ),
        )

    sampled_ids = []
    for row_idx, is_active in enumerate(active):
        if is_active:
            sampled_ids.append(
                xp.array(
                    xp.sample_categorical(behavior_logits[row_idx]),
                    dtype=xp.int32,
                )
            )
        else:
            # Finished rows keep feeding EOS so batch shapes stay fixed. The
            # caller ignores these padded tokens when building the result.
            sampled_ids.append(xp.array(eos_token_id, dtype=xp.int32))
    token_ids = xp.stack(sampled_ids, axis=0)

    if not compute_logprobs:
        return token_ids, None

    # The reported logprob must come from the exact distribution used for
    # sampling, after temperature and top-k have changed the logits. Subtracting
    # the row max is the standard numerically stable log-softmax trick.
    shifted = behavior_logits - xp.max(behavior_logits, axis=-1, keepdims=True)
    log_denom = xp.log(xp.sum(xp.exp(shifted), axis=-1))
    row_idx = xp.arange(num_generations)
    return token_ids, shifted[row_idx, token_ids] - log_denom


def generate_text(
    model: nn.Module,
    prediction_func: nn.AbstractLLMForwardFn,
    bpe: BytePairEncoder,
    start_tokens: Optional[str],
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_token: str = "<|endoftext|>",
) -> str:
    """Generate text from a string prompt.

    This is a convenience wrapper around `generate`: it handles tokenizer
    encode/decode, switches the model to eval mode for generation, and restores
    the prior training mode. It shows a token-level progress bar and a one-line
    wall-time / tok/s summary; the generated text itself is only returned, so
    callers decide how to display it (e.g. `print(generate_text(...))`).

    Args:
        model: Language model used for generation.
        prediction_func: Forward function passed through to `generate`.
        bpe: Tokenizer used to encode the prompt and decode generated tokens.
        start_tokens: Prompt text. Defaults to "<SOS>" when omitted.
        max_length: Maximum total token length, including prompt tokens.
        temperature: Sampling temperature passed through to `generate`.
        top_k: Optional top-k filter passed through to `generate`.
        stop_token: Token string that stops generation when sampled. Chat
            checkpoints stop at the turn separator instead of `<|endoftext|>`.

    Returns:
        Decoded prompt plus generated completion text.
    """
    was_training = getattr(model, "_is_training", None)
    model.eval()
    try:
        start_tokens = start_tokens or "<SOS>"
        output_ids = list(bpe.encode(start_tokens))
        prompt_len = len(output_ids)
        t0 = time.perf_counter()
        result = generate(
            model=model,
            prediction_func=prediction_func,
            prompt_tokens=output_ids,
            max_new_tokens=max_length - prompt_len,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=bpe.encode(stop_token)[0],
            num_generations=1,
            # Text generation only needs the tokens; skip the per-step
            # log-softmax work since nothing reads the logprobs.
            compute_logprobs=False,
        )[0]
        elapsed = time.perf_counter() - t0
        output_ids.extend(result.completion_tokens)
        text = bpe.decode(output_ids)
        n_generated = len(result.completion_tokens)
        tok_per_sec = (n_generated / elapsed) if elapsed > 0 else float("nan")
        print(
            f"[prompt {prompt_len} tokens + {n_generated} new tokens "
            f"in {elapsed:.2f}s, {tok_per_sec:.1f} tok/s]\n"
        )
        return text
    finally:
        if was_training:
            model.train()


def teacher_force(
    model: nn.Module,
    prediction_func: nn.AbstractLLMForwardFn,
    bpe: BytePairEncoder,
    groundtruth_data: Array,
    max_length: int = 50,
) -> str:
    """Run teacher forcing over ground-truth token ids and print predictions.

    At each step the model receives the ground-truth prefix and the decoded
    argmax prediction is appended to the returned text. This is intentionally
    separate from `generate`, because the model input is fixed by the dataset
    rather than by previously sampled tokens.

    Args:
        model: Language model used for the forward pass.
        prediction_func: Forward function called with `mode="sample"`.
        bpe: Tokenizer used to decode predicted token ids.
        groundtruth_data: Ground-truth token ids used as model inputs.
        max_length: Maximum number of ground-truth tokens to evaluate.

    Returns:
        Decoded text from the model's argmax predictions.
    """
    was_training = getattr(model, "_is_training", None)
    model.eval()
    try:
        num_steps = max(0, min(max_length, len(groundtruth_data)) - 1)
        output_ids = [int(groundtruth_data[0])]

        groundtruth_tokens = [int(token) for token in groundtruth_data.tolist()]
        groundtruth_text = "\n".join(
            bpe.decode(groundtruth_tokens).split("<|endoftext|>")
        )
        logger.info(f"Teacher forcing mode on!!\nGroundtruth:\n{groundtruth_text}")
        logger.info("Model:\n")

        for i in range(num_steps):
            # Teacher forcing feeds the ground-truth prefix at each step instead
            # of feeding back the model's own previous predictions.
            current_input = groundtruth_data[: i + 1]
            batch_data = xp.expand_dims(xp.array(current_input, dtype=xp.int32), axis=0)
            prediction = prediction_func(
                model=model, batch_data=batch_data, mode="sample"
            )
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            logits = prediction.data[0, -1]
            # We still inspect the model's next-token prediction, but the next
            # loop input will come from groundtruth_data, not this prediction.
            next_token = int(xp.argmax(logits))
            output_ids.append(next_token)
            token_str = bpe.decode([next_token])
            print(token_str, end="", flush=True)
            if token_str == "<|endoftext|>":
                break

        print("\n--------------------------------------------------------------\n")
        return bpe.decode(output_ids)
    finally:
        if was_training:
            model.train()


def create_vocabulary(
    texts: List[str],
    max_features: Optional[int],
    custom_tokenizer: Optional[Callable[[str], List[str]]] = None,
    special_tokens: List[str] = ["<PAD>", "<SOS>", "<UNK>"],
) -> Dict[str, int]:
    """
    Create a vocabulary (word->index) from given texts,
    keeping up to max_features most common words.

    Examples:
        >>> texts = ["Hello world", "Hello there", "World peace"]
        >>> vocab = create_vocabulary(texts, max_features=5)
        >>> print(vocab)
        {'hello': 0, 'world': 1, 'there': 2, 'peace': 3}  # Order and exact indices may vary.
    """
    token_freq = defaultdict(float)
    for text in texts:
        if custom_tokenizer is None:
            tokens = text.lower().split()
        else:
            tokens = custom_tokenizer(text)

        for t in tokens:
            token_freq[t] += 1

    for i, st in enumerate(special_tokens):
        token_freq[st] = float("inf") - i

    # Sort by frequency
    sorted_words = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    if max_features is not None:
        sorted_words = sorted_words[:max_features]

    # Create word->index mapping
    vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    return vocab


def text_to_one_hot_and_sparse(
    texts: List[str],
    vocabulary: Dict[str, int],
    max_sequence_length: int,
    pad_str: str = "<PAD>",
) -> Tuple[Array, Array]:
    """
    Convert list of texts into a sequential feature matrix using the vocabulary.
    It will do the padding/truncation based on max_sequence_length, then convert to one-hot encoding
    Shape: (batch_size, sequence_length, vocab_size)

    Args:
        texts (list of str): The input sentences or documents.
        vocabulary (dict): A mapping of word -> index. We'll also add "<PAD>"
                           if it’s not already present.
        max_sequence_length (int): The maximum sequence length for truncation/padding.
        pad_str (str): The padding string.

    Returns:
        one_hot (Array): shape (batch_size, max_sequence_length, vocab_size)
        matrix  (Array): shape (batch_size, max_sequence_length) of integer IDs

    Examples:
        >>> texts = ["Hello world", "Hello there"]
        >>> vocab = {"hello": 0, "world": 1, "there": 2, "<PAD>": 3, "<UNK>": 4}
        >>> one_hot, matrix = text_to_one_hot_and_sparse(texts, vocabulary=vocab, max_sequence_length=4)
        >>> print(matrix)
        [[0, 1, 3, 3],
         [0, 2, 3, 3]]
        >>> print(one_hot.shape)
        (2, 4, 5)
    """
    batch_size = len(texts)
    vocab_size = len(vocabulary)
    pad_idx = vocabulary[pad_str]

    # Create an integer matrix of shape (batch_size, max_sequence_length)
    # filled with pad_idx initially, then we overwrite with actual indices later
    matrix = xp.full((batch_size, max_sequence_length), pad_idx, dtype=xp.int32)

    for i, text in enumerate(texts):
        # Split text into words and convert to indices
        words = text.lower().split()
        # Truncate or pad sequence to max_sequence_length
        words = words[:max_sequence_length]

        for j, word in enumerate(words):
            if word in vocabulary:
                matrix[i, j] = vocabulary[word]
            else:
                matrix[i, j] = vocabulary.get("<UNK>", pad_idx)

    # Convert to one-hot encoding
    # Shape: (batch_size, sequence_length, vocab_size)
    one_hot = xp.eye(vocab_size, dtype=xp.float32)[matrix]
    return one_hot, matrix


def create_causal_mask(
    seq_len: int,
    batch_size: int,
    lookback: bool = False,
    mask_diagonal: bool = False,
) -> Array:
    """
    Creates a causal mask that prevents positions from attending to future (lookforward)
    or past (lookback) positions. 1.0 => masked.

    Args:
        seq_len (int): Length of the sequence
        batch_size (int): Size of the batch
        lookback (bool): If True, masks "past" (i>j). If False, masks "future" (i<j).
        mask_diagonal (bool): If True, the main diagonal is also masked.

    Returns:
        Array: shape (batch_size, 1, seq_len, seq_len) with 1.0 in masked positions.

    Examples:
        >>> mask = create_causal_mask(seq_len=5, batch_size=2)
        >>> print(mask.shape)
        (2, 1, 5, 5)
    """
    # We want to produce a matrix M of shape (seq_len, seq_len) where
    # M[i,j] = 1 if it is masked, else 0 if it's allowed.

    if lookback:
        # Mask the lower triangle => i>j => row>column => can't attend to "past"
        # If mask_diagonal=True => includes diagonal => i>=j
        # If mask_diagonal=False => strictly below diagonal => i>j
        k_ = 0 if mask_diagonal else -1
        # np.tril(..., k=0) includes diagonal; np.tril(..., k=-1) excludes diagonal
        mask_2d = xp.tril(xp.ones((seq_len, seq_len), dtype=xp.float32), k=k_)
    else:
        # Mask the upper triangle => i<j => can't attend to "future"
        # If mask_diagonal=True => includes diagonal => i<=j => so we do k=0 in np.triu
        # If mask_diagonal=False => strictly above diagonal => i<j => so we do k=1
        k_ = 0 if mask_diagonal else 1
        mask_2d = xp.triu(xp.ones((seq_len, seq_len), dtype=xp.float32), k=k_)

    # "mask" means 1.0 in forbidden positions.
    # Add batch dimension: (batch_size, 1, seq_len, seq_len)
    mask_4d = xp.expand_dims(xp.expand_dims(mask_2d, axis=0), axis=0)
    mask_4d = xp.repeat(mask_4d, batch_size, axis=0)
    return mask_4d


def prepare_mlx_attention_mask(
    mask: Optional[Union[Tensor, ArrayLike]],
    *,
    query_shape: Tuple[int, ...],
    key_shape: Tuple[int, ...],
) -> Tuple[
    Literal["none", "causal", "explicit_bool", "explicit_additive", "dense_fallback"],
    Optional[Array],
]:
    """
    Translate repo mask semantics into the narrower MLX attention contracts.

    The MLX custom attention path uses this classification to decide whether it
    can take the optimized causal self-attention fast path or must fall back to
    the dense contract implementation.

    Mask inputs intentionally accept either:
    - repo `Tensor` masks that already follow the dense additive-mask
      contract (`1.0 == forbidden`, `0.0 == allowed`)
    - raw backend arrays for bool-mask cases (`True == keep`, `False == masked`)

    TODO: revisit this mixed `Tensor`/raw-array mask contract only if the repo
    adopts dtype-preserving `Tensor` semantics. Today, `Tensor` construction
    coerces data to `float32`, which would erase explicit-bool mask intent.
    """
    if mask is None:
        return "none", None

    raw_mask = mask.data if isinstance(mask, Tensor) else xp.array(mask)
    target_shape = (
        int(query_shape[0]),
        int(query_shape[1]),
        int(query_shape[-2]),
        int(key_shape[-2]),
    )
    if raw_mask.ndim > 4:
        return "dense_fallback", None

    try:
        broadcast_mask = xp.broadcast_to(raw_mask, target_shape)
    except ValueError:
        return "dense_fallback", None

    if broadcast_mask.dtype == xp.bool_:
        # MLX bool masks use "True means keep". Fully-masked rows stay on the
        # dense path because that remains the contract oracle.
        if xp.to_scalar(xp.any(xp.all(~broadcast_mask, axis=-1))):
            return "dense_fallback", None
        return "explicit_bool", raw_mask

    float_mask = xp.array(broadcast_mask, dtype=xp.float32)
    seq_q = target_shape[-2]
    seq_k = target_shape[-1]
    standard_causal = xp.broadcast_to(
        xp.triu(xp.ones((seq_q, seq_k), dtype=xp.float32), k=1),
        target_shape,
    )
    if xp.to_scalar(xp.all(float_mask == standard_causal)):
        return "causal", None

    unsupported_structural_masks = (
        xp.triu(xp.ones((seq_q, seq_k), dtype=xp.float32), k=0),
        xp.tril(xp.ones((seq_q, seq_k), dtype=xp.float32), k=-1),
        xp.tril(xp.ones((seq_q, seq_k), dtype=xp.float32), k=0),
    )
    if any(
        xp.to_scalar(
            xp.all(float_mask == xp.broadcast_to(structural_mask, target_shape))
        )
        for structural_mask in unsupported_structural_masks
    ):
        return "dense_fallback", None

    if xp.to_scalar(xp.any(xp.all(float_mask > 0, axis=-1))):
        return "dense_fallback", None

    return "explicit_additive", xp.array(raw_mask, dtype=xp.float32) * -1e9


def clean_and_tokenize(
    text: str, pattern: str = r"\w+|[^\w\s]|[\n\s]", lowercase: bool = True
) -> List[str]:
    """
    Naive tokenizer split by words

    Args:
        text (str): The entire input text to be tokenized
        pattern (str): Regular expression pattern used for tokenization.
                       Default splits on words, punctuation and whitespace.
        lowercase (bool): Whether to convert tokens to lowercase. Default True.

    Returns:
        list of tokens (str)

    Examples:
        >>> text = "Hello, world! \nNew line."
        >>> tokens = clean_and_tokenize(text)
        >>> print(tokens)
        ['hello', ',', 'world', '!', 'new', 'line', '.']
    """
    tokens = re.findall(pattern, text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return [token for token in tokens if token not in {" ", "\n"}]


def validate_batches(x: Array, y: Array) -> None:
    batch_size, seq_len = x.shape
    for b in range(min(4, batch_size)):
        for seq_idx in range(seq_len):
            print("[X]: ", x[b, : seq_idx + 1])
            print("[y]: ", y[b, seq_idx])


def token_batch_to_indices(
    token_batch: List[List[str]],
    vocab: Dict[Union[str, bytes], int],
) -> Array:
    """
    Convert a batch of token lists to a matrix of token indices using a given vocabulary.

    Args:
        token_batch (List[List[str]]): A list of tokenized sentences (each a list of strings).
        vocab (Dict[Union[str, bytes], int]): A vocabulary mapping tokens to integer indices.

    Returns:
        Array: A matrix of shape (batch_size, sequence_length) containing token indices.

    Examples:
        >>> token_batch = [["hello", "world"], ["this", "test"]]
        >>> vocab = {"hello": 0, "world": 1, "this": 2, "test": 3, b"<UNK>": 4}
        >>> indices = token_batch_to_indices(token_batch, vocab)
        >>> print(indices)
        [[0, 1],
         [2, 3]]
    """
    X: List[List[int]] = []
    for batch in token_batch:
        seq: List[int] = []
        for token in batch:
            seq.append(vocab.get(token, vocab[b"<UNK>"]))
        X.append(seq)
    return xp.array(X, dtype=xp.int32)


def load_wiki_simple() -> str:
    from autograd.data.utils import load_data

    if not os.path.exists("training_data/wiki_simple_english.txt"):
        print("Downloading data...")
        os.system(
            "curl -L -o examples/plain-text-wikipedia-simpleenglish.zip https://www.kaggle.com/api/v1/datasets/download/ffatty/plain-text-wikipedia-simpleenglish"
        )
        os.system("unzip examples/plain-text-wikipedia-simpleenglish.zip -d examples")
        os.system("rm -rf examples/1of2")
        os.system("rm -rf examples/2of2")
        os.system("mv examples/AllCombined.txt training_data/wiki_simple_english.txt")

    data = load_data(
        "training_data/wiki_simple_english.txt",
        "training_data/wiki_simple_english.txt",
    )
    assert isinstance(data, str)
    logger.info(f"{len(data)} characters in the entire dataset. Sample: \n{data[:100]}")
    return data


def load_shakespeare_mini() -> str:
    from autograd.data.utils import load_data

    data = load_data(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "training_data/tinyshakespeare.txt",
    )
    assert isinstance(data, str)
    logger.info(f"{len(data)} characters in the entire dataset. Sample: \n{data[:100]}")
    return data


def load_openwebtext(
    parquet_shards_per_batch: int = 1,
    *,
    start_token: str,
    split_token: str,
) -> OpenWebTextSource:
    """Return a streaming OpenWebText source backed by public parquet shards."""
    if parquet_shards_per_batch < 1:
        raise ValueError(
            f"parquet_shards_per_batch must be >= 1, got {parquet_shards_per_batch}"
        )
    parquet_manifest_url = "https://datasets-server.huggingface.co/parquet?dataset=Skylion007%2Fopenwebtext"

    os.makedirs("training_data", exist_ok=True)
    manifest_path = "training_data/openwebtext_parquet_manifest.json"
    parquet_dir = "training_data/openwebtext_parquet"
    os.makedirs(parquet_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print("Downloading OpenWebText parquet manifest...")
        tmp_manifest_path = f"{manifest_path}.tmp"
        with urlopen(parquet_manifest_url, timeout=60) as response:
            with open(tmp_manifest_path, "wb") as f:
                f.write(response.read())
        os.replace(tmp_manifest_path, manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    parquet_files = [
        parquet_file
        for parquet_file in manifest.get("parquet_files", [])
        if parquet_file.get("split") == "train"
    ]
    if not parquet_files:
        raise ValueError("OpenWebText parquet manifest has no train split files")

    return OpenWebTextSource(
        parquet_files=parquet_files,
        parquet_dir=parquet_dir,
        start_token=start_token,
        split_token=split_token,
        parquet_shards_per_batch=parquet_shards_per_batch,
    )


def _ensure_openwebtext_shard(parquet_file: dict[str, Any], parquet_dir: str) -> str:
    filename = parquet_file["filename"]
    expected_size = parquet_file.get("size")
    if not isinstance(expected_size, int):
        raise ValueError(
            f"OpenWebText manifest entry for {filename!r} is missing an integer 'size'"
        )
    parquet_path = os.path.join(parquet_dir, filename)
    if os.path.exists(parquet_path) and os.path.getsize(parquet_path) != expected_size:
        os.remove(parquet_path)

    if not os.path.exists(parquet_path):
        print(
            f"Downloading OpenWebText shard {filename} "
            f"({expected_size / 1_000_000:.0f} MB)..."
        )
        _download_url(parquet_file["url"], parquet_path, expected_size=expected_size)
    return parquet_path


def _download_url(url: str, filename: str, *, expected_size: int) -> None:
    """Download `url` to `filename` using parallel HTTP Range requests.

    Assumes the server honors `Range` (HuggingFace's CDN does). Each worker
    fetches one contiguous byte range; parts are reassembled in order.
    """
    parent_dir = os.path.dirname(filename)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    workers = min(8, max(1, expected_size))
    part_size = (expected_size + workers - 1) // workers

    def download_part(start: int) -> bytes:
        end = min(start + part_size - 1, expected_size - 1)
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError(
                f"range {start}-{end} returned {len(data)} bytes from {url!r}"
            )
        return data

    tmp_filename = f"{filename}.tmp"
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = executor.map(download_part, range(0, expected_size, part_size))
            with open(tmp_filename, "wb") as f:
                for part in parts:
                    f.write(part)
        os.replace(tmp_filename, filename)
    except Exception as exc:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        raise RuntimeError(
            f"Failed to download {url!r} to {filename!r}. "
            "The dataset is public, but unauthenticated HuggingFace downloads can "
            "still be rate-limited; retry later or keep cached parquet shards in "
            "training_data/openwebtext_parquet."
        ) from exc


def _iter_batches(
    values: Sequence[dict[str, Any]],
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])
