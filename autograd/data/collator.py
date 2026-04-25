from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Tuple

from autograd.backend import Array, xp
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.functional import IGNORE_INDEX
from autograd.text import utils as text_utils


def pack_tokens(tokens: Array, max_tokens: int, pad_idx: int) -> Array:
    # This is fixed-window truncation/padding, not true multi-example packing.
    # Left truncation assumes the tail of each SFT record is most valuable, which
    # usually keeps the final assistant answer but can drop earlier conversation
    # turns, the latest user prompt, or the start of an overlong assistant answer.
    # Use a conversation-aware packer when those boundaries matter.
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    if len(tokens) < max_tokens:
        pad_width = max_tokens - len(tokens)
        tokens = xp.concatenate(
            [tokens, xp.full((pad_width,), pad_idx, dtype=xp.int32)],
            axis=0,
        )
    return tokens


def truncate_aligned_left(
    values: Sequence[Array],
    max_tokens: int,
) -> Tuple[Array, ...]:
    """
    Keep the last `max_tokens` from each aligned 1D value.

    Shapes:
    - input: each value is `(T,)`
    - output: each value is `(min(T, max_tokens),)`

    Example:
        values=([10, 11, 12, 13], [0, 0, 1, 1]), max_tokens=3
        -> [11, 12, 13], [0, 1, 1]

    Tradeoff: this is simple and usually preserves the answer tail, but it can
    cut through conversation boundaries and remove the latest user prompt.
    This operation preserves alignment only; it does not know what any field
    means.

    Caller contract: `values` is a non-empty sequence of same-length 1D arrays.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    first_len = len(values[0])
    if first_len <= max_tokens:
        return tuple(values)
    return tuple(value[-max_tokens:] for value in values)


def pad_aligned_right(
    values: Sequence[Array],
    max_tokens: int,
    pad_values: Sequence[int],
) -> Tuple[Array, ...]:
    """
    Right-pad aligned 1D values to a fixed length.

    Example:
        values=([10, 11], [0, 1]), pad_values=(257, 0), max_tokens=4
        -> [10, 11, 257, 257], [0, 1, 0, 0]

    Tradeoff: fixed shapes are backend-friendly and easy to reason about, but
    dynamic batch padding can avoid wasted compute when packed lengths vary.
    Pad semantics live at the call site through `pad_values`.

    Caller contract: `values` and `pad_values` have the same length, `values`
    are same-length 1D arrays, and each value is no longer than `max_tokens`.
    """
    first_len = len(values[0])
    if first_len == max_tokens:
        return tuple(values)

    pad_width = max_tokens - first_len
    padded_values = []
    for value, pad_value in zip(values, pad_values):
        value_array = xp.array(value)
        padded_values.append(
            xp.concatenate(
                [
                    value_array,
                    xp.full((pad_width,), pad_value, dtype=value_array.dtype),
                ],
                axis=0,
            )
        )
    return tuple(padded_values)


def greedy_pack_aligned_examples(
    examples: Sequence[dict[str, Array]],
    fields: Sequence[str],
    max_tokens: int,
) -> list[dict[str, Array]]:
    """
    Greedily pack examples while preserving alignment across selected fields.

    Example lengths with max_tokens=8:
        [3, 4, 5] -> packs [3+4], [5]

    This operation preserves token/mask alignment and should run before
    collation because packing changes the dataset examples themselves.

    Tradeoff: for causal LM rows, tokens can attend across earlier packed
    examples after an EOS delimiter. This is efficient and simple, but not
    block-diagonal conversation isolation.

    Caller contract: `fields` is non-empty, selected fields are aligned, and
    overlong examples have already been truncated.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    packed_examples: list[dict[str, Array]] = []
    current_values = {field: [] for field in fields}
    current_len = 0

    def flush_current() -> None:
        nonlocal current_values, current_len
        if not current_values[fields[0]]:
            return
        packed_examples.append(
            {field: xp.concatenate(current_values[field], axis=0) for field in fields}
        )
        current_values = {field: [] for field in fields}
        current_len = 0

    for example in examples:
        values = {field: xp.array(example[field], dtype=xp.int32) for field in fields}
        example_len = len(values[fields[0]])
        if example_len > max_tokens:
            raise ValueError(
                "example is longer than max_tokens; truncate before packing"
            )

        if current_len and current_len + example_len > max_tokens:
            flush_current()

        for field in fields:
            current_values[field].append(values[field])
        current_len += example_len

    flush_current()
    return packed_examples


def build_causal_lm_inputs_and_labels(
    tokens: Array,
    loss_mask: Array,
) -> Tuple[Array, Array]:
    """
    Shift one token/mask row into causal-LM inputs and labels.

    Example:
        tokens    = [10, 20, 30]
        loss_mask = [ 0,  1,  1]
        input_ids = [10, 20]
        labels    = [20, 30]

    Any shifted label whose source token has `loss_mask == 0` becomes
    `IGNORE_INDEX`, so prompt, role-marker, and pad tokens do not train loss.

    Caller contract: `tokens` and `loss_mask` are aligned and contain at least
    two tokens after padding.
    """
    input_ids = tokens[:-1]
    labels = xp.array(tokens[1:], dtype=xp.int32)
    labels[xp.array(loss_mask[1:], dtype=xp.float32) == 0] = IGNORE_INDEX
    return input_ids, labels


class Collator(ABC):
    """
    Abstract interface for collators that turn example lists into batches.
    """

    @abstractmethod
    def __call__(self, examples: Sequence[Any]) -> Any:
        pass


class PairedCollator(Collator):
    """
    Batches paired examples from `PairedIterableDataset`.

    Examples:
        >>> collator = PairedCollator()
        >>> batch_X, batch_y = collator(
        ...     [
        ...         {"inputs": xp.array([1, 2]), "targets": xp.array(0)},
        ...         {"inputs": xp.array([3, 4]), "targets": xp.array(1)},
        ...     ]
        ... )
        >>> batch_X.shape, batch_y.shape
        ((2, 2), (2,))
    """

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Tuple[Array, Array]:
        batch_X = xp.stack(
            [xp.array(example["inputs"]) for example in examples], axis=0
        )
        batch_y = xp.stack(
            [xp.array(example["targets"]) for example in examples],
            axis=0,
        )
        return batch_X, batch_y


class OneHotCollator(Collator):
    """
    Batches token-id examples and materializes one-hot inputs.

    This keeps token IDs in memory and expands them to one-hot features at
    batch time.

    Examples:
        >>> collator = OneHotCollator(num_classes=4)
        >>> batch_X, batch_y = collator(
        ...     [
        ...         {"inputs": xp.array([0, 2]), "targets": xp.array(1)},
        ...         {"inputs": xp.array([1, 3]), "targets": xp.array(0)},
        ...     ]
        ... )
        >>> batch_X.shape, batch_y.shape
        ((2, 2, 4), (2,))
    """

    def __init__(self, num_classes: int, dtype: Array = xp.float32) -> None:
        self.num_classes = num_classes
        self.dtype = dtype

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Tuple[Array, Array]:
        # TODO: replace batch-time one-hot with direct token-id model inputs.
        eye = xp.eye(self.num_classes, dtype=self.dtype)
        batch_tokens = xp.stack(
            [xp.array(example["inputs"], dtype=xp.int32) for example in examples],
            axis=0,
        )
        batch_y = xp.stack(
            [xp.array(example["targets"]) for example in examples],
            axis=0,
        )
        return eye[batch_tokens], batch_y


class CausalLMWindowCollator(Collator):
    def __call__(self, examples: Sequence[TokenWindowExample]) -> CausalLMBatch:
        stream = examples[0].stream
        window_len = examples[0].window_len
        if window_len < 2:
            raise ValueError("window_len must be >= 2 for causal LM shifting")

        offsets = xp.array([ex.offset for ex in examples], dtype=xp.int32)
        positions = xp.arange(window_len, dtype=xp.int32)

        windows = stream[offsets[:, None] + positions[None, :]]

        return CausalLMBatch(
            input_ids=windows[:, :-1],
            labels=windows[:, 1:],
        )


class CausalLMCollator(Collator):
    def __init__(
        self,
        max_tokens: int,
        pad_idx: int,
        *,
        truncator: Callable[[Sequence[Array], int], Tuple[Array, ...]],
        padder: Callable[[Sequence[Array], int, Sequence[int]], Tuple[Array, ...]],
        label_builder: Callable[[Array, Array], Tuple[Array, Array]],
    ) -> None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2 for causal LM")
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.truncator = truncator
        self.padder = padder
        self.label_builder = label_builder

    def __call__(self, examples: Sequence[dict[str, Array]]) -> CausalLMBatch:
        batch_inputs = []
        batch_labels = []

        for example in examples:
            tokens = xp.array(example["tokens"], dtype=xp.int32)
            loss_mask = xp.array(example["loss_mask"], dtype=xp.int32)

            truncated_tokens, truncated_loss_mask = self.truncator(
                (tokens, loss_mask),
                self.max_tokens,
            )
            padded_tokens, padded_loss_mask = self.padder(
                (truncated_tokens, truncated_loss_mask),
                self.max_tokens,
                (self.pad_idx, 0),
            )
            input_tokens, labels = self.label_builder(padded_tokens, padded_loss_mask)

            batch_inputs.append(input_tokens)
            batch_labels.append(labels)

        return CausalLMBatch(
            input_ids=xp.stack(batch_inputs, axis=0),
            labels=xp.stack(batch_labels, axis=0),
        )


class Seq2SeqCollator(Collator):
    """
    Builds encoder-decoder LM batches with decoder inputs and padding masks.
    """

    def __init__(
        self,
        max_tokens: int,
        pad_idx: int,
        sos_idx: int,
        *,
        packer: Callable[[Array, int, int], Array],
    ) -> None:
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.packer = packer

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Seq2SeqBatch:
        batch_inputs = []
        batch_decoder_targets = []
        batch_labels = []

        for example in examples:
            input_ids = xp.array(example["input_ids"], dtype=xp.int32)
            labels = xp.array(example["labels"], dtype=xp.int32)

            packed_input_ids = self.packer(input_ids, self.max_tokens, self.pad_idx)
            packed_target_tokens = self.packer(labels, self.max_tokens, self.pad_idx)

            masked_labels = xp.array(packed_target_tokens, dtype=xp.int32)
            masked_labels[masked_labels == self.pad_idx] = IGNORE_INDEX

            batch_inputs.append(packed_input_ids)
            batch_decoder_targets.append(packed_target_tokens)
            batch_labels.append(masked_labels)

        input_ids = xp.stack(batch_inputs, axis=0)
        decoder_targets = xp.stack(batch_decoder_targets, axis=0)
        labels = xp.stack(batch_labels, axis=0)
        batch_size = input_ids.shape[0]
        sos_column = xp.full((batch_size, 1), self.sos_idx, dtype=decoder_targets.dtype)
        decoder_input_ids = xp.concatenate(
            [sos_column, decoder_targets[:, :-1]], axis=1
        )
        return Seq2SeqBatch(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            src_mask=text_utils.create_padding_mask(input_ids, self.pad_idx),
            tgt_mask=text_utils.create_padding_mask(decoder_input_ids, self.pad_idx),
        )
