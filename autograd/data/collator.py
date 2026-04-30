from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

from autograd.backend import Array, xp
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.functional import IGNORE_INDEX


def create_padding_mask(
    token_indices: Array,
    pad_idx: int = 0,
    dims: tuple[int, ...] | None = None,
) -> Array:
    """
    Create a float padding mask where pad-token positions are `1.0`.

    With `dims=None`, a `(batch_size, seq_len)` token matrix becomes the
    standard attention-mask shape `(batch_size, 1, 1, seq_len)`. Pass `dims`
    when a collator is formatting one sequence at a time and needs an explicit
    broadcast shape.
    """
    token_indices = xp.array(token_indices)
    pad_positions = (token_indices == pad_idx).astype(xp.float32)

    if dims is None:
        # Default shape for attention over batched token IDs.
        return xp.expand_dims(xp.expand_dims(pad_positions, axis=1), axis=1)
    return pad_positions.reshape(dims)


def pad_right_1d(values: Array, target_length: int, pad_value: Any) -> Array:
    """
    Right-pad a 1D array to `target_length`.

    This never truncates. Callers that allow truncation should truncate
    explicitly before calling this helper.
    """
    if len(values) > target_length:
        raise ValueError(
            f"cannot right-pad length {len(values)} to shorter target_length "
            f"{target_length}"
        )

    if len(values) == target_length:
        return values

    pad_width = target_length - len(values)
    return xp.concatenate(
        [
            values,
            xp.full((pad_width,), pad_value, dtype=values.dtype),
        ],
        axis=0,
    )


class Collator(ABC):
    """
    Abstract interface for collators that turn example lists into batches.
    """

    @abstractmethod
    def __call__(self, examples: Sequence[Any]) -> Any:
        pass


class PairedCollator(Collator):
    """
    Batches paired examples from `PairedMapDataset`.

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
            loss_total_weight=xp.array(
                len(examples) * (window_len - 1),
                dtype=xp.float32,
            ),
        )


class _CausalLMBaseCollator(Collator):
    """
    Shared mechanics for causal LM collators.

    Each example contains aligned `tokens` and `loss_mask` arrays. The shared
    flow is: left-truncate both arrays together, right-pad them together, shift
    tokens into `(input_ids, labels)`, and stack the final batch.
    """

    def __init__(
        self,
        max_tokens: int,
        *,
        pad_idx: int,
    ) -> None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2 for causal LM")
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx

    def _truncate_examples(
        self,
        examples: Sequence[dict[str, Array]],
    ) -> list[Tuple[Array, Array]]:
        truncated_examples = []

        for example in examples:
            tokens = xp.array(example["tokens"], dtype=xp.int32)
            loss_mask = xp.array(example["loss_mask"], dtype=xp.int32)
            if len(tokens) != len(loss_mask):
                raise ValueError("tokens and loss_mask must have the same length")

            if len(tokens) > self.max_tokens:
                tokens = tokens[-self.max_tokens :]
                loss_mask = loss_mask[-self.max_tokens :]

            truncated_tokens, truncated_loss_mask = tokens, loss_mask
            truncated_examples.append((truncated_tokens, truncated_loss_mask))

        return truncated_examples

    def _build_inputs_and_labels(
        self,
        tokens: Array,
        loss_mask: Array,
    ) -> Tuple[Array, Array]:
        input_ids = tokens[:-1]
        labels = xp.array(tokens[1:], dtype=xp.int32)
        labels[xp.array(loss_mask[1:], dtype=xp.float32) == 0] = IGNORE_INDEX
        return input_ids, labels

    def _stack_padded_examples(
        self,
        padded_examples: Sequence[Tuple[Array, Array]],
    ) -> CausalLMBatch:
        batch_inputs = []
        batch_labels = []
        loss_total_weight = xp.array(0.0, dtype=xp.float32)

        for padded_tokens, padded_loss_mask in padded_examples:
            input_tokens, labels = self._build_inputs_and_labels(
                padded_tokens,
                padded_loss_mask,
            )

            batch_inputs.append(input_tokens)
            batch_labels.append(labels)
            loss_total_weight = loss_total_weight + xp.sum(padded_loss_mask[1:] != 0)

        return CausalLMBatch(
            input_ids=xp.stack(batch_inputs, axis=0),
            labels=xp.stack(batch_labels, axis=0),
            loss_total_weight=loss_total_weight,
        )


class FixedLengthCausalLMCollator(_CausalLMBaseCollator):
    """
    Builds causal-LM batches padded to `max_tokens`.

    The batch-time flow is: left-truncate each `(tokens, loss_mask)` row, then
    right-pad every row to `max_tokens`, then shift tokens into inputs/labels.
    """

    def __call__(self, examples: Sequence[dict[str, Array]]) -> CausalLMBatch:
        truncated_examples = self._truncate_examples(examples)
        padded_examples = tuple(
            (
                pad_right_1d(tokens, self.max_tokens, self.pad_idx),
                pad_right_1d(loss_mask, self.max_tokens, 0),
            )
            for tokens, loss_mask in truncated_examples
        )
        return self._stack_padded_examples(padded_examples)


class BatchMaxLengthCausalLMCollator(_CausalLMBaseCollator):
    """
    Builds causal-LM batches padded to the longest row in this batch.

    The batch-time flow is: left-truncate each `(tokens, loss_mask)` row, then
    right-pad every row to the longest truncated row in this batch, then shift
    tokens into inputs/labels.
    """

    def __call__(self, examples: Sequence[dict[str, Array]]) -> CausalLMBatch:
        truncated_examples = self._truncate_examples(examples)
        longest_row_length = max(
            2, max(len(tokens) for tokens, _ in truncated_examples)
        )
        padded_examples = tuple(
            (
                pad_right_1d(tokens, longest_row_length, self.pad_idx),
                pad_right_1d(loss_mask, longest_row_length, 0),
            )
            for tokens, loss_mask in truncated_examples
        )
        return self._stack_padded_examples(padded_examples)


class Seq2SeqCollator(Collator):
    """
    Builds encoder-decoder LM batches with decoder inputs and padding masks.

    Each example has source `input_ids` and target `labels`. The collator pads
    both to `max_tokens`, masks padded labels with `IGNORE_INDEX`, and builds
    decoder inputs by shifting target tokens right with an SOS token.
    """

    def __init__(
        self,
        max_tokens: int,
        pad_idx: int,
        sos_idx: int,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 for seq2seq")
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx

    def __call__(self, examples: Sequence[dict[str, Array]]) -> Seq2SeqBatch:
        batch_inputs = []
        batch_decoder_inputs = []
        batch_labels = []
        batch_src_masks = []
        batch_tgt_masks = []

        for example in examples:
            input_ids = xp.array(example["input_ids"], dtype=xp.int32)
            labels = xp.array(example["labels"], dtype=xp.int32)

            if len(input_ids) > self.max_tokens:
                input_ids = input_ids[-self.max_tokens :]
            if len(labels) > self.max_tokens:
                labels = labels[-self.max_tokens :]

            formatted_input_ids = pad_right_1d(
                input_ids,
                self.max_tokens,
                self.pad_idx,
            )

            decoder_targets = pad_right_1d(
                labels,
                self.max_tokens,
                self.pad_idx,
            )
            masked_labels = xp.array(decoder_targets, dtype=xp.int32)
            masked_labels[masked_labels == self.pad_idx] = IGNORE_INDEX

            sos_token = xp.array([self.sos_idx], dtype=decoder_targets.dtype)
            decoder_input_ids = xp.concatenate(
                [sos_token, decoder_targets[:-1]],
                axis=0,
            )

            src_mask = create_padding_mask(
                formatted_input_ids,
                self.pad_idx,
                dims=(1, 1, len(formatted_input_ids)),
            )
            tgt_mask = create_padding_mask(
                decoder_input_ids,
                self.pad_idx,
                dims=(1, 1, len(decoder_input_ids)),
            )

            batch_inputs.append(formatted_input_ids)
            batch_decoder_inputs.append(decoder_input_ids)
            batch_labels.append(masked_labels)
            batch_src_masks.append(src_mask)
            batch_tgt_masks.append(tgt_mask)

        input_ids = xp.stack(batch_inputs, axis=0)
        decoder_input_ids = xp.stack(batch_decoder_inputs, axis=0)
        labels = xp.stack(batch_labels, axis=0)
        return Seq2SeqBatch(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            src_mask=xp.stack(batch_src_masks, axis=0),
            tgt_mask=xp.stack(batch_tgt_masks, axis=0),
        )
