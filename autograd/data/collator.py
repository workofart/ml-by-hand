from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple

from autograd.backend import Array, xp
from autograd.data.types import CausalLMBatch, Seq2SeqBatch, TokenWindowExample
from autograd.functional import IGNORE_INDEX
from autograd.text import utils as text_utils


def pack_tokens(tokens: Array, max_tokens: int, pad_idx: int) -> Array:
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    if len(tokens) < max_tokens:
        pad_width = max_tokens - len(tokens)
        tokens = xp.concatenate(
            [tokens, xp.full((pad_width,), pad_idx, dtype=xp.int32)],
            axis=0,
        )
    return tokens


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
        packer: Optional[Callable[[Array, int, int], Array]] = None,
    ) -> None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2 for causal LM")
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.packer = packer or pack_tokens

    def __call__(self, examples: Sequence[dict[str, Array]]) -> CausalLMBatch:
        if not examples:
            raise ValueError("examples must not be empty")
        batch_inputs = []
        batch_labels = []

        for example in examples:
            tokens = xp.array(example["tokens"], dtype=xp.int32)
            loss_mask = xp.array(
                example.get(
                    "loss_mask",
                    xp.ones(tokens.shape, dtype=xp.int32),
                ),
                dtype=xp.int32,
            )
            if len(tokens) != len(loss_mask):
                raise ValueError("tokens and loss_mask must have the same length")

            packed_tokens = self.packer(tokens, self.max_tokens, self.pad_idx)
            packed_loss_mask = self.packer(loss_mask, self.max_tokens, 0)

            input_tokens = packed_tokens[:-1]
            labels = xp.array(packed_tokens[1:], dtype=xp.int32)
            labels[xp.array(packed_loss_mask[1:], dtype=xp.float32) == 0] = IGNORE_INDEX

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
        packer: Optional[Callable[[Array, int, int], Array]] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.packer = packer or pack_tokens

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
