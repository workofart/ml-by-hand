"""
Data pipeline boundary:

1. Per-sample transforms/tokenization
- dataset / transforms

2. Batch assembly and batch-time shaping
- collator

3. Training-specific logic
- trainer / training loop / model forward
"""

from numbers import Integral
from typing import Any, Callable, Sequence

from autograd.data.dataset import MapDataset
from autograd.data.sampler import Sampler

CollateFn = Callable[[Sequence[Any]], Any]


class DataLoader:
    """
    Generic example-batching loader.

    Map-style datasets iterate in stored order unless a sampler is supplied.
    DataLoader groups examples.
    Collator creates batches.
    """

    def __init__(
        self,
        dataset: MapDataset,
        batch_size: int,
        collator: CollateFn | None = None,
        *,
        sampler: Sampler | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.batch_size = batch_size
        self.collator = collator
        self.sampler = sampler
        self.drop_last = drop_last

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()
        if self.sampler is not None:
            self.sampler.on_epoch_start()

    def __iter__(self):
        examples = []
        yielded_batches = 0

        if self.sampler is not None:
            dataset_len = len(self.dataset)
            iterable = (
                self.dataset[self._validate_sampler_index(index, dataset_len)]
                for index in self.sampler
            )
        else:
            iterable = iter(self.dataset)
        for example in iterable:
            examples.append(example)

            if len(examples) == self.batch_size:
                yielded_batches += 1
                yield self.collator(examples) if self.collator else examples
                examples = []

        if examples and not self.drop_last:
            yielded_batches += 1
            yield self.collator(examples) if self.collator else examples

        if yielded_batches == 0:
            raise ValueError(
                "DataLoader yielded no batches. The dataset may be empty, may "
                "have yielded no examples for this pass, or drop_last=True may "
                "have dropped the only partial batch."
            )

    def __len__(self) -> int:
        # A sampler can yield a subset, repeat examples, or shard data, so the
        # number of rows seen by this loader is not always len(dataset).
        n = len(self.sampler) if self.sampler is not None else len(self.dataset)

        if self.drop_last:
            batch_count = n // self.batch_size
        else:
            batch_count = (n + self.batch_size - 1) // self.batch_size

        if batch_count < 1:
            raise ValueError(
                "DataLoader yielded no batches. The dataset may be empty, may "
                "have yielded no examples for this pass, or drop_last=True may "
                "have dropped the only partial batch."
            )

        return batch_count

    def _validate_sampler_index(self, index: Any, dataset_len: int) -> int:
        if not isinstance(index, Integral):
            raise TypeError(
                f"sampler yielded non-integer index {index!r} "
                f"of type {type(index).__name__}"
            )
        index_int = int(index)
        if index_int < 0 or index_int >= dataset_len:
            raise IndexError(
                f"sampler yielded index {index_int} outside dataset length "
                f"{dataset_len}"
            )
        return index_int
