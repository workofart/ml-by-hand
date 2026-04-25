"""
Data pipeline boundary:

1. Per-sample transforms/tokenization
- dataset / transforms

2. Batch assembly and batch-time shaping
- collate_fn

3. Training-specific logic
- trainer / training loop / model forward
"""

from typing import Any, Callable, Sequence

from autograd.data.dataset import IterableDataset

CollateFn = Callable[[Sequence[Any]], Any]


class DataLoader:
    """
    Generic example-batching loader.

    Dataset yields examples.
    DataLoader groups examples.
    Collator creates batches.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        collate_fn: CollateFn | None = None,
        *,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()

    def __iter__(self):
        examples = []
        yielded_batches = 0

        for example in self.dataset:
            examples.append(example)

            if len(examples) == self.batch_size:
                yielded_batches += 1
                yield self.collate_fn(examples) if self.collate_fn else examples
                examples = []

        if examples and not self.drop_last:
            yielded_batches += 1
            yield self.collate_fn(examples) if self.collate_fn else examples

        if yielded_batches == 0:
            raise ValueError(
                "DataLoader yielded no batches. The dataset may be empty, may "
                "have yielded no examples for this pass, or drop_last=True may "
                "have dropped the only partial batch."
            )

        try:
            expected_batches = len(self)
        except TypeError:
            return
        if yielded_batches != expected_batches:
            raise ValueError(
                "DataLoader yielded a different number of batches than len(DataLoader)."
            )

    def __len__(self) -> int:
        n = len(self.dataset)

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
