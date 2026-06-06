"""
Data pipeline boundary:

1. Per-sample transforms/tokenization
- dataset / transforms

2. Batch assembly and batch-time shaping
- collator

3. Training-specific logic
- trainer / training loop / model forward
"""

import queue
import threading
from numbers import Integral
from typing import Any, Callable, Iterator, Sequence

from autograd.backend import pin_cuda_device
from autograd.data.dataset import MapDataset
from autograd.data.sampler import Sampler

CollateFn = Callable[[Sequence[Any]], Any]


class DataLoader:
    """
    Generic example-batching loader.

    Map-style datasets iterate in stored order unless a sampler is supplied.
    DataLoader groups examples.
    Collator creates batches.

    With prefetch=True a background daemon thread produces batches into a
    bounded queue so production (sampler indexing, collation, and for GPU
    backends the host->device copy in the collator) overlaps the consumer's
    compute instead of running synchronously right before each batch is used.
    Opt-in; default off keeps iteration fully synchronous and single-threaded.

    Reproducibility caveat: with prefetch=True the sampler's RNG draws run on
    the producer thread and interleave, in thread-scheduling order, with any
    global-RNG draws on the training thread (e.g. numpy-backend dropout) — so
    seeded runs are not bit-reproducible the way prefetch=False runs are.
    """

    def __init__(
        self,
        dataset: MapDataset,
        batch_size: int,
        collator: CollateFn | None = None,
        *,
        sampler: Sampler | None = None,
        drop_last: bool = False,
        prefetch: bool = False,
        prefetch_depth: int = 4,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")

        self.dataset = dataset
        self.batch_size = batch_size
        self.collator = collator
        self.sampler = sampler
        self.drop_last = drop_last
        self.prefetch = prefetch
        self.prefetch_depth = prefetch_depth

    def on_epoch_start(self) -> None:
        self.dataset.on_epoch_start()
        if self.sampler is not None:
            self.sampler.on_epoch_start()

    def __iter__(self) -> Iterator[Any]:
        if not self.prefetch:
            return self._iter_batches()
        return self._iter_prefetched()

    def _iter_batches(self):
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

    def _iter_prefetched(self):
        """Yield batches produced by a background thread (see class docstring).

        One producer thread iterates `_iter_batches` into a bounded queue while
        the consumer trains. The producer always emits a final sentinel (even on
        error or early stop), and any producer exception is re-raised on the
        consumer side so failures surface instead of hanging — including when
        the consumer abandons iteration early (break / GeneratorExit before the
        sentinel), in which case we signal stop and drain until the sentinel so
        the producer can never block forever on a full queue (which would leak
        the thread).
        """
        batch_queue: "queue.Queue" = queue.Queue(maxsize=self.prefetch_depth)
        sentinel = object()
        producer_error: list[BaseException] = []
        stop = threading.Event()

        def produce():
            try:
                # CuPy's current device is thread-local: re-pin so the
                # collator's host->device copies land on this process's DDP
                # device, not device 0.
                pin_cuda_device()
                for batch in self._iter_batches():
                    if stop.is_set():
                        break
                    batch_queue.put(batch)
            except BaseException as exc:  # surfaced to the consumer below
                producer_error.append(exc)
            finally:
                batch_queue.put(sentinel)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()

        sentinel_seen = False
        try:
            while True:
                batch = batch_queue.get()
                if batch is sentinel:
                    sentinel_seen = True
                    break
                yield batch
        finally:
            if not sentinel_seen:
                stop.set()
                # Unblock a producer stuck on a full queue. The timeout only
                # fires if the producer can no longer respond at all (e.g.
                # daemon threads frozen at interpreter shutdown) — give up
                # then instead of hanging the process.
                try:
                    while batch_queue.get(timeout=5.0) is not sentinel:
                        pass
                except queue.Empty:
                    pass
            # Raising in the finally covers both the normal path (sentinel
            # reached) and an early-stopping consumer, who would otherwise
            # never learn the producer died.
            if producer_error:
                raise producer_error[0]

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
