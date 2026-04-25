import pytest

from autograd.backend import xp
from autograd.data.dataset import PairedMapDataset
from autograd.data.sampler import (
    RandomSampler,
    SequentialSampler,
    TokenLengthGroupedRandomSampler,
)


def make_token_dataset(token_sequences):
    return PairedMapDataset(
        token_sequences,
        [xp.ones((len(tokens),), dtype=xp.int32) for tokens in token_sequences],
        input_key="tokens",
        target_key="loss_mask",
        dtype=xp.int32,
    )


def test_sequential_sampler_yields_indices_in_order():
    dataset = make_token_dataset([xp.arange(2), xp.arange(3), xp.arange(4)])

    assert list(SequentialSampler(dataset)) == [0, 1, 2]


def test_random_sampler_yields_cpu_permutation():
    dataset = make_token_dataset([xp.arange(2), xp.arange(3), xp.arange(4)])

    sampler = RandomSampler(dataset)

    assert sorted(sampler) == [0, 1, 2]


def test_random_sampler_replacement_respects_num_samples():
    dataset = make_token_dataset([xp.arange(2), xp.arange(3), xp.arange(4)])

    sampler = RandomSampler(dataset, replacement=True, num_samples=8)
    indices = list(sampler)

    assert len(sampler) == 8
    assert len(indices) == 8
    assert all(0 <= index < len(dataset) for index in indices)


def test_random_sampler_without_replacement_respects_num_samples():
    dataset = make_token_dataset([xp.arange(2), xp.arange(3), xp.arange(4)])

    sampler = RandomSampler(dataset, num_samples=2)

    assert len(sampler) == 2
    assert len(list(sampler)) == 2


def test_random_sampler_rejects_invalid_num_samples():
    dataset = make_token_dataset([xp.arange(2)])

    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        RandomSampler(dataset, num_samples=0)


def test_random_sampler_rejects_empty_dataset():
    with pytest.raises(ValueError, match="requires a non-empty dataset"):
        RandomSampler(PairedMapDataset([], []))


def test_sequential_sampler_requires_map_style_dataset():
    class NonMapDataset:
        def __iter__(self):
            return iter(())

    with pytest.raises(TypeError, match="SequentialSampler requires MapDataset"):
        SequentialSampler(NonMapDataset())


def test_length_grouped_sampler_shuffles_then_sorts_local_buffers(monkeypatch):
    dataset = make_token_dataset(
        [
            xp.arange(2, dtype=xp.int32),
            xp.arange(8, dtype=xp.int32),
            xp.arange(3, dtype=xp.int32),
            xp.arange(7, dtype=xp.int32),
        ]
    )

    def fixed_shuffle(indices):
        indices[:] = [1, 0, 3, 2]

    monkeypatch.setattr("autograd.data.sampler.np.random.shuffle", fixed_shuffle)
    sampler = TokenLengthGroupedRandomSampler(
        dataset,
        sort_buffer_size=2,
    )

    assert list(sampler) == [0, 1, 2, 3]


def test_length_grouped_sampler_requires_sort_buffer_size():
    with pytest.raises(TypeError, match="required keyword-only argument"):
        TokenLengthGroupedRandomSampler(
            make_token_dataset([xp.arange(2), xp.arange(3)]),
        )


def test_length_grouped_sampler_rejects_shuffle_arg():
    with pytest.raises(TypeError, match="unexpected keyword argument 'shuffle'"):
        TokenLengthGroupedRandomSampler(
            make_token_dataset([xp.arange(2), xp.arange(3)]),
            shuffle=True,
            sort_buffer_size=2,
        )


def test_length_grouped_sampler_rejects_invalid_sort_buffer():
    with pytest.raises(ValueError, match="sort_buffer_size must be >= 1"):
        TokenLengthGroupedRandomSampler(
            make_token_dataset([xp.arange(2), xp.arange(3)]),
            sort_buffer_size=0,
        )


def test_length_grouped_sampler_requires_map_dataset():
    with pytest.raises(TypeError, match="requires MapDataset"):
        TokenLengthGroupedRandomSampler([10, 20], sort_buffer_size=2)


def test_length_grouped_sampler_requires_tokens_field():
    with pytest.raises(ValueError, match="requires examples with a 'tokens' field"):
        TokenLengthGroupedRandomSampler(
            PairedMapDataset([xp.arange(2)], [xp.arange(2)]),
            sort_buffer_size=2,
        )
