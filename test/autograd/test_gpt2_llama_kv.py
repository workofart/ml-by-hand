import numpy as np
import pytest

from autograd.backend import NAME, xp
from autograd.tensor import Tensor
from examples import gpt_2_llama
from examples.gpt_2 import GPT2
from examples.gpt_2_llama import GPT2Llama


def _to_numpy(x):
    if hasattr(xp, "to_numpy"):
        return xp.to_numpy(x)
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(x)
    return np.asarray(x)


def test_forward_kv_cuda_dispatch_matches_full_forward(monkeypatch):
    monkeypatch.setattr(gpt_2_llama, "IS_MLX", False)
    monkeypatch.setattr(gpt_2_llama, "IS_CUPY", True)

    xp.random.seed(0)
    model = GPT2Llama(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        ff_hidden_size=16,
    )
    input_ids = xp.array([[1, 2, 3]], dtype=xp.int32)

    full_logits = model(Tensor(input_ids)).data[:, -1:, :]
    kv_logits, caches = model.forward_kv(input_ids)

    np.testing.assert_allclose(
        _to_numpy(kv_logits),
        _to_numpy(full_logits),
        rtol=5e-3,
        atol=5e-3,
    )
    assert len(caches) == 1
    assert caches[0][0].shape == (1, 1, 3, 4)
    assert caches[0][1].shape == (1, 1, 3, 4)

    prefix = input_ids[:, :2]
    next_token = input_ids[:, 2:]
    _, prefix_cache = model.forward_kv(prefix)
    step_logits, _ = model.forward_kv(next_token, kv_cache=prefix_cache, offset=2)

    np.testing.assert_allclose(
        _to_numpy(step_logits),
        _to_numpy(full_logits),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(
    NAME not in ("cupy", "mlx"),
    reason="GPT2.forward_kv requires MLX or CUDA/CuPy",
)
def test_gpt2_batched_forward_kv_matches_full_forward():
    xp.random.seed(0)
    model = GPT2(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        use_packed_qkv=True,
    )
    input_ids = xp.array([[1, 2, 3], [1, 4, 5]], dtype=xp.int32)

    full_logits = model(Tensor(input_ids)).data[:, -1:, :]
    kv_logits, caches = model.forward_kv(input_ids)

    np.testing.assert_allclose(
        _to_numpy(kv_logits),
        _to_numpy(full_logits),
        rtol=5e-3,
        atol=5e-3,
    )
    assert len(caches) == 1
    assert caches[0][0].shape == (2, 2, 3, 4)
    assert caches[0][1].shape == (2, 2, 3, 4)

    prefix = input_ids[:, :2]
    next_token = input_ids[:, 2:]
    _, prefix_cache = model.forward_kv(prefix)
    step_logits, _ = model.forward_kv(next_token, kv_cache=prefix_cache, offset=2)

    np.testing.assert_allclose(
        _to_numpy(step_logits),
        _to_numpy(full_logits),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(
    NAME not in ("cupy", "mlx"),
    reason="GPT2.forward_kv requires MLX or CUDA/CuPy",
)
def test_gpt2_repeated_prompt_kv_matches_batched_prompt_kv_decode():
    xp.random.seed(0)
    model = GPT2(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        use_packed_qkv=True,
    )
    prompt = [1, 2, 3]
    batch_prompt = xp.array([prompt, prompt], dtype=xp.int32)
    next_token = xp.array([[4], [5]], dtype=xp.int32)

    _, batched_cache = model.forward_kv(batch_prompt)
    batched_logits, _ = model.forward_kv(
        next_token,
        kv_cache=batched_cache,
        offset=len(prompt),
    )

    single_logits, single_cache = model.forward_kv(xp.array([prompt], dtype=xp.int32))
    repeated_cache = [
        (xp.repeat(k, 2, axis=0), xp.repeat(v, 2, axis=0)) for k, v in single_cache
    ]
    repeated_logits = xp.repeat(single_logits, 2, axis=0)

    np.testing.assert_allclose(
        _to_numpy(repeated_logits),
        _to_numpy(model.forward_kv(batch_prompt)[0]),
        rtol=5e-3,
        atol=5e-3,
    )

    repeated_step_logits, _ = model.forward_kv(
        next_token,
        kv_cache=repeated_cache,
        offset=len(prompt),
    )

    np.testing.assert_allclose(
        _to_numpy(repeated_step_logits),
        _to_numpy(batched_logits),
        rtol=5e-3,
        atol=5e-3,
    )


def test_gpt2_forward_selected_indices_matches_full_logits():
    xp.random.seed(0)
    model = GPT2(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        use_packed_qkv=True,
    )
    input_ids = xp.array([[1, 2, 3], [1, 4, 5]], dtype=xp.int32)
    selected_indices = xp.array([1, 3, 5], dtype=xp.int32)

    full_logits = model(Tensor(input_ids))
    selected_logits = model(
        Tensor(input_ids),
        selected_token_indices=selected_indices,
    )
    expected = full_logits.reshape(-1, full_logits.shape[-1])[selected_indices]

    np.testing.assert_allclose(
        _to_numpy(selected_logits.data),
        _to_numpy(expected.data),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(
    NAME not in ("cupy", "mlx"),
    reason="GPT2.forward_kv requires MLX or CUDA/CuPy",
)
def test_gpt2_warmup_kv_accepts_batch_size_without_changing_outputs():
    xp.random.seed(0)
    model = GPT2(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        use_packed_qkv=True,
    )
    input_ids = xp.array([[1, 2, 3], [1, 4, 5]], dtype=xp.int32)

    before, _ = model.forward_kv(input_ids)
    model.warmup_kv(prompt_len=3, decode_steps=2, batch_size=2)
    after, _ = model.forward_kv(input_ids)

    np.testing.assert_allclose(
        _to_numpy(after),
        _to_numpy(before),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.skipif(NAME != "cupy", reason="requires CUDA/CuPy bf16")
def test_forward_kv_cupy_bfloat16_matches_full_forward():
    xp.random.seed(0)
    model = GPT2Llama(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        ff_hidden_size=32,
        parameter_dtype="bfloat16",
    )
    input_ids = xp.array([[1, 2, 3]], dtype=xp.int32)

    full_logits = model(Tensor(input_ids)).data[:, -1:, :]
    kv_logits, _ = model.forward_kv(input_ids)

    np.testing.assert_allclose(
        _to_numpy(kv_logits).astype(np.float32),
        _to_numpy(full_logits).astype(np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
