import numpy as np
import pytest

from autograd.backend import xp
from autograd.tensor import Tensor
from examples import gpt_2
from examples.gpt_2 import GPT2


def _to_numpy(x):
    if hasattr(xp, "to_numpy"):
        return xp.to_numpy(x)
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(x)
    return np.asarray(x)


def _to_numpy_float32(x):
    return _to_numpy(x.astype(xp.float32))


@pytest.mark.parametrize("use_packed_qkv", [True, False])
def test_gpt2_forward_kv_array_dispatch_matches_full_forward(
    monkeypatch,
    use_packed_qkv,
):
    monkeypatch.setattr(gpt_2, "NAME", "cupy")

    xp.random.seed(0)
    model = GPT2(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=2,
        max_seq_len=8,
        dropout_prob=0.0,
        num_decoder_layers=1,
        parameter_dtype="bfloat16" if use_packed_qkv else None,
        use_packed_qkv=use_packed_qkv,
    )
    input_ids = xp.array([[1, 2, 3], [1, 4, 5]], dtype=xp.int32)

    full_logits = model(Tensor(input_ids)).data[:, -1:, :]
    kv_logits, caches = model.forward_kv(input_ids)

    np.testing.assert_allclose(
        _to_numpy_float32(kv_logits),
        _to_numpy_float32(full_logits),
        rtol=2e-2 if use_packed_qkv else 5e-3,
        atol=2e-2 if use_packed_qkv else 5e-3,
    )
    assert len(caches) == 1
    assert caches[0][0].shape == (2, 2, 3, 8)
    assert caches[0][1].shape == (2, 2, 3, 8)

    prefix = input_ids[:, :2]
    next_token = input_ids[:, 2:]
    _, prefix_cache = model.forward_kv(prefix)
    step_logits, _ = model.forward_kv(next_token, kv_cache=prefix_cache, offset=2)

    np.testing.assert_allclose(
        _to_numpy_float32(step_logits),
        _to_numpy_float32(full_logits),
        rtol=2e-2 if use_packed_qkv else 5e-3,
        atol=2e-2 if use_packed_qkv else 5e-3,
    )
