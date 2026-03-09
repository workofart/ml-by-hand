from typing import Optional
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

from autograd.backend import IS_MLX, xp
from autograd.nn import ScaledDotProductAttention
from autograd.tensor import Tensor
from autograd.text.utils import create_causal_mask
from test.helpers import allclose


class TestScaledDotProductAttention(TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.num_heads = 3
        self.seq_len = 4
        self.head_dim = 5

        xp.random.seed(7)
        self.query = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            )
        )
        self.key = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            )
        )
        self.value = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            )
        )

    def _run_attention(self, implementation: str, mask=None):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation=implementation,
        )
        return attention(self.query, self.key, self.value, mask=mask)

    def _causal_mask(self, *, batch_size: int, seq_len: int) -> Tensor:
        return Tensor(
            create_causal_mask(seq_len=seq_len, batch_size=batch_size),
            requires_grad=False,
        )

    def _run_custom_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        return attention(query, key, value, mask=mask)

    @skipUnless(IS_MLX, "mlx_fast_reference parity requires the MLX backend")
    def test_mlx_reference_matches_dense_without_mask(self):
        dense = self._run_attention("dense")
        mlx_reference = self._run_attention("mlx_fast_reference")

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_fast_reference parity requires the MLX backend")
    def test_mlx_reference_matches_dense_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        dense = self._run_attention("dense", mask=mask)
        mlx_reference = self._run_attention("mlx_fast_reference", mask=mask)

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_fast_reference parity requires the MLX backend")
    def test_mlx_reference_matches_dense_for_explicit_additive_mask(self):
        additive_mask = xp.zeros(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.float32,
        )
        additive_mask[:, :, 0, -1] = 1.0
        additive_mask[:, :, 1, 0] = 1.0
        mask = Tensor(additive_mask, requires_grad=False)

        dense = self._run_attention("dense", mask=mask)
        mlx_reference = self._run_attention("mlx_fast_reference", mask=mask)

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_fast_reference parity requires the MLX backend")
    def test_mlx_reference_matches_dense_for_explicit_bool_mask(self):
        bool_mask = xp.ones(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.bool_,
        )
        bool_mask[:, :, 0, -1] = False
        bool_mask[:, :, -1, 0] = False

        dense_mask = Tensor((~bool_mask).astype(xp.float32), requires_grad=False)
        dense = self._run_attention("dense", mask=dense_mask)
        mlx_reference = self._run_attention("mlx_fast_reference", mask=bool_mask)

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

    def test_mlx_reference_falls_back_to_dense_for_reverse_causal_mask(self):
        mask = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                lookback=True,
            ),
            requires_grad=False,
        )

        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXReference.apply",
            side_effect=AssertionError("mlx reference path should not be used"),
        ):
            result = self._run_attention("mlx_fast_reference", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_reference_falls_back_to_dense_for_masked_diagonal(self):
        mask = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                mask_diagonal=True,
            ),
            requires_grad=False,
        )

        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXReference.apply",
            side_effect=AssertionError("mlx reference path should not be used"),
        ):
            result = self._run_attention("mlx_fast_reference", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_reference_falls_back_to_dense_for_fully_masked_rows(self):
        additive_mask = xp.zeros(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.float32,
        )
        additive_mask[:, :, 0, :] = 1.0
        mask = Tensor(additive_mask, requires_grad=False)

        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXReference.apply",
            side_effect=AssertionError("mlx reference path should not be used"),
        ):
            result = self._run_attention("mlx_fast_reference", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_fast_reference parity requires the MLX backend")
    def test_mlx_reference_backward_raises_exact_error(self):
        output = self._run_attention("mlx_fast_reference")

        with self.assertRaisesRegex(
            NotImplementedError,
            "mlx_fast_reference is forward-only",
        ):
            output.sum().backward()

    def test_mlx_reference_requires_mlx_backend(self):
        import builtins

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"mlx.core", "mlx.core.fast"}:
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(
                RuntimeError,
                "mlx_fast_reference requires the MLX backend",
            ):
                self._run_attention("mlx_fast_reference")

    def test_mlx_custom_falls_back_to_dense_when_dropout_is_nonzero(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.1,
            _implementation="mlx_custom",
        )
        dense_attention = ScaledDotProductAttention(
            dropout_prob=0.1,
            _implementation="dense",
        )

        result = attention(self.query, self.key, self.value)
        dense = dense_attention(self.query, self.key, self.value)

        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_without_mask(self):
        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom")

        dense = self._run_attention("dense")
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_for_reverse_causal_mask(self):
        mask = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                lookback=True,
            ),
            requires_grad=False,
        )

        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_for_explicit_additive_mask(self):
        additive_mask = xp.zeros(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.float32,
        )
        additive_mask[:, :, 0, -1] = 1.0
        additive_mask[:, :, 1, 0] = 1.0
        mask = Tensor(additive_mask, requires_grad=False)

        with patch(
            "autograd.nn._ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_matches_dense_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        dense = self._run_attention("dense", mask=mask)
        mlx_custom = self._run_attention("mlx_custom", mask=mask)

        assert allclose(dense.data, mlx_custom.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_matches_mlx_reference_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        mlx_reference = self._run_attention("mlx_fast_reference", mask=mask)
        mlx_custom = self._run_attention("mlx_custom", mask=mask)

        assert allclose(mlx_reference.data, mlx_custom.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_matches_pytorch_math_sdpa_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )
        mlx_custom = self._run_attention("mlx_custom", mask=mask)

        torch_query = torch.tensor(xp.to_numpy(self.query.data), dtype=torch.float32)
        torch_key = torch.tensor(xp.to_numpy(self.key.data), dtype=torch.float32)
        torch_value = torch.tensor(xp.to_numpy(self.value.data), dtype=torch.float32)
        torch_output = torch.nn.functional.scaled_dot_product_attention(
            torch_query,
            torch_key,
            torch_value,
            is_causal=True,
            dropout_p=0.0,
        )

        assert allclose(mlx_custom.data, torch_output, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_backward_raises_exact_error(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )
        output = self._run_attention("mlx_custom", mask=mask)

        with self.assertRaisesRegex(
            NotImplementedError,
            "mlx_custom is forward-only",
        ):
            output.sum().backward()

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_reduces_to_prefix_average_when_queries_and_keys_are_zero(self):
        test_shapes = (
            (1, 1, 4, 1),
            (2, 2, 5, 3),
        )

        for shape in test_shapes:
            with self.subTest(shape=shape):
                batch_size, _, seq_len, _ = shape
                query = Tensor(xp.zeros(shape, dtype=xp.float32))
                key = Tensor(xp.zeros(shape, dtype=xp.float32))
                value = Tensor(xp.random.normal(shape=shape))
                mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

                output = self._run_custom_attention(query, key, value, mask=mask)
                prefix_sums = xp.cumsum(value.data, axis=2)
                prefix_lengths = xp.arange(1, seq_len + 1, dtype=xp.float32).reshape(
                    1, 1, seq_len, 1
                )
                expected = prefix_sums / prefix_lengths

                assert allclose(output.data, expected, atol=1e-6, rtol=1e-6)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_does_not_let_future_tokens_change_past_outputs(self):
        shape = (2, 2, 5, 3)
        batch_size, _, seq_len, _ = shape
        mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

        query = Tensor(xp.random.normal(shape=shape))
        key = Tensor(xp.random.normal(shape=shape))
        value = Tensor(xp.random.normal(shape=shape))
        baseline = self._run_custom_attention(query, key, value, mask=mask)

        cutoff = 2
        modified_query = xp.array(query.data)
        modified_key = xp.array(key.data)
        modified_value = xp.array(value.data)

        modified_query[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_query[:, :, cutoff + 1 :, :].shape
        )
        modified_key[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_key[:, :, cutoff + 1 :, :].shape
        )
        modified_value[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_value[:, :, cutoff + 1 :, :].shape
        )

        updated = self._run_custom_attention(
            Tensor(modified_query),
            Tensor(modified_key),
            Tensor(modified_value),
            mask=mask,
        )

        assert allclose(
            baseline.data[:, :, : cutoff + 1, :],
            updated.data[:, :, : cutoff + 1, :],
            atol=1e-5,
            rtol=1e-5,
        )

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_keeps_batches_and_heads_independent(self):
        shape = (2, 3, 4, 2)
        batch_size, _, seq_len, _ = shape
        mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

        query = Tensor(xp.random.normal(shape=shape))
        key = Tensor(xp.random.normal(shape=shape))
        value = Tensor(xp.random.normal(shape=shape))
        baseline = self._run_custom_attention(query, key, value, mask=mask)

        target_batch = 1
        target_head = 2
        modified_query = xp.array(query.data)
        modified_key = xp.array(key.data)
        modified_value = xp.array(value.data)

        modified_query[target_batch, target_head] = xp.random.normal(
            shape=modified_query[target_batch, target_head].shape
        )
        modified_key[target_batch, target_head] = xp.random.normal(
            shape=modified_key[target_batch, target_head].shape
        )
        modified_value[target_batch, target_head] = xp.random.normal(
            shape=modified_value[target_batch, target_head].shape
        )

        updated = self._run_custom_attention(
            Tensor(modified_query),
            Tensor(modified_key),
            Tensor(modified_value),
            mask=mask,
        )

        baseline_data = xp.to_numpy(baseline.data)
        updated_data = xp.to_numpy(updated.data)
        baseline_data[target_batch, target_head] = 0.0
        updated_data[target_batch, target_head] = 0.0

        assert allclose(baseline_data, updated_data, atol=1e-5, rtol=1e-5)
