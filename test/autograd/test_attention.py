from typing import Optional
from unittest import TestCase, skipUnless
from unittest.mock import patch

import torch

from autograd.backend import IS_MLX, xp
from autograd.functional import ScaledDotProductAttentionMLXCustom
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

    def _backward_grads(
        self,
        implementation: str,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        upstream: Optional[Tensor] = None,
    ):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation=implementation,
        )
        output = attention(query, key, value, mask=mask)
        output.backward(upstream)
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        return query.grad.data, key.grad.data, value.grad.data

    def _attention_objective(
        self,
        implementation: str,
        *,
        query,
        key,
        value,
        mask: Optional[Tensor],
        upstream: Tensor,
    ) -> float:
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation=implementation,
        )
        output = attention(Tensor(query), Tensor(key), Tensor(value), mask=mask)
        return xp.to_scalar(xp.sum(output.data * upstream.data))

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
            "autograd.functional.ScaledDotProductAttentionMLXReference.apply",
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
            "autograd.functional.ScaledDotProductAttentionMLXReference.apply",
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
            "autograd.functional.ScaledDotProductAttentionMLXReference.apply",
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

    def test_mlx_custom_falls_back_to_dense_without_mask(self):
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
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
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
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
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_for_explicit_bool_mask(self):
        bool_mask = xp.ones(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.bool_,
        )
        bool_mask[:, :, 0, -1] = False
        bool_mask[:, :, -1, 0] = False

        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom", mask=bool_mask)

        dense = self._run_attention("dense", mask=bool_mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_for_masked_diagonal(self):
        mask = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                mask_diagonal=True,
            ),
            requires_grad=False,
        )

        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            result = self._run_attention("mlx_custom", mask=mask)

        dense = self._run_attention("dense", mask=mask)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_falls_back_to_dense_for_fully_masked_rows(self):
        additive_mask = xp.zeros(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.float32,
        )
        additive_mask[:, :, 0, :] = 1.0
        mask = Tensor(additive_mask, requires_grad=False)

        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
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
    def test_mlx_custom_uses_custom_path_in_eval_mode_with_configured_dropout(self):
        mask = self._causal_mask(batch_size=self.batch_size, seq_len=self.seq_len)
        dense_attention = ScaledDotProductAttention(
            dropout_prob=0.25,
            _implementation="dense",
        )
        custom_attention = ScaledDotProductAttention(
            dropout_prob=0.25,
            _implementation="mlx_custom",
        )
        dense_attention.eval()
        custom_attention.eval()

        dense = dense_attention(self.query, self.key, self.value, mask=mask)
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            wraps=ScaledDotProductAttentionMLXCustom.apply,
        ) as custom_apply:
            result = custom_attention(self.query, self.key, self.value, mask=mask)

        self.assertEqual(custom_apply.call_count, 1)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_train_mode_dropout_matches_dense_for_standard_causal_mask(self):
        mask = self._causal_mask(batch_size=self.batch_size, seq_len=self.seq_len)
        dropout_prob = 0.25
        seed = 123

        dense_attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
            _implementation="dense",
        )
        custom_attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
            _implementation="mlx_custom",
        )
        dense_attention.train()
        custom_attention.train()

        xp.random.seed(seed)
        dense = dense_attention(self.query, self.key, self.value, mask=mask)

        xp.random.seed(seed)
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            wraps=ScaledDotProductAttentionMLXCustom.apply,
        ) as custom_apply:
            result = custom_attention(self.query, self.key, self.value, mask=mask)

        self.assertEqual(custom_apply.call_count, 1)
        assert allclose(dense.data, result.data, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_backward_matches_dense_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )
        upstream = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            ),
            requires_grad=False,
        )

        dense_query = Tensor(xp.array(self.query.data))
        dense_key = Tensor(xp.array(self.key.data))
        dense_value = Tensor(xp.array(self.value.data))
        mlx_query = Tensor(xp.array(self.query.data))
        mlx_key = Tensor(xp.array(self.key.data))
        mlx_value = Tensor(xp.array(self.value.data))

        dense_grads = self._backward_grads(
            "dense",
            query=dense_query,
            key=dense_key,
            value=dense_value,
            mask=mask,
            upstream=upstream,
        )
        mlx_grads = self._backward_grads(
            "mlx_custom",
            query=mlx_query,
            key=mlx_key,
            value=mlx_value,
            mask=mask,
            upstream=upstream,
        )

        for dense_grad, mlx_grad in zip(dense_grads, mlx_grads):
            assert allclose(dense_grad, mlx_grad, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_train_mode_dropout_backward_matches_dense(self):
        mask = self._causal_mask(batch_size=self.batch_size, seq_len=self.seq_len)
        dropout_prob = 0.25
        seed = 321
        upstream = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            ),
            requires_grad=False,
        )

        dense_query = Tensor(xp.array(self.query.data))
        dense_key = Tensor(xp.array(self.key.data))
        dense_value = Tensor(xp.array(self.value.data))
        custom_query = Tensor(xp.array(self.query.data))
        custom_key = Tensor(xp.array(self.key.data))
        custom_value = Tensor(xp.array(self.value.data))

        dense_attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
            _implementation="dense",
        )
        custom_attention = ScaledDotProductAttention(
            dropout_prob=dropout_prob,
            _implementation="mlx_custom",
        )
        dense_attention.train()
        custom_attention.train()

        xp.random.seed(seed)
        dense_output = dense_attention(
            dense_query,
            dense_key,
            dense_value,
            mask=mask,
        )
        dense_output.backward(upstream)

        xp.random.seed(seed)
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            wraps=ScaledDotProductAttentionMLXCustom.apply,
        ) as custom_apply:
            custom_output = custom_attention(
                custom_query,
                custom_key,
                custom_value,
                mask=mask,
            )

        self.assertEqual(custom_apply.call_count, 1)
        custom_output.backward(upstream)

        assert dense_query.grad is not None
        assert dense_key.grad is not None
        assert dense_value.grad is not None
        assert custom_query.grad is not None
        assert custom_key.grad is not None
        assert custom_value.grad is not None

        assert allclose(dense_output.data, custom_output.data, atol=1e-5, rtol=1e-5)
        assert allclose(
            dense_query.grad.data, custom_query.grad.data, atol=1e-5, rtol=1e-5
        )
        assert allclose(dense_key.grad.data, custom_key.grad.data, atol=1e-5, rtol=1e-5)
        assert allclose(
            dense_value.grad.data, custom_value.grad.data, atol=1e-5, rtol=1e-5
        )

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_train_mode_dropout_p1_matches_dense_and_returns_zero(self):
        mask = self._causal_mask(batch_size=self.batch_size, seq_len=self.seq_len)
        dense_attention = ScaledDotProductAttention(
            dropout_prob=1.0,
            _implementation="dense",
        )
        custom_attention = ScaledDotProductAttention(
            dropout_prob=1.0,
            _implementation="mlx_custom",
        )
        dense_attention.train()
        custom_attention.train()

        dense = dense_attention(self.query, self.key, self.value, mask=mask)
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            wraps=ScaledDotProductAttentionMLXCustom.apply,
        ) as custom_apply:
            result = custom_attention(self.query, self.key, self.value, mask=mask)

        self.assertEqual(custom_apply.call_count, 1)
        assert allclose(dense.data, result.data, atol=1e-6, rtol=1e-6)
        assert allclose(result.data, xp.zeros_like(result.data), atol=1e-6, rtol=1e-6)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_train_mode_dropout_p1_backward_returns_zero_grads(self):
        mask = self._causal_mask(batch_size=self.batch_size, seq_len=self.seq_len)
        upstream = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            ),
            requires_grad=False,
        )

        dense_query = Tensor(xp.array(self.query.data))
        dense_key = Tensor(xp.array(self.key.data))
        dense_value = Tensor(xp.array(self.value.data))
        custom_query = Tensor(xp.array(self.query.data))
        custom_key = Tensor(xp.array(self.key.data))
        custom_value = Tensor(xp.array(self.value.data))

        dense_attention = ScaledDotProductAttention(
            dropout_prob=1.0,
            _implementation="dense",
        )
        custom_attention = ScaledDotProductAttention(
            dropout_prob=1.0,
            _implementation="mlx_custom",
        )
        dense_attention.train()
        custom_attention.train()

        dense_output = dense_attention(
            dense_query,
            dense_key,
            dense_value,
            mask=mask,
        )
        dense_output.backward(upstream)

        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            wraps=ScaledDotProductAttentionMLXCustom.apply,
        ) as custom_apply:
            custom_output = custom_attention(
                custom_query,
                custom_key,
                custom_value,
                mask=mask,
            )

        self.assertEqual(custom_apply.call_count, 1)
        custom_output.backward(upstream)

        assert dense_query.grad is not None
        assert dense_key.grad is not None
        assert dense_value.grad is not None
        assert custom_query.grad is not None
        assert custom_key.grad is not None
        assert custom_value.grad is not None

        zero_output = xp.zeros_like(custom_output.data)
        zero_query_grad = xp.zeros_like(custom_query.grad.data)
        zero_key_grad = xp.zeros_like(custom_key.grad.data)
        zero_value_grad = xp.zeros_like(custom_value.grad.data)

        assert allclose(dense_output.data, custom_output.data, atol=1e-6, rtol=1e-6)
        assert allclose(custom_output.data, zero_output, atol=1e-6, rtol=1e-6)
        assert allclose(
            dense_query.grad.data, custom_query.grad.data, atol=1e-6, rtol=1e-6
        )
        assert allclose(dense_key.grad.data, custom_key.grad.data, atol=1e-6, rtol=1e-6)
        assert allclose(
            dense_value.grad.data, custom_value.grad.data, atol=1e-6, rtol=1e-6
        )
        assert allclose(custom_query.grad.data, zero_query_grad, atol=1e-6, rtol=1e-6)
        assert allclose(custom_key.grad.data, zero_key_grad, atol=1e-6, rtol=1e-6)
        assert allclose(custom_value.grad.data, zero_value_grad, atol=1e-6, rtol=1e-6)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_backward_matches_dense_across_supported_shapes(self):
        test_shapes = (
            (1, 1, 4, 1),
            (2, 2, 5, 3),
            (2, 3, 4, 5),
        )

        for shape in test_shapes:
            with self.subTest(shape=shape):
                batch_size, _, seq_len, _ = shape
                query_data = xp.random.normal(shape=shape)
                key_data = xp.random.normal(shape=shape)
                value_data = xp.random.normal(shape=shape)
                upstream = Tensor(xp.random.normal(shape=shape), requires_grad=False)
                mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

                dense_grads = self._backward_grads(
                    "dense",
                    query=Tensor(xp.array(query_data)),
                    key=Tensor(xp.array(key_data)),
                    value=Tensor(xp.array(value_data)),
                    mask=mask,
                    upstream=upstream,
                )
                mlx_grads = self._backward_grads(
                    "mlx_custom",
                    query=Tensor(xp.array(query_data)),
                    key=Tensor(xp.array(key_data)),
                    value=Tensor(xp.array(value_data)),
                    mask=mask,
                    upstream=upstream,
                )

                for dense_grad, mlx_grad in zip(dense_grads, mlx_grads):
                    assert allclose(dense_grad, mlx_grad, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_fallback_backward_matches_dense_for_unsupported_masks(self):
        reverse_causal = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                lookback=True,
            ),
            requires_grad=False,
        )
        explicit_additive = Tensor(
            xp.zeros(
                (self.batch_size, 1, self.seq_len, self.seq_len),
                dtype=xp.float32,
            ),
            requires_grad=False,
        )
        explicit_additive.data[:, :, 0, -1] = 1.0
        explicit_additive.data[:, :, 1, 0] = 1.0
        masked_diagonal = Tensor(
            create_causal_mask(
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                mask_diagonal=True,
            ),
            requires_grad=False,
        )
        fully_masked_rows = Tensor(
            xp.zeros(
                (self.batch_size, 1, self.seq_len, self.seq_len),
                dtype=xp.float32,
            ),
            requires_grad=False,
        )
        fully_masked_rows.data[:, :, 0, :] = 1.0

        upstream = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            ),
            requires_grad=False,
        )

        for mask in (
            reverse_causal,
            explicit_additive,
            masked_diagonal,
            fully_masked_rows,
        ):
            with self.subTest(mask_shape=mask.shape, mask=xp.to_numpy(mask.data)):
                dense_grads = self._backward_grads(
                    "dense",
                    query=Tensor(xp.array(self.query.data)),
                    key=Tensor(xp.array(self.key.data)),
                    value=Tensor(xp.array(self.value.data)),
                    mask=mask,
                    upstream=upstream,
                )

                with patch(
                    "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
                    side_effect=AssertionError("mlx custom path should not be used"),
                ):
                    fallback_grads = self._backward_grads(
                        "mlx_custom",
                        query=Tensor(xp.array(self.query.data)),
                        key=Tensor(xp.array(self.key.data)),
                        value=Tensor(xp.array(self.value.data)),
                        mask=mask,
                        upstream=upstream,
                    )

                for dense_grad, fallback_grad in zip(dense_grads, fallback_grads):
                    assert allclose(dense_grad, fallback_grad, atol=1e-5, rtol=1e-5)

    def test_mlx_custom_fallback_backward_matches_dense_for_explicit_bool_mask(self):
        bool_mask = xp.ones(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            dtype=xp.bool_,
        )
        bool_mask[:, :, 0, -1] = False
        bool_mask[:, :, -1, 0] = False
        upstream = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len,
                    self.head_dim,
                )
            ),
            requires_grad=False,
        )

        dense_grads = self._backward_grads(
            "dense",
            query=Tensor(xp.array(self.query.data)),
            key=Tensor(xp.array(self.key.data)),
            value=Tensor(xp.array(self.value.data)),
            mask=bool_mask,
            upstream=upstream,
        )

        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("mlx custom path should not be used"),
        ):
            query = Tensor(xp.array(self.query.data))
            key = Tensor(xp.array(self.key.data))
            value = Tensor(xp.array(self.value.data))
            output = attention(query, key, value, mask=bool_mask)
            output.backward(upstream)

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        fallback_grads = (query.grad.data, key.grad.data, value.grad.data)

        for dense_grad, fallback_grad in zip(dense_grads, fallback_grads):
            assert allclose(dense_grad, fallback_grad, atol=1e-5, rtol=1e-5)

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_backward_matches_finite_difference_on_tiny_case(self):
        shape = (1, 1, 3, 2)
        batch_size, _, seq_len, _ = shape
        epsilon = 1e-3
        mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

        query_data = xp.random.normal(shape=shape)
        key_data = xp.random.normal(shape=shape)
        value_data = xp.random.normal(shape=shape)
        upstream = Tensor(xp.random.normal(shape=shape), requires_grad=False)

        grad_query, grad_key, grad_value = self._backward_grads(
            "mlx_custom",
            query=Tensor(xp.array(query_data)),
            key=Tensor(xp.array(key_data)),
            value=Tensor(xp.array(value_data)),
            mask=mask,
            upstream=upstream,
        )

        for name, tensor_data, grad_data, index in (
            ("query", query_data, grad_query, (0, 0, 1, 0)),
            ("key", key_data, grad_key, (0, 0, 1, 1)),
            ("value", value_data, grad_value, (0, 0, 2, 0)),
        ):
            with self.subTest(input=name):
                positive = xp.array(tensor_data)
                negative = xp.array(tensor_data)
                positive[index] += epsilon
                negative[index] -= epsilon

                if name == "query":
                    positive_loss = self._attention_objective(
                        "mlx_custom",
                        query=positive,
                        key=key_data,
                        value=value_data,
                        mask=mask,
                        upstream=upstream,
                    )
                    negative_loss = self._attention_objective(
                        "mlx_custom",
                        query=negative,
                        key=key_data,
                        value=value_data,
                        mask=mask,
                        upstream=upstream,
                    )
                elif name == "key":
                    positive_loss = self._attention_objective(
                        "mlx_custom",
                        query=query_data,
                        key=positive,
                        value=value_data,
                        mask=mask,
                        upstream=upstream,
                    )
                    negative_loss = self._attention_objective(
                        "mlx_custom",
                        query=query_data,
                        key=negative,
                        value=value_data,
                        mask=mask,
                        upstream=upstream,
                    )
                else:
                    positive_loss = self._attention_objective(
                        "mlx_custom",
                        query=query_data,
                        key=key_data,
                        value=positive,
                        mask=mask,
                        upstream=upstream,
                    )
                    negative_loss = self._attention_objective(
                        "mlx_custom",
                        query=query_data,
                        key=key_data,
                        value=negative,
                        mask=mask,
                        upstream=upstream,
                    )

                numerical_grad = (positive_loss - negative_loss) / (2 * epsilon)
                assert allclose(
                    grad_data[index],
                    numerical_grad,
                    atol=5e-3,
                    rtol=5e-3,
                )

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_mlx_custom_backward_keeps_causal_gradients_local(self):
        shape = (2, 2, 5, 3)
        batch_size, _, seq_len, _ = shape
        cutoff = 2
        mask = self._causal_mask(batch_size=batch_size, seq_len=seq_len)

        query_data = xp.random.normal(shape=shape)
        key_data = xp.random.normal(shape=shape)
        value_data = xp.random.normal(shape=shape)
        upstream_data = xp.zeros(shape, dtype=xp.float32)
        upstream_data[:, :, : cutoff + 1, :] = xp.random.normal(
            shape=upstream_data[:, :, : cutoff + 1, :].shape
        )
        upstream = Tensor(upstream_data, requires_grad=False)

        baseline_grads = self._backward_grads(
            "mlx_custom",
            query=Tensor(xp.array(query_data)),
            key=Tensor(xp.array(key_data)),
            value=Tensor(xp.array(value_data)),
            mask=mask,
            upstream=upstream,
        )

        modified_query = xp.array(query_data)
        modified_key = xp.array(key_data)
        modified_value = xp.array(value_data)
        modified_query[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_query[:, :, cutoff + 1 :, :].shape
        )
        modified_key[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_key[:, :, cutoff + 1 :, :].shape
        )
        modified_value[:, :, cutoff + 1 :, :] = xp.random.normal(
            shape=modified_value[:, :, cutoff + 1 :, :].shape
        )

        updated_grads = self._backward_grads(
            "mlx_custom",
            query=Tensor(modified_query),
            key=Tensor(modified_key),
            value=Tensor(modified_value),
            mask=mask,
            upstream=upstream,
        )

        for baseline_grad, updated_grad in zip(baseline_grads, updated_grads):
            assert allclose(
                baseline_grad[:, :, : cutoff + 1, :],
                updated_grad[:, :, : cutoff + 1, :],
                atol=1e-5,
                rtol=1e-5,
            )

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
