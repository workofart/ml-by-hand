from unittest import TestCase
from unittest.mock import patch

from autograd.backend import xp
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

    def test_mlx_reference_matches_dense_without_mask(self):
        dense = self._run_attention("dense")
        mlx_reference = self._run_attention("mlx_fast_reference")

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

    def test_mlx_reference_matches_dense_for_standard_causal_mask(self):
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        dense = self._run_attention("dense", mask=mask)
        mlx_reference = self._run_attention("mlx_fast_reference", mask=mask)

        assert allclose(dense.data, mlx_reference.data, atol=1e-5, rtol=1e-5)

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

    def test_mlx_reference_backward_raises_exact_error(self):
        output = self._run_attention("mlx_fast_reference")

        with self.assertRaisesRegex(
            NotImplementedError,
            "mlx_fast_reference is forward-only in milestone 0",
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
