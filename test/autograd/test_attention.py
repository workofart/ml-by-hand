from unittest import TestCase, skipUnless
from unittest.mock import patch

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

    def test_mlx_custom_requires_dropout_prob_zero(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.1,
            _implementation="mlx_custom",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "mlx_custom requires dropout_prob == 0",
        ):
            attention(self.query, self.key, self.value)

    def test_mlx_custom_rejects_mask_in_milestone_1(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "mlx_custom does not support mask in milestone 1",
        ):
            attention(self.query, self.key, self.value, mask=mask)

    @skipUnless(IS_MLX, "mlx_custom prototype requires the MLX backend")
    def test_mlx_custom_matches_toy_oracle(self):
        test_shapes = (
            (1, 1, 1, 1),
            (1, 2, 3, 4),
            (2, 3, 4, 5),
        )
        for shape in test_shapes:
            with self.subTest(shape=shape):
                query = Tensor(xp.random.normal(shape=shape))
                key = Tensor(xp.random.normal(shape=shape))
                value = Tensor(xp.random.normal(shape=shape))
                attention = ScaledDotProductAttention(
                    dropout_prob=0.0,
                    _implementation="mlx_custom",
                )

                output = attention(query, key, value)
                expected = (
                    query.data
                    + value.data
                    + xp.mean(
                        key.data,
                        axis=2,
                        keepdims=True,
                    )
                )

                assert output.shape == shape
                assert allclose(output.data, expected, atol=1e-6, rtol=1e-6)

    @skipUnless(IS_MLX, "mlx_custom prototype requires the MLX backend")
    def test_mlx_custom_matches_hand_computed_toy_example(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        query = Tensor(xp.array([[[[1.0], [2.0], [3.0]]]], dtype=xp.float32))
        key = Tensor(xp.array([[[[4.0], [7.0], [10.0]]]], dtype=xp.float32))
        value = Tensor(xp.array([[[[0.5], [1.5], [2.5]]]], dtype=xp.float32))

        output = attention(query, key, value)
        expected = xp.array([[[[8.5], [10.5], [12.5]]]], dtype=xp.float32)

        assert allclose(output.data, expected, atol=1e-6, rtol=1e-6)

    @skipUnless(IS_MLX, "mlx_custom prototype requires the MLX backend")
    def test_mlx_custom_backward_raises_exact_error(self):
        output = self._run_attention("mlx_custom")

        with self.assertRaisesRegex(
            NotImplementedError,
            "mlx_custom is forward-only in milestone 1",
        ):
            output.sum().backward()

    @skipUnless(IS_MLX, "mlx_custom prototype requires the MLX backend")
    def test_mlx_custom_requires_rank_4_inputs(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        query = Tensor(
            xp.random.normal(
                shape=(self.batch_size, self.seq_len, self.num_heads * self.head_dim)
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "mlx_custom expects query, key, and value to be 4D tensors",
        ):
            attention(query, query, query)

    @skipUnless(IS_MLX, "mlx_custom prototype requires the MLX backend")
    def test_mlx_custom_requires_matching_shapes(self):
        attention = ScaledDotProductAttention(
            dropout_prob=0.0,
            _implementation="mlx_custom",
        )
        key = Tensor(
            xp.random.normal(
                shape=(
                    self.batch_size,
                    self.num_heads,
                    self.seq_len + 1,
                    self.head_dim,
                )
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "mlx_custom requires query, key, and value to share the same shape",
        ):
            attention(self.query, key, self.value)
