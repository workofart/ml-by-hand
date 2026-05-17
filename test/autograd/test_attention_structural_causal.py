from unittest import TestCase, skipUnless
from unittest.mock import patch

from autograd.backend import IS_CUPY, IS_MLX, xp
from autograd.functional import scaled_dot_product_attention_cudnn
from autograd.nn import ScaledDotProductAttention
from autograd.tensor import Tensor
from autograd.text.utils import create_causal_mask
from test.helpers import allclose


class TestStructuralCausalAttention(TestCase):
    def setUp(self) -> None:
        xp.random.seed(11)
        self.batch_size = 2
        self.num_heads = 3
        self.seq_len = 4
        self.head_dim = 5
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

    def test_dense_structural_causal_matches_explicit_causal_mask(self):
        attention = ScaledDotProductAttention(dropout_prob=0.0)
        explicit_mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )

        with patch(
            "autograd.nn.scaled_dot_product_attention_mlx_custom",
            side_effect=RuntimeError("force dense fallback"),
        ):
            structural = attention(
                self.query,
                self.key,
                self.value,
                mask=None,
                is_causal=True,
            )
            explicit = attention(
                self.query,
                self.key,
                self.value,
                mask=explicit_mask,
                is_causal=False,
            )

        assert allclose(structural.data, explicit.data, atol=1e-5, rtol=1e-5)

    def test_dense_structural_causal_backward_matches_explicit_causal_mask(self):
        attention = ScaledDotProductAttention(dropout_prob=0.0)
        explicit_mask = Tensor(
            create_causal_mask(seq_len=self.seq_len, batch_size=self.batch_size),
            requires_grad=False,
        )
        upstream = Tensor(xp.random.normal(shape=self.query.shape))

        structural_query = Tensor(xp.array(self.query.data))
        structural_key = Tensor(xp.array(self.key.data))
        structural_value = Tensor(xp.array(self.value.data))
        structural = attention(
            structural_query,
            structural_key,
            structural_value,
            mask=None,
            is_causal=True,
        )
        (structural * upstream).sum().backward()

        explicit_query = Tensor(xp.array(self.query.data))
        explicit_key = Tensor(xp.array(self.key.data))
        explicit_value = Tensor(xp.array(self.value.data))
        explicit = attention(
            explicit_query,
            explicit_key,
            explicit_value,
            mask=explicit_mask,
            is_causal=False,
        )
        (explicit * upstream).sum().backward()

        assert allclose(structural.data, explicit.data, atol=1e-5, rtol=1e-5)
        assert allclose(
            structural_query.grad.data, explicit_query.grad.data, atol=1e-5, rtol=1e-5
        )
        assert allclose(
            structural_key.grad.data, explicit_key.grad.data, atol=1e-5, rtol=1e-5
        )
        assert allclose(
            structural_value.grad.data,
            explicit_value.grad.data,
            atol=1e-5,
            rtol=1e-5,
        )

    @skipUnless(IS_MLX, "mlx_custom requires the MLX backend")
    def test_structural_causal_attention_stays_on_dense_path(self):
        attention = ScaledDotProductAttention(dropout_prob=0.0)

        with patch(
            "autograd.functional.ScaledDotProductAttentionMLXCustom.apply",
            side_effect=AssertionError("structural causal should not use mlx_custom"),
        ) as apply_mock:
            attention(
                self.query,
                self.key,
                self.value,
                mask=None,
                is_causal=True,
            )

        self.assertEqual(apply_mock.call_count, 0)

    @skipUnless(IS_CUPY, "cuDNN SDPA requires the CuPy backend")
    def test_cudnn_structural_causal_attention_matches_dense_bf16(self):
        try:
            import cudnn  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("nvidia-cudnn-frontend is not installed")

        shape = (2, 3, 4, 8)
        dtype = xp.dtype("bfloat16")
        query_data = xp.array(xp.random.normal(shape=shape), dtype=dtype)
        key_data = xp.array(xp.random.normal(shape=shape), dtype=dtype)
        value_data = xp.array(xp.random.normal(shape=shape), dtype=dtype)
        upstream_data = xp.array(xp.random.normal(shape=shape), dtype=dtype)

        cudnn_query = Tensor(query_data)
        cudnn_key = Tensor(key_data)
        cudnn_value = Tensor(value_data)
        cudnn_out = scaled_dot_product_attention_cudnn(
            cudnn_query,
            cudnn_key,
            cudnn_value,
        )
        (cudnn_out * Tensor(upstream_data)).sum().backward()

        attention = ScaledDotProductAttention(dropout_prob=0.0)
        with patch(
            "autograd.nn.scaled_dot_product_attention_cudnn",
            side_effect=RuntimeError("force dense fallback"),
        ):
            dense_query = Tensor(query_data)
            dense_key = Tensor(key_data)
            dense_value = Tensor(value_data)
            dense_out = attention(
                dense_query,
                dense_key,
                dense_value,
                mask=None,
                is_causal=True,
            )
            (dense_out * Tensor(upstream_data)).sum().backward()

        assert allclose(
            cudnn_out.data.astype(xp.float32),
            dense_out.data.astype(xp.float32),
            atol=1e-2,
            rtol=1e-2,
        )
        assert allclose(
            cudnn_query.grad.data.astype(xp.float32),
            dense_query.grad.data.astype(xp.float32),
            atol=1e-2,
            rtol=1e-2,
        )
        assert allclose(
            cudnn_key.grad.data.astype(xp.float32),
            dense_key.grad.data.astype(xp.float32),
            atol=1e-2,
            rtol=1e-2,
        )
        assert allclose(
            cudnn_value.grad.data.astype(xp.float32),
            dense_value.grad.data.astype(xp.float32),
            atol=1e-2,
            rtol=1e-2,
        )

    @skipUnless(IS_CUPY, "cuDNN SDPA requires the CuPy backend")
    def test_cudnn_output_stride_makes_head_combine_a_view(self):
        try:
            import cudnn  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("nvidia-cudnn-frontend is not installed")

        shape = (2, 3, 4, 8)
        dtype = xp.dtype("bfloat16")
        query = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))
        key = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))
        value = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))

        out = scaled_dot_product_attention_cudnn(query, key, value)
        combined = out.permute(0, 2, 1, 3).view(2, 4, 24)

        self.assertTrue(combined.data.flags.c_contiguous)
        self.assertEqual(combined.data.data.ptr, out.data.data.ptr)

        upstream = Tensor(xp.array(xp.random.normal(shape=combined.shape), dtype=dtype))
        (combined * upstream).sum().backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

    @skipUnless(IS_CUPY, "cuDNN SDPA requires the CuPy backend")
    def test_cudnn_backward_releases_saved_output_and_stats(self):
        try:
            import cudnn  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("nvidia-cudnn-frontend is not installed")

        shape = (2, 3, 4, 8)
        dtype = xp.dtype("bfloat16")
        query = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))
        key = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))
        value = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))
        upstream = Tensor(xp.array(xp.random.normal(shape=shape), dtype=dtype))

        out = scaled_dot_product_attention_cudnn(query, key, value)
        creator = out.creator
        self.assertIsNotNone(getattr(creator, "output", None))
        self.assertIsNotNone(getattr(creator, "stats", None))

        grad_query, grad_key, grad_value = creator.backward(upstream)

        self.assertEqual(grad_query.shape, query.shape)
        self.assertEqual(grad_key.shape, key.shape)
        self.assertEqual(grad_value.shape, value.shape)
        self.assertIsNone(getattr(creator, "output", None))
        self.assertIsNone(getattr(creator, "stats", None))
