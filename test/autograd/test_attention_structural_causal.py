from unittest import TestCase, skipUnless
from unittest.mock import patch

from autograd.backend import IS_MLX, xp
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
