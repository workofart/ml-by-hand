import math
from unittest import TestCase

from autograd.backend import xp
from autograd.init import compute_in_out_tensor_count, xavier_uniform
from autograd.tensor import Tensor


class TestComputeInOutTensorCount(TestCase):
    def test_linear_weight_fans(self):
        # Linear weights use the (input_size, output_size) convention
        t = Tensor(xp.zeros((5, 10)))
        assert compute_in_out_tensor_count(t) == (5, 10)

    def test_conv_kernel_fans(self):
        # Conv2d weights use the (out_channels, in_channels, kH, kW) convention,
        # so fan_in = in_channels * receptive field, fan_out = out_channels * receptive field
        t = Tensor(xp.zeros((16, 8, 3, 3)))
        assert compute_in_out_tensor_count(t) == (8 * 9, 16 * 9)

    def test_xavier_uniform_bound_uses_conv_fans(self):
        xp.random.seed(42)
        t = xavier_uniform(Tensor(xp.zeros((16, 8, 3, 3))))
        limit = math.sqrt(6.0 / (8 * 9 + 16 * 9))
        assert float(t.data.max()) <= limit
        assert float(t.data.min()) >= -limit
