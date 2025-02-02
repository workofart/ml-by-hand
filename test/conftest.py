import sys

import numpy as real_numpy
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def patch_torch_numpy():
    """
    Monkey-patch torch.Tensor.numpy() so it always
    returns a CPU NumPy array (avoiding errors if
    the tensor is on CUDA).
    """
    old_numpy = torch.Tensor.numpy  # Keep a reference to the original

    def new_numpy(t):
        # Just in case you rely on .detach():
        t = t.detach()
        if t.is_cuda:
            t = t.cpu()
        return old_numpy(t)

    torch.Tensor.numpy = new_numpy

    yield  # Run the tests with this patch in place

    # revert back after tests
    torch.Tensor.numpy = old_numpy


# Replace 'cupy' in sys.modules with numpy
sys.modules["cupy"] = real_numpy
