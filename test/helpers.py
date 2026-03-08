import numbers

import numpy as np
import torch

from autograd.backend import xp


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (numbers.Number, bool)):
        return np.asarray(value)
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except (TypeError, ValueError):
            return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except (TypeError, ValueError):
            return value
    if hasattr(value, "shape") or hasattr(value, "tolist"):
        try:
            return xp.to_numpy(value)
        except (TypeError, ValueError):
            return value
    return value


def allclose(a, b, *args, **kwargs):
    return np.allclose(_to_numpy(a), _to_numpy(b), *args, **kwargs)


def array_equal(a, b, *args, **kwargs):
    try:
        return np.array_equal(_to_numpy(a), _to_numpy(b), *args, **kwargs)
    except (TypeError, ValueError):
        return a == b


def isclose(a, b, *args, **kwargs):
    return np.isclose(_to_numpy(a), _to_numpy(b), *args, **kwargs)
