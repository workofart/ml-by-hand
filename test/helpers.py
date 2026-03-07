import numbers

import mlx.core as mx
import torch

MX_ARRAY_TYPE = type(mx.array(0))


def _to_mx_array(value):
    if isinstance(value, MX_ARRAY_TYPE):
        return value
    if isinstance(value, torch.Tensor):
        return mx.array(value.detach().cpu().tolist())
    if isinstance(value, (numbers.Number, bool)):
        return mx.array(value)
    if isinstance(value, (list, tuple)):
        try:
            return mx.array(value)
        except (TypeError, ValueError):
            return value
    if hasattr(value, "tolist"):
        try:
            return mx.array(value.tolist())
        except (TypeError, ValueError):
            return value
    return value


def allclose(a, b, *args, **kwargs):
    return mx.allclose(_to_mx_array(a), _to_mx_array(b), *args, **kwargs)


def array_equal(a, b, *args, **kwargs):
    try:
        return mx.array_equal(_to_mx_array(a), _to_mx_array(b), *args, **kwargs)
    except (TypeError, ValueError):
        return a == b


def isclose(a, b, *args, **kwargs):
    return mx.isclose(_to_mx_array(a), _to_mx_array(b), *args, **kwargs)
