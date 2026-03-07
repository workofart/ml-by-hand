from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import mlx.core as mx

ArrayLike = Union[mx.array, bool, int, float, Sequence[Any]]
