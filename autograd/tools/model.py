import json
import os
from typing import Any, Dict

import numpy  # need this for loading checkpoint

try:
    # drop-in replacement for numpy for GPU acceleration
    import cupy as np  # type: ignore

    _ = np.cuda.runtime.getDeviceCount()  # Check if a CUDA device is available
except Exception:
    import numpy as np

from autograd.tensor import Tensor

# Define a type for the serialized metadata structure
SerializedMeta = Dict[str, Any]


def save_checkpoint(
    obj: Any, json_path: str = "checkpoint.json", npz_path: str = "checkpoint.npz"
) -> None:
    """
    Serialize any Python object (model checkpoint, etc.) to:
      1. JSON file with nested structure (metadata).
      2. NPZ file with numeric array data.

    Args:
       obj: The object to serialize (usually the state_dict of a model or optimizer).
            You can also add custom metadata to this object.
       json_path: Path to save the JSON metadata file.
       npz_path: Path to save the NPZ data file.
    """

    def _serialize(
        obj: Any, arrays: Dict[str, np.ndarray], prefix: str = ""
    ) -> SerializedMeta:
        """
        Recursively serialize a Python object into a structure
        that can be stored as JSON + NPZ.
        """
        # Dictionary
        if isinstance(obj, dict):
            return {
                "_type": "dict",
                "items": {
                    k: _serialize(v, arrays, f"{prefix}.{k}" if prefix else k)
                    for k, v in obj.items()
                },
            }

        # List or Tuple
        if isinstance(obj, (list, tuple)):
            items = [
                _serialize(v, arrays, f"{prefix}.{i}" if prefix else str(i))
                for i, v in enumerate(obj)
            ]
            return {
                "_type": "list" if isinstance(obj, list) else "tuple",
                "items": items,
            }

        # Tensor or NumPy array
        if isinstance(obj, (Tensor, np.ndarray)):
            key = prefix if prefix else "root_array"
            arr = obj.data if isinstance(obj, Tensor) else obj
            arrays[key] = arr
            return {
                "_type": "tensor" if isinstance(obj, Tensor) else "np.ndarray",
                "key": key,
            }

        # Primitive scalar types
        if isinstance(obj, (int, float, str, bool, type(None))):
            return {"_type": "scalar", "value": obj}

        # Fallback for other types
        return {"_type": "raw", "value": str(obj)}

    arrays_dict: Dict[str, np.ndarray] = {}
    meta: SerializedMeta = _serialize(obj, arrays_dict)

    # Save metadata to JSON
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Save arrays to NPZ
    np.savez_compressed(npz_path, **arrays_dict)


def load_checkpoint(
    json_path: str = "checkpoint.json",
    npz_path: str = "checkpoint.npz",
    weights_only: bool = False,
) -> Any:
    """
    Load an object saved by `save_model`, reconstructing Tensors, arrays, etc.

    Args:
       json_path: Path to the JSON file containing the serialized object.
       npz_path: Path to the NPZ file containing the serialized arrays.
       weights_only: If True, only load the weights (e.g., "parameters" key), ignoring other states.
    """

    def _deserialize(meta: SerializedMeta, data: Dict[str, np.ndarray]) -> Any:
        """
        Recursively reconstruct an object from its serialized form.
        """
        t = meta["_type"]

        if t == "dict":
            return {k: _deserialize(v, data) for k, v in meta["items"].items()}

        if t in ("list", "tuple"):
            items = [_deserialize(x, data) for x in meta["items"]]
            return items if t == "list" else tuple(items)

        if t in ("tensor", "np.ndarray"):
            arr = data[meta["key"]]
            return Tensor(arr) if t == "tensor" else arr

        if t in ("scalar", "raw"):
            return meta["value"]

        raise ValueError(f"Unknown meta type: {t}")

    if not os.path.exists(json_path):
        raise ValueError(f"Checkpoint file not found: {json_path}")
    if not os.path.exists(npz_path):
        raise ValueError(f"Checkpoint file not found: {npz_path}")

    # Load metadata
    with open(json_path, "r") as f:
        meta: SerializedMeta = json.load(f)

    # Load NPZ data
    # Need to use numpy's load API
    with numpy.load(npz_path, allow_pickle=True) as npz_data:
        data_dict = {key: np.array(npz_data[key]) for key in npz_data.files}

    deserialized_data = _deserialize(meta, data_dict)

    # If requested, return only the "parameters" sub-dictionary
    if weights_only and isinstance(deserialized_data, dict):
        return deserialized_data.get("parameters", deserialized_data)

    return deserialized_data
