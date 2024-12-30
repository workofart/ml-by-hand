import json
import os
import logging
import numpy as np
from autograd.tensor import Tensor
from typing import Dict, Any, List


logger = logging.getLogger(__name__)
# Define a type for the serialized metadata structure
SerializedMeta = Dict[str, Any]


def save_model(
    obj: Any, json_path: str = "checkpoint.json", npz_path: str = "checkpoint.npz"
) -> None:
    """
    Serialize any Python object (model checkpoint, etc.) to:
      1. JSON file with nested structure (metadata).
      2. NPZ file with numeric array data.
    """

    def _serialize_object(
        obj: Any, arrays_dict: Dict[str, np.ndarray], prefix: str = ""
    ) -> SerializedMeta:
        """
        Serialize a single object (recursively).
        """
        if isinstance(obj, (dict, list, tuple)):
            if isinstance(obj, dict):
                meta: SerializedMeta = {"_type": "dict", "items": {}}
                for k, v in obj.items():
                    sub_prefix = f"{prefix}.{k}" if prefix else k
                    meta["items"][k] = _serialize_object(v, arrays_dict, sub_prefix)
            else:
                meta = {
                    "_type": "list" if isinstance(obj, list) else "tuple",
                    "items": [],
                }
                for i, v in enumerate(obj):
                    sub_prefix = f"{prefix}.{str(i)}" if prefix else str(i)
                    meta["items"].append(_serialize_object(v, arrays_dict, sub_prefix))
            return meta
        elif isinstance(obj, (Tensor, np.ndarray)):
            key = prefix if prefix else "root_array"
            # Extract raw numpy data:
            arr = obj.data if isinstance(obj, Tensor) else obj
            arrays_dict[key] = arr

            return {
                "_type": "tensor" if isinstance(obj, Tensor) else "np.ndarray",
                "key": key,
            }
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return {"_type": "scalar", "value": obj}
        return {"_type": "raw", "value": str(obj)}

    arrays_dict: Dict[str, np.ndarray] = {}
    meta: SerializedMeta = _serialize_object(obj, arrays_dict)

    # Save metadata to JSON
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Save arrays to NPZ
    np.savez_compressed(npz_path, **arrays_dict)


def load_model(
    json_path: str = "checkpoint.json",
    npz_path: str = "checkpoint.npz",
    weights_only=False,
) -> Any:
    """
    Load an object saved by `save_model`, reconstructing Tensors, arrays, etc.

    Args:
       json_path (str): Path to the JSON file containing the serialized object.
       npz_path (str): Path to the NPZ file containing the serialized arrays.
       weights_only (bool): If True, only load the weights and not the states.
    """

    def _deserialize_object(
        meta: SerializedMeta, data_dict: Dict[str, np.ndarray]
    ) -> Any:
        """
        Recursively reconstruct an object from its serialized form.
        """
        t: str = meta["_type"]
        if t in ("dict", "list", "tuple"):
            if t == "dict":
                result: Dict[str, Any] = {}
                for k, v in meta["items"].items():
                    result[k] = _deserialize_object(v, data_dict)
                return result
            else:
                items: List[Any] = [
                    _deserialize_object(x, data_dict) for x in meta["items"]
                ]
                return items if t == "list" else tuple(items)
        elif t in ("tensor", "np.ndarray"):
            arr = data_dict[meta["key"]]
            return Tensor(arr) if meta["_type"] == "tensor" else arr
        elif t in ("scalar", "raw"):
            if t == "scalar":
                return meta["value"]
            elif t == "raw":
                return meta["value"]
            else:
                raise ValueError(f"Unknown primitive type: {t}")
        else:
            raise ValueError(f"Unknown meta type: {t}")

    if not os.path.exists(json_path):
        raise ValueError(f"Checkpoint file not found: {json_path}")
    if not os.path.exists(npz_path):
        raise ValueError(f"Checkpoint file not found: {npz_path}")

    # Load metadata
    with open(json_path, "r") as f:
        meta: SerializedMeta = json.load(f)

    # Load arrays
    with np.load(npz_path, allow_pickle=True) as data:
        data_dict: Dict[str, np.ndarray] = dict(data.items())

    deserialized_data = _deserialize_object(meta, data_dict)

    if (
        weights_only
        and isinstance(deserialized_data, dict)
        and "parameters" in deserialized_data
    ):
        return deserialized_data["parameters"]

    return deserialized_data
