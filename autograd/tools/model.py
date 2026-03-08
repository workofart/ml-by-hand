"""Utilities for saving and loading model checkpoints."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from autograd.backend import ARRAY_TYPE, xp
from autograd.tensor import Tensor

# Define a type for the serialized metadata structure
SerializedMeta = Dict[str, Any]


def save_checkpoint(
    obj: Any, json_path: str = "checkpoint.json", npz_path: str = "checkpoint.npz"
) -> None:
    """Save a Python object (model state, etc.) into JSON and NPZ files.

    This function splits the saved content into:
    - A JSON file for the metadata or structure.
    - A NPZ file for numeric array data (e.g., model weights).

    Args:
        obj: The Python object to serialize, typically a model's state_dict or
            other checkpoint data structure.
        json_path (str): The file path to save the JSON metadata. Defaults to
            'checkpoint.json'.
        npz_path (str): The file path to save the NPZ data. Defaults to
            'checkpoint.npz'.

    Raises:
        OSError: If there is an error writing to the specified files.

    Example:
        >>> from autograd.tensor import Tensor
        >>> obj_to_save = {
        ...     "parameters": {
        ...         "weight": Tensor([1.0, 2.0, 3.0]),
        ...         "bias": Tensor([0.5])
        ...     },
        ...     "epoch": 5
        ... }
        >>> save_checkpoint(obj_to_save, "my_model.json", "my_model.npz")
    """

    def _serialize(
        obj: Any, arrays: Dict[str, Any], prefix: str = ""
    ) -> SerializedMeta:
        """Recursively serialize a Python object into a structure
        that can be stored as JSON + NPZ.

        Args:
            obj: The object (dictionary, list, tuple, Tensor, etc.) to serialize.
            arrays (Dict[str, Any]): A dictionary to store array data keyed
                by string identifiers.
            prefix (str): A prefix for naming arrays in `arrays`.

        Returns:
            SerializedMeta: A dictionary representing the serialized metadata.
        """
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

        # Tensor or backend array
        if isinstance(obj, Tensor):
            key = prefix if prefix else "root_array"
            arrays[key] = xp.array(obj.data)
            return {
                "_type": "tensor",
                "key": key,
            }

        if isinstance(obj, ARRAY_TYPE) or hasattr(obj, "__array__"):
            key = prefix if prefix else "root_array"
            arrays[key] = xp.array(obj)
            return {
                "_type": "array",
                "key": key,
            }

        # Primitive scalar types
        if isinstance(obj, (int, float, str, bool, type(None))):
            return {"_type": "scalar", "value": obj}

        # Fallback for other types
        return {"_type": "raw", "value": str(obj)}

    arrays_dict: Dict[str, Any] = {}
    meta: SerializedMeta = _serialize(obj, arrays_dict)

    # Save metadata to JSON
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Save arrays to NPZ
    xp.savez_compressed(npz_path, **arrays_dict)


def load_checkpoint(
    json_path: str = "checkpoint.json",
    npz_path: str = "checkpoint.npz",
    weights_only: bool = False,
) -> Any:
    """Load an object from saved checkpoint files.

    This function reconstructs the Python object that was previously serialized
    by `save_checkpoint`. It reads:
    - A JSON file for the metadata or structure.
    - A NPZ file for numeric array data.

    Args:
        json_path (str): The file path of the JSON metadata. Defaults to
            'checkpoint.json'.
        npz_path (str): The file path of the NPZ data. Defaults to
            'checkpoint.npz'.
        weights_only (bool): If True, only loads the "parameters" sub-dictionary
            (commonly used for model weights). Defaults to False.

    Returns:
        Any: The reconstructed Python object, which may be a nested dictionary,
        list, Tensor, etc.

    Raises:
        ValueError: If either JSON or NPZ checkpoint file does not exist.
        ValueError: If the metadata contains unknown types that cannot be deserialized.

    Example:
        >>> checkpoint_data = load_checkpoint("my_model.json", "my_model.npz")
        >>> # If only model parameters are needed:
        >>> model_weights = load_checkpoint("my_model.json", "my_model.npz", weights_only=True)
    """

    def _deserialize(meta: SerializedMeta, data: Dict[str, Any]) -> Any:
        """Recursively reconstruct an object from its serialized form.

        Args:
            meta (SerializedMeta): The serialized metadata.
            data (Dict[str, Any]): A dictionary of array data loaded from NPZ.

        Returns:
            Any: The reconstructed Python object (dict, list, Tensor, etc.).

        Raises:
            ValueError: If the metadata has an unknown '_type'.
        """
        t = meta["_type"]

        if t == "dict":
            return {k: _deserialize(v, data) for k, v in meta["items"].items()}

        if t in ("list", "tuple"):
            items = [_deserialize(x, data) for x in meta["items"]]
            return items if t == "list" else tuple(items)

        if t == "tensor":
            return Tensor(data[meta["key"]])

        if t == "array":
            return data[meta["key"]]

        if "key" in meta and "items" not in meta and "value" not in meta:
            return data[meta["key"]]

        if t in ("scalar", "raw"):
            return meta["value"]

        raise ValueError(f"Unknown meta type: {t}")

    if not os.path.exists(json_path):
        raise ValueError(
            f"Checkpoint file not found: {json_path}, double-check your 'resume_epoch' field in the config. Set to None if you don't want to load from checkpoint."
        )
    if not os.path.exists(npz_path):
        raise ValueError(
            f"Checkpoint file not found: {npz_path}, double-check your 'resume_epoch' field in the config. Set to None if you don't want to load from checkpoint."
        )

    # Load metadata
    with open(json_path, "r") as f:
        meta: SerializedMeta = json.load(f)

    # Load NPZ data
    data_dict: Any = xp.load(npz_path)

    deserialized_data = _deserialize(meta, data_dict)

    # If requested, return only the "parameters" sub-dictionary
    if weights_only and isinstance(deserialized_data, dict):
        return deserialized_data.get("parameters", deserialized_data)

    return deserialized_data
