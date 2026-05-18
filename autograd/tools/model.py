"""Utilities for saving and loading model checkpoints."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, Iterable

from autograd.backend import ARRAY_TYPE, resolve_dtype, xp
from autograd.tensor import Tensor

# Define a type for the serialized metadata structure
SerializedMeta = Dict[str, Any]
_ARRAY_META_TYPES = {"array", "tensor", "np.ndarray"}


def _check_checkpoint_paths(json_path: str, npz_path: str | None = None) -> None:
    if not os.path.exists(json_path):
        raise ValueError(
            f"Checkpoint file not found: {json_path}, double-check your 'resume_epoch' field in the config. Set to None if you don't want to load from checkpoint."
        )
    if npz_path is not None and not os.path.exists(npz_path):
        raise ValueError(
            f"Checkpoint file not found: {npz_path}, double-check your 'resume_epoch' field in the config. Set to None if you don't want to load from checkpoint."
        )


def _read_checkpoint_meta(json_path: str) -> SerializedMeta:
    with open(json_path, "r") as f:
        return json.load(f)


def _dtype_name(arr: Any) -> str:
    """Return a backend-neutral short dtype name (e.g. "bfloat16").

    Needed because numpy/CuPy report dtypes as "bfloat16" but MLX reports
    them as "mlx.core.bfloat16" — keying off the short tail makes the JSON
    sidecar portable across backends.
    """
    return str(arr.dtype).rsplit(".", 1)[-1]


def _restore_array(meta: SerializedMeta, data: Dict[str, Any]) -> Any:
    # numpy's .npy format can't natively store extension dtypes like
    # ml_dtypes.bfloat16 — those round-trip as opaque void bytes (e.g.
    # |V2). The sidecar carries each array's original dtype so we can
    # view-reinterpret bytes back to it instead of guessing.
    arr = data[meta["key"]]
    target = resolve_dtype(meta["dtype"])
    if arr.dtype != target:
        arr = arr.view(target)
    return arr


def _deserialize(meta: SerializedMeta, data: Dict[str, Any]) -> Any:
    """Recursively reconstruct an object from its serialized form."""
    t = meta["_type"]

    if t == "dict":
        return {k: _deserialize(v, data) for k, v in meta["items"].items()}

    if t in ("list", "tuple"):
        items = [_deserialize(x, data) for x in meta["items"]]
        return items if t == "list" else tuple(items)

    if t == "tensor":
        return Tensor(_restore_array(meta, data))

    if t in ("array", "np.ndarray"):
        return _restore_array(meta, data)

    if t == "class":
        module = importlib.import_module(meta["module"])
        resolved = module
        for attr in meta["qualname"].split("."):
            resolved = getattr(resolved, attr)
        return resolved

    if "key" in meta and "items" not in meta and "value" not in meta:
        return _restore_array(meta, data)

    if t in ("scalar", "raw"):
        return meta["value"]

    raise ValueError(f"Unknown meta type: {t}")


def _deserialize_metadata(meta: SerializedMeta) -> Any:
    """Deserialize JSON-only checkpoint fields without touching NPZ arrays."""
    t = meta["_type"]

    if t == "dict":
        return {k: _deserialize_metadata(v) for k, v in meta["items"].items()}

    if t in ("list", "tuple"):
        items = [_deserialize_metadata(x) for x in meta["items"]]
        return items if t == "list" else tuple(items)

    if t == "class":
        module = importlib.import_module(meta["module"])
        resolved = module
        for attr in meta["qualname"].split("."):
            resolved = getattr(resolved, attr)
        return resolved

    if t in ("scalar", "raw"):
        return meta["value"]

    if t in _ARRAY_META_TYPES or "key" in meta:
        raise ValueError("Metadata-only checkpoint load encountered an array field.")

    raise ValueError(f"Unknown meta type: {t}")


def _dict_items(meta: SerializedMeta, *, label: str) -> Dict[str, SerializedMeta]:
    if meta["_type"] != "dict":
        raise ValueError(f"{label} must be a dict in checkpoint metadata.")
    return meta["items"]


def _leaf_array(meta: SerializedMeta, data: Dict[str, Any]) -> Any:
    if meta["_type"] == "tensor":
        return _restore_array(meta, data)
    return _deserialize(meta, data)


def _array_dtype(meta: SerializedMeta) -> Any:
    if "dtype" in meta:
        return resolve_dtype(meta["dtype"])
    raise ValueError("Checkpoint array metadata is missing dtype.")


def _module_ref(root: Any, full_name: str) -> tuple[Any, str]:
    parts = full_name.split(".")
    *path, var_name = parts
    module_ref = root
    for part in path:
        module_ref = module_ref._modules[part]
    return module_ref, var_name


def _check_named_values(
    *,
    current: Dict[str, Any],
    loaded_items: Dict[str, SerializedMeta],
    kind: str,
    strict: bool,
) -> Iterable[str]:
    missing = set(current) - set(loaded_items)
    extra = set(loaded_items) - set(current)
    if strict and (missing or extra):
        raise ValueError(
            f"state_dict {kind} key mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return set(current) & set(loaded_items)


def _load_module_state_from_meta(
    module: Any,
    state_meta: SerializedMeta,
    data: Dict[str, Any] | None,
    *,
    strict: bool = True,
) -> None:
    unexpected_top_keys = set(_dict_items(state_meta, label="model_state_dict")) - {
        "parameters",
        "states",
    }
    if unexpected_top_keys:
        raise ValueError(
            f"state_dict has unexpected top-level keys: {sorted(unexpected_top_keys)}"
        )

    state_items = _dict_items(state_meta, label="model_state_dict")
    if strict and "parameters" not in state_items and module.parameters:
        raise ValueError(
            "state_dict parameter key mismatch: "
            f"missing={sorted(module.parameters)}, extra=[]"
        )
    if strict and "states" not in state_items and module.states:
        raise ValueError(
            f"state_dict state key mismatch: missing={sorted(module.states)}, extra=[]"
        )

    if "parameters" in state_items:
        param_items = _dict_items(state_items["parameters"], label="parameters")
        for name in _check_named_values(
            current=module.parameters,
            loaded_items=param_items,
            kind="parameter",
            strict=strict,
        ):
            current_param = module.parameters[name]
            loaded = (
                _leaf_array(param_items[name], data)
                if data is not None
                else xp.empty(
                    current_param.data.shape,
                    dtype=_array_dtype(param_items[name]),
                )
            )
            if loaded.shape != current_param.data.shape:
                raise ValueError(
                    f"parameter {name!r} shape mismatch: "
                    f"model={current_param.data.shape}, ckpt={loaded.shape}"
                )
            if loaded.dtype != current_param.data.dtype:
                raise ValueError(
                    f"parameter {name!r} dtype mismatch: "
                    f"model={current_param.data.dtype}, ckpt={loaded.dtype}"
                )
            module_ref, var_name = _module_ref(module, name)
            module_ref._parameters[var_name].data = loaded

    if "states" in state_items:
        loaded_state_items = _dict_items(state_items["states"], label="states")
        for name in _check_named_values(
            current=module.states,
            loaded_items=loaded_state_items,
            kind="state",
            strict=strict,
        ):
            current_state = module.states[name]
            loaded = (
                _leaf_array(loaded_state_items[name], data)
                if data is not None
                else xp.empty(
                    current_state.shape,
                    dtype=_array_dtype(loaded_state_items[name]),
                )
            )
            if loaded.shape != current_state.shape:
                raise ValueError(
                    f"state {name!r} shape mismatch: "
                    f"model={current_state.shape}, ckpt={loaded.shape}"
                )
            if loaded.dtype != current_state.dtype:
                raise ValueError(
                    f"state {name!r} dtype mismatch: "
                    f"model={current_state.dtype}, ckpt={loaded.dtype}"
                )
            module_ref, var_name = _module_ref(module, name)
            module_ref._states[var_name] = loaded
            object.__setattr__(module_ref, var_name, loaded)


def _load_optimizer_state_from_meta(
    optimizer: Any,
    state_meta: SerializedMeta,
    data: Dict[str, Any] | None,
) -> None:
    state_items = _dict_items(state_meta, label="optimizer_state_dict")
    optimizer._states.clear()

    if "hyperparams" in state_items:
        for key, value in _deserialize_metadata(state_items["hyperparams"]).items():
            optimizer._hyperparams[key] = value

    if "states" not in state_items:
        return

    for state_key, value_meta in _dict_items(
        state_items["states"], label="states"
    ).items():
        if value_meta["_type"] != "dict":
            optimizer._states[state_key] = _deserialize_metadata(value_meta)
            continue

        optimizer._states[state_key] = {}
        for param_name, param_meta in value_meta["items"].items():
            if param_name in optimizer.model_parameters:
                param = optimizer.model_parameters[param_name]
                optimizer._states[state_key][param_name] = (
                    _leaf_array(param_meta, data)
                    if data is not None
                    else xp.empty(param.data.shape, dtype=_array_dtype(param_meta))
                )


def load_checkpoint_metadata(
    json_path: str,
    npz_path: str | None = None,
    *,
    skip_keys: Iterable[str] = ("model_state_dict", "optimizer_state_dict"),
) -> Any:
    """Load checkpoint metadata without deserializing NPZ-backed arrays."""
    _check_checkpoint_paths(json_path, npz_path)
    meta = _read_checkpoint_meta(json_path)
    if meta["_type"] != "dict":
        return _deserialize_metadata(meta)

    skipped = set(skip_keys)
    return {
        key: _deserialize_metadata(value)
        for key, value in meta["items"].items()
        if key not in skipped
    }


def load_checkpoint_state_into(
    *,
    model: Any,
    optimizer: Any | None,
    json_path: str,
    npz_path: str,
    load_arrays: bool = True,
) -> None:
    """Load model/optimizer arrays directly from checkpoint NPZ members.

    This avoids materializing the whole checkpoint object in GPU memory during
    resume. Metadata is read from JSON; each array is read and assigned to its
    target slot once.
    """
    _check_checkpoint_paths(json_path, npz_path)
    root_items = _dict_items(
        _read_checkpoint_meta(json_path),
        label="checkpoint",
    )

    data: Any | None = xp.load(npz_path) if load_arrays else None
    try:
        if optimizer is not None:
            optimizer._states.clear()
        _load_module_state_from_meta(model, root_items["model_state_dict"], data)
        if optimizer is not None:
            _load_optimizer_state_from_meta(
                optimizer,
                root_items["optimizer_state_dict"],
                data,
            )
    finally:
        close = getattr(data, "close", None)
        if close is not None:
            close()


def save_checkpoint(
    obj: Any,
    *,
    checkpoint_dir: str = ".",
    checkpoint_name: str = "checkpoint",
) -> tuple[str, str]:
    """Save a Python object (model state, etc.) into JSON and NPZ files.

    This function splits the saved content into:
    - A JSON file for the metadata or structure.
    - A NPZ file for numeric array data (e.g., model weights).

    Args:
        obj: The Python object to serialize, typically a model's state_dict or
            other checkpoint data structure.
        checkpoint_dir (str): Directory where checkpoint files are written.
            Defaults to the current directory.
        checkpoint_name (str): Basename for the checkpoint files written inside
            `checkpoint_dir`. Defaults to `checkpoint`.

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
        >>> save_checkpoint(obj_to_save, checkpoint_name="my_model")
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
    npz_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.npz")

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
            arr = xp.array(obj.data)
            arrays[key] = arr
            return {
                "_type": "tensor",
                "key": key,
                "dtype": _dtype_name(arr),
            }

        if isinstance(obj, ARRAY_TYPE) or hasattr(obj, "__array__"):
            key = prefix if prefix else "root_array"
            arr = xp.array(obj)
            arrays[key] = arr
            return {
                "_type": "array",
                "key": key,
                "dtype": _dtype_name(arr),
            }

        # Primitive scalar types
        if isinstance(obj, (int, float, str, bool, type(None))):
            return {"_type": "scalar", "value": obj}

        if isinstance(obj, type):
            return {
                "_type": "class",
                "module": obj.__module__,
                "qualname": obj.__qualname__,
            }

        # Fallback for other types
        return {"_type": "raw", "value": str(obj)}

    arrays_dict: Dict[str, Any] = {}
    meta: SerializedMeta = _serialize(obj, arrays_dict)

    # Save metadata to JSON
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Save arrays without ZIP compression. GPT-sized checkpoints are mostly
    # dense weights/state, so compression burns CPU on the training path while
    # saving little disk.
    xp.savez(npz_path, **arrays_dict)
    return json_path, npz_path


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

    _check_checkpoint_paths(json_path, npz_path)
    meta = _read_checkpoint_meta(json_path)

    # Load NPZ data
    data_dict: Any = xp.load(npz_path)

    deserialized_data = _deserialize(meta, data_dict)

    # If requested, return only the "parameters" sub-dictionary
    if weights_only and isinstance(deserialized_data, dict):
        return deserialized_data.get("parameters", deserialized_data)

    return deserialized_data
