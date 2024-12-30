import importlib
import json
import numpy as np
from autograd.tensor import Tensor

"""
Tool for serializing/deserializing the model to disk

metadata:
    {
    "_class": "Module",
    "submodules": {
        "layer1": {
        "_class": "Linear",
        "submodules": {},
        "parameters": ["weight", "bias"]
        },
        "layer2": {
        "_class": "Linear",
        "submodules": {},
        "parameters": ["weight", "bias"]
        }
    },
    "parameters": []
    }

param_dict:
    {
    "layer1.weight": np.array(...),
    "layer1.bias": np.array(...),
    "layer2.weight": np.array(...),
    "layer2.bias": np.array(...),
    }
"""


def module_to_metadata_and_params(module, prefix=""):
    """
    Recursively convert a Module into:
      1) A metadata dictionary describing its structure.
      2) A parameter dictionary with string keys -> NumPy arrays.
    """
    # The metadata for this module
    args, kwargs = getattr(module, "_constructor_args", ([], {}))
    metadata = {
        "_class": module.__class__.__module__ + "." + module.__class__.__name__,
        "constructor_args": {"args": args, "kwargs": kwargs},
        "submodules": {},
        "parameters": [],
    }

    # 1. Gather parameters from the current module:
    #    We store them in the param_dict as {prefix + param_name: array}
    #    We also remember the 'param_name' in metadata["parameters"] for reference
    param_dict = {}
    for param_name, tensor in module._parameters.items():
        full_key = f"{prefix}.{param_name}" if prefix else param_name
        metadata["parameters"].append(param_name)
        param_dict[full_key] = (
            tensor.data if isinstance(tensor, Tensor) else tensor
        )  # This is a NumPy array

    # 2. Gather submodules:
    #    For each submodule, we recurse. We also build a submodule key
    #    e.g. if prefix="layer1" and sub_name="layer2", combined_prefix="layer1.layer2"
    for sub_name, submodule in module._modules.items():
        sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
        sub_metadata, sub_param_dict = module_to_metadata_and_params(
            submodule, sub_prefix
        )
        # Save that submodule's metadata in the parent's 'submodules' dict
        metadata["submodules"][sub_name] = sub_metadata
        # Merge parameter dict from submodule
        param_dict.update(sub_param_dict)

    return metadata, param_dict


def save_model(module, json_path="model_structure.json", npz_path="model_params.npz"):
    """
    Save a Module to two files:
      - A JSON file with module structure.
      - An NPZ file with all the parameter arrays.
    """
    metadata, param_dict = module_to_metadata_and_params(module)

    # 1. Save the JSON structure
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 2. Save the parameters in compressed NumPy format
    np.savez_compressed(npz_path, **param_dict)


def module_from_metadata_and_params(metadata, params, prefix=""):
    """
    Rebuild a Module (recursively) from:
      - metadata: dictionary describing the structure
      - params: dict-like with {param_full_name: np.array}
    """

    # Grab the class name from metadata
    module_name, class_name = metadata["_class"].rsplit(".", 1)  # split on last dot
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    # Retrieve the constructor args
    constructor_args = metadata["constructor_args"]
    args = constructor_args["args"]
    kwargs = constructor_args["kwargs"]
    module = cls(*args, **kwargs)

    # Load the parameters for this module
    for param_name in metadata["parameters"]:
        full_key = f"{prefix}.{param_name}" if prefix else param_name
        # Create a Tensor with the loaded NumPy array
        val = params[full_key]
        param_tensor = (
            None if val is None or np.array_equal(val, np.array(None)) else Tensor(val)
        )
        module._parameters[param_name] = param_tensor

    # Recursively rebuild submodules
    for sub_name, sub_metadata in metadata["submodules"].items():
        sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
        submodule = module_from_metadata_and_params(sub_metadata, params, sub_prefix)
        module._modules[sub_name] = submodule

    return module


def load_model(json_path="model_structure.json", npz_path="model_params.npz"):
    """
    Load a Module from the two files created by save_model:
      - JSON structure
      - NPZ parameters
    """
    # 1. Load the JSON structure
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # 2. Load the NPZ (parameters)
    with np.load(npz_path, allow_pickle=True) as param_arrays:
        # param_arrays is an NpzFile, which works like a dict of arrays
        module = module_from_metadata_and_params(metadata, param_arrays)

    return module
