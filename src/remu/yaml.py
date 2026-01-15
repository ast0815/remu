"""YAML compatibility utilities for ReMU.

This module provides custom YAML serialization/deserialization with
NumPy 2.x compatibility.
"""

import numpy as np
import yaml


def represent_numpy_scalar(dumper, data):
    """Custom representer for NumPy scalars to fix compatibility issues."""
    # Convert NumPy scalars to Python scalars
    if hasattr(data, "item"):
        value = data.item()
    else:
        value = data
    if isinstance(value, bool):
        return dumper.represent_bool(value)
    elif isinstance(value, int):
        return dumper.represent_int(value)
    elif isinstance(value, float):
        return dumper.represent_float(value)
    else:
        return dumper.represent_str(str(value))


# Register the custom representer
yaml.add_multi_representer(np.generic, represent_numpy_scalar)


# Re-export all public symbols from PyYAML
__all__ = []
for name in dir(yaml):
    if not name.startswith("_"):
        globals()[name] = getattr(yaml, name)
        __all__.append(name)


# Override load to set default Loader
def load(stream, **kwargs):
    if "Loader" not in kwargs:
        kwargs["Loader"] = yaml.FullLoader
    return yaml.load(stream, **kwargs)


globals()["load"] = load
__all__.append("load")
