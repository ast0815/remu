"""YAML compatibility utilities for ReMU.

This module provides custom YAML serialization/deserialization with
NumPy 2.x compatibility.
"""

from typing import cast

import numpy as np
import yaml

# Static re-exports for type checkers; the loop at the end of this module
# re-exports the full PyYAML API at runtime.
from yaml import FullLoader, YAMLObject


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


# Register the custom representer (np.generic is used as an isinstance tag
# here, not instantiated).
yaml.add_multi_representer(cast(type, np.generic), represent_numpy_scalar)


# Re-export all public symbols from PyYAML. The names used across the package
# are listed statically so that type checkers and linters can see them; the
# loop appends the rest at runtime.
__all__ = ["FullLoader", "YAMLObject"]
for name in dir(yaml):
    if not name.startswith("_"):
        globals()[name] = getattr(yaml, name)
        if name not in __all__:
            __all__.append(name)
