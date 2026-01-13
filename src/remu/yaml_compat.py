"""YAML compatibility utilities for ReMU.

This module provides custom YAML serialization/deserialization with
NumPy 2.x compatibility.
"""

import numpy as np
import yaml


def represent_numpy_scalar(dumper, data):
    """Custom representer for NumPy scalars to fix compatibility issues."""
    # Convert NumPy scalars to Python scalars
    if hasattr(data, 'item'):
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


def dump(data, stream=None, **kwargs):
    """Dump data to YAML with NumPy compatibility."""
    return yaml.dump(data, stream=stream, **kwargs)


def full_load(stream, **kwargs):
    """Load data from YAML with NumPy compatibility.""" 
    return yaml.full_load(stream, **kwargs)


def load(stream, **kwargs):
    """Load data from YAML with NumPy compatibility."""
    if 'Loader' not in kwargs:
        kwargs['Loader'] = yaml.FullLoader
    return yaml.load(stream, **kwargs)


# Export the yaml functions for compatibility
FullLoader = yaml.FullLoader