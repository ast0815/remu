"""ReMU - Response Matrix Utilities

A framework for likelihood calculations and hypothesis testing using binned
events and response matrices.

"""

from typing import Optional

__version__: Optional[str]

try:
    from ._version import __version__
except ImportError:
    __version__ = None
