"""
Compatibility helpers for environments running NumPy 2.x.

Some third-party libraries (e.g., chromadb) still import deprecated aliases such
as `np.float_` or `np.uint`. This module restores those aliases when missing so
the rest of the codebase can run without forcing users to downgrade NumPy.
"""

from __future__ import annotations

import numpy as np


def _alias(missing_name: str, target_name: str) -> None:
    if hasattr(np, missing_name):
        return
    target = getattr(np, target_name, None)
    if target is not None:
        setattr(np, missing_name, target)


_alias("float_", "float64")
_alias("int_", "int64")
_alias("uint", "uint64")
