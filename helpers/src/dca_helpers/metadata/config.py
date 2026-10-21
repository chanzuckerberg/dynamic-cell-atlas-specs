"""Load and layer the ``[dca]`` table of one or more TOML config files."""

import copy
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dca_helpers.metadata.resolve import _is_leaf


def _merge(base: Any, over: Any) -> Any:
    """Merge *over* onto *base*; only two plain mappings merge, anything else is replaced.

    A leaf spec on either side is replaced whole, so a ``{ value = ... }`` or a bare
    literal over a base ``{ column = ... }`` never leaves a ``{column, value}`` hybrid.
    Lists, including arrays of tables, are replaced rather than appended.
    """
    both_trees = isinstance(base, Mapping) and isinstance(over, Mapping)
    if not both_trees or _is_leaf(base) or _is_leaf(over):
        return copy.deepcopy(over)
    merged = {key: copy.deepcopy(val) for key, val in base.items()}
    for key, val in over.items():
        merged[key] = _merge(merged[key], val) if key in merged else copy.deepcopy(val)
    return merged


def _read_dca(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if "dca" not in data:
        raise ValueError(f"{path} has no [dca] table")
    return data["dca"]


def load_dca_config(*paths: Path) -> dict[str, Any]:
    """Return the ``[dca]`` table of *paths*, later files overriding earlier ones.

    Tables merge key by key; a leaf spec, a scalar or an array is replaced whole by the
    later value. The result is a deep copy. A file without a ``[dca]`` table raises
    ``ValueError`` naming that file.
    """
    result: dict[str, Any] = {}
    for path in paths:
        result = _merge(result, _read_dca(Path(path)))
    return result
