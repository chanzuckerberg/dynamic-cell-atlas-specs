"""Load and layer the ``[dca]`` table of one or more TOML config files."""

import copy
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dca_helpers.metadata.resolve import _flatten_provenance, _is_leaf


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
    """Return the ``[dca]`` table of *path* with its provenance keyed by dotted path.

    A TOML syntax error is re-raised as ``ValueError`` prefixed with *path*, so a run that
    layers several config files says which one is broken.
    """
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{path}: {error}") from error
    if "dca" not in data:
        raise ValueError(f"{path} has no [dca] table")
    block = data["dca"]
    provenance = block.get("provenance")
    if isinstance(provenance, Mapping):
        block["provenance"] = _flatten_provenance(provenance)
    return block


def _merge_provenance(base: Any, over: Any) -> Any:
    """Merge two flattened provenance maps; a later record replaces an earlier one whole."""
    if not (isinstance(base, Mapping) and isinstance(over, Mapping)):
        return copy.deepcopy(over)
    merged = {key: copy.deepcopy(val) for key, val in base.items()}
    merged.update(copy.deepcopy(dict(over)))
    return merged


def load_dca_config(*paths: Path) -> dict[str, Any]:
    """Return the ``[dca]`` table of *paths*, later files overriding earlier ones.

    Tables merge key by key; a leaf spec, a scalar or an array is replaced whole by the
    later value. The result is a deep copy. A file without a ``[dca]`` table raises
    ``ValueError`` naming that file.

    Each file's ``provenance`` is flattened to dotted field paths (``"study.release_date"``)
    before merging, so the nested and quoted TOML spellings name the same entry. A
    provenance record is replaced whole by a later file's record for the same path, so a
    confirmed record keeps no key, such as an agent ``citation``, from the base one.
    """
    result: dict[str, Any] = {}
    for path in paths:
        block = _read_dca(Path(path))
        merged = _merge(result, {k: v for k, v in block.items() if k != "provenance"})
        if "provenance" in block:
            merged["provenance"] = _merge_provenance(
                result.get("provenance"), block["provenance"]
            )
        result = merged
    return result
