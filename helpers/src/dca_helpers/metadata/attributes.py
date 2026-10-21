"""Writing the resolved DCA block and normalization statistics onto a zarr node.

:func:`merge_dca_attributes` puts a resolved payload on a node's ``attributes.dca``.
:func:`write_normalization_statistics` replaces ``attributes.dca.normalization_statistics``.
Both mutate the attributes dict in place and leave every other key alone.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dca_helpers import CURRENT_SPEC_VERSION
from dca_helpers.validation.versions.v0_2.models import DCAChannelNormalization

logger = logging.getLogger(__name__)

#: The ``dca.url`` written on every node: the JSON Schema of the current spec version.
DCA_SCHEMA_URL = (
    "https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/tree/main/docs/"
    f"v{CURRENT_SPEC_VERSION}/schema.json"
)


def merge_dca_attributes(
    attrs: dict[str, Any],
    payload: Mapping[str, Any] | None,
    channels: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Merge a resolved DCA payload into *attrs* in place.

    ``update`` rather than assignment: the payload's top-level keys are exactly the ones
    it owns (it is validated against a model with ``extra="forbid"``), so anything else
    already on the node, ``normalization_statistics`` above all, survives.

    ``color`` is an omero field, not a DCA one; it rides in the same channel entry so a
    config declares each channel once, and is dropped on the way into ``dca``.
    """
    if channels is None and not payload:
        return
    dca = attrs.setdefault("dca", {})
    dca["version"] = CURRENT_SPEC_VERSION
    dca["url"] = DCA_SCHEMA_URL
    if channels is not None:
        dca["channels"] = [
            {k: v for k, v in entry.items() if k != "color"} for entry in channels
        ]
    dca.update({k: v for k, v in (payload or {}).items() if k != "channels"})


def _assert_finite(value: Any, where: str) -> None:
    """Raise ``ValueError`` if *value* holds a NaN or infinite number at any depth."""
    if isinstance(value, bool):
        return
    if isinstance(value, float | int):
        if not math.isfinite(value):
            raise ValueError(f"{where}: value {value!r} is not finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{where}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _assert_finite(item, f"{where}[{index}]")


def write_normalization_statistics(
    attrs: dict[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Replace ``attrs["dca"]["normalization_statistics"]`` with validated *stats*.

    *stats* maps a channel index string (``"0"``, ``"1"``) to a channel entry. Each entry
    must hold finite numbers and validate as ``DCAChannelNormalization`` (required
    ``dataset_statistics`` with ``p1``, ``p5``, ``p95``, ``p99``). The block is replaced
    wholesale, not merged, so a re-run cannot keep a stale channel. *metadata*, when given,
    is stored under the ``"metadata"`` key.
    """
    block: dict[str, Any] = {}
    for key, entry in stats.items():
        if not str(key).isdigit():
            raise ValueError(
                f"normalization statistics key {key!r} is not a channel index "
                "(a decimal string such as '0')"
            )
        _assert_finite(entry, f"normalization statistics for channel {key}")
        block[str(key)] = DCAChannelNormalization.model_validate(entry).model_dump(
            exclude_none=True
        )
    if metadata is not None:
        block["metadata"] = dict(metadata)
    attrs.setdefault("dca", {})["normalization_statistics"] = block
    logger.debug("wrote normalization statistics for %d channels", len(stats))
