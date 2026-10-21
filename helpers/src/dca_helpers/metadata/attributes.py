"""Writing the resolved DCA block and normalization statistics onto a zarr node.

:func:`merge_dca_attributes` puts a resolved payload on a node's ``attributes.dca``.
:func:`write_normalization_statistics` replaces ``attributes.dca.normalization_statistics``.
Both mutate the attributes dict in place and leave every other key alone, except the run
provenance keys a caller passes to :func:`merge_dca_attributes` as ``run_attrs``.
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


#: Top-level attribute keys that run provenance may not claim.
_RESERVED_ATTRS = frozenset({"dca", "ome"})


def merge_dca_attributes(
    attrs: dict[str, Any],
    payload: Mapping[str, Any] | None,
    channels: Sequence[Mapping[str, Any]] | None,
    *,
    run_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Merge a resolved DCA payload into *attrs* in place.

    ``update`` rather than assignment: the payload's top-level keys are exactly the ones
    it owns (it is validated against a model with ``extra="forbid"``), so anything else
    already on the node, ``normalization_statistics`` above all, survives.

    ``color`` is an omero field, not a DCA one; it rides in the same channel entry so a
    config declares each channel once, and is dropped on the way into ``dca``.

    *run_attrs* records which code wrote the block. Its shape is
    ``{"<tool>": {"version": ..., "code_git_sha": ..., ...}}``, and each key is set as a
    top-level sibling of ``dca``, never inside it, because the DCA models would reject
    the extra keys. A key is replaced wholesale, not merged, so a re-run without a git
    sha cannot inherit the previous run's. ``None`` (the default) leaves any existing
    sibling untouched. A key of ``"dca"`` or ``"ome"`` raises ``ValueError`` before
    anything is written.
    """
    if run_attrs is not None:
        reserved = sorted(_RESERVED_ATTRS.intersection(run_attrs))
        if reserved:
            raise ValueError(f"run_attrs cannot use the reserved keys {reserved}")
        for key, value in run_attrs.items():
            attrs[key] = dict(value)
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
    must validate as ``DCAChannelNormalization`` (required ``dataset_statistics`` with
    ``p1``, ``p5``, ``p95``, ``p99``) and hold only finite numbers after validation, so a
    ``"nan"`` string that pydantic coerces to a float is rejected too. The block is replaced
    wholesale, not merged, so a re-run cannot keep a stale channel. *metadata*, when given,
    must hold only finite numbers and is stored under the ``"metadata"`` key.
    """
    block: dict[str, Any] = {}
    for key, entry in stats.items():
        if not str(key).isdigit():
            raise ValueError(
                f"normalization statistics key {key!r} is not a channel index "
                "(a decimal string such as '0')"
            )
        dumped = DCAChannelNormalization.model_validate(entry).model_dump(
            exclude_none=True
        )
        _assert_finite(dumped, f"normalization statistics for channel {key}")
        block[str(key)] = dumped
    if metadata is not None:
        _assert_finite(metadata, "normalization statistics metadata")
        block["metadata"] = dict(metadata)
    attrs.setdefault("dca", {})["normalization_statistics"] = block
    logger.debug("wrote normalization statistics for %d channels", len(stats))
