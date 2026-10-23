"""Writing the resolved DCA block and normalization statistics onto a zarr node.

:func:`merge_dca_attributes` puts a resolved payload on a node's ``attributes.dca``.
:func:`write_normalization_statistics` replaces ``attributes.dca.normalization_statistics``.
Both mutate the attributes dict in place and leave every other key alone, except the run
provenance keys a caller passes to :func:`merge_dca_attributes` as ``run_attrs``.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, get_args

from dca_helpers.validation.versions.v0_2.models import (
    DCAChannelNormalization,
    DCAGroupMetadataV0_2,
    DCANormalizationStatistics,
)

logger = logging.getLogger(__name__)

#: The ``dca.version`` written on every node. Read from the ``version`` literal of the
#: v0.2 models that resolve and validate against, so the stamp cannot drift from them.
(SPEC_VERSION,) = get_args(DCAGroupMetadataV0_2.model_fields["version"].annotation)

#: The ``dca.url`` written on every node: the JSON Schema of :data:`SPEC_VERSION`.
DCA_SCHEMA_URL = (
    "https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/tree/main/docs/"
    f"v{SPEC_VERSION}/schema.json"
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
    dca["version"] = SPEC_VERSION
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

    *stats* maps a channel index string (``"0"``, ``"1"``) to a channel entry, and
    *metadata*, when given, is stored under the ``"metadata"`` key. The assembled block
    is validated once with ``DCANormalizationStatistics``, which rejects any other key,
    any channel entry that is not a ``DCAChannelNormalization`` (required
    ``dataset_statistics`` with ``p1``, ``p5``, ``p95``, ``p99``) and a ``metadata.grid``
    below 1. Other ``metadata`` keys are free form.

    Each channel entry is written as its model dump, so numeric strings become floats.
    The input and the written block must hold only finite numbers, because JSON cannot
    hold NaN or infinity; checking after the dump catches a ``"nan"`` string that
    pydantic coerces to a float. The block is replaced wholesale, not merged, so a re-run
    cannot keep a stale channel.
    """
    _assert_finite(stats, "normalization statistics")
    block: dict[str, Any] = {str(key): entry for key, entry in stats.items()}
    if "metadata" in block:
        raise ValueError(
            "normalization statistics key 'metadata' is not a channel index; "
            "pass it as metadata="
        )
    if metadata is not None:
        _assert_finite(metadata, "normalization statistics metadata")
        block["metadata"] = dict(metadata)
    model = DCANormalizationStatistics.model_validate(block)
    written: dict[str, Any] = {
        key: DCAChannelNormalization.model_validate(entry).model_dump(exclude_none=True)
        for key, entry in (model.model_extra or {}).items()
    }
    if model.metadata is not None:
        written["metadata"] = model.metadata.model_dump(exclude_unset=True)
    _assert_finite(written, "normalization statistics")
    attrs.setdefault("dca", {})["normalization_statistics"] = written
    logger.debug("wrote normalization statistics for %d channels", len(stats))
