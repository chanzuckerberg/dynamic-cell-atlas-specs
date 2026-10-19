"""
Pydantic models for Dynamic Cell Atlas spec v0.3.

v0.3 keeps every v0.2 image and label rule and adds projections
(docs/v0.3/projections.rst, docs/v0.3/schema.json ``$defs/Projection``):

  - ``dca.version`` is "0.3".
  - ``dca.projection`` (optional on the model; its presence is governed by
    ``check_projection_v0_3`` below) holds the projection parameters:
    ``method`` and ``source_store`` MUST be present; ``z_dim_source`` /
    ``z_start`` / ``z_end`` MAY be, all three or none, with
    ``0 <= z_start < z_end <= z_dim_source``.

The rules that relate ``dca.projection`` to the rest of the group (the
processing-variant agreement, z of size 1 at every level, the z scale, the
v0.2-era top-level ``mip`` attribute) live in ``check_projection_v0_3`` rather
than in more ``mode="after"`` validators on ``DCAStoreSpecV0_3``: pydantic stops
running a model's after-validators at the first one that raises, so a SHOULD
warning from an inherited v0.2 rule (e.g. the channel-count check) would
silently skip them. The function runs independently, like the
experimental-metadata layer.

Cross-store rules (the projection's source exists and matches; ``--require-mip``)
need every store in a run and live in ``validation/pairing.py``.

MUST violations  → raise ValueError (or ERROR Issue) → ERROR.
SHOULD violations → "[SHOULD]" prefix (or WARNING Issue) → WARNING.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dca_helpers.validation.versions.v0_2.models import (
    DCAGroupMetadataV0_2,
    DCALabelStoreSpecV0_2,
    DCAStoreSpecV0_2,
)
from dca_helpers.validation.versions.v0_3.experimental_metadata import MIP_STEP

if TYPE_CHECKING:
    from dca_helpers.validation.result import Issue

# Top-level attribute that v0.2-era projection stores wrote beside ``ome`` and
# ``dca``. v0.3 moves it to ``dca.projection`` (projections.rst § Migration).
LEGACY_MIP_ATTR = "mip"

_Z_RANGE_FIELDS = ("z_dim_source", "z_start", "z_end")


class DCAProjectionV0_3(BaseModel):
    """``dca.projection`` — schema.json ``$defs/Projection``."""

    model_config = ConfigDict(extra="allow")

    # projections.rst § Projection block — MUST
    method: Literal["max"]
    source_store: str = Field(min_length=1)

    # MAY — all three or none
    z_dim_source: int | None = Field(default=None, ge=1)
    z_start: int | None = Field(default=None, ge=0)
    z_end: int | None = Field(default=None, ge=1)
    z_fraction: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def check_z_range(self) -> "DCAProjectionV0_3":
        present = [f for f in _Z_RANGE_FIELDS if getattr(self, f) is not None]
        if present and len(present) != len(_Z_RANGE_FIELDS):
            missing = [f for f in _Z_RANGE_FIELDS if f not in present]
            raise ValueError(
                f"dca.projection has {present} but not {missing}; z_dim_source, "
                "z_start and z_end MUST be given together or not at all "
                "(projections.rst § Projection block)"
            )
        if present and not (0 <= self.z_start < self.z_end <= self.z_dim_source):
            raise ValueError(
                f"dca.projection range is invalid: z_start={self.z_start}, "
                f"z_end={self.z_end}, z_dim_source={self.z_dim_source}; it MUST "
                "satisfy 0 <= z_start < z_end <= z_dim_source (z_end is exclusive)"
            )
        return self


class DCAGroupMetadataV0_3(DCAGroupMetadataV0_2):
    """The image-group ``dca`` object for v0.3: v0.2 plus ``projection``."""

    version: Literal["0.3"]
    projection: DCAProjectionV0_3 | None = None


class DCAStoreSpecV0_3(DCAStoreSpecV0_2):
    """
    Root DCA v0.3 image model. Same input as DCAStoreSpecV0_2 with
    spec_version "0.3"; every v0.2 rule is inherited unchanged.
    """

    spec_version: Literal["0.3"]
    dca: DCAGroupMetadataV0_3 | None = Field(default=None, validate_default=True)

    # Same rule as v0.2, restated so the message names the right version.
    @field_validator("dca", mode="before")
    @classmethod
    def check_dca_present(
        cls, v: "DCAGroupMetadataV0_3 | dict | None"
    ) -> "DCAGroupMetadataV0_3 | dict | None":
        if v is None:
            raise ValueError(
                "Missing 'dca' key in zarr.json. "
                "DCA v0.3 MUST include a 'dca' object containing "
                "version, url, channels, and normalization_statistics "
                "(schema.json dca.required)."
            )
        return v


class DCALabelStoreSpecV0_3(DCALabelStoreSpecV0_2):
    """v0.3 label image model: label rules are unchanged from v0.2."""

    spec_version: Literal["0.3"]


def _lists_mip(dca: dict) -> bool:
    image_data = dca.get("image_data")
    steps = (
        image_data.get("processing_variant") if isinstance(image_data, dict) else None
    )
    if isinstance(steps, list):
        return MIP_STEP in steps
    # A bare string is a type error reported by the experimental-metadata layer;
    # still read it so this check does not add a second, confusing error.
    return steps == MIP_STEP


def check_projection_v0_3(
    node_dict: dict, dca: object, group_attrs: dict
) -> "list[Issue]":
    """Projection rules that relate ``dca.projection`` to the rest of an image group.

    ``node_dict`` is the dict built by ``validator._build_node_dict``; ``group_attrs``
    is the group's full ``zarr.json`` attributes (for the top-level ``mip`` check).
    """
    from dca_helpers.validation.result import Issue, Severity

    issues: list[Issue] = []

    def error(loc: tuple, message: str) -> None:
        issues.append(Issue(loc=loc, message=message, severity=Severity.ERROR))

    def warning(loc: tuple, message: str) -> None:
        issues.append(Issue(loc=loc, message=message, severity=Severity.WARNING))

    if LEGACY_MIP_ATTR in group_attrs:
        error(
            (LEGACY_MIP_ATTR,),
            f"Top-level '{LEGACY_MIP_ATTR}' attribute found. v0.3 stores carry "
            "projection parameters in dca.projection; move the object there "
            "unchanged and remove the top-level key (projections.rst § Migration).",
        )

    if not isinstance(dca, dict):
        return issues

    projection = dca.get("projection")
    has_block = projection is not None
    lists_mip = _lists_mip(dca)

    if lists_mip and not has_block:
        error(
            ("dca", "projection"),
            f"image_data.processing_variant lists '{MIP_STEP}' but dca.projection is "
            "missing; every projection MUST carry dca.projection "
            "(projections.rst § Projection block).",
        )
    if has_block and not lists_mip:
        error(
            ("dca", "image_data", "processing_variant"),
            f"dca.projection is present but image_data.processing_variant does not "
            f"list '{MIP_STEP}'; a projection MUST list it, and an image that is not "
            "a projection MUST NOT carry dca.projection.",
        )
    if not has_block:
        return issues

    axes = [a.lower() for a in node_dict.get("axes", [])]
    if "z" in axes:
        zi = axes.index("z")
        bad = [
            (i, shape[zi])
            for i, shape in enumerate(node_dict.get("array_shapes", []))
            if len(shape) > zi and shape[zi] != 1
        ]
        if bad:
            error(
                ("levels",),
                f"A projection MUST have z of size 1 at every level; got "
                f"(level, z) = {bad} (projections.rst § Dimensions and Scale).",
            )
        z_scale = {
            k.lower(): v for k, v in node_dict.get("axis_scales", {}).items()
        }.get("z")
        if z_scale is not None and z_scale != 1.0:
            warning(
                ("axis_scales", "z"),
                f"Projection z scale is {z_scale}; it should be 1.0, since a "
                "projection has no physical z spacing "
                "(projections.rst § Dimensions and Scale).",
            )

    if isinstance(projection, dict):
        zd, z0, z1 = (projection.get(f) for f in _Z_RANGE_FIELDS)
        frac = projection.get("z_fraction")
        if (
            all(isinstance(v, int) for v in (zd, z0, z1))
            and isinstance(frac, (int, float))
            and zd > 0
            # Rounding a slab to whole planes can move each edge by one plane.
            and not math.isclose(frac, (z1 - z0) / zd, abs_tol=2 / zd)
        ):
            warning(
                ("dca", "projection", "z_fraction"),
                f"z_fraction={frac} disagrees with the range "
                f"({z1} - {z0}) / {zd} = {(z1 - z0) / zd:.3f}; z_start and z_end "
                "are authoritative.",
            )
    return issues
