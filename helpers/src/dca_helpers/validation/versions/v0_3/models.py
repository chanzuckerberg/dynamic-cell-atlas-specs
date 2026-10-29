"""
Pydantic models for Dynamic Cell Atlas spec v0.3.

v0.3 keeps every v0.2 image and label rule and adds projections
(docs/v0.3/projections.rst, docs/v0.3/schema.json ``$defs/Projection``):

  - ``dca.version`` is "0.3".
  - ``dca.projection`` (optional on the model; its presence is governed by
    ``check_projection_v0_3`` below) records what was done: the method (``max``,
    ``mean``, ``sum``, ``slice``, shared with biahub's ``project`` command) and the
    source planes actually used, as absolute half-open ``[z_start, z_end)`` ranges.
    The uniform form is ``method`` plus the ``z_start`` / ``z_end`` / ``z_dim_source``
    / ``z_fraction`` that v0.2-era stores already carry; the per-channel form is
    ``channels``, keyed by projected output channel name (a subset of the image's
    channels), where ``z_start`` / ``z_end`` may be per-timepoint lists. ``focus`` is
    optional provenance about how planes were chosen. ``source_store`` is required only
    when the 3D source is in the DCA, which the cross-store pass checks.

The rules that relate ``dca.projection`` to the rest of the group (the
processing-variant agreement, z of size 1 at every level, channel names, per-timepoint
list lengths, the v0.2-era top-level ``mip`` attribute) live in
``check_projection_v0_3`` rather than in more ``mode="after"`` validators on
``DCAStoreSpecV0_3``: pydantic stops running a model's after-validators at the first
one that raises, so a SHOULD warning from an inherited v0.2 rule (e.g. the
channel-count check) would silently skip them. The function runs independently,
like the experimental-metadata layer.

Cross-store rules (the projection's source exists and matches;
``--require-projection``) need every store in a run and live in
``validation/pairing.py``.

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
from dca_helpers.validation.versions.v0_3.experimental_metadata import (
    PROJECTION_STEP_BY_METHOD,
    PROJECTION_STEPS,
)

if TYPE_CHECKING:
    from dca_helpers.validation.result import Issue

# Top-level attribute that v0.2-era projection stores wrote beside ``ome`` and
# ``dca``. v0.3 moves it to ``dca.projection`` (projections.rst § Migration).
LEGACY_MIP_ATTR = "mip"

ProjectionMethod = Literal["max", "mean", "sum", "slice"]

# projections.rst § Dimensions and Scale: methods whose output values are source
# values, so a projection using only these keeps the source dtype.
VALUE_PRESERVING_METHODS = frozenset({"max", "slice"})

_UNIFORM_ONLY_FIELDS = ("z_start", "z_end", "z_step", "z_fraction")


def _plane_range_errors(
    where: str,
    method: str | None,
    z_start: "int | list[int] | None",
    z_end: "int | list[int] | None",
    z_step: int | None,
    z_dim: int | None,
) -> list[str]:
    """projections.rst § Plane ranges, for one form's (or one channel's) fields."""
    errors: list[str] = []
    if (z_start is None) != (z_end is None):
        errors.append(f"{where}: z_start and z_end MUST be given together")
        return errors
    if z_start is None:
        if z_step is not None:
            errors.append(
                f"{where}: z_step MUST NOT be given without z_start and z_end"
            )
        if method == "slice":
            errors.append(
                f"{where}: method slice MUST give z_start and z_end (one plane)"
            )
        return errors
    if isinstance(z_start, list) != isinstance(z_end, list):
        errors.append(
            f"{where}: z_start and z_end MUST both be integers or both be lists"
        )
        return errors
    starts = z_start if isinstance(z_start, list) else [z_start]
    ends = z_end if isinstance(z_end, list) else [z_end]
    if len(starts) != len(ends):
        errors.append(
            f"{where}: z_start has {len(starts)} entries and z_end {len(ends)}; they "
            "MUST have one entry per timepoint each"
        )
        return errors
    bad = [
        (t, a, b)
        for t, (a, b) in enumerate(zip(starts, ends))
        if not (0 <= a < b and (z_dim is None or b <= z_dim))
    ]
    if bad:
        bound = f" <= z_dim_source={z_dim}" if z_dim is not None else ""
        errors.append(
            f"{where}: plane range(s) {bad[:5]} (t, z_start, z_end) are invalid; each "
            f"MUST satisfy 0 <= z_start < z_end{bound} (z_end is exclusive)"
        )
    if method == "slice":
        wide = [(t, b - a) for t, (a, b) in enumerate(zip(starts, ends)) if b - a != 1]
        if wide:
            errors.append(
                f"{where}: method slice selects one plane, so z_end - z_start MUST be 1; "
                f"got (t, planes) {wide[:5]}"
            )
    return errors


class ProjectionChannelV0_3(BaseModel):
    """``channels.<name>`` — how one output channel was projected."""

    model_config = ConfigDict(extra="allow")

    method: ProjectionMethod
    source_channel: str | None = Field(default=None, min_length=1)
    z_start: int | list[int] | None = None
    z_end: int | list[int] | None = None
    z_step: int | None = Field(default=None, ge=1)


class ProjectionFocusV0_3(BaseModel):
    """``dca.projection.focus`` — how the in-focus plane used to choose planes was found.

    ``z`` / ``z_per_t`` entries are ``None`` (JSON ``null``) where no in-focus plane was
    found; the field itself must still be present when ``per`` requires it.
    """

    model_config = ConfigDict(extra="allow")

    channel: str = Field(min_length=1)
    per: Literal["fov", "timepoint"]
    z: int | None = Field(default=None, ge=0)
    z_per_t: list[int | None] | None = None
    parameters: dict | None = None

    @model_validator(mode="after")
    def check_planes_recorded(self) -> "ProjectionFocusV0_3":
        if self.per == "fov" and "z" not in self.model_fields_set:
            raise ValueError(
                "focus.per is 'fov' but focus.z, the in-focus plane (null if none was "
                "found), is missing"
            )
        if self.per == "timepoint" and self.z_per_t is None:
            raise ValueError(
                "focus.per is 'timepoint' but focus.z_per_t, the in-focus plane of each "
                "timepoint (null where none was found), is missing"
            )
        if self.z_per_t is not None and any(
            z is not None and z < 0 for z in self.z_per_t
        ):
            raise ValueError("focus.z_per_t entries MUST be plane indices >= 0 or null")
        return self

    def planes(self) -> list[int]:
        """The in-focus planes that were found (nulls left out)."""
        found = [self.z] if self.z is not None else []
        return found + [z for z in (self.z_per_t or []) if z is not None]


class DCAProjectionV0_3(BaseModel):
    """``dca.projection`` — schema.json ``$defs/Projection``, uniform or per-channel."""

    model_config = ConfigDict(extra="allow")

    # projections.rst § Projection block — shared
    # MUST when the 3D source is in the DCA, which only the cross-store pass can see
    # (pairing.py); omitted for an image projected before deposit.
    source_store: str | None = Field(default=None, min_length=1)
    z_dim_source: int | None = Field(default=None, ge=1)
    focus: ProjectionFocusV0_3 | None = None

    # Uniform form
    method: ProjectionMethod | None = None
    z_start: int | None = None
    z_end: int | None = None
    z_step: int | None = Field(default=None, ge=1)
    z_fraction: float | None = Field(default=None, gt=0, le=1)

    # Per-channel form
    channels: dict[str, ProjectionChannelV0_3] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def check_form(self) -> "DCAProjectionV0_3":
        errors: list[str] = []
        if (self.method is None) == (self.channels is None):
            errors.append(
                "dca.projection MUST have exactly one of 'method' (uniform form) and "
                "'channels' (per-channel form) (projections.rst § Projection block)"
            )
        if self.channels is not None:
            stray = [f for f in _UNIFORM_ONLY_FIELDS if getattr(self, f) is not None]
            if stray:
                errors.append(
                    f"{stray} belong to the uniform form; in the per-channel form each "
                    "channel's planes go in channels.<name>"
                )
        ranges = []
        if self.method is not None:
            ranges.append(
                ("dca.projection", self.method, self.z_start, self.z_end, self.z_step)
            )
        for name, c in (self.channels or {}).items():
            ranges.append((f"channels.{name}", c.method, c.z_start, c.z_end, c.z_step))
        if self.z_dim_source is None and any(r[2] is not None for r in ranges):
            errors.append(
                "z_dim_source MUST be given when any z_start / z_end is "
                "(projections.rst § Projection block)"
            )
        for where, method, z0, z1, step in ranges:
            errors.extend(
                _plane_range_errors(where, method, z0, z1, step, self.z_dim_source)
            )
        if self.focus is not None and self.z_dim_source is not None:
            outside = sorted({z for z in self.focus.planes() if z >= self.z_dim_source})
            if outside:
                errors.append(
                    f"focus plane(s) {outside} are outside the {self.z_dim_source}-plane "
                    "source; in-focus planes MUST lie within the source stack "
                    "(projections.rst § Focus)"
                )
        if errors:
            raise ValueError("\n".join(errors))
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


def projection_methods(projection: dict) -> set[str]:
    """The methods a raw ``dca.projection`` dict uses (either form); unknown values kept."""
    channels = projection.get("channels")
    if isinstance(channels, dict):
        return {c.get("method") for c in channels.values() if isinstance(c, dict)} - {
            None
        }
    method = projection.get("method")
    return {method} if isinstance(method, str) else set()


def _listed_steps(dca: dict) -> list[str] | None:
    image_data = dca.get("image_data")
    steps = (
        image_data.get("processing_variant") if isinstance(image_data, dict) else None
    )
    # A bare string is a type error reported by the experimental-metadata layer;
    # still read it so this check does not add a second, confusing error.
    if isinstance(steps, str):
        return [steps]
    return steps if isinstance(steps, list) else None


def check_projection_v0_3(
    node_dict: dict, dca: object, group_attrs: dict
) -> "list[Issue]":
    """Projection rules that relate ``dca.projection`` to the rest of an image group.

    ``node_dict`` is the dict built by ``validator._build_node_dict``; ``group_attrs``
    is the group's full ``zarr.json`` attributes (for the top-level ``mip`` check).
    Structural errors inside the block itself are reported by ``DCAProjectionV0_3``;
    this function tolerates a malformed block and checks what it can.
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
    steps = _listed_steps(dca) or []
    listed = [s for s in steps if s in PROJECTION_STEPS]

    if listed and not has_block:
        # Allowed for an image projected before deposit; an error only when its 3D
        # source is in the dataset, which pairing.py checks across stores.
        warning(
            ("dca", "projection"),
            f"image_data.processing_variant lists projection steps {listed} but there "
            "is no dca.projection. If the 3D source is part of this dataset, the "
            "projection MUST carry dca.projection; if it was projected before deposit, "
            "this can be ignored, or dca.projection can record known parameters "
            "(projections.rst § Projections made before deposit).",
        )
    if has_block and not listed:
        error(
            ("dca", "image_data", "processing_variant"),
            "dca.projection is present but image_data.processing_variant lists no "
            "projection step; a projection MUST list the step of each method it uses, "
            "and an image that is not a projection MUST NOT carry dca.projection.",
        )
    if not isinstance(projection, dict):
        return issues

    methods = projection_methods(projection)
    expected = {
        PROJECTION_STEP_BY_METHOD[m] for m in methods if m in PROJECTION_STEP_BY_METHOD
    }
    if listed and expected and set(listed) != expected:
        error(
            ("dca", "image_data", "processing_variant"),
            f"processing_variant lists projection steps {sorted(set(listed))} but the "
            f"methods used ({sorted(methods)}) call for exactly {sorted(expected)} "
            "(projections.rst § Processing variant).",
        )

    axes = [a.lower() for a in node_dict.get("axes", [])]
    shapes = node_dict.get("array_shapes", [])
    if "z" in axes:
        zi = axes.index("z")
        bad = [
            (i, sh[zi]) for i, sh in enumerate(shapes) if len(sh) > zi and sh[zi] != 1
        ]
        if bad:
            error(
                ("levels",),
                f"A projection MUST have z of size 1 at every level; got "
                f"(level, z) = {bad} (projections.rst § Dimensions and Scale).",
            )
    t_size = shapes[0][axes.index("t")] if shapes and "t" in axes else None

    channels = projection.get("channels")
    if isinstance(channels, dict):
        names = [
            c.get("name") for c in dca.get("channels") or [] if isinstance(c, dict)
        ]
        extra = sorted(k for k in channels if k not in names)
        if extra:
            error(
                ("dca", "projection", "channels"),
                f"dca.projection.channels keys {extra} are not channels of this image "
                f"({names}); every key MUST be one of its dca.channels names "
                "(projections.rst § Per-channel form).",
            )
        if t_size is not None:
            for name, c in channels.items():
                if not isinstance(c, dict):
                    continue
                for f in ("z_start", "z_end"):
                    v = c.get(f)
                    if isinstance(v, list) and len(v) != t_size:
                        error(
                            ("dca", "projection", "channels", name, f),
                            f"channels.{name}.{f} has {len(v)} entries but the level-0 "
                            f"array has T={t_size}; a per-timepoint list MUST have one "
                            "entry per timepoint.",
                        )

    focus = projection.get("focus")
    z_per_t = focus.get("z_per_t") if isinstance(focus, dict) else None
    if isinstance(z_per_t, list) and t_size is not None and len(z_per_t) != t_size:
        error(
            ("dca", "projection", "focus", "z_per_t"),
            f"focus.z_per_t has {len(z_per_t)} entries but the level-0 array has "
            f"T={t_size}; it MUST have one plane per timepoint.",
        )

    if projection.get("method") is not None:
        zd, z0, z1 = (projection.get(f) for f in ("z_dim_source", "z_start", "z_end"))
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

    ranges = [((), projection)]
    if isinstance(channels, dict):
        ranges += [
            (("channels", n), c) for n, c in channels.items() if isinstance(c, dict)
        ]
    for sub, spec in ranges:
        loose = _loose_z_ends(
            spec.get("z_start"), spec.get("z_end"), spec.get("z_step")
        )
        if loose:
            prefix = ".".join(sub) + "." if sub else ""
            warning(
                ("dca", "projection", *sub, "z_end"),
                f"{prefix}z_end should be the last plane used + 1, so each set of "
                f"planes has one spelling; (t, z_end, expected) = {loose[:5]} "
                "(projections.rst § Plane ranges).",
            )
    return issues


def _loose_z_ends(
    z_start: object, z_end: object, z_step: object
) -> "list[tuple[int, int, int]]":
    """Valid ranges whose ``z_end`` is not the last plane used + 1, as
    ``(t, z_end, expected)``. With a step, any ``z_end`` in (last, last + step] selects
    the same planes; only last + 1 is the recommended spelling."""
    if not isinstance(z_step, int) or isinstance(z_step, bool) or z_step <= 1:
        return []
    if isinstance(z_start, list) and isinstance(z_end, list):
        pairs = list(zip(z_start, z_end))
    elif isinstance(z_start, int) and isinstance(z_end, int):
        pairs = [(z_start, z_end)]
    else:
        return []
    loose = []
    for t, (a, b) in enumerate(pairs):
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b):
            continue  # invalid ranges are reported by DCAProjectionV0_3
        expected = a + (b - 1 - a) // z_step * z_step + 1
        if b != expected:
            loose.append((t, b, expected))
    return loose
