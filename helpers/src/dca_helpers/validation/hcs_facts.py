"""Versioned metadata evidence for HCS consistency (never image pixels)."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FACTS_VERSION = "1.0"


def canonical_json(value: object) -> str:
    """Stable JSON encoding, not a definition of biological equivalence."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metadata_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class _Facts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    def evidence_json(self) -> str | None:
        """Do not let ancillary evidence serialization break ordinary validation.

        Non-JSON/nonfinite metadata cannot support a trustworthy persisted
        rollup, but existing validation should still return its own findings.
        """
        try:
            return canonical_json(self.model_dump())
        except (TypeError, ValueError):
            return None


class HCSFieldFacts(_Facts):
    format_version: Literal["1.0"] = FACTS_VERSION
    kind: Literal["field"] = "field"
    read_success: bool
    axes: list[str] | None = None
    chunk_shape: list[int] | None = None
    perturbation: Any = None


class HCSWellFacts(_Facts):
    metadata_digest: str
    expected_fields: list[str]
    inventory_complete: bool
    perturbation: Any = None


class HCSPlateFacts(_Facts):
    format_version: Literal["1.0"] = FACTS_VERSION
    kind: Literal["plate"] = "plate"
    metadata_digest: str
    declared_wells: list[str]
    inventory_complete: bool
    wells: dict[str, HCSWellFacts]


class HCSManifest(_Facts):
    """Explicit coverage contract for one immutable plate snapshot.

    snapshot_id identifies a storage snapshot/revision supplied by the caller.
    A timestamp or matching token alone cannot freeze a mutable dataset.
    Root/well fingerprints detect metadata drift, but do not independently
    establish immutability of disjoint fields. The caller must arrange that.
    """

    format_version: Literal["1.0"] = FACTS_VERSION
    plate_path: str
    snapshot_id: str
    spec_version: str
    validator_version: str
    plate: HCSPlateFacts
