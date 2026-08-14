"""
ZarrNodeType — classification of zarr.json group node types in OME-NGFF v0.5 stores.

Five structurally distinct group schemas are recognised:

  IMAGE       — ome.multiscales present (raw image or label image — identical schema)
  IMAGE_LABEL — ome.multiscales + ome.image-label present
  HCS_PLATE   — ome.plate present
  LABELS_LIST — ome.labels present (container listing label names)
  GEFF        — geff key present at root attrs level (GEFF tracking store, not OME-Zarr)

ome-zarr-models does not enforce mutual exclusivity between ome.plate and
ome.multiscales (BaseOMEAttrs uses extra="allow"), so classify_group performs an
explicit check before dispatching.
"""

from __future__ import annotations

from enum import Enum, auto

from ome_zarr_models.v05.hcs import HCS
from ome_zarr_models.v05.image import Image
from ome_zarr_models.v05.image_label import ImageLabel
from ome_zarr_models.v05.labels import Labels


class ZarrNodeType(Enum):
    IMAGE = auto()  # group with ome.multiscales (raw or label image — same schema)
    IMAGE_LABEL = auto()  # group with ome.multiscales + ome.image-label
    HCS_PLATE = auto()  # group with ome.plate
    LABELS_LIST = auto()  # group with ome.labels (container listing label names)
    GEFF = (
        auto()
    )  # GEFF tracking store (geff key at root attrs); skipped, not validated


def classify_group(path: str, ome_obj, raw_ome_attrs: dict) -> ZarrNodeType:
    """
    Determine ZarrNodeType for a group node returned by open_ome_zarr().

    raw_ome_attrs must be the dict at group.attrs["ome"] read before calling
    open_ome_zarr(), so that the mutual exclusivity check runs against the raw
    metadata rather than the already-dispatched ome-zarr-models object.

    Raises ValueError if both 'plate' and 'multiscales' are present in the
    ome attributes — ome-zarr-models would silently accept this as HCS due to
    extra="allow" on BaseOMEAttrs, masking a malformed store.
    """
    has_plate = "plate" in raw_ome_attrs
    has_multiscales = "multiscales" in raw_ome_attrs

    if has_plate and has_multiscales:
        raise ValueError(
            f"Ambiguous OME metadata at {path}: both 'plate' and 'multiscales' "
            f"are present. A group must be either an HCS plate or a multiscale "
            f"image, not both."
        )

    if isinstance(ome_obj, HCS):
        return ZarrNodeType.HCS_PLATE
    if isinstance(ome_obj, ImageLabel):
        return ZarrNodeType.IMAGE_LABEL
    if isinstance(ome_obj, Labels):
        return ZarrNodeType.LABELS_LIST
    if isinstance(ome_obj, Image):
        return ZarrNodeType.IMAGE

    raise ValueError(
        f"Unrecognised ome-zarr-models type {type(ome_obj).__name__} at {path}"
    )
