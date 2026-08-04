"""
Maps DCA spec version strings to their root Pydantic model class.

Two registries exist — one for image groups, one for label image groups.
To add support for a new spec version:
  1. Create src/validation/versions/vX_Y/models.py with DCAStoreSpecVX_Y
     and DCALabelStoreSpecVX_Y classes.
  2. Add one entry to _REGISTRY and one to _LABEL_REGISTRY below.
  No other files change.
"""

from pydantic import BaseModel

from dca_helpers.validation.versions.v0_1.models import DCAStoreSpecV0_1
from dca_helpers.validation.versions.v0_2.models import (
    DCALabelStoreSpecV0_2,
    DCAStoreSpecV0_2,
)


class UnsupportedSpecVersionError(ValueError):
    def __init__(self, version: str, *, label: bool = False) -> None:
        registry = _LABEL_REGISTRY if label else _REGISTRY
        supported = sorted(registry)
        kind = "label " if label else ""
        super().__init__(
            f"DCA spec version '{version}' is not supported for {kind}validation. "
            f"Supported versions: {supported}"
        )


_REGISTRY: dict[str, type[BaseModel]] = {
    "0.1": DCAStoreSpecV0_1,
    "0.2": DCAStoreSpecV0_2,
}

# Label image model registry.
_LABEL_REGISTRY: dict[str, type[BaseModel]] = {
    "0.2": DCALabelStoreSpecV0_2,
}


def get_model(spec_version: str) -> type[BaseModel]:
    """Return the DCAStoreSpec model class for the given spec version string."""
    if spec_version not in _REGISTRY:
        raise UnsupportedSpecVersionError(spec_version)
    return _REGISTRY[spec_version]


def get_label_model(spec_version: str) -> type[BaseModel]:
    """Return the DCALabelStoreSpec model class for the given spec version string."""
    if spec_version not in _LABEL_REGISTRY:
        raise UnsupportedSpecVersionError(spec_version, label=True)
    return _LABEL_REGISTRY[spec_version]
