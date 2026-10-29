"""Dynamic Cell Atlas helper utilities."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path


def _read_version() -> str:
    try:
        return _dist_version("dca_helpers")
    except PackageNotFoundError:
        version_file = Path(__file__).resolve().parents[3] / "CURRENT_VERSION"
        return version_file.read_text().strip() if version_file.is_file() else "0.0.0"


__version__ = _read_version()


#: The spec version the validator uses when none is given. Pinned rather than read
#: from the package version: v0.3 is released but still being developed, and the
#: metadata engine writes v0.2 blocks, so the default stays 0.2 until v0.3 is ready.
CURRENT_SPEC_VERSION = "0.2"

__all__ = ["__version__", "CURRENT_SPEC_VERSION"]
