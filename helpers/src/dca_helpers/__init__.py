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


def _spec_version(version: str) -> str:
    """The DCA spec version — package version truncated to ``major.minor``."""
    return ".".join(version.split(".")[:2])


CURRENT_SPEC_VERSION = _spec_version(__version__)

__all__ = ["__version__", "CURRENT_SPEC_VERSION"]
