"""
Pytest configuration and shared fixtures for documentation tests.
"""

import sys
from pathlib import Path

import pytest

# Add docs directory to Python path so we can import conf.py
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"
sys.path.insert(0, str(DOCS_DIR))


@pytest.fixture(scope="session")
def docs_dir():
    """Path to the documentation directory."""
    return DOCS_DIR


@pytest.fixture(scope="session")
def root_dir():
    """Path to the repository root directory."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def sphinx_config():
    """Import and return the Sphinx configuration."""
    import conf

    return conf


@pytest.fixture(scope="session")
def rst_files(docs_dir):
    """List of all RST files in the documentation."""
    return list(docs_dir.rglob("*.rst"))
