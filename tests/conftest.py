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

# Valid perturbation identifiers shared by the validation and metadata tests.
# Aspirin's fingerprint was computed once with RDKit 2026.09.1 using the parameters
# the schema pins: GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
# on "CC(=O)OC1=CC=CC=C1C(=O)O", encoded with DataStructs.BitVectToFPSText.
ASPIRIN_PUBCHEM_CID = "PubChem:2244"
ASPIRIN_MORGAN_FINGERPRINT = (
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000000000000020000000000000000001000000000000"
    "0000000000000000000000000000000000040000000080000000000000000000"
    "0000000080000000000000000000000000200000000000000000000000000002"
    "0008800002000000010000000000000000000000008000000000000000000000"
    "0000000000000000000000001000000004000000800000100000000000000000"
    "0000000000000000000001000000000000000000000000000200400000800000"
    "0000000000000000000002000000002000000000000004008000000000000000"
)
TP53_ENSEMBL_GENE_ID = "ensembl:ENSG00000141510"


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


def pytest_addoption(parser):
    """Add --live, which opts in to tests that hit real external APIs.

    Everything else in the suite mocks its transport, so the suite is
    deterministic and offline. That leaves nobody watching the real services:
    a mocked test still passes after OLS changes its response shape or an
    ontology renames a term. The `live` tests close that gap, and they are
    opt-in because CI should not fail on someone else's outage.
    """
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests marked `live`, which make real network calls to "
        "external ontology registries (OLS4, Cellosaurus).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `live`-marked tests unless --live was passed."""
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="needs --live (makes real network calls)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
