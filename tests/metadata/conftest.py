"""Shared helpers for the DCA metadata test suite."""

import copy
import importlib
import sys

import pytest

from dca_helpers.validation import ontology_lookup

from ._registry import RegistryReplay

#: The canonical minimal valid DCA v0.2 block in ``[dca.*]`` shape
#: (U2OS / CVCL:0042). Never handed out directly (use :func:`min_dca_block`).
_DCA_MIN = {
    "study": {
        "source": "Lab",
        "study_type": ["Screen"],
        "description": "d",
        "year_imaged": "2020",
        "experimenter": [{"first_name": "A", "last_name": "B"}],
        "dataset_name": "D",
        "release_date": "2020-01-01",
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
    },
    "biosample": {
        "group_id": "test",
        "organism": [{"ontology_term_id": "NCBITaxon:9606", "label": "Homo sapiens"}],
        "tissue_type": "cell line",
        "cell_line_name": "U2OS",
        "tissue": [{"ontology_term_id": "CVCL:0042", "label": "U2OS"}],
        "disease": [{"ontology_term_id": "MONDO:0009807", "label": "osteosarcoma"}],
        "development_stage": [{"ontology_term_id": "unknown", "label": "unknown"}],
    },
    "specimen": {"specimen_state": "fixed"},
    # `[dca.*]` injects nothing, so a config authors `provenance` itself —
    # empty is valid.
    "provenance": {},
    "study_component": {
        "imaging_method": [
            {"ontology_term_id": "FBbi:00000246", "label": "fluorescence microscopy"}
        ],
        "assay": [{"ontology_term_id": "EFO:0007550", "label": "high content screen"}],
    },
}


def min_dca_block(**sections) -> dict:
    """A fresh copy of the canonical minimal valid block.

    Keyword arguments replace (or add) top-level sections, e.g.
    ``min_dca_block(perturbation={...})``.
    """
    block = copy.deepcopy(_DCA_MIN)
    block.update(sections)
    return block


@pytest.fixture
def replay_registry(monkeypatch) -> RegistryReplay:
    """Serve recorded registry responses wherever ``propose`` makes an HTTP call."""
    replay = RegistryReplay()
    for name in ("propose", "perturbation_ids"):
        module = importlib.import_module(f"dca_helpers.metadata.{name}")
        monkeypatch.setattr(module, "_get_with_retry", replay)
    monkeypatch.setattr(ontology_lookup, "_get_with_retry", replay)
    ontology_lookup.lookup_ols.cache_clear()
    yield replay
    ontology_lookup.lookup_ols.cache_clear()


@pytest.fixture
def hide_rdkit(monkeypatch) -> None:
    """Make ``import rdkit`` fail, as it does without the ``chem`` extra."""
    for name in ("rdkit", "rdkit.Chem", "rdkit.DataStructs"):
        monkeypatch.setitem(sys.modules, name, None)
