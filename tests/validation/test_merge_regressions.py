"""Integration regressions for HCS rollup and upstream projection validation."""

from unittest.mock import MagicMock

import obstore.exceptions as obs_exc
import pytest
from click.testing import CliRunner

from dca_helpers.validation.cli import main
from dca_helpers.validation.runner import validate
from dca_helpers.validation.validator import _gather_plate_fields

from .projection_fixtures import make_pair


@pytest.mark.parametrize("stage", ["opening", "listing"])
def test_s3_failure_retains_six_value_well_contract(monkeypatch, stage):
    group = MagicMock()
    group.attrs = {"ome": {"well": {"images": [{"path": "0"}]}}}
    if stage == "listing":
        group.keys.side_effect = obs_exc.GenericError("slow down")

    def open_group(*args, **kwargs):
        if stage == "opening":
            raise obs_exc.GenericError("slow down")
        return group

    monkeypatch.setattr("dca_helpers.validation.validator.open_zarr_group", open_group)
    facts = {}
    issues, fields, selected, perturbations = _gather_plate_fields(
        "/p", [{"path": "A/1"}], spec_version="0.3", well_facts=facts
    )
    assert [i.loc for i in issues] == [("s3_io",)]
    assert stage in issues[0].message
    assert fields == selected == []
    assert perturbations == facts == {}


@pytest.mark.functional
def test_snapshot_and_projection_evidence_coexist(tmp_path):
    make_pair(tmp_path)
    run = validate(
        str(tmp_path),
        spec_version="0.3",
        show_progress=False,
        require_projection=True,
        snapshot_id="merge-regression",
    )
    assert run.summary.stores_failed == 0
    assert run.snapshot_id == "merge-regression"
    fields = [r for r in run if r.image_facts is not None]
    assert fields
    assert all(r.hcs_facts is not None for r in fields)


@pytest.mark.functional
def test_cli_passes_both_snapshot_and_projection(tmp_path):
    make_pair(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            str(tmp_path),
            "--spec-version",
            "0.3",
            "--require-projection",
            "--snapshot-id",
            "merge-regression",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
