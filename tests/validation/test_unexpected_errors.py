import logging
import threading

import pytest

from dca_helpers.validation import validator
from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate
from dca_helpers.validation.versions.v0_2.models import DCAStoreSpecV0_2

FIXTURES = __import__("pathlib").Path(__file__).parent.parent / "fixtures"


class ExplodingModel:
    @classmethod
    def model_validate(cls, data):
        raise TypeError("boom")


@pytest.mark.unit
class TestRunDcaValidationGuard:
    def test_unexpected_model_error_becomes_one_error_issue(self):
        issues = validator._run_dca_validation({}, None, "0.2", ExplodingModel, False)
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR
        assert issues[0].loc == ("model validation",)
        assert "TypeError" in issues[0].message
        assert "boom" in issues[0].message

    def test_unexpected_metadata_error_keeps_the_model_issues(self, monkeypatch):
        def explode(dca, check_ontology):
            raise RuntimeError("bad metadata")

        monkeypatch.setattr(
            validator, "validate_dca_experimental_metadata_v0_2", explode
        )
        issues = validator._run_dca_validation({}, {}, "0.2", ExplodingModel, False)
        assert [i.loc for i in issues] == [
            ("model validation",),
            ("experimental metadata",),
        ]
        assert "RuntimeError" in issues[1].message


@pytest.mark.unit
def test_unexpected_error_is_logged_with_its_traceback(caplog):
    with caplog.at_level(logging.WARNING, logger=validator.logger.name):
        validator._run_dca_validation({}, None, "0.2", ExplodingModel, False)
    records = [r for r in caplog.records if "Unexpected TypeError" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info[0] is TypeError


@pytest.mark.functional
class TestPlateSurvivesAnUnexpectedFieldError:
    def test_one_failing_field_does_not_abort_the_plate(self, monkeypatch):
        original = DCAStoreSpecV0_2.model_validate
        lock = threading.Lock()
        calls = []

        def fail_the_first_field(data, *args, **kwargs):
            with lock:
                calls.append(1)
                first = len(calls) == 1
            if first:
                raise TypeError("malformed field")
            return original(data, *args, **kwargs)

        monkeypatch.setattr(DCAStoreSpecV0_2, "model_validate", fail_the_first_field)
        run = validate(
            str(FIXTURES / "v0_2/valid_plate_multiwell.ome.zarr"),
            spec_version="0.2",
            show_progress=False,
        )
        failed = [
            r
            for r in run.results
            if any("Unexpected TypeError" in i.message for i in r.issues)
        ]
        assert len(calls) > 1
        assert len(failed) == 1
        assert not failed[0].passed
        assert any(r.passed for r in run.results if r is not failed[0])
