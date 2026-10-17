import json
import shutil
from pathlib import Path

import pytest

from dca_helpers.validation.result import Severity
from dca_helpers.validation.runner import validate
from dca_helpers.validation.zarr_node import ZarrNodeType

FIXTURES = Path(__file__).parent.parent / "fixtures"
PYRAMID_V02 = FIXTURES / "v0_2/valid_two_level_pyramid.ome.zarr"
SINGLE_V01 = FIXTURES / "v0_1/valid_single_image.ome.zarr"
MULTIWELL_V02 = FIXTURES / "v0_2/valid_plate_multiwell.ome.zarr"


def _write(path: Path, doc: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(json.dumps(doc))


def _group(ome: dict) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": {"version": "0.5", **ome}},
    }


def add_labels(
    image_dir: Path,
    names: list[str],
    *,
    create: "list[str] | None" = None,
    dtype: str = "uint32",
    image_label: bool = True,
    chunk_shape: "list[int] | None" = None,
    extra_dirs: "list[str]" = (),
) -> None:
    """Add labels/ to an image group: ome.labels lists `names`; groups are
    created for `create` (default: all of names), mirroring the image's own
    multiscale and level arrays."""
    image_doc = json.loads((image_dir / "zarr.json").read_text())
    multiscales = image_doc["attributes"]["ome"]["multiscales"]
    level_paths = [d["path"] for d in multiscales[0]["datasets"]]
    labels_dir = image_dir / "labels"
    _write(labels_dir, _group({"labels": names}))
    for name in names if create is None else create:
        ome = {"multiscales": multiscales}
        if image_label:
            ome["image-label"] = {}
        _write(labels_dir / name, _group(ome))
        for level in level_paths:
            array = json.loads((image_dir / level / "zarr.json").read_text())
            array["data_type"] = dtype
            if chunk_shape is not None:
                array["chunk_grid"]["configuration"]["chunk_shape"] = chunk_shape
            _write(labels_dir / name / level, array)
    for extra in extra_dirs:
        _write(labels_dir / extra, _group({}))


def run(path: Path, spec_version: str = "0.2", **kwargs):
    return validate(str(path), spec_version=spec_version, show_progress=False, **kwargs)


def label_rows(results):
    return [r for r in results if r.node_type == ZarrNodeType.IMAGE_LABEL]


@pytest.fixture
def image_v02(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    image = root / "image.ome.zarr"
    shutil.copytree(PYRAMID_V02, image)
    return root, image


@pytest.mark.functional
class TestImageLabels:
    def test_discovery_run_validates_declared_label(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"])
        results = run(root).results
        (label,) = label_rows(results)
        assert label.node_path == f"{image}/labels/nuclei"
        assert label.parent_path == str(image)
        assert label.passed
        assert all(r.passed for r in results)

    def test_image_without_labels_is_unchanged(self, image_v02):
        root, image = image_v02
        results = run(root).results
        assert len(results) == 1
        assert results[0].node_type == ZarrNodeType.IMAGE

    def test_label_defect_is_reported_on_label_row(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"], chunk_shape=[1, 1, 1, 2, 2])
        results = run(root).results
        (label,) = label_rows(results)
        assert not label.passed
        assert label.parent_path == str(image)

    def test_ngff_level_label_defect_keeps_label_row_dca_findings(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"], dtype="float32")
        results = run(root).results
        assert any(
            r.parent_path is None and r.errors[0].loc == ("ngff_validation",)
            for r in results
        )
        (label,) = label_rows(results)
        assert label.warnings[0].loc[0] == "levels"

    def test_declared_label_without_group_is_error(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei", "ghost"], create=["nuclei"])
        results = label_rows(run(root).results)
        ghost = next(r for r in results if r.node_path.endswith("/ghost"))
        assert not ghost.passed
        assert ghost.errors[0].loc == ("labels", "ghost")
        assert next(r for r in results if r.node_path.endswith("/nuclei")).passed

    def test_listed_label_without_image_label_is_error(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"], image_label=False)
        (label,) = label_rows(run(root).results)
        assert not label.passed
        assert "ome.image-label" in label.errors[0].message

    def test_undeclared_label_dir_is_warning_only(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"], extra_dirs=["stray"])
        results = run(root).results
        (container,) = [r for r in results if r.node_type == ZarrNodeType.LABELS_LIST]
        assert container.passed
        assert [i.severity for i in container.issues] == [Severity.WARNING]
        assert "stray" in container.issues[0].message

    def test_duplicate_label_name_validated_once(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei", "nuclei"], create=["nuclei"])
        assert len(label_rows(run(root).results)) == 1

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", ".."])
    def test_unsafe_label_name_is_error(self, image_v02, bad):
        root, image = image_v02
        add_labels(image, [bad], create=[])
        (row,) = label_rows(run(root).results)
        assert not row.passed
        assert "Invalid label name" in row.errors[0].message

    def test_labels_container_pointed_at_validates_children(self, image_v02):
        _, image = image_v02
        add_labels(image, ["nuclei"])
        results = run(image / "labels").results
        assert any(r.node_type == ZarrNodeType.LABELS_LIST for r in results)
        (label,) = label_rows(results)
        assert label.node_path == f"{image}/labels/nuclei"
        assert label.passed

    def test_label_pointed_at_directly_is_validated_once(self, image_v02):
        _, image = image_v02
        add_labels(image, ["nuclei"])
        results = run(image / "labels" / "nuclei").results
        assert len(results) == 1
        assert results[0].node_type == ZarrNodeType.IMAGE_LABEL
        assert results[0].parent_path is None
        assert results[0].passed

    def test_broken_parent_still_reports_labels(self, image_v02):
        root, image = image_v02
        add_labels(image, ["nuclei"])
        doc = json.loads((image / "zarr.json").read_text())
        doc["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] = "missing"
        (image / "zarr.json").write_text(json.dumps(doc))
        results = run(root).results
        assert any(not r.passed and r.parent_path is None for r in results)
        assert any(r.node_type == ZarrNodeType.IMAGE_LABEL for r in results)

    def test_v01_label_reports_not_supported_warning(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        image = root / "image.ome.zarr"
        shutil.copytree(SINGLE_V01, image)
        add_labels(image, ["nuclei", "cells"])
        results = label_rows(run(root, "0.1").results)
        assert len(results) == 2
        for r in results:
            assert r.passed
            assert "not yet supported" in r.warnings[0].message


@pytest.mark.functional
class TestPlateFieldLabels:
    @pytest.fixture
    def plate(self, tmp_path):
        plate = tmp_path / "plate.ome.zarr"
        shutil.copytree(MULTIWELL_V02, plate)
        return plate

    def test_field_labels_validated_through_plate(self, plate):
        add_labels(plate / "A/1/0", ["nuclei"])
        results = run(plate).results
        (label,) = label_rows(results)
        assert label.node_path == f"{plate}/A/1/0/labels/nuclei"
        assert label.parent_path == f"{plate}/A/1/0"
        assert label.passed

    def test_field_label_defect_fails_run(self, plate):
        add_labels(plate / "B/1/1", ["nuclei"], chunk_shape=[1, 1, 1, 2, 2])
        results = run(plate).results
        (label,) = label_rows(results)
        assert not label.passed

    def test_field_limit_excludes_labels_of_unselected_fields(self, plate):
        add_labels(plate / "A/1/0", ["nuclei"])
        add_labels(plate / "B/1/1", ["nuclei"])
        results = run(plate, field_limit=1).results
        (label,) = label_rows(results)
        assert label.node_path == f"{plate}/A/1/0/labels/nuclei"
