"""
Unit tests for ZarrNodeType, classify_group, and classify_array.
"""

from unittest.mock import MagicMock

import pytest
from ome_zarr_models.v05.hcs import HCS
from ome_zarr_models.v05.image import Image
from ome_zarr_models.v05.image_label import ImageLabel
from ome_zarr_models.v05.labels import Labels

from dca_helpers.validation.zarr_node import ZarrNodeType, classify_group

# ---------------------------------------------------------------------------
# classify_group — mutual exclusivity check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyGroupMutualExclusivity:
    def test_plate_and_multiscales_raises(self):
        raw_attrs = {"plate": {}, "multiscales": [{}]}
        ome_obj = MagicMock()
        with pytest.raises(ValueError, match="both 'plate' and 'multiscales'"):
            classify_group("s3://bucket/store", ome_obj, raw_attrs)

    def test_only_plate_ok(self):
        raw_attrs = {"plate": {}}
        ome_obj = MagicMock(spec=HCS)
        result = classify_group("/store", ome_obj, raw_attrs)
        assert result == ZarrNodeType.HCS_PLATE

    def test_only_multiscales_ok(self):
        raw_attrs = {"multiscales": [{}]}
        ome_obj = MagicMock(spec=Image)
        result = classify_group("/store", ome_obj, raw_attrs)
        assert result == ZarrNodeType.IMAGE

    def test_empty_attrs_ok(self):
        raw_attrs = {}
        ome_obj = MagicMock(spec=Image)
        result = classify_group("/store", ome_obj, raw_attrs)
        assert result == ZarrNodeType.IMAGE


# ---------------------------------------------------------------------------
# classify_group — type dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyGroupDispatch:
    def test_hcs_returns_hcs_plate(self):
        ome_obj = MagicMock(spec=HCS)
        result = classify_group("/store", ome_obj, {})
        assert result == ZarrNodeType.HCS_PLATE

    def test_image_label_returns_image_label(self):
        ome_obj = MagicMock(spec=ImageLabel)
        result = classify_group("/store", ome_obj, {"multiscales": [{}]})
        assert result == ZarrNodeType.IMAGE_LABEL

    def test_labels_returns_labels_list(self):
        ome_obj = MagicMock(spec=Labels)
        result = classify_group("/store", ome_obj, {"labels": []})
        assert result == ZarrNodeType.LABELS_LIST

    def test_image_returns_image(self):
        ome_obj = MagicMock(spec=Image)
        result = classify_group("/store", ome_obj, {"multiscales": [{}]})
        assert result == ZarrNodeType.IMAGE

    def test_unknown_type_raises(self):
        class WeirdOMEObj:
            pass

        ome_obj = WeirdOMEObj()
        with pytest.raises(ValueError, match="Unrecognised"):
            classify_group("/store", ome_obj, {})
