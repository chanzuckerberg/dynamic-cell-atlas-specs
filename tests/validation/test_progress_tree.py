from unittest.mock import MagicMock

import pytest

from dca_helpers.validation.progress_tree import _TreeDisplay
from dca_helpers.validation.result import ZarrNodeValidationResult
from dca_helpers.validation.zarr_node import ZarrNodeType


def _result(path, parent, node_type):
    return ZarrNodeValidationResult(
        node_path=path,
        spec_version="0.2",
        passed=True,
        parent_path=parent,
        node_type=node_type,
    )


@pytest.mark.unit
def test_label_results_are_not_counted_as_fields():
    display = _TreeDisplay("/root", 1)
    live = MagicMock()
    display.set_live(live)
    display.start_store("/root/plate")
    display.complete_store(
        "/root/plate",
        [
            _result("/root/plate", None, ZarrNodeType.HCS_PLATE),
            _result("/root/plate/A/1/0", "/root/plate", ZarrNodeType.IMAGE),
            _result(
                "/root/plate/A/1/0/labels/n",
                "/root/plate/A/1/0",
                ZarrNodeType.IMAGE_LABEL,
            ),
        ],
    )
    printed = str(live.console.print.call_args[0][0])
    assert "1 fields" in printed
    assert "1 labels" in printed
