import pytest

from dca_helpers.metadata import STORE_COLUMNS, name_date


@pytest.mark.parametrize(
    "name, expected",
    [
        ("2026_09_25_A549_PCNA_DENV_ZIKV", "2026-09-25"),
        ("screen0056_20250721", "2025-07-21"),
        ("screen9004", None),
        ("livescreen_20260626_190942", None),
        ("run_20251399", None),
        ("2026_13_40_x", None),
        ("", None),
    ],
)
def test_name_date(name, expected):
    assert name_date(name) == expected


def test_store_columns_are_the_four_reserved_names():
    assert STORE_COLUMNS == {
        "store_dataset_name",
        "store_group_id",
        "store_name_date",
        "store_time_increment_s",
    }
