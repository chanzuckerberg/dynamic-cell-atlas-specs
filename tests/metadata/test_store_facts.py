from dca_helpers.metadata import STORE_COLUMNS


def test_store_columns_are_the_four_reserved_names():
    assert STORE_COLUMNS == {
        "store_dataset_name",
        "store_group_id",
        "store_name_date",
        "store_time_increment_s",
    }
