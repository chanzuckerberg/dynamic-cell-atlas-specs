import pytest

from dca_helpers.metadata.propose import search_cellosaurus


@pytest.mark.live
def test_live_a549():
    assert search_cellosaurus("A549")[0] == ("CVCL:0023", "A549")


@pytest.mark.live
def test_live_hek293_is_the_parent_line():
    assert search_cellosaurus("HEK293")[0][0] == "CVCL:0045"
