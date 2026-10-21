import pytest

from dca_helpers.metadata.propose import search_cellosaurus, search_ncbitaxon


@pytest.mark.live
def test_live_a549_and_zika():
    assert search_cellosaurus("A549")[0] == ("CVCL:0023", "A549")
    assert search_ncbitaxon("Zika virus")[0][0] == "NCBITaxon:64320"
