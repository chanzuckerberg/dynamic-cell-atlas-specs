from pathlib import Path

import pytest

from dca_helpers.metadata import load_dca_config, resolve_dca


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_later_file_wins_at_a_scalar_leaf(tmp_path):
    base = _write(
        tmp_path,
        "base.toml",
        '[dca.study]\nlicense = "CC-BY-4.0"\ndataset_name = { column = "store_dataset_name" }\n',
    )
    over = _write(tmp_path, "over.toml", '[dca.study]\nlicense = "CC0-1.0"\n')
    assert load_dca_config(base, over)["study"] == {
        "license": "CC0-1.0",
        "dataset_name": {"column": "store_dataset_name"},
    }


def test_a_literal_replaces_a_column_leaf_whole(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    over = _write(tmp_path, "o.toml", '[dca.biosample]\ncell_line_name = "A549"\n')
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == "A549"


def test_a_value_leaf_replaces_a_column_leaf_whole(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    over = _write(
        tmp_path, "o.toml", '[dca.biosample]\ncell_line_name = { value = "A549" }\n'
    )
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == {
        "value": "A549"
    }


def test_a_column_leaf_replaces_a_value_leaf_whole(tmp_path):
    base = _write(
        tmp_path, "b.toml", '[dca.biosample]\ncell_line_name = { value = "A549" }\n'
    )
    over = _write(
        tmp_path,
        "o.toml",
        '[dca.biosample]\ncell_line_name = { column = "cell_line" }\n',
    )
    assert load_dca_config(base, over)["biosample"]["cell_line_name"] == {
        "column": "cell_line"
    }


def test_an_array_of_tables_is_replaced_not_appended(tmp_path):
    base = _write(
        tmp_path,
        "b.toml",
        '[[dca.study.experimenter]]\nname = "A"\n[[dca.study.experimenter]]\nname = "B"\n',
    )
    over = _write(tmp_path, "o.toml", '[[dca.study.experimenter]]\nname = "C"\n')
    assert load_dca_config(base, over)["study"]["experimenter"] == [{"name": "C"}]


def test_a_quoted_channel_key_survives(tmp_path):
    path = _write(
        tmp_path,
        "c.toml",
        '[dca.channels."raw GFP EX488 EM525-45"]\ndescription = "d"\n',
    )
    assert "raw GFP EX488 EM525-45" in load_dca_config(path)["channels"]


def test_a_file_without_a_dca_table_is_an_error(tmp_path):
    (tmp_path / "a.toml").write_text("[other]\nx = 1\n")
    with pytest.raises(ValueError, match=r"a\.toml.*\[dca\]"):
        load_dca_config(tmp_path / "a.toml")


def test_inputs_are_not_mutated_across_calls(tmp_path):
    base = _write(tmp_path, "b.toml", '[dca.study]\nlicense = "CC-BY-4.0"\n')
    over = _write(tmp_path, "o.toml", '[dca.study]\nlicense = "CC0-1.0"\n')
    first = load_dca_config(base, over)
    first["study"]["license"] = "changed"
    second = load_dca_config(base, over)
    assert second == {"study": {"license": "CC0-1.0"}}
    assert first is not second


_BASE_PROVENANCE = """
[dca.study]
release_date = "2026-01-15"

[dca.provenance.study.release_date]
source_kind = "agent"
citation = { source_url = "dataset name", quote = "from the name", match_status = "exact" }
"""

_CONFIRMED_RECORD = (
    'source_kind = "human"\n'
    'citation = { source_url = "curator note", quote = "checked", match_status = "exact" }\n'
)

_CONFIRMED = {
    "quoted": '[dca.provenance."study.release_date"]\n' + _CONFIRMED_RECORD,
    "nested": "[dca.provenance.study.release_date]\n" + _CONFIRMED_RECORD,
}


@pytest.mark.parametrize("spelling", sorted(_CONFIRMED))
def test_a_confirmed_provenance_record_replaces_the_base_record_whole(
    tmp_path, spelling
):
    base = _write(tmp_path, "base.toml", _BASE_PROVENANCE)
    over = _write(tmp_path, "confirmed.toml", _CONFIRMED[spelling])
    block = load_dca_config(base, over)
    confirmed = {
        "source_kind": "human",
        "citation": {
            "source_url": "curator note",
            "quote": "checked",
            "match_status": "exact",
        },
    }
    assert block["provenance"] == {"study.release_date": confirmed}
    assert resolve_dca(block, {})["provenance"] == {"study.release_date": confirmed}


def test_a_confirmed_record_drops_base_keys_it_does_not_repeat(tmp_path):
    base = _write(tmp_path, "base.toml", _BASE_PROVENANCE)
    over = _write(
        tmp_path,
        "confirmed.toml",
        '[dca.provenance.study.release_date]\nsource_kind = "human"\n',
    )
    block = load_dca_config(base, over)
    assert resolve_dca(block, {})["provenance"] == {
        "study.release_date": {"source_kind": "human"}
    }


def test_provenance_from_one_file_is_keyed_by_dotted_path(tmp_path):
    base = _write(tmp_path, "base.toml", _BASE_PROVENANCE)
    assert list(load_dca_config(base)["provenance"]) == ["study.release_date"]


def test_one_path_spelled_both_ways_in_one_file_is_still_an_error(tmp_path):
    path = _write(
        tmp_path,
        "both.toml",
        _BASE_PROVENANCE
        + '\n[dca.provenance."study.release_date"]\nsource_kind = "human"\n',
    )
    with pytest.raises(ValueError, match="authored twice"):
        load_dca_config(path)
