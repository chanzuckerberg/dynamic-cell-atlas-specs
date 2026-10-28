import copy

import pydantic
import pytest

from dca_helpers.validation.versions.v0_1 import models as models_v01
from dca_helpers.validation.versions.v0_1.models import (
    DCAStoreSpecV0_1,
    raw_array_rule_messages,
    raw_downsampling_messages,
    raw_level_view,
    raw_shape_views,
)

from .test_models import valid_level, valid_store, validate_level

pytestmark = pytest.mark.unit

BIG_ARRAY = [1, 1, 512, 1024, 1024]
SMALL_SHARD = [1, 1, 128, 128, 128]


def without_prefix(messages):
    return sorted(m.removeprefix("[SHOULD] ") for m in messages)


LEVELS = {
    "valid": valid_level(),
    "chunk_should": valid_level(chunk_shape=[1, 1, 64, 64, 64]),
    "shard_must": valid_level(shape=BIG_ARRAY, shard_shape=SMALL_SHARD),
    "chunk_should_and_shard_must": valid_level(
        shape=BIG_ARRAY, chunk_shape=[1, 1, 64, 64, 64], shard_shape=SMALL_SHARD
    ),
    "channel_chunk": valid_level(chunk_shape=[1, 2, 128, 128, 128]),
    "compression_level": valid_level(codec_level=9),
    "dtype": valid_level(dtype="int32"),
    "downsampled_level": valid_level(level_index=1, chunk_shape=[1, 1, 1, 64, 64]),
}


@pytest.mark.parametrize("level", LEVELS.values(), ids=LEVELS.keys())
def test_raw_array_rules_match_the_model(level):
    expected = without_prefix(i.message for i in validate_level(level))
    assert without_prefix(raw_array_rule_messages(level)) == expected


def test_raw_level_view_lowercases_the_codec():
    view = raw_level_view(valid_level(codec_id="ZSTD"))
    assert view.codec_id == "zstd"


MALFORMED_LEVELS = {
    "none": None,
    "string": "level",
    "list": [],
    "empty_dict": {},
    "shape_string": {**valid_level(), "shape": "1,1,1,1,1"},
    "shape_too_short": {**valid_level(), "shape": [1, 1, 1]},
    "shape_float": {**valid_level(), "shape": [1, 1, 1.5, 1, 1]},
    "shape_bool": {**valid_level(), "shape": [True, 1, 1, 1, 1]},
    "chunk_shape_none": {**valid_level(), "chunk_shape": None},
    "shard_shape_short": {**valid_level(), "shard_shape": [1, 2]},
    "dtype_unknown": {**valid_level(), "dtype": "notatype"},
    "dtype_number": {**valid_level(), "dtype": 5},
    "codec_none": {**valid_level(), "codec_id": None},
    "level_index_bool": {**valid_level(), "level_index": True},
    "level_index_string": {**valid_level(), "level_index": "0"},
}


@pytest.mark.parametrize("data", MALFORMED_LEVELS.values(), ids=MALFORMED_LEVELS.keys())
def test_malformed_level_gives_no_view_and_no_messages(data):
    assert raw_level_view(data) is None
    assert raw_array_rule_messages(data) == []


def test_codec_level_that_is_not_an_integer_is_ignored():
    level = {**valid_level(), "codec_level": "high"}
    assert raw_level_view(level).codec_level is None


def test_shape_views_skip_every_rule_when_one_level_is_malformed():
    levels = valid_store()["levels"]
    assert raw_shape_views(levels) is not None
    assert raw_shape_views([*levels, {"shape": "x"}]) is None
    assert raw_shape_views("levels") is None
    assert raw_shape_views([]) == []


def downsampled_levels():
    levels = copy.deepcopy(valid_store()["levels"])
    levels[1]["shape"][0] = 2
    levels[2]["shape"][3] = levels[1]["shape"][3] // 4
    return levels


def test_raw_downsampling_reports_the_must_and_the_should():
    messages = raw_downsampling_messages(downsampled_levels())
    assert any("downsampling factor for 't'" in m for m in messages)
    assert any(m.startswith("[SHOULD]") for m in messages)


@pytest.mark.parametrize("levels", [None, "x", [{"shape": 1}], [1, 2]])
def test_raw_downsampling_tolerates_malformed_levels(levels):
    assert raw_downsampling_messages(levels) == []


def test_failing_v01_downsampling_raw_rule_keeps_the_original_errors(monkeypatch):
    def failing_rule(levels):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(models_v01, "raw_downsampling_messages", failing_rule)
    store = valid_store(n_levels=2)
    store["levels"][1]["shape"][0] = 2
    with pytest.raises(pydantic.ValidationError) as info:
        DCAStoreSpecV0_1.model_validate(store)
    assert any("At least 3 spatial scales" in e["msg"] for e in info.value.errors())
