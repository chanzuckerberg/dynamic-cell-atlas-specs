import copy

import pydantic
import pytest

from dca_helpers.validation.versions.v0_1.models import (
    DCAScaleLevelSpec,
    DCAStoreSpecV0_1,
)
from dca_helpers.validation.versions.v0_2.models import (
    DCALabelScaleLevelSpecV0_2,
    DCALabelStoreSpecV0_2,
    DCAScaleLevelSpecV0_2,
    DCAStoreSpecV0_2,
)

from .v0_1.test_models import valid_level as valid_level_v01
from .v0_1.test_models import valid_store as valid_store_v01
from .v0_2.test_models import valid_level, valid_store

pytestmark = pytest.mark.unit

LENGTHS = [[], [1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [-1] * 5]
DTYPES = ["foo", "", "uint16x", "U", "O", "V0", "float128", "datetime64"]
LEVEL_MODELS = [
    (DCAScaleLevelSpec, valid_level_v01),
    (DCAScaleLevelSpecV0_2, valid_level),
    (DCALabelScaleLevelSpecV0_2, valid_level),
]


def reports_or_passes(model, data):
    try:
        model.model_validate(copy.deepcopy(data))
    except pydantic.ValidationError:
        pass


def label_store():
    store = valid_store()
    return {key: store[key] for key in ("spec_version", "axes", "levels")}


@pytest.mark.parametrize(
    "model,make", LEVEL_MODELS, ids=lambda x: getattr(x, "__name__", "")
)
@pytest.mark.parametrize("field", ["shape", "chunk_shape", "shard_shape"])
@pytest.mark.parametrize("value", LENGTHS, ids=str)
def test_level_with_a_malformed_array_shape_reports_instead_of_raising(
    model, make, field, value
):
    reports_or_passes(model, {**make(), field: value})


@pytest.mark.parametrize(
    "model,make", LEVEL_MODELS, ids=lambda x: getattr(x, "__name__", "")
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_level_with_an_unparseable_dtype_reports_instead_of_raising(model, make, dtype):
    reports_or_passes(model, {**make(), "dtype": dtype})


@pytest.mark.parametrize(
    "model,make", LEVEL_MODELS, ids=lambda x: getattr(x, "__name__", "")
)
@pytest.mark.parametrize("level_index", [-1, 5, 10**9])
@pytest.mark.parametrize("codec_level", [-5, 10**6, None])
def test_level_with_out_of_range_numbers_reports_instead_of_raising(
    model, make, level_index, codec_level
):
    reports_or_passes(
        model, {**make(), "level_index": level_index, "codec_level": codec_level}
    )


STORES = [
    (DCAStoreSpecV0_2, valid_store),
    (DCALabelStoreSpecV0_2, label_store),
]


@pytest.mark.parametrize("model,make", STORES, ids=lambda x: getattr(x, "__name__", ""))
@pytest.mark.parametrize("field", ["shape", "chunk_shape"])
@pytest.mark.parametrize("value", LENGTHS, ids=str)
@pytest.mark.parametrize("level", [0, 1])
def test_store_with_a_malformed_level_reports_instead_of_raising(
    model, make, field, value, level
):
    store = make()
    store["levels"][level][field] = value
    reports_or_passes(model, store)


@pytest.mark.parametrize("model,make", STORES, ids=lambda x: getattr(x, "__name__", ""))
@pytest.mark.parametrize("levels", [[], "levels", [None], [{}]], ids=str)
def test_store_with_malformed_levels_reports_instead_of_raising(model, make, levels):
    store = make()
    store["levels"] = levels
    reports_or_passes(model, store)


@pytest.mark.parametrize("value", LENGTHS, ids=str)
def test_v01_store_with_a_malformed_level_reports_instead_of_raising(value):
    store = valid_store_v01()
    store["levels"][1]["chunk_shape"] = value
    reports_or_passes(DCAStoreSpecV0_1, store)


@pytest.mark.parametrize(
    "model,make", LEVEL_MODELS[:2], ids=lambda x: getattr(x, "__name__", "")
)
def test_unparseable_dtype_is_a_named_error(model, make):
    with pytest.raises(pydantic.ValidationError, match="not a valid NumPy dtype"):
        model.model_validate({**make(), "dtype": "foo"})


@pytest.mark.parametrize(
    "model,make", LEVEL_MODELS[:2], ids=lambda x: getattr(x, "__name__", "")
)
def test_short_shape_is_a_named_error(model, make):
    with pytest.raises(pydantic.ValidationError, match="MUST have 5 dimensions"):
        model.model_validate({**make(), "chunk_shape": [1]})
