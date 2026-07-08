"""
Unit tests for validation.registry.
"""

import pytest

from dca_helpers.validation.registry import (
    UnsupportedSpecVersionError,
    get_label_model,
    get_model,
)
from dca_helpers.validation.versions.v0_1.models import DCAStoreSpecV0_1
from dca_helpers.validation.versions.v0_2.models import (
    DCALabelStoreSpecV0_2,
    DCAStoreSpecV0_2,
)


@pytest.mark.unit
class TestGetModel:
    def test_v0_1_returns_correct_class(self):
        assert get_model("0.1") is DCAStoreSpecV0_1

    def test_v0_2_returns_correct_class(self):
        assert get_model("0.2") is DCAStoreSpecV0_2

    def test_unsupported_version_raises(self):
        with pytest.raises(UnsupportedSpecVersionError):
            get_model("0.3")

    def test_unsupported_version_message_contains_version(self):
        with pytest.raises(UnsupportedSpecVersionError, match="0.9"):
            get_model("0.9")

    def test_unsupported_version_message_lists_supported(self):
        with pytest.raises(UnsupportedSpecVersionError, match="0.1"):
            get_model("0.3")

    def test_empty_string_raises(self):
        with pytest.raises(UnsupportedSpecVersionError):
            get_model("")

    def test_unsupported_version_is_value_error(self):
        with pytest.raises(ValueError):
            get_model("0.3")


@pytest.mark.unit
class TestGetLabelModel:
    def test_v0_1_not_registered(self):
        with pytest.raises(UnsupportedSpecVersionError):
            get_label_model("0.1")

    def test_v0_2_returns_correct_class(self):
        assert get_label_model("0.2") is DCALabelStoreSpecV0_2

    def test_unsupported_version_raises(self):
        with pytest.raises(UnsupportedSpecVersionError):
            get_label_model("0.3")

    def test_error_message_contains_version(self):
        with pytest.raises(UnsupportedSpecVersionError, match="0.1"):
            get_label_model("0.1")

    def test_error_message_mentions_label(self):
        with pytest.raises(UnsupportedSpecVersionError, match="label"):
            get_label_model("0.1")

    def test_error_is_value_error(self):
        with pytest.raises(ValueError):
            get_label_model("0.1")
