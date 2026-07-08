"""
Unit tests for dca_helpers.validation.discovery.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dca_helpers.validation.discovery import (
    _discover_local,
    _list_dir_skip_markers,
    discover_zarr_stores,
    is_zarr_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store(root: Path, *rel_parts: str) -> Path:
    """Create a minimal zarr store (just zarr.json) at root / rel_parts."""
    store = root.joinpath(*rel_parts)
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").touch()
    return store


# ---------------------------------------------------------------------------
# _discover_local
# ---------------------------------------------------------------------------


class TestDiscoverLocal:
    def test_single_store_at_root(self, tmp_path):
        make_store(tmp_path, "plate.ome.zarr")
        result = _discover_local(str(tmp_path))
        assert result == [str(tmp_path / "plate.ome.zarr")]

    def test_multiple_stores(self, tmp_path):
        make_store(tmp_path, "plate1.ome.zarr")
        make_store(tmp_path, "plate2.ome.zarr")
        result = sorted(_discover_local(str(tmp_path)))
        assert result == sorted(
            [
                str(tmp_path / "plate1.ome.zarr"),
                str(tmp_path / "plate2.ome.zarr"),
            ]
        )

    def test_nested_store_excluded(self, tmp_path):
        """Sub-groups inside a store must not be returned as separate stores."""
        outer = make_store(tmp_path, "plate.ome.zarr")
        # Simulate an internal sub-group that also has zarr.json
        make_store(outer, "0")
        result = _discover_local(str(tmp_path))
        assert result == [str(outer)]

    def test_empty_directory(self, tmp_path):
        assert _discover_local(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# is_zarr_store
# ---------------------------------------------------------------------------


class TestIsZarrStore:
    def test_true_when_zarr_json_present(self, tmp_path):
        store = make_store(tmp_path, "plate.ome.zarr")
        assert is_zarr_store(str(store)) is True

    def test_false_when_no_zarr_json(self, tmp_path):
        (tmp_path / "not_a_store").mkdir()
        assert is_zarr_store(str(tmp_path / "not_a_store")) is False

    def test_false_for_parent_directory(self, tmp_path):
        make_store(tmp_path, "plate.ome.zarr")
        assert is_zarr_store(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# discover_zarr_stores — local routing
# ---------------------------------------------------------------------------


class TestDiscoverZarrStoresLocal:
    def test_local_path_uses_walk(self, tmp_path):
        make_store(tmp_path, "plate.ome.zarr")
        result = discover_zarr_stores(str(tmp_path))
        assert len(result) == 1

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            # ValueError is raised before any S3 call, so no patching needed
            discover_zarr_stores(
                "s3://bucket/prefix", strategy="bad"  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# discover_zarr_stores — S3 routing
# ---------------------------------------------------------------------------

_OBJ = {"size": 2, "last_modified": None, "e_tag": None, "version": None}


class TestDiscoverZarrStoresS3:
    def test_glob_strategy(self):
        chunks = [[{**_OBJ, "path": "prefix/plate.ome.zarr/zarr.json"}]]
        with (
            patch("dca_helpers.validation.discovery.S3Store"),
            patch(
                "dca_helpers.validation.discovery.obstore.list",
                return_value=iter(chunks),
            ),
        ):
            result = discover_zarr_stores("s3://bucket/prefix", strategy="glob")
        assert result == ["s3://bucket/prefix/plate.ome.zarr"]

    def test_walk_strategy(self):
        # First call (prefix="prefix/"): no zarr.json, one subdirectory
        # Second call (prefix="prefix/plate.ome.zarr/"): zarr.json found
        side_effects = [
            {"common_prefixes": ["prefix/plate.ome.zarr"], "objects": []},
            {
                "common_prefixes": [],
                "objects": [{**_OBJ, "path": "prefix/plate.ome.zarr/zarr.json"}],
            },
        ]
        with (
            patch("dca_helpers.validation.discovery.S3Store"),
            patch(
                "dca_helpers.validation.discovery.obstore.list_with_delimiter",
                side_effect=side_effects,
            ),
        ):
            result = discover_zarr_stores("s3://bucket/prefix", strategy="walk")
        assert result == ["s3://bucket/prefix/plate.ome.zarr"]

    def test_glob_outermost_only(self):
        """Glob must return only the outermost zarr.json, not internal sub-groups."""
        chunks = [
            [
                {**_OBJ, "path": "prefix/plate.ome.zarr/zarr.json"},
                {**_OBJ, "path": "prefix/plate.ome.zarr/0/zarr.json"},
                {**_OBJ, "path": "prefix/plate.ome.zarr/labels/zarr.json"},
            ]
        ]
        with (
            patch("dca_helpers.validation.discovery.S3Store"),
            patch(
                "dca_helpers.validation.discovery.obstore.list",
                return_value=iter(chunks),
            ),
        ):
            result = discover_zarr_stores("s3://bucket/prefix", strategy="glob")
        assert result == ["s3://bucket/prefix/plate.ome.zarr"]


# ---------------------------------------------------------------------------
# _list_dir_skip_markers — S3 directory-marker tolerance (issue #107)
# ---------------------------------------------------------------------------


def _collect(async_gen) -> list[str]:
    """Drain an async generator into a list."""

    async def _drain():
        return [item async for item in async_gen]

    return asyncio.run(_drain())


def _list_dir(prefix: str, listing: dict) -> list[str]:
    """Run _list_dir_skip_markers against a mocked obstore listing."""
    with patch(
        "dca_helpers.validation.discovery.obstore.list_with_delimiter_async",
        AsyncMock(return_value=listing),
    ):
        return _collect(_list_dir_skip_markers(store=None, prefix=prefix))


class TestListDirSkipMarkers:
    def test_skips_directory_marker(self):
        """The zero-byte self-marker object must not appear as a child."""
        prefix = "p/well.ome.zarr/A/1"
        listing = {
            "common_prefixes": ["p/well.ome.zarr/A/1/0/"],
            "objects": [
                {"path": "p/well.ome.zarr/A/1/"},  # directory marker / self entry
                {"path": "p/well.ome.zarr/A/1/zarr.json"},
            ],
        }
        # The marker is dropped; the real field group and metadata remain.
        assert sorted(_list_dir(prefix, listing)) == ["0/", "zarr.json"]

    def test_no_marker_unchanged(self):
        """A store without a marker lists exactly its children (regression guard)."""
        prefix = "p/well.ome.zarr/B/1"
        listing = {
            "common_prefixes": [
                "p/well.ome.zarr/B/1/0/",
                "p/well.ome.zarr/B/1/1/",
            ],
            "objects": [{"path": "p/well.ome.zarr/B/1/zarr.json"}],
        }
        assert sorted(_list_dir(prefix, listing)) == ["0/", "1/", "zarr.json"]

    def test_marker_with_trailing_slash_prefix(self):
        """A trailing slash on the prefix still identifies the marker."""
        prefix = "p/well.ome.zarr/A/1/"
        listing = {
            "common_prefixes": ["p/well.ome.zarr/A/1/0/"],
            "objects": [{"path": "p/well.ome.zarr/A/1/"}],  # marker
        }
        assert _list_dir(prefix, listing) == ["0/"]
