"""
Discover Zarr store roots under a local path or an S3 URL.

A directory is treated as a Zarr store root if it contains zarr.json at its
top level.  Nested stores (multiscale sub-arrays) are excluded — only the
outermost group root is returned.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Literal

import obstore
import zarr
from boto3 import Session as Boto3Session
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store
from zarr.storage import ObjectStore as _ZarrObjectStore


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def well_of_field(field_path: str) -> str:
    """Return the parent well path of a "<row>/<col>/<field>" field path."""
    return field_path.rsplit("/", 1)[0]


def _parse_s3_url(url: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from an s3://bucket/prefix URL."""
    stripped = url.rstrip("/").removeprefix("s3://")
    bucket, _, prefix = stripped.partition("/")
    return bucket, prefix


@lru_cache(maxsize=None)
def _bucket_region(bucket: str) -> str | None:
    """Detect the AWS region for a bucket via an unauthenticated HEAD request."""
    try:
        req = urllib.request.Request(
            f"https://{bucket}.s3.amazonaws.com/", method="HEAD"
        )
        response = urllib.request.urlopen(req)
        return response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as e:
        return e.headers.get("x-amz-bucket-region")
    except Exception:
        pass
    return None


@lru_cache(maxsize=None)
def _s3_bucket_store(bucket: str) -> S3Store:
    """Return a cached S3Store for bucket with auto-refreshing boto3 credentials."""
    session = Boto3Session()
    credential_provider = Boto3CredentialProvider(session)
    region = _bucket_region(bucket)
    config = {"region": region} if region else {}
    return S3Store(bucket, config=config, credential_provider=credential_provider)


def _s3_store(url: str) -> tuple[S3Store, str]:
    """Split an s3://bucket/prefix URL into a bucket-rooted S3Store and key prefix."""
    bucket, prefix = _parse_s3_url(url)
    return _s3_bucket_store(bucket), prefix


async def _list_dir_skip_markers(
    store: S3Store, prefix: str
) -> AsyncGenerator[str, None]:
    """List a prefix's immediate children, skipping S3 "directory marker" objects.

    A directory marker is a zero-byte object whose key is the prefix itself
    (e.g. ``.../A/1/``) — created by some upload tools to represent an explicit
    "folder". It is legal on S3 but is not a child of the prefix; it relativizes
    to the prefix itself. zarr's stock ``ObjectStore.list_dir`` relativizes every
    returned key unconditionally and raises ``ValueError`` on such an entry,
    which makes ``Group.keys()`` blow up on any store that carries these markers.
    Plate validation enumerates well children via ``keys()``, so the crash
    surfaced as a false "field has no group on disk". See issue #107.

    This mirrors zarr's ``_transform_list_dir`` but drops any entry equal to the
    prefix before relativizing.
    """
    result = await obstore.list_with_delimiter_async(store, prefix=prefix)
    norm = prefix.rstrip("/")
    prefix_slash = f"{norm}/" if norm else ""
    for path in chain(
        result["common_prefixes"], (obj["path"] for obj in result["objects"])
    ):
        if norm and path.rstrip("/") == norm:
            continue  # directory-marker / self entry — not a child
        yield path.removeprefix(prefix_slash) if prefix_slash else path


class _MarkerTolerantObjectStore(_ZarrObjectStore):
    """zarr ``ObjectStore`` that tolerates S3 directory-marker objects.

    Overrides ``list_dir`` so a directory-marker object (a key equal to the
    listed prefix) is skipped instead of crashing relativization. See issue #107
    and :func:`_list_dir_skip_markers`.
    """

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        return _list_dir_skip_markers(self.store, prefix)


def open_zarr_group(path: str) -> zarr.Group:
    """Open a Zarr group at `path`, using obstore for s3:// URLs."""
    if _is_s3(path):
        bucket, prefix = _parse_s3_url(path)
        return zarr.open_group(
            _MarkerTolerantObjectStore(_s3_bucket_store(bucket), read_only=True),
            path=prefix or None,
            mode="r",
        )
    return zarr.open_group(path, mode="r")


def _discover_local(root: str) -> list[str]:
    """Walk a local directory tree and return Zarr store roots."""
    stores: list[str] = []
    root_path = Path(root)

    for dirpath, dirnames, filenames in os.walk(root_path):
        if "zarr.json" in filenames:
            stores.append(str(dirpath))
            # Do not descend into sub-groups of this store; they are not
            # independent stores.  Clear dirnames in-place so os.walk skips them.
            dirnames.clear()

    return stores


def _discover_s3_walk(root: str) -> list[str]:
    """Return Zarr store roots under an S3 prefix using BFS with list_with_delimiter.

    BFS: one ListObjectsV2-with-delimiter call per directory level.
    Prunes descent into a store once its root is found, so it avoids
    enumerating internal sub-groups.  More efficient for single/few plates;
    less efficient for deeply nested or widely scattered single images.
    """
    bucket, start_prefix = _parse_s3_url(root)
    store = _s3_bucket_store(bucket)

    stores: list[str] = []
    # queue holds directory key prefixes WITHOUT trailing slash; "" = bucket root
    queue: list[str] = [start_prefix]

    while queue:
        current = queue.pop(0)
        # Add "/" suffix to ensure directory-scoped listing, not string-prefix match
        list_prefix = f"{current}/" if current else None
        result = obstore.list_with_delimiter(store, prefix=list_prefix)

        zarr_key = f"{current}/zarr.json" if current else "zarr.json"
        has_zarr = any(obj["path"] == zarr_key for obj in result["objects"])

        if has_zarr:
            stores.append(f"s3://{bucket}/{current}" if current else f"s3://{bucket}")
        else:
            # common_prefixes are full key paths without trailing slash
            queue.extend(result["common_prefixes"])

    return stores


def _discover_s3_glob(root: str) -> list[str]:
    """Return Zarr store roots under an S3 prefix using a single flat list scan.

    Issues one paginated ListObjectsV2 (no delimiter) for the entire prefix,
    returning all keys before filtering outermost store roots in memory.
    More efficient for deeply nested or widely scattered single images;
    less efficient for large plates where all internal objects are returned
    before filtering can occur.
    """
    bucket, prefix = _parse_s3_url(root)
    store = _s3_bucket_store(bucket)

    # Use "prefix/" to avoid false matches from sibling keys sharing a string prefix
    list_prefix = f"{prefix}/" if prefix else None

    all_keys: list[str] = []
    for chunk in obstore.list(store, prefix=list_prefix):
        for obj in chunk:
            path = obj["path"]
            if path.endswith("/zarr.json") or path == "zarr.json":
                all_keys.append(path)

    # Sort by depth first (fewest slashes = outermost), then alphabetically.
    # Alphabetical-only sorting can place deep paths before shallow siblings
    # (e.g. "labels/nuclei/0" before "labels"), breaking the ancestor filter.
    all_keys = list(dict.fromkeys(all_keys))
    all_keys.sort(key=lambda k: (k.count("/"), k))

    # The store root is the parent directory of zarr.json
    seen: set[str] = set()
    stores: list[str] = []
    for key in all_keys:
        store_key = key[: -len("/zarr.json")]  # parent of zarr.json
        # Keep only outermost: skip if any ancestor is already a store
        if any(store_key.startswith(s + "/") for s in seen):
            continue
        seen.add(store_key)
        stores.append(f"s3://{bucket}/{store_key}" if store_key else f"s3://{bucket}")

    return stores


def is_zarr_store(path: str) -> bool:
    """
    Return True if `path` is itself a Zarr store root (has zarr.json at its top level).

    Does not recurse.  Used by the runner to skip discovery when the caller
    supplies an exact store path rather than a parent directory.
    """
    if _is_s3(path):
        store, prefix = _s3_store(path)
        zarr_key = f"{prefix}/zarr.json" if prefix else "zarr.json"
        try:
            obstore.head(store, zarr_key)
            return True
        except FileNotFoundError:
            return False
    return (Path(path) / "zarr.json").exists()


def discover_zarr_stores(
    root: str, strategy: Literal["glob", "walk"] = "glob"
) -> list[str]:
    """
    Return a flat list of Zarr store root paths found under `root`.

    Parameters
    ----------
    root     : local filesystem path or s3://bucket/prefix URL
    strategy : S3 discovery strategy — "glob" (default) or "walk".
               Ignored for local paths, which always use os.walk.

               "glob" issues a single paginated ListObjectsV2 scan and filters
               outermost roots in memory.  More efficient for deeply nested or
               widely scattered single images.

               "walk" uses BFS with per-level API calls, pruning descent once a
               store root is found.  More efficient for single/few plates where
               early pruning avoids enumerating internal sub-groups.
    """
    if _is_s3(root):
        if strategy == "glob":
            return _discover_s3_glob(root)
        if strategy == "walk":
            return _discover_s3_walk(root)
        raise ValueError(f"Invalid strategy {strategy!r}; expected 'glob' or 'walk'.")
    return _discover_local(root)
