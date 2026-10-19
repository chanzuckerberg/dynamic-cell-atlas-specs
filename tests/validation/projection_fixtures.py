"""
Builders for v0.3 source / projection store pairs, for the projection tests.

Starts from the committed v0.2 plate fixture (zarr.json files only, a 3D plate
with z=16 at level 0) and writes, under a temporary directory:

* ``plate.ome.zarr``     — the source, upgraded to dca.version "0.3"
* ``plate_mip.ome.zarr`` — its maximum intensity projection: z=1 at every
  level, z scale 1.0, ``dca.projection`` and the MIP processing step

Each builder takes ``edit_source`` / ``edit_mip`` callbacks that receive the
store root ``Path`` so a test can break exactly one thing.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"
BASE_PLATE = FIXTURES / "v0_2" / "valid_plate.ome.zarr"
FIELD = "A/1/0"
MIP_STEP = "maximum intensity projection"
SOURCE_Z = 16


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def edit_field_attrs(
    store: Path, fn: Callable[[dict], None], field: str = FIELD
) -> None:
    """Apply ``fn`` to one image group's zarr.json ``attributes`` in place."""
    p = store / field / "zarr.json"
    group = read_json(p)
    fn(group["attributes"])
    write_json(p, group)


def _to_v03_source(store: Path, steps: list[str]) -> None:
    def upgrade(attrs: dict) -> None:
        attrs["dca"]["version"] = "0.3"
        attrs["dca"]["image_data"] = {"processing_variant": steps}

    edit_field_attrs(store, upgrade)


def _project(store: Path, source_name: str, steps: list[str]) -> None:
    field_dir = store / FIELD
    group = read_json(field_dir / "zarr.json")
    attrs = group["attributes"]
    ms = attrs["ome"]["multiscales"][0]
    zi = [a["name"].lower() for a in ms["axes"]].index("z")
    for ds in ms["datasets"]:
        for t in ds["coordinateTransformations"]:
            if t["type"] == "scale":
                t["scale"][zi] = 1.0
        arr_path = field_dir / ds["path"] / "zarr.json"
        arr = read_json(arr_path)
        arr["shape"][zi] = 1
        write_json(arr_path, arr)
    attrs["dca"]["image_data"] = {"processing_variant": [*steps, MIP_STEP]}
    attrs["dca"]["projection"] = {
        "source_store": source_name,
        "method": "max",
        "z_fraction": 1.0,
        "z_start": 0,
        "z_end": SOURCE_Z,
        "z_dim_source": SOURCE_Z,
    }
    write_json(field_dir / "zarr.json", group)


def make_pair(
    root: Path,
    *,
    source_name: str = "plate.ome.zarr",
    mip_name: str = "plate_mip.ome.zarr",
    steps: tuple[str, ...] = ("deconvolved",),
    edit_source: Callable[[Path], None] | None = None,
    edit_mip: Callable[[Path], None] | None = None,
    with_mip: bool = True,
) -> tuple[Path, Path]:
    """Write a v0.3 source plate and (unless ``with_mip`` is False) its MIP."""
    source = root / source_name
    shutil.copytree(BASE_PLATE, source)
    _to_v03_source(source, list(steps))
    mip = root / mip_name
    if with_mip:
        shutil.copytree(source, mip)
        _project(mip, source_name, list(steps))
    if edit_source:
        edit_source(source)
    if with_mip and edit_mip:
        edit_mip(mip)
    return source, mip
