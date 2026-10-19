Projections
===========

Overview
--------

A **projection** collapses one axis of an image array into a single plane. This page
specifies how a projected image array is laid out and described. Maximum intensity
projection (MIP) along ``z`` is the only projection this version defines.

A projection is a derived image, not a new acquisition. It carries the same
experimental metadata as its source and **MUST** satisfy every image-array requirement
in this specification (array standard, ``dca`` block, required floor). This page adds
only what is specific to projections.

Store Layout
------------

* A projection **MUST** be written to its own store, a sibling of its source store. It
  **MUST NOT** be written into the source store (for example as an extra resolution
  level or a ``labels/`` entry).
* The projection store name **SHOULD** be the source store name with ``_mip`` inserted
  before the ``.ome.zarr`` extension, e.g. ``plate_01.ome.zarr`` →
  ``plate_01_mip.ome.zarr``.
* The projection store **MUST** mirror the hierarchy of its source. For an HCS plate
  this means the same ``ome.plate`` rows, columns, wells and field-of-view paths, so
  that ``A/3/0`` in the projection is the projection of ``A/3/0`` in the source.

Dimensions and Scale
--------------------

* The projected array **MUST** keep all five dimensions (``t, c, z, y, x``). The
  projected axis is kept with size 1; it is not dropped.
* ``t``, ``c``, ``y`` and ``x`` **MUST** have the same size as level 0 of the source,
  and the ``y``, ``x`` and ``t`` entries of the level-0 ``scale`` transform **MUST**
  equal the source's.
* The ``z`` entry of the ``scale`` transform **SHOULD** be ``1.0``. A projection has no
  meaningful ``z`` spacing; consumers **MUST NOT** interpret the ``z`` scale of a
  projection as a physical voxel size. The projected extent is recorded in
  ``dca.projection`` (below), not in the coordinate transforms.
* The projection **MUST** be computed from level 0 of the source, never from a
  downsampled level. Lower resolution levels of the projection are then produced by
  ordinary downsampling (mean filter, as for any raw image array).
* The projection **MUST** keep the source data type. A maximum preserves it exactly.

Because ``z`` has size 1, chunk and shard sizes follow the low-``z`` edge cases of the
:doc:`array standard <array-standard>` (e.g. chunks of ``16, 1, 1, 128, 128``).

Metadata
--------

Processing variant
~~~~~~~~~~~~~~~~~~

``dca.image_data.processing_variant`` is an ordered list of processing steps (see
:doc:`experimental-metadata`). A projection **MUST** list its source's steps followed by
``"maximum intensity projection"``, so the full chain is readable on the projection
itself; e.g. a projection of a deconvolved, registered stack lists
``["deconvolved", "registered", "maximum intensity projection"]``. An image lists
``"maximum intensity projection"`` exactly when it carries ``dca.projection``: each
**MUST NOT** appear without the other. A projection of an unprocessed stack lists only
``["maximum intensity projection"]``, since ``"raw"`` means no processing at all.

Projection block
~~~~~~~~~~~~~~~~

Each image group of a projection **MUST** carry a
``dca.projection`` object in its ``zarr.json``. It records the parameters a reader
needs to interpret the array.

``dca.projection`` describes the array, like ``dca.channels`` and
``dca.normalization_statistics``. It is **not** part of the experimental-metadata
record: it is defined by the DCA :download:`JSON Schema <schema.json>`
(``$defs/Projection``), lives in ``zarr.json`` only, and is not mirrored to the
experimental-metadata Parquet. What the image *is* (REMBI Image Data "Type") is
``dca.image_data.processing_variant`` above. The full derivation (pipeline, versions)
belongs to catalog lineage, as described for ``processing_variant`` in
:doc:`experimental-metadata`.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Requirement
     - Description
   * - ``method``
     - string
     - **MUST**
     - Reduction applied along ``z``. **MUST** be ``"max"`` in this version.
   * - ``z_dim_source``
     - integer
     - MAY
     - Size of ``z`` in level 0 of the source.
   * - ``z_start``
     - integer
     - MAY
     - First source ``z`` index included in the projection (0-based, inclusive).
   * - ``z_end``
     - integer
     - MAY
     - Last source ``z`` index included, exclusive. ``z_end - z_start`` is the number
       of planes projected; a full-depth projection has ``z_start = 0`` and
       ``z_end = z_dim_source``.
   * - ``source_store``
     - string
     - **MUST**
     - The source store, as either its name (e.g. ``"plate_01.ome.zarr"``), resolved
       as a sibling of this store, or its full URL or absolute path (e.g.
       ``"s3://bucket/datasets/plate_01.ome.zarr"``). The name form **SHOULD** be
       preferred, since it stays valid when the stores are moved together. The image
       group inside the source has the same path as this one (see `Store Layout`_).
   * - ``z_fraction``
     - number
     - MAY
     - Fraction of the source depth requested, in ``(0, 1]``, taken as a central slab.
       ``1.0`` is a full-depth projection. When it disagrees with ``z_start`` /
       ``z_end`` (through rounding), ``z_start`` and ``z_end`` are authoritative.

``z_dim_source``, ``z_start`` and ``z_end`` **MUST** be given together or not at all.
When they are absent the projected range is not known; readers **MUST NOT** assume a
full-depth projection.

A projection over a central slab rather than the full depth is expressed through
``z_start`` and ``z_end``, e.g. the middle 60% of a 50-plane stack is
``z_fraction = 0.6``, ``z_start = 10``, ``z_end = 40``.

Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

A projection has a different intensity distribution from its source, since a maximum
shifts every percentile upward. ``dca.normalization_statistics`` of a projection
**MUST** be computed from the projected array. It **MUST NOT** be copied from the
source.

Provenance
~~~~~~~~~~

``dca.provenance`` is inherited from the source. If a provenance quote states that no
projection was applied (true of the source, false of the projection), the projection
**SHOULD** drop or annotate that quote rather than carry it unchanged.

Example
-------

``plate_01_mip.ome.zarr/A/3/0/zarr.json`` (abridged):

.. code-block:: json

   {
     "zarr_format": 3,
     "node_type": "group",
     "attributes": {
       "ome": {
         "version": "0.5",
         "multiscales": [ "..." ]
       },
       "dca": {
         "version": "0.3",
         "url": "...",
         "channels": [ "..." ],
         "normalization_statistics": { "..." },
         "projection": {
           "source_store": "plate_01.ome.zarr",
           "method": "max",
           "z_fraction": 1.0,
           "z_start": 0,
           "z_end": 50,
           "z_dim_source": 50
         },
         "image_data": {
           "processing_variant": ["deconvolved", "maximum intensity projection"]
         },
         "study": { "..." },
         "biosample": { "..." },
         "specimen": { "..." },
         "study_component": { "..." },
         "provenance": { "..." }
       }
     }
   }

Migration from v0.2
-------------------

v0.2 did not specify projections. Projection stores written under v0.2 carry the same
fields, with the same names, in a top-level ``mip`` attribute beside ``ome`` and
``dca``. Upgrading such a store to v0.3 (which rewrites ``dca.version`` in every image
group anyway) **MUST** move that object to ``dca.projection`` unchanged and remove the
top-level ``mip``. Readers **MAY** fall back to a top-level ``mip`` when reading stores
that declare ``dca.version`` ``"0.2"``.
