Projections
===========

Overview
--------

The DCA includes 2D projections of 3D images along ``z`` (maximum, mean or sum intensity
projections, or a single selected plane) for easier rendering and to provide additional
input options for model training.

A projection is a derived image, not a new acquisition. It carries the same
experimental metadata as its source and **MUST** satisfy every image-array requirement
in this specification (array standard, ``dca`` block, required floor). This page adds
only what is specific to projections.

The requirements below apply to a projection whose **3D source is also in the DCA**.
An image that was projected before it was deposited, and whose stack is not shared, is
simply a 2D image: it **MAY** carry ``dca.projection`` to record the parameters when
they are known (see `Projections made before deposit`_), and nothing else on this page
applies to it.

``dca.projection`` records what was done, not how it was configured: the method and the
source planes actually used for each output channel, as absolute indices. The method
vocabulary is shared with the ``project`` command in
`biahub <https://github.com/czbiohub-sf/biahub>`_ (see `Relation to biahub`_).

Store Layout
------------

* A projection of a 3D image in the DCA **MUST** be written to its own store, a sibling
  of its source store. It
  **MUST NOT** be written into the source store (for example as an extra resolution
  level or a ``labels/`` entry).
* The projection store name **SHOULD** be the source store name with ``_proj`` inserted
  before the ``.ome.zarr`` extension, e.g. ``plate_01.ome.zarr`` →
  ``plate_01_proj.ome.zarr``. A projection whose every channel uses ``max`` **MAY** use
  ``_mip`` instead, as stores written before this version do.
* The projection store **MUST** mirror the hierarchy of its source. For an HCS plate
  this means the same ``ome.plate`` rows, columns, wells and field-of-view paths, so
  that ``A/3/0`` in the projection is the projection of ``A/3/0`` in the source.

Dimensions and Scale
--------------------

* The projected array **MUST** keep all five dimensions (``t, c, z, y, x``). The
  projected axis is kept with size 1 at every level; it is not dropped.
* ``t``, ``y`` and ``x`` **MUST** have the same size as level 0 of the source, and the
  ``t``, ``y`` and ``x`` entries of the level-0 ``scale`` transform **MUST** equal the
  source's.
* Each output channel is projected from one source channel. A projection **MAY** keep
  only some of the source's channels, in any order (see `Per-channel form`_).
* The ``z`` entry of the ``scale`` transform carries no meaning for a projection;
  consumers **MUST NOT** interpret it as a physical voxel size. Copying the source's
  ``z`` scale and writing ``1.0`` are both acceptable. The projected planes are recorded
  in ``dca.projection`` (below), not in the coordinate transforms.
* The projection **MUST** be computed from level 0 of the source, never from a
  downsampled level. Lower resolution levels of the projection are then produced by
  ordinary downsampling (mean filter, as for any raw image array).
* When every channel uses ``max`` or ``slice``, the projection **MUST** keep the source
  data type, since each output value is a source value. When any channel uses ``mean``
  or ``sum``, the projection **MAY** use a floating-point type (e.g. ``float32``) for the
  whole array.

Because ``z`` has size 1, chunk and shard sizes follow the low-``z`` edge cases of the
:doc:`array standard <array-standard>` (e.g. chunks of ``16, 1, 1, 128, 128``).

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - ``method``
     - Output value per pixel
     - ``processing_variant`` step
   * - ``max``
     - Maximum over the selected planes
     - ``"maximum intensity projection"``
   * - ``mean``
     - Mean over the selected planes
     - ``"mean intensity projection"``
   * - ``sum``
     - Sum over the selected planes
     - ``"sum intensity projection"``
   * - ``slice``
     - The value at one selected plane
     - ``"z slice"``

Metadata
--------

Processing variant
~~~~~~~~~~~~~~~~~~

``dca.image_data.processing_variant`` is an ordered list of processing steps (see
:doc:`experimental-metadata`). A projection **MUST** list its source's steps followed by
the step of every method it uses (table above), so the full chain is readable on the
projection itself; e.g. a maximum intensity projection of a deconvolved, registered
stack lists ``["deconvolved", "registered", "maximum intensity projection"]``, and a
projection using ``max`` for some channels and ``slice`` for others lists both steps.
An image that carries ``dca.projection`` **MUST** list exactly the projection steps of
the methods it uses. An image **MAY** list a projection step without carrying
``dca.projection`` only when it was projected before deposit and its 3D source is not in
the DCA (see `Projections made before deposit`_). A projection
of an unprocessed stack lists only its projection steps, since ``"raw"`` means no
processing at all. A projection copies its source's ``image_data.processing_description``
unchanged, so the source's full description is available on the projection too.

Projection block
~~~~~~~~~~~~~~~~

Each image group of a projection whose 3D source is in the DCA **MUST** carry a
``dca.projection`` object in its ``zarr.json``. It records the parameters a reader needs
to interpret the array.

``dca.projection`` describes the array, like ``dca.channels`` and
``dca.normalization_statistics``. It is **not** part of the experimental-metadata
record: it is defined by the DCA :download:`JSON Schema <schema.json>`
(``$defs/Projection``), lives in ``zarr.json`` only, and is not mirrored to the
experimental-metadata Parquet. What the image *is* (REMBI Image Data "Type") is
``dca.image_data.processing_variant`` above. The full derivation (pipeline, versions)
belongs to catalog lineage, as described for ``processing_variant`` in
:doc:`experimental-metadata`.

The block takes one of two forms: the **uniform form**, when every channel is projected
the same way over the same planes, or the **per-channel form**. Exactly one of
``method`` (uniform) and ``channels`` (per-channel) **MUST** be present. Both forms
share these fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Requirement
     - Description
   * - ``source_store``
     - string
     - **MUST** when the 3D source is in the DCA; omitted otherwise
     - The source store, as either its name (e.g. ``"plate_01.ome.zarr"``), resolved
       as a sibling of this store, or its full URL or absolute path (e.g.
       ``"s3://bucket/datasets/plate_01.ome.zarr"``). The name form **SHOULD** be
       preferred, since it stays valid when the stores are moved together. The image
       group inside the source has the same path as this one (see `Store Layout`_).
   * - ``z_dim_source``
     - integer
     - **MUST** when any ``z_start`` / ``z_end`` is given; MAY otherwise
     - Size of ``z`` in level 0 of the source.
   * - ``focus``
     - object
     - **SHOULD** when planes were chosen around a detected in-focus plane
     - How the in-focus plane was found. See `Focus`_.

Plane ranges
^^^^^^^^^^^^

Both forms record the planes used with the same fields, as absolute source ``z``
indices:

* ``z_start``: first plane used (0-based, inclusive).
* ``z_end``: last plane used, exclusive, so ``z_end - z_start`` planes are spanned and
  ``0 <= z_start < z_end <= z_dim_source``.
* ``z_step``: every ``z_step``-th plane of the range was used (default 1). **MUST NOT**
  be given without ``z_start`` and ``z_end``.

``z_start`` and ``z_end`` **MUST** be given together. For ``slice`` they **MUST** be given,
with ``z_end - z_start = 1``. When they are absent the planes used are not known; readers
**MUST NOT** assume a full-depth projection.

Uniform form
^^^^^^^^^^^^

Every channel is projected with the same method over the same planes, and the projection
keeps every source channel, in the source's order.

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
     - One of the `Methods`_.
   * - ``z_start``, ``z_end``, ``z_step``
     - integer
     - MAY; **MUST** for ``slice``
     - The planes used (see `Plane ranges`_).
   * - ``z_fraction``
     - number
     - MAY
     - Fraction of the source depth requested, in ``(0, 1]``, taken as a central slab.
       ``1.0`` is a full-depth projection. When it disagrees with ``z_start`` /
       ``z_end`` (through rounding), ``z_start`` and ``z_end`` are authoritative.

A projection over a central slab rather than the full depth is expressed through
``z_start`` and ``z_end``, e.g. the middle 60% of a 50-plane stack is
``z_fraction = 0.6``, ``z_start = 10``, ``z_end = 40``.

Per-channel form
^^^^^^^^^^^^^^^^

``channels`` maps each projected output channel, by its ``dca.channels[].name``, to how
it was projected. Every key **MUST** be one of the image's channel names. A channel that
is not a key was not projected along ``z`` (for example a reconstruction or a virtual
stain computed another way), and its ``processing_variant`` steps describe it. A source
channel that is not projected is simply absent.

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
     - One of the `Methods`_.
   * - ``source_channel``
     - string
     - MAY
     - Name of the source channel this output channel was projected from. Defaults to
       the output channel's own name.
   * - ``z_start``, ``z_end``
     - integer, or list of integers
     - MAY; **MUST** for ``slice``
     - The planes used (see `Plane ranges`_). When the planes differ between timepoints
       (e.g. a window that follows ``z`` drift), each is a list with one entry per
       level-0 timepoint; ``z_start[t]`` and ``z_end[t]`` are then the planes used at
       timepoint ``t``. ``z_start`` and ``z_end`` **MUST** both be integers or both be
       lists of that length.
   * - ``z_step``
     - integer
     - MAY
     - See `Plane ranges`_.

Focus
^^^^^

When the planes of a channel were chosen around an in-focus plane detected on the data,
``focus`` **SHOULD** record how. It is provenance: the planes themselves are always the
ones in ``z_start`` / ``z_end``, and nothing else depends on this block.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Requirement
     - Description
   * - ``channel``
     - string
     - **MUST**
     - Source channel the focus was measured on. It need not be an output channel.
   * - ``per``
     - string
     - **MUST**
     - ``"fov"``: one in-focus plane for the whole field of view (e.g. the median over
       timepoints). ``"timepoint"``: one plane per timepoint.
   * - ``z``
     - integer
     - **MUST** when ``per`` is ``"fov"``
     - The in-focus plane.
   * - ``z_per_t``
     - list of integers
     - **MUST** when ``per`` is ``"timepoint"``; MAY otherwise
     - The in-focus plane of each timepoint, one entry per level-0 timepoint.
   * - ``parameters``
     - object
     - MAY
     - Settings of the focus-finding method (e.g. a detection numerical aperture,
       illumination wavelength or crop), recorded as given by the tool.

Relation to biahub
~~~~~~~~~~~~~~~~~~

biahub's ``project`` command uses the same four methods (``max``, ``mean``, ``sum``,
``slice``) and projects each output channel separately, so its output fits the
per-channel form. Its settings describe how to choose planes (a range, or a window
around a fixed or detected in-focus plane); ``dca.projection`` records the planes that
choice resolved to. For a ``project`` run:

.. list-table::
   :header-rows: 1

   * - ``project`` settings / output
     - ``dca.projection``
   * - ``channels.<name>.method``
     - ``channels.<name>.method``
   * - ``channels.<name>.z`` (``range``, ``center``, ``step``)
     - ``channels.<name>.z_start`` / ``z_end`` (resolved and clipped to the stack; lists
       when the window follows a per-timepoint focus), ``z_step``
   * - ``focus.channel``, ``focus.per``
     - ``focus.channel``, ``focus.per``
   * - ``focus_z``, ``focus_z_per_t`` (recorded per position)
     - ``focus.z``, ``focus.z_per_t``
   * - focus-finding settings (``na_det``, ``lambda_ill``, ``center_crop_xy``)
     - ``focus.parameters``
   * - (not recorded)
     - ``source_store``, ``z_dim_source``

Projections made before deposit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some datasets are deposited as 2D images that were projected from stacks that are not
shared. These are 2D datasets: the store-layout, pairing and source rules above do not
apply, and ``dca.projection`` is not required.

* ``processing_variant`` **MAY** list the projection step as part of the image's
  history, with the detail in ``processing_description``.
* When the parameters are known, ``dca.projection`` **MAY** record them, without
  ``source_store``: e.g. ``z_dim_source`` for the depth of the original stack and, per
  projected channel, the method and planes used. Channels that were not projected are
  left out of ``channels``.

Whether a projection's 3D source is in the DCA is decided when a dataset is validated as
a whole: a projection store named after a 3D sibling (``plate_01_mip.ome.zarr`` or
``plate_01_proj.ome.zarr`` next to ``plate_01.ome.zarr``) is a projection of that store,
and **MUST** then carry ``dca.projection`` with ``source_store``.

Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

A projection has a different intensity distribution from its source (a maximum shifts
every percentile upward, a sum scales with the number of planes).
``dca.normalization_statistics`` of a projection **MUST** be computed from the projected
array. It **MUST NOT** be copied from the source.

Examples
--------

A maximum intensity projection in the uniform form,
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
           "processing_variant": ["deconvolved", "maximum intensity projection"],
           "processing_description": "deconvolved with a measured PSF"
         },
         "study": { "..." },
         "biosample": { "..." },
         "specimen": { "..." },
         "study_component": { "..." },
         "provenance": { "..." }
       }
     }
   }

A per-channel projection of a three-timepoint stack that keeps three of the source's
channels. ``nuclei`` is the mean over a window of 20 planes below to 40 planes above the
in-focus plane found on a phase channel at each timepoint (planes 19, 18 and 20), clipped
at the bottom of the stack. ``plate_01_proj.ome.zarr/A/3/0/zarr.json`` (``dca`` only,
abridged):

.. code-block:: json

   {
     "version": "0.3",
     "channels": [
       {"name": "membrane", "index": 0, "...": "..."},
       {"name": "nuclei", "index": 1, "...": "..."},
       {"name": "mCherry", "index": 2, "...": "..."}
     ],
     "projection": {
       "source_store": "plate_01.ome.zarr",
       "z_dim_source": 86,
       "channels": {
         "membrane": {"method": "max", "z_start": 0, "z_end": 86},
         "nuclei": {"method": "mean", "z_start": [0, 0, 0], "z_end": [59, 58, 60]},
         "mCherry": {"method": "slice", "z_start": 42, "z_end": 43}
       },
       "focus": {
         "channel": "Phase3D",
         "per": "timepoint",
         "z_per_t": [19, 18, 20],
         "parameters": {"na_det": 1.35, "lambda_ill": 0.5, "center_crop_xy": [800, 800]}
       }
     },
     "image_data": {
       "processing_variant": [
         "maximum intensity projection", "mean intensity projection", "z slice"
       ]
     }
   }

An image projected before deposit, recording the known parameters for its two projected
channels; its label-free channel, a reconstruction, is not a key (``dca`` only,
abridged):

.. code-block:: json

   {
     "channels": [
       {"name": "Phase2D", "index": 0, "...": "..."},
       {"name": "GFP", "index": 1, "...": "..."},
       {"name": "mCherry", "index": 2, "...": "..."}
     ],
     "projection": {
       "z_dim_source": 7,
       "channels": {
         "GFP": {"method": "max", "z_start": 0, "z_end": 7},
         "mCherry": {"method": "max", "z_start": 0, "z_end": 7}
       }
     },
     "image_data": {
       "processing_variant": ["maximum intensity projection"],
       "processing_description": "Maximum intensity projection of a 7-plane z-stack (2 um steps)"
     }
   }

Migration from v0.2
-------------------

v0.2 did not specify projections. Projection stores written under v0.2 carry the
uniform-form fields, with the same names, in a top-level ``mip`` attribute beside
``ome`` and ``dca``. Upgrading such a store to v0.3 (which rewrites ``dca.version`` in
every image group anyway) **MUST** move that object to ``dca.projection`` unchanged and
remove the top-level ``mip``. Readers **MAY** fall back to a top-level ``mip`` when
reading stores that declare ``dca.version`` ``"0.2"``.
