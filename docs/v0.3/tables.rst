Tables
======

Overview
--------

Many datasets include tabular data to store measurements or feature-extraction outputs
at the field-of-view, store or object level. This page outlines where tables live, their
layout, and how a unique object can be found from its pixels, its rows in any table and
its node in a tracking graph.

The DCA spec follows the `ngio table specification
<https://biovisioncenter.github.io/ngio/stable/table_specs/overview/>`_ as it is commonly
used in the `Fractal <https://fractal-analytics-platform.github.io/>`_ ecosystem, and
also stores data as Parquet. A DCA table is an ngio table kept at the root of the store,
as ngio keeps a plate's tables, with rows for every image group of the store. Each label
image has an **object index**, which lists its segmented objects, so that every object
can be counted and joined across tables and tracking graphs.

Conformance:

* Every instance label image in a store **MUST** have an object index (see
  `Object Index`_).
* A store without label images needs no tables beyond ``tables/obs``.
* Any other table is optional, but a table that is present **MUST** follow this page.

Table layout
------------

A table is a Zarr group, ``tables/<name>/``, laid out as the `ngio table specification
<https://biovisioncenter.github.io/ngio/stable/table_specs/overview/>`_ defines. In short, ngio describes a table in the attributes of its
group's ``zarr.json``: its ``type`` (one of the table types below), its ``table_version``,
its ``backend`` (the `format the data is stored in <https://biovisioncenter.github.io/ngio/stable/table_specs/backend/>`_), and, for a table
that describes the objects of a label image, its ``region`` (which label image) and its
``instance_key`` (the column that holds each object's label value). DCA uses these
attributes unchanged and adds none of its own. On top of ngio, a DCA table follows these
rules:

* **Location (a DCA choice).** ngio allows a table in an image group or in a plate. DCA
  uses only the second, for every store: each table lives at the root of the store,
  ``<store>/tables/<name>/``, so that one read finds all of a store's tables. Tables
  **MUST NOT** be stored inside image groups. When a tool writes a table into each image
  group, as Fractal tasks do, those tables are concatenated into one table of the same
  name at the store root (for a plate, what ngio's ``concatenate_image_tables``
  produces), and the per-image copies are deleted. In a store whose root is itself the
  image group, the two locations are the same.
* **Listing.** ``<store>/tables/zarr.json`` **MUST** list every table of the store by name
  in a ``tables`` attribute (``{"tables": ["obs", "nuclei_ROI_table", ...]}``), so that
  the tables can be found with one read, without listing the store.
* **Parquet.** ``backend`` is ``"parquet"`` and ``table_version`` is ``"1"``: the data is
  one Parquet file, ``tables/<name>/table.parquet``.
* **Types.** ``type`` is one of ngio's table types, each specified on its own ngio page:

  * `feature_table <https://biovisioncenter.github.io/ngio/stable/table_specs/table_types/feature_table/>`_: measurements of the objects of one
    label image (area, intensities, shape, a perturbation call, a classification).
  * `masking_roi_table <https://biovisioncenter.github.io/ngio/stable/table_specs/table_types/masking_roi_table/>`_: the objects of one label
    image, with their bounding boxes in micrometres; the type of the `Object Index`_.
  * `roi_table <https://biovisioncenter.github.io/ngio/stable/table_specs/table_types/roi_table/>`_: regions that are not objects, such as
    fields of view, as bounding boxes in micrometres.
  * `condition_table <https://biovisioncenter.github.io/ngio/stable/table_specs/table_types/condition_table/>`_: experimental conditions. A
    store has exactly one, ``tables/obs`` (see `The condition table`_).
  * `generic_table <https://biovisioncenter.github.io/ngio/stable/table_specs/table_types/generic_table/>`_: anything else.

``obs`` is reserved for the condition table.

.. code-block:: text

   plate.ome.zarr/
     zarr.json
     A/1/0/                         image group
       labels/nuclei/  labels/cells/
     tables/                        every table of the store
       zarr.json                    {"tables": ["obs", "nuclei_ROI_table", ...]}
       obs/                  zarr.json  table.parquet  condition table, one row per image group
       nuclei_ROI_table/     zarr.json  table.parquet  object index of nuclei
       cells_ROI_table/      zarr.json  table.parquet  object index of cells
       cellprofiler_nuclei/  zarr.json  table.parquet  nuclei features

Every path a table records is **relative**. No table records an absolute URL, so a store
that is copied or moved keeps working.

Tables of objects
-----------------

A table of objects (a ``feature_table`` or a ``masking_roi_table``) describes the objects
of **one label image**, as in ngio, in every image group of the store that has it:

* ``region`` names the label image, as ngio writes it: ``{"path":
  "../labels/<label_name>"}``, resolved from each row's image group.
* ``index_key`` and ``instance_key`` are ``"label"``. ``index_type`` is ``"int"`` or
  ``"str"``, as ngio writes it (ngio writes ``"str"`` when it combines per-image tables).

Because one table spans many image groups, each row also names its image group and
timepoint. These are the **key columns** of every table of objects:

.. list-table::
   :header-rows: 1
   :widths: 18 12 22 48

   * - Column
     - Type
     - Requirement
     - Meaning
   * - ``zarr_path``
     - string
     - **MUST**
     - The image group, relative to the store root, without a leading or trailing ``/``
       (e.g. ``A/1/0``). The same value as the image group's ``zarr_path`` in
       ``tables/obs``.
   * - ``row``, ``column``, ``path_in_well``
     - string
     - **SHOULD** in a plate
     - The image group's well row, well column and path in the well, the columns ngio
       adds when it combines a plate's per-image tables. Together they name the same
       image group as ``zarr_path``.
   * - ``label``
     - integer
     - **MUST**
     - The object's pixel value in the label image, greater than 0. It **MUST NOT**
       encode anything else; a value that combines a timepoint and an object number
       belongs in ``t`` and ``label`` separately.
   * - ``t``
     - int64
     - **MUST** when the label image has more than one timepoint
     - The zero-based timepoint index of the object.

A table **MAY** keep columns that a tool writes, such as CellProfiler's ``ImageNumber``
and ``ObjectNumber``. These are not keys: CellProfiler renumbers objects 1..N per image,
so its ``ObjectNumber`` is not in general the object's ``label``.

A perturbation call for an object, such as the guide detected in one cell of a pooled
screen, is a measurement of that object and belongs in a ``feature_table``. It does not
replace the image group's perturbation metadata, which records the conditions the image
group was designed with.

A value that an object has several of, such as several guide calls for one cell, is
recorded as **one row per value**, never as a list joined into one string.

Object Index
------------

The object index of a label image, ``tables/<label_name>_ROI_table/``, lists **every
object of that label image exactly once**, most often cells or nuclei: one row per
object (at one timepoint) in every image group that has the label image. It is an ngio
``masking_roi_table``, so that the rows of one image group form an ngio masking ROI table
that Fractal tools can use as they are.

Every **instance** label image, one whose pixel values each identify one object, **MUST**
have an object index. A label image that is not an instance segmentation, such as a class
map whose values are cell types or a binary foreground mask, has none.

.. list-table::
   :header-rows: 1
   :widths: 26 10 18 46

   * - Column
     - Type
     - Requirement
     - Meaning
   * - ``zarr_path``, ``label``, ``t``
     - see `Tables of objects`_
     - see `Tables of objects`_
     - The object's key. No two rows **MAY** share one.
   * - ``x_micrometer``, ``y_micrometer``, ``z_micrometer``, ``len_x_micrometer``,
       ``len_y_micrometer``, ``len_z_micrometer``
     - float64
     - **SHOULD**
     - The object's bounding box, in ngio's ROI columns: its lower corner and its size
       in micrometres, in the image group's physical coordinates.
   * - ``centroid_x_micrometer``, ``centroid_y_micrometer``, ``centroid_z_micrometer``
     - float64
     - **SHOULD**
     - The object's centroid (the mean position of its pixels) in the same coordinates.
       ``centroid_z_micrometer`` applies to a 3D label image.
   * - ``parent_label_name``, ``parent_label``
     - string, int64
     - **SHOULD** when the object lies within an object of another label image
     - That object, such as the cell a nucleus belongs to, in the same image group and
       timepoint: the name of its label image and its pixel value there. Given together
       or not at all; the parent has its own row in that label image's object index.

Linking objects across labels, tables and tracks
------------------------------------------------

An object's identity in a store is its image group, its label image, its timepoint and
its pixel value. Every structure that refers to the object uses that same pixel value,
under the name its own specification gives it:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Where
     - How it refers to the object
   * - The label image
     - The pixel value itself, in ``<image group>/labels/<label_name>``.
   * - A table of objects
     - ``region`` names the label image; ``zarr_path``, ``t`` and ``label`` (ngio's
       ``instance_key``) name the object in it.
   * - A `GEFF <https://liveimagetrackingtools.org/geff/>`_ tracking graph
     - A ``related_objects`` entry of type ``labels`` names the label image, and its
       ``node_prop`` names the node property that holds the object's pixel value
       (``seg_id`` in GEFF's example). The node's time property gives its timepoint;
       pixel values repeat across timepoints, so a node is matched with ``t`` as well.

A tracking graph's nodes join the object index of the label image its ``related_objects``
names, on (``zarr_path``, ``t``, ``label``), where ``zarr_path`` is that label image's
image group. Where a tracking graph is stored is outside this page. Whether two objects
at different timepoints are the same cell is a statement of the graph, not of these keys.

Across stores, an object's identity is its store's identity plus these keys. A store is
identified by a stable name, not by its location: a catalog that maps each store to where
it currently lives is updated when a store moves, and no table changes.

The condition table
-------------------

``tables/obs`` is the store's condition table: an ngio ``condition_table`` with **one row
per image group**, recording the conditions each image group was acquired under. Every
store has one.

* **Attributes.** ``type`` ``"condition_table"``, ``table_version`` ``"1"``, ``backend``
  ``"parquet"``, ``index_key`` ``"zarr_path"`` and ``index_type`` ``"str"``, so that ngio
  opens the table indexed by image group.
* **Data.** ``tables/obs/table.parquet``, the file ngio's Parquet backend reads. (A v0.2
  store names it ``obs.parquet``; ngio cannot open that file.)
* **Key.** ``zarr_path`` (**MUST**), the image group, as in `Tables of objects`_, with
  exactly one row per image group.
* **Conditions.** An image group's experimental metadata, its perturbations included,
  is recorded in its ``dca`` metadata (see :doc:`experimental-metadata`), and the row
  carries the same record.

Examples
--------

The ``zarr.json`` of the object index of a plate's ``nuclei`` label image:

.. code-block:: json

   {
     "zarr_format": 3,
     "node_type": "group",
     "attributes": {
       "type": "masking_roi_table",
       "table_version": "1",
       "backend": "parquet",
       "index_key": "label",
       "index_type": "int",
       "instance_key": "label",
       "region": {"path": "../labels/nuclei"}
     }
   }

A ``feature_table``'s ``zarr.json`` is the same, with ``type`` ``"feature_table"``.

Rows of the two object indexes of a plate's image group ``A/1/0``, a cell and the nucleus
inside it (bounding-box columns left out for space). Each label image has its own table;
the nucleus row names its cell by label image and label value, and that cell's own row is
in ``cells_ROI_table``.

``tables/cells_ROI_table/table.parquet``:

.. list-table::
   :header-rows: 1

   * - zarr_path
     - label
     - t
     - centroid_x_micrometer
     - centroid_y_micrometer
   * - ``A/1/0``
     - 17
     - 0
     - 332.8
     - 150.1

``tables/nuclei_ROI_table/table.parquet``:

.. list-table::
   :header-rows: 1

   * - zarr_path
     - label
     - t
     - centroid_x_micrometer
     - centroid_y_micrometer
     - parent_label_name
     - parent_label
   * - ``A/1/0``
     - 9
     - 0
     - 330.3
     - 148.2
     - ``cells``
     - 17

The ``zarr.json`` of the condition table, ``tables/obs``:

.. code-block:: json

   {
     "zarr_format": 3,
     "node_type": "group",
     "attributes": {
       "type": "condition_table",
       "table_version": "1",
       "backend": "parquet",
       "index_key": "zarr_path",
       "index_type": "str"
     }
   }

Validation
----------

``dca-validate --spec-version 0.3`` checks the tables of every store it validates from a
few metadata reads: the ``tables/`` listing, each table's ``zarr.json`` and Parquet
footer, and only the key columns where values are needed. v0.2 stores are not checked.

Errors (MUST):

* The store has no ``tables/obs`` condition table, or ``tables/obs`` keeps the v0.2
  ``obs.parquet`` instead of ``table.parquet``; ``tables/obs`` is not a
  ``condition_table``, or another table is.
* ``tables/zarr.json`` is missing, or its ``tables`` attribute does not list exactly the
  table groups present.
* A table group has no ``zarr.json``, or its attributes do not match
  ``$defs/StoreTable`` of the v0.3 schema; ``table.parquet`` is missing or unreadable.
* ``tables/obs`` has no ``zarr_path`` column, a ``zarr_path`` twice, or ``zarr_path``
  values that are not the store's image groups (for a plate, the fields its wells
  declare; for a store whose root is the image group, one row). When ``--wells``
  selects some wells, only their fields are checked for a row.
* A table of objects has no ``zarr_path`` or ``label`` column, a ``label`` that is not
  an integer greater than 0, only one of ``parent_label_name`` and ``parent_label``, or
  no ``t`` column while its label image has more than one timepoint.
* Two rows of a ``masking_roi_table`` share (``zarr_path``, ``t``, ``label``).
* A validated image group of a plate has its own ``tables`` group.

Warnings (SHOULD):

* A table of objects in a plate has no ``row``, ``column`` or ``path_in_well`` column.
* A label image of a validated image group has no ``masking_roi_table`` whose ``region``
  names it. Only an instance label image needs an object index, and the metadata does not
  say which label images are instance segmentations, so this is not an error.

Relation to other conventions
-----------------------------

* **GEFF.** Tracking graphs, and where they are stored, are GEFF's; this page only states
  how their nodes match the object index.
* **OME-NGFF.** OME-NGFF does not define tables yet. Its proposed collections RFC
  (`RFC-8 <https://ngff.openmicroscopy.org/rfc/8/index.html>`_) names ngio's tables as an
  existing convention and leaves table types to later proposals.
