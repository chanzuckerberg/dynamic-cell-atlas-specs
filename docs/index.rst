Dynamic Cell Atlas Specification
==================================

Welcome to the Dynamic Cell Atlas (DCA) data resource specifications.

These specifications define standards for storing 5D fluorescence microscopy datasets (3D+time+channels) in a standardized format that makes them interchangeable for further processing, with a focus on self-supervised pre-training AI models.

**Version 0.3** (draft) — Last updated: |last_updated_v03|

**Version 0.2** (latest release) — Last updated: |last_updated_v02|

.. toctree::
   :maxdepth: 2

   🚧 v0.3 - Draft <v0.3/index>
   v0.2 <v0.2/index>

.. toctree::
   :hidden:

   v0.1 <v0.1/index>

About
-----

The Dynamic Cell Atlas standard aims to be optimized for handling 5D datasets (3D+time+channels). It also supports 4D data (2D+time+channel, 3D+channel), but will not be optimized for it.

The DCA standard builds upon and aims to be compatible with:

* `Zarr v3 <https://zarr.readthedocs.io/en/stable/user-guide/v3_migration.html>`_
* `OME-NGFF v0.5 <https://ngff.openmicroscopy.org/0.5/>`_

Throughout these specifications, conformance requirements are expressed using :rfc:`2119` terminology (MUST, SHOULD, MAY, etc.).
