OME-Zarr Metadata
=================

* OME-zarr v0.5 metadata **MUST** be included in ``zarr.json`` at the group and array level, with the fields described below:

Dimensions
----------

* The ``axes`` are used as part of OME-ngff v0.5 ``multiscales`` `metadata <https://ngff.openmicroscopy.org/0.5/#multiscale-md>`_.
  The length of ``axes`` **MUST** be equal to the number of dimensions of the arrays that contain the image data.
* The ``dimension_names`` attribute **MUST** be included in the ``zarr.json`` of the Zarr array of a multiscale level and **MUST** match the names in the ``axes`` metadata.

.. note::

   Axis names and ``dimension_names`` values MUST be all lowercase (``t``, ``c``, ``z``, ``y``, ``x``) or all uppercase (``T``, ``C``, ``Z``, ``Y``, ``X``). Mixed casing such as ``T``, ``C``, ``Z``, ``y``, ``x`` is rejected. The original casing is preserved.

Units
-----

* For spatial and temporal dimensions, units from the `OME-ngff v0.5 specification <https://ngff.openmicroscopy.org/0.5/#axes-md>`_ **MUST** be used.
* The temporal axis has one exception. OME-ngff v0.5 expresses spacing as a single uniform scale and its temporal vocabulary has no ``frame``, so an acquisition with unevenly spaced timepoints cannot name a unit honestly. Such a group **MAY** omit the ``t`` unit if it declares ``acquisition.time_sampling = "variable"``.
* A group declaring ``variable`` **MUST** carry ``acquisition.timestamps``, one non-decreasing entry per timepoint of the level-0 array in seconds since the Unix epoch (UTC), and **MUST** set the level-0 ``t`` scale to 1 so the axis reads as frame indices. Spatial axes are unaffected.

Multiresolution
---------------

* The used downsampling method **MUST** be documented in the ``zarr.json`` file.

Additional Metadata
-------------------

Datasets may provide additional metadata that is not captured by this standard.
The original metadata **MAY** additionally be stored in human-readable format next to individual arrays, i.e. next to ``zarr.json`` files containing ``ome.multiscales`` metadata.
