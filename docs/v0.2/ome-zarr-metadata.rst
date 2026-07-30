OME-Zarr Metadata
=================

* OME-zarr v0.5 metadata **MUST** be included in ``zarr.json`` at the group and array level, with the fields described below:

Dimensions
----------

* The ``axes`` are used as part of OME-ngff v0.5 ``multiscales`` `metadata <https://ngff.openmicroscopy.org/0.5/#multiscale-md>`_.
  The length of ``axes`` **MUST** be equal to the number of dimensions of the arrays that contain the image data.
* The ``dimension_names`` attribute **MUST** be included in the ``zarr.json`` of the Zarr array of a multiscale level and **MUST** match the names in the ``axes`` metadata.

Units
-----

* For spatial and temporal dimensions, units from the `OME-ngff v0.5 specification <https://ngff.openmicroscopy.org/0.5/#axes-md>`_ **MUST** be used.

Multiresolution
---------------

* The used downsampling method **MUST** be documented in the ``zarr.json`` file.

Additional Metadata
-------------------

Datasets may provide additional metadata that is not captured by this standard.
The original metadata **MAY** additionally be stored in human-readable format next to individual arrays, i.e. next to ``zarr.json`` files containing ``ome.multiscales`` metadata.
