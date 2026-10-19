DCA Metadata
============

DCA metadata lives in the ``zarr.json`` of the **multiscales image group** (i.e. the same level as ``ome.multiscales``), under the ``dca`` key,
enabling varying channel composition at arbitrary levels of the OME-Zarr hierarchy.
In an OME-Zarr HCS layout this is the field-of-view level (e.g. ``plate.ome.zarr/A/3/0/zarr.json``).

The ``dca`` object **MUST** contain:

* ``version`` — a string specifying the version of this specification (currently ``"0.3"``)
* ``url`` — a URL pointing to the specification in the GitHub repository
* ``channels`` — an array of channel metadata objects (see :doc:`channel-metadata`)
* ``normalization_statistics`` — per-channel normalization statistics keyed by channel index (see :doc:`normalization-statistics`)

For image groups it **MUST** additionally carry the experimental-metadata
**required floor** — the ``study``, ``biosample``, ``specimen``,
``study_component``, and ``provenance`` blocks — mirrored from the curated Parquet
(see :ref:`v03-experimental-metadata-storage` below and :doc:`experimental-metadata`).

Two further keys are **conditionally required**:

* ``projection`` — the parameters of a projected image (e.g. a maximum intensity
  projection). **MUST** be present on every projection and
  **MUST NOT** be present on an image that is not a projection. Like ``channels`` and
  ``normalization_statistics`` it describes the array and is not part of the
  experimental-metadata record. See :doc:`projections`.

* ``perturbation`` — required for datasets with perturbations applied; a dataset
  without perturbations omits it, and omitting it is correct for them. It is the
  ``PerturbationAssignment`` applied to this image group's material: the
  ``Perturbation`` it references (its ``definition`` slot), plus dosing / control
  class / timing (see
  :ref:`Where it is written <v03-experimental-perturbation-placement>`). A dataset **with**
  perturbations MUST carry it on every image group; a dataset with **none** simply
  omits the key, and omitting it is not an error. Whether a dataset "has
  perturbations" is a judgement about the experiment, not a machine-checkable property
  of the store, so the validator checks the block's *contents* when present and never
  requires its presence.

.. code-block:: json

   {
     "ome": {
       "multiscales": [ "..." ]
     },
     "dca": {
       "version": "...",
       "url": "...",
       "channels": [ "..." ],
       "normalization_statistics": { "..." },
       "study": { "..." },
       "biosample": { "..." },
       "specimen": { "..." },
       "study_component": { "..." },
       "provenance": { "..." }
     }
   }

The :download:`JSON Schema <schema.json>` is the authoritative definition of the
``version``, ``url``, ``channels``, ``normalization_statistics`` and ``projection``
fields; the
required floor is defined by a separate schema
(``schema/v0.3/dca_experimental_metadata.yaml``, see :doc:`experimental-metadata`).

.. _v03-experimental-metadata-storage:

Experimental metadata storage
-----------------------------

The **full** experimental-metadata record is curated and written to two places:
this ``dca`` block, and a sibling Parquet that is the source of truth. Both carry the
same record — the **required floor** (``study``, ``biosample``, ``specimen``,
``study_component``, ``provenance``) together with the recommended and optional
fields and the remaining blocks (e.g. ``acquisition``, ``image_data``) — and the
``zarr.json`` copy is derived from the Parquet.

What the floor names is not a location but a conformance class: those blocks **MUST**
be present, which is what makes a store self-validating. Recommended and optional
fields are written to the same block; a store is never failed for omitting one, but
whatever is present is validated like the rest (see :doc:`experimental-metadata`). Label image groups are exempt from the ``dca``
metadata requirements, including the required floor.

An **arrayed HCS screen** may additionally write a ``WellPerturbation`` — the same
assignment plus its ``(plate, well)`` key — to the **well group's** ``zarr.json``, so a
plate's design can be enumerated without opening every field of view. That row is
*derived*: the image-group ``perturbation`` block above is authoritative, and it is the
placement every layout can carry, including a plateless ``image_collection``.

.. toctree::
   :maxdepth: 2

   channel-metadata
   normalization-statistics
   experimental-metadata
   experimental-metadata-example
