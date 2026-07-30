DCA Metadata
============

DCA metadata lives in the ``zarr.json`` of the **multiscales image group** (i.e. the same level as ``ome.multiscales``), under the ``dca`` key,
enabling varying channel composition at arbitrary levels of the OME-Zarr hierarchy.
In an OME-Zarr HCS layout this is the field-of-view level (e.g. ``plate.ome.zarr/A/3/0/zarr.json``).

The ``dca`` object **MUST** contain:

* ``version`` — a string specifying the version of this specification (currently ``"0.2"``)
* ``url`` — a URL pointing to the specification in the GitHub repository
* ``channels`` — an array of channel metadata objects (see :doc:`channel-metadata`)
* ``normalization_statistics`` — per-channel normalization statistics keyed by channel index (see :doc:`normalization-statistics`)

For image groups it **MUST** additionally carry the experimental-metadata
**required floor** — the ``study``, ``biosample``, ``specimen``,
``study_component``, and ``provenance`` blocks — mirrored from the curated Parquet
(see :ref:`experimental-metadata-storage` below and :doc:`experimental-metadata`).

One further key is **conditionally required** — required when present, in the same
sense as the existing ``WellPerturbation``. **Most datasets will not have it, and
omitting it is correct for them:**

* ``perturbation`` — the ``PerturbationAssignment`` applied to this image group's
  material: the ``Perturbation`` it references (its ``definition`` slot), plus dosing
  / control class / timing (see
  :ref:`Where it is written <experimental-perturbation-placement>`). A dataset **with**
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
``version``, ``url``, ``channels``, and ``normalization_statistics`` fields; the
required floor is defined by a separate schema
(``schema/v0.2/dca_experimental_metadata.yaml``, see :doc:`experimental-metadata`).

.. _experimental-metadata-storage:

Experimental metadata storage
-----------------------------

The **full** experimental-metadata record is agent-curated into a sibling Parquet
(the source of truth). Its **required floor** (``study``, ``biosample``,
``specimen``, ``study_component``, ``provenance``) is additionally mirrored into
this ``dca`` block so that a store is self-validating. Recommended/Optional fields
and non-floor blocks (e.g. ``acquisition``, ``image_data``) live **only** in the
Parquet (see :doc:`experimental-metadata`). Label image groups are exempt from the
``dca`` metadata requirements, including the required floor.

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
