from __future__ import annotations

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer
)


metamodel_version = "1.11.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )





class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'dca',
     'default_range': 'string',
     'description': 'LinkML schema for Dynamic Cell Atlas experimental metadata '
                    '(v0.2) — the single source of truth for the acquisition, '
                    'study/sample, and perturbation context curated for DCA '
                    'datasets.\n'
                    'Required metadata is written to the ``zarr.json`` ``dca`` '
                    'block (alongside the machine-measured channels + '
                    'normalization statistics); optional / conditional metadata is '
                    'agent-curated into a sibling Parquet. Every value carries its '
                    'provenance as plain columns alongside it (citation incl. '
                    'grounding match + grounded flag + source_kind), blank when '
                    'not applicable, in both stores. A field is promoted into '
                    '``zarr.json`` when verifiable, schema-stable, and '
                    'consistently scoped.\n'
                    'Anchoring: the overall Study / Biosample / Specimen / '
                    'acquisition structure follows the REMBI reporting guideline; '
                    'within it, instrument / acquisition fields → Bio-Formats / '
                    'OME; study & people → OME ``Experiment`` / ``Experimenter`` + '
                    'Broad CellPainting Gallery; cell line & organism → '
                    'CellPainting Gallery + NCBITaxon; perturbation → OME '
                    '``Reagent`` + CPG + JUMP. Concepts with no equivalent in '
                    'those sources are added as DCA-native extensions '
                    '(REMBI-derived).',
     'id': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
     'imports': ['linkml:types'],
     'license': 'MIT',
     'name': 'dca_experimental_metadata',
     'prefixes': {'CL': {'prefix_prefix': 'CL',
                         'prefix_reference': 'http://purl.obolibrary.org/obo/CL_'},
                  'CVCL': {'prefix_prefix': 'CVCL',
                           'prefix_reference': 'https://www.cellosaurus.org/CVCL_'},
                  'EFO': {'prefix_prefix': 'EFO',
                          'prefix_reference': 'http://www.ebi.ac.uk/efo/EFO_'},
                  'FBbi': {'prefix_prefix': 'FBbi',
                           'prefix_reference': 'http://purl.obolibrary.org/obo/FBbi_'},
                  'FBdv': {'prefix_prefix': 'FBdv',
                           'prefix_reference': 'http://purl.obolibrary.org/obo/FBdv_'},
                  'HGNC': {'prefix_prefix': 'HGNC',
                           'prefix_reference': 'http://identifiers.org/hgnc.symbol/'},
                  'HsapDv': {'prefix_prefix': 'HsapDv',
                             'prefix_reference': 'http://purl.obolibrary.org/obo/HsapDv_'},
                  'MONDO': {'prefix_prefix': 'MONDO',
                            'prefix_reference': 'http://purl.obolibrary.org/obo/MONDO_'},
                  'MmusDv': {'prefix_prefix': 'MmusDv',
                             'prefix_reference': 'http://purl.obolibrary.org/obo/MmusDv_'},
                  'NCBIGene': {'prefix_prefix': 'NCBIGene',
                               'prefix_reference': 'http://purl.obolibrary.org/obo/NCBIGene_'},
                  'NCBITaxon': {'prefix_prefix': 'NCBITaxon',
                                'prefix_reference': 'http://purl.obolibrary.org/obo/NCBITaxon_'},
                  'OME': {'prefix_prefix': 'OME',
                          'prefix_reference': 'https://www.openmicroscopy.org/Schemas/OME/2016-06#'},
                  'PATO': {'prefix_prefix': 'PATO',
                           'prefix_reference': 'http://purl.obolibrary.org/obo/PATO_'},
                  'PubChem': {'prefix_prefix': 'PubChem',
                              'prefix_reference': 'http://identifiers.org/pubchem.compound/'},
                  'REMBI': {'prefix_prefix': 'REMBI',
                            'prefix_reference': 'https://www.nature.com/articles/s41592-021-01166-8#'},
                  'UBERON': {'prefix_prefix': 'UBERON',
                             'prefix_reference': 'http://purl.obolibrary.org/obo/UBERON_'},
                  'WBls': {'prefix_prefix': 'WBls',
                           'prefix_reference': 'http://purl.obolibrary.org/obo/WBls_'},
                  'ZFA': {'prefix_prefix': 'ZFA',
                          'prefix_reference': 'http://purl.obolibrary.org/obo/ZFA_'},
                  'ZFS': {'prefix_prefix': 'ZFS',
                          'prefix_reference': 'http://purl.obolibrary.org/obo/ZFS_'},
                  'bioformats': {'prefix_prefix': 'bioformats',
                                 'prefix_reference': 'https://bio-formats.readthedocs.io/en/v8.5.0/metadata-summary.html#'},
                  'cdp': {'prefix_prefix': 'cdp',
                          'prefix_reference': 'https://github.com/chanzuckerberg/cryoet-data-portal-backend/blob/main/schema/core/v2.0.0/common.yaml#'},
                  'cpg': {'prefix_prefix': 'cpg',
                          'prefix_reference': 'https://github.com/broadinstitute/cellpainting-gallery-metadata/blob/main/harmonized_ontology.json#'},
                  'cxmod': {'prefix_prefix': 'cxmod',
                            'prefix_reference': 'https://github.com/chanzuckerberg/data-guidance/blob/main/standards/cross-modality/1.1.0/schema.md#'},
                  'dca': {'prefix_prefix': 'dca',
                          'prefix_reference': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata/'},
                  'ensembl': {'prefix_prefix': 'ensembl',
                              'prefix_reference': 'http://identifiers.org/ensembl/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'ops': {'prefix_prefix': 'ops',
                          'prefix_reference': 'https://github.com/chanzuckerberg/ops-schema/blob/main/standards/ops/0.1.0/experimental-metadata.md#'},
                  'sc': {'prefix_prefix': 'sc',
                         'prefix_reference': 'https://schema.org/'},
                  'sdc': {'prefix_prefix': 'sdc',
                          'prefix_reference': 'https://github.com/chanzuckerberg/dataset-catalog#'}},
     'source_file': 'schema/v0.2/dca_experimental_metadata.yaml',
     'title': 'DCA Experimental Metadata'} )

class MicroscopeType(str, Enum):
    """
    OME Microscope.Type vocabulary.
    """
    Upright = "Upright"
    Inverted = "Inverted"
    Dissection = "Dissection"
    Electrophysiology = "Electrophysiology"
    Other = "Other"


class ObjectiveImmersion(str, Enum):
    """
    OME Objective.Immersion vocabulary.
    """
    Oil = "Oil"
    Water = "Water"
    WaterDipping = "WaterDipping"
    Air = "Air"
    Multi = "Multi"
    Glycerol = "Glycerol"
    Other = "Other"


class ObjectiveCorrection(str, Enum):
    """
    OME Objective.Correction vocabulary.
    """
    UV = "UV"
    PlanApo = "PlanApo"
    PlanFluor = "PlanFluor"
    SuperFluor = "SuperFluor"
    VioletCorrected = "VioletCorrected"
    Achro = "Achro"
    Achromat = "Achromat"
    Fluor = "Fluor"
    Fl = "Fl"
    Neofluar = "Neofluar"
    Apo = "Apo"
    Other = "Other"


class DetectorType(str, Enum):
    """
    OME Detector.Type vocabulary.
    """
    CCD = "CCD"
    EMCCD = "EMCCD"
    CMOS = "CMOS"
    PMT = "PMT"
    Photodiode = "Photodiode"
    Other = "Other"


class ProvenanceSourceKind(str, Enum):
    """
    How the value was obtained, which determines whether the provenance columns (citation + grounding) are **required**: required for ``agent`` values, optional for ``human`` and ``original_data``. DCA-defined.
    """
    agent = "agent"
    """
    Extracted by an agent from a source document. The provenance columns (citation + grounding-gate result) are REQUIRED.
    """
    human = "human"
    """
    Manually entered by a person. Provenance is optional (the citation MAY be blank).
    """
    original_data = "original_data"
    """
    Taken directly from the data source's own provided metadata (e.g. a metadata CSV on the source website), not prose-extracted. Provenance optional.
    """


class MatchStatus(str, Enum):
    """
    Result of an agent's deterministic grounding gate locating a cited quote in its source page. Carried with the value so the grounding result stays auditable downstream, where the raw source corpus is unavailable.
    """
    exact = "exact"
    """
    Quote found verbatim in the cited source page.
    """
    fuzzy = "fuzzy"
    """
    Quote located above the difflib similarity threshold (minor whitespace / punctuation differences) — still grounded.
    """
    none = "none"
    """
    Quote not found in the cited source page (ungrounded).
    """


class PerturbationModality(str, Enum):
    """
    How the perturbation acts. Normalized superset of CellPainting Gallery ``Treatment_Category`` ([ORF, Compound, CRISPR, shRNA, miRNA, None]) and JUMP ``perturbation_modality`` ([compound, orf, crispr, unknown]). Values are lowercased to reconcile the two casings; ``unknown`` from JUMP is retained.
    """
    compound = "compound"
    """
    Small-molecule / chemical perturbation.
    """
    orf = "orf"
    """
    Open reading frame overexpression.
    """
    crispr = "crispr"
    """
    CRISPR knockout / interference / activation.
    """
    shrna = "shrna"
    """
    Short hairpin RNA knockdown (CPG only).
    """
    mirna = "mirna"
    """
    microRNA perturbation (CPG only).
    """
    sirna = "sirna"
    """
    Small interfering RNA knockdown — transient, transfected synthetic duplex, distinct from the vector-delivered, stably-expressed ``shrna``. Not in the CPG/JUMP source vocabularies (e.g. RxRx1's ``sirna`` column).
    """
    physical = "physical"
    """
    Physical perturbation (e.g. heat shock, irradiation).
    """
    biological = "biological"
    """
    Biological perturbation (e.g. pathogen, ligand, co-culture).
    """
    other = "other"
    """
    Perturbation not covered by the categories above.
    """
    unknown = "unknown"
    """
    Modality not determinable (JUMP only).
    """


class PerturbationControlClass(str, Enum):
    """
    Control role of a well. Reconciles CellPainting Gallery ``Treatment_Control_Class`` ([Control, Treatment, NegCon, PosCon]) and JUMP control ``pert_type`` ([negcon, poscon, empty]). CPG's redundant "Control" is dropped in favor of the explicit negcon/poscon; JUMP's ``empty`` is kept.
    """
    treatment = "treatment"
    """
    Experimental (non-control) perturbation.
    """
    negcon = "negcon"
    """
    Negative control.
    """
    poscon = "poscon"
    """
    Positive control.
    """
    empty = "empty"
    """
    Empty / untreated well (JUMP).
    """


class TissueType(str, Enum):
    """
    CZI cross-modality ``tissue_type`` (v1.1.0), plus ``unknown`` for sources that do not record what the sample was.
    """
    tissue = "tissue"
    cell_culture = "cell culture"
    cell_line = "cell line"
    organoid = "organoid"
    unknown = "unknown"
    """
    Sample type not determinable from the source. For image corpora assembled without a per-image manifest (e.g. segmentation training sets pooled from many studies), where the sample type genuinely is not recorded. Carries the same meaning as the ``unknown`` sentinel on the ontology-term slots, and is likewise not a shortcut for "not looked up yet".
    """


class SpecimenState(str, Enum):
    """
    Specimen state at imaging time — the coarse, mutually-exclusive live-vs-fixed axis. DCA controlled vocabulary (REMBI-derived); no source ontology structures this state cleanly across specimen types. Specific preparation methods (fixation subtype, permeabilization, sectioning, whole-mount, clearing, expansion) are recorded in ``Specimen.preparation_method``.
    """
    live = "live"
    """
    Imaged alive / unfixed (e.g. live-cell or live organoid imaging).
    """
    fixed = "fixed"
    """
    Chemically or cryo-fixed before imaging.
    """
    unknown = "unknown"
    """
    State not determinable from the source.
    """


class StudyType(str, Enum):
    """
    OME ``Experiment.Type`` vocabulary for the study-level experiment type / mode, plus ``Other`` (which requires a free-text ``Study.study_type_other``).
    """
    FP = "FP"
    FRET = "FRET"
    TimeLapse = "TimeLapse"
    FourDPlus = "FourDPlus"
    Screen = "Screen"
    Immunocytochemistry = "Immunocytochemistry"
    Immunofluorescence = "Immunofluorescence"
    FISH = "FISH"
    Electrophysiology = "Electrophysiology"
    IonImaging = "IonImaging"
    Colocalization = "Colocalization"
    PGIDocumentation = "PGIDocumentation"
    FluorescenceLifetime = "FluorescenceLifetime"
    SpectralImaging = "SpectralImaging"
    Photobleaching = "Photobleaching"
    SPIM = "SPIM"
    Other = "Other"



class ExperimentalMetadata(ConfiguredBaseModel):
    """
    Root container for one image group's experimental metadata (a flattened REMBI Study + Study Component): the study, the ``study_component`` (assay class + imaging method), the sample (``biosample`` + ``specimen``), acquisition, and image-data context, the ``perturbation`` assignment, plus the per-value ``provenance`` map. This is the object curated into a Parquet row (one per ``Biosample.group_id``); the required fields are mirrored into the ``zarr.json`` ``dca`` block. Arrayed HCS screens ALSO index the same assignment per well as :class:`WellPerturbation` (well-level); that well-level form is a second placement of this block, not a different fact.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'tree_root': True})

    study: Study = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    biosample: Biosample = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    specimen: Specimen = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    perturbation: Optional[PerturbationAssignment] = Field(default=None, description="""The perturbation applied to this image group's material — dosing / control class / timing plus the :class:`Perturbation` it references (its ``definition`` slot). CONDITIONALLY REQUIRED: present for every image group of a dataset that has perturbations, absent for a dataset that has none. Mirrored into the ``zarr.json`` ``dca`` block like the other required blocks, so a perturbation is readable from the image group a consumer already has open, without resolving a plate / well key — and so it is expressible for a plateless ``image_collection`` store. This is the AUTHORITATIVE placement: an arrayed HCS screen may additionally index the same assignment per well as a :class:`WellPerturbation`, which is derived from this block.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    acquisition: Optional[AcquisitionMetadata] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    image_data: Optional[ImageData] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    study_component: StudyComponent = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata']} })
    provenance: dict[str, FieldProvenance] = Field(default=..., description="""Per-value provenance, as a map keyed by each value's dotted ``field_path`` (e.g. ``biosample.organism``). A parallel provenance layer; a deterministic post-ingest step folds each entry onto its value's provenance columns in the Parquet and the ``zarr.json`` mirror. REQUIRED for ``agent``-curated values, optional for ``human`` / ``original_data`` (see :class:`FieldProvenance` / :class:`Provenance`).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata',
                       'AcquisitionMetadata',
                       'PerturbationAssignment']} })


class StudyComponent(ConfiguredBaseModel):
    """
    REMBI Study Component: the experiment-class descriptors that organise this image group, namely the assay / experiment class (EFO) and the acquisition modality (FBbi). Distinct from the instrument attributes in :class:`AcquisitionMetadata` and from the project-level :class:`Study`.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    imaging_method: list[ImagingMethodTerm] = Field(default=..., description="""Acquisition **modality** as an ``ontology_term_id`` + ``label`` pair, where the id is a Biological Imaging Methods Ontology (FBbi) term — e.g. ``FBbi:00000251`` (confocal microscopy), ``FBbi:00000369`` (light-sheet / selective plane illumination), ``FBbi:00000246`` (fluorescence microscopy). **FBbi only.** The technique used to acquire the image data, distinct from the EFO assay-class in ``assay`` (EFO gives the experiment class, this gives how it was imaged) and from the instrument attributes in :class:`AcquisitionMetadata`. REQUIRED for every imaging dataset; use the ``unavailable`` sentinel with a free-text ``label`` **only** when no FBbi term fits the modality. ``unknown`` and ``na`` are not permitted: the acquiring lab always knows the modality and it always applies, so a real FBbi term or ``unavailable`` is always determinable.""", json_schema_extra = { "linkml_meta": {'domain_of': ['StudyComponent']} })
    assay: list[AssayTerm] = Field(default=..., description="""The **assay / experiment class** as an ``ontology_term_id`` + ``label`` pair, where the id is any EFO assay term, not limited to these examples: ``EFO:0009918`` (smFISH), ``EFO:0022955`` (imaging-based spatial transcriptomics), or ``EFO:0002909`` (microscopy assay) as the coarse fallback. **EFO only** (cross-modality / CELLxGENE-aligned). \"What kind of experiment\". The specific acquisition **modality** (confocal, light-sheet, …) is *not* recorded here — it lives in ``imaging_method`` (FBbi). One ontology per field: EFO answers \"what kind of experiment\", FBbi \"how was it imaged\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['StudyComponent']} })


class AcquisitionMetadata(ConfiguredBaseModel):
    """
    Image-acquisition metadata (REMBI Image Acquisition): instrument attributes (microscope / objective / detector / imaging environment) and acquisition parameters (acquisition date, time increment, …), all OME-aligned. All optional, so stored in the Parquet (not ``zarr.json``). The acquisition **modality** (FBbi ``imaging_method``) is not here: REMBI scopes it to the Study Component, so it lives on :class:`ExperimentalMetadata`. Populate what the source supports; omit the rest rather than guessing.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    provenance: Optional[Provenance] = Field(default=None, description="""Per-value provenance (see Provenance).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata',
                       'AcquisitionMetadata',
                       'PerturbationAssignment']} })
    acquisition_date: Optional[datetime ] = Field(default=None, description="""Acquisition timestamp (ISO 8601).""", json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata'],
         'exact_mappings': ['bioformats:Image.AcquisitionDate']} })
    time_increment_s: Optional[float] = Field(default=None, description="""Time between successive timepoints, in seconds. Required for time-lapse data; the validator derives the time-lapse condition from the ``zarr.json`` array shape (time dimension T > 1) rather than from a metadata flag, so this is enforced by the validator, not a LinkML ``if/then`` rule.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata'],
         'exact_mappings': ['bioformats:Image.TimeIncrement'],
         'unit': {'ucum_code': 's'}} })
    timepoint_acquisition: Optional[str] = Field(default=None, description="""Acquisition timepoint (elapsed acquisition time).""", json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata'],
         'exact_mappings': ['cpg:Timepoint_Acquisition']} })
    plate_type: Optional[str] = Field(default=None, description="""Plate or imaging vessel used. SHOULD include well count, substrate, and catalog identifier (e.g. ``\"96-well glass-bottom (Cellvis P96-1.5H-N)\"``). Maps to OPS ``cellular.plate_type``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata'],
         'exact_mappings': ['ops:cellular.plate_type']} })
    microscope: Optional[Microscope] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata']} })
    objective: Optional[Objective] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata']} })
    detector: Optional[Detector] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata']} })
    imaging_environment: Optional[ImagingEnvironment] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['AcquisitionMetadata']} })


class Microscope(ConfiguredBaseModel):
    """
    Imaging system make/model. Maps to OME Microscope.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Microscope',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    manufacturer: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Microscope.Manufacturer']} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Microscope.Model', 'cpg:Microscope_Name']} })
    type: Optional[MicroscopeType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Microscope.Type']} })
    modality: Optional[str] = Field(default=None, description="""Imaging modality (e.g. fluorescence, brightfield).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Perturbation'],
         'exact_mappings': ['cpg:Microscope_Modality']} })
    binning: Optional[str] = Field(default=None, description="""Detector binning (e.g. \"2x2\").""", json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope'], 'exact_mappings': ['cpg:Microscope_Binning']} })


class Objective(ConfiguredBaseModel):
    """
    Objective lens. Maps to OME Objective.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Objective',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    nominal_magnification: Optional[float] = Field(default=None, description="""Nominal (engraved) magnification, e.g. 63 for a 63x lens.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Objective'],
         'exact_mappings': ['bioformats:Objective.NominalMagnification',
                            'cpg:Microscope_Objective_Magnification']} })
    lens_na: Optional[float] = Field(default=None, description="""Numerical aperture of the objective.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Objective'],
         'exact_mappings': ['bioformats:Objective.LensNA',
                            'cpg:Microscope_Objective_NA']} })
    immersion: Optional[ObjectiveImmersion] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Objective'],
         'exact_mappings': ['bioformats:Objective.Immersion']} })
    correction: Optional[ObjectiveCorrection] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Objective'],
         'exact_mappings': ['bioformats:Objective.Correction']} })
    working_distance_mm: Optional[float] = Field(default=None, description="""Working distance in millimeters.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Objective'],
         'exact_mappings': ['bioformats:Objective.WorkingDistance'],
         'unit': {'ucum_code': 'mm'}} })


class Detector(ConfiguredBaseModel):
    """
    Detector / camera. Maps to OME Detector.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Detector',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    type: Optional[DetectorType] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Detector.Type']} })
    model: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Detector.Model']} })
    manufacturer: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Detector'],
         'exact_mappings': ['bioformats:Detector.Manufacturer']} })


class ImagingEnvironment(ConfiguredBaseModel):
    """
    Environmental conditions during acquisition. Maps to OME ImagingEnvironment. RECOMMENDED for live-cell datasets.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:ImagingEnvironment',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    temperature_c: Optional[float] = Field(default=None, description="""Temperature in degrees Celsius.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ImagingEnvironment'],
         'exact_mappings': ['bioformats:ImagingEnvironment.Temperature'],
         'unit': {'ucum_code': 'Cel'}} })
    co2_percent: Optional[float] = Field(default=None, description="""CO2 concentration, percent (0-100).""", ge=0, le=100, json_schema_extra = { "linkml_meta": {'domain_of': ['ImagingEnvironment'],
         'exact_mappings': ['bioformats:ImagingEnvironment.CO2Percent']} })
    humidity_percent: Optional[float] = Field(default=None, description="""Relative humidity, percent (0-100).""", ge=0, le=100, json_schema_extra = { "linkml_meta": {'domain_of': ['ImagingEnvironment'],
         'exact_mappings': ['bioformats:ImagingEnvironment.Humidity']} })
    air_pressure: Optional[float] = Field(default=None, description="""Air pressure in millibar.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ImagingEnvironment'],
         'exact_mappings': ['bioformats:ImagingEnvironment.AirPressure']} })


class Study(ConfiguredBaseModel):
    """
    Study-level descriptors. Anchored on OME ``Experiment`` + the Broad CellPainting Gallery study fields + schema.org. Required fields are written to ``zarr.json``; optional fields to the Parquet.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Experiment',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'rules': [{'description': 'A JUMP-CP dataset (identified by its '
                                   '``dataset_name``) MUST state the CellProfiler '
                                   'version ``cp_version`` it was processed with. The '
                                   'condition is keyed on ``dataset_name`` — a schema '
                                   'slot — so this is a true ``if/then`` rule rather '
                                   'than a validator-only check.',
                    'postconditions': {'slot_conditions': {'cp_version': {'name': 'cp_version',
                                                                          'required': True}}},
                    'preconditions': {'slot_conditions': {'dataset_name': {'name': 'dataset_name',
                                                                           'pattern': '.*[Jj][Uu][Mm][Pp][ '
                                                                                      '_-]?[Cc][Pp].*'}}}}]})

    accession_id: Optional[str] = Field(default=None, description="""Repository accession, when the dataset was deposited in a public archive (e.g. BioImage Archive ``S-BIAD####``, IDR ``idr####``). RECOMMENDED — SHOULD be provided when the dataset is deposited; omitted for in-house data with no repository accession (such datasets are addressed by their store location instead).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'recommended': True} })
    source: str = Field(default=..., description="""Data-generating center / source.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'exact_mappings': ['cpg:Source']} })
    study_type: list[StudyType] = Field(default=..., description="""The overall **study type** (REMBI \"Study type\"; OME ``Experiment.Type``): the experimental design / modes, from the OME ``Experiment.Type`` vocabulary (:class:`StudyType`). Multivalued, so a multimodal study lists several (e.g. ``FourDPlus`` + ``TimeLapse`` + ``SPIM`` + ``FP`` + ``Immunofluorescence``). Study-level and may span the study's components, distinct from the per-component acquisition **modality** (the FBbi ``StudyComponent.imaging_method``) and **assay class** (the EFO ``StudyComponent.assay``). Use ``Other`` and fill ``study_type_other`` only when the vocabulary is insufficient.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'exact_mappings': ['bioformats:Experiment.Type']} })
    study_type_other: Optional[str] = Field(default=None, description="""Free-text study type, REQUIRED when ``study_type`` includes ``Other`` (the OME ``Experiment.Type`` vocabulary did not adequately describe the study); omit otherwise. The \"required-when-``Other``\" condition is enforced by the :doc:`validator <validator>`, not a LinkML ``if/then`` rule, because ``gen-json-schema`` cannot express an \"array contains a value\" precondition.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study']} })
    description: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Study', 'Perturbation'],
         'exact_mappings': ['bioformats:Experiment.Description']} })
    year_imaged: str = Field(default=..., description="""Four-digit year the data were acquired. Required; use the ``unknown`` sentinel when the acquisition year cannot be determined. Stored as a string so the ``unknown`` / ``na`` sentinel is expressible — mirroring the cross-modality term-id fields, so a missing value is stated explicitly rather than silently dropped or invented.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'exact_mappings': ['cpg:Year_Imaged']} })
    cp_version: Optional[str] = Field(default=None, description="""CellProfiler version used to process the dataset. Conditional — required for JUMP-CP datasets, enforced by a schema ``if/then`` rule keyed on ``dataset_name`` (which names the JUMP-CP dataset); see the class rule above.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['ops:pipeline.version'],
         'domain_of': ['Study'],
         'exact_mappings': ['cpg:CP_Version']} })
    protocol_url: Optional[str] = Field(default=None, description="""URL to the full experimental protocol (e.g. protocols.io).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study']} })
    experimenter: list[Experimenter] = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Study']} })
    dataset_name: str = Field(default=..., description="""The dataset's **own** human-readable name — the title of the *data*, not of any paper (distinct from ``Publication.title``). Searchable; present even for unpublished datasets. schema.org ``sc:name`` / catalog ``Dataset.name``.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:name'],
         'domain_of': ['Study'],
         'exact_mappings': ['sc:name', 'ops:experiment.title']} })
    release_date: date = Field(default=..., description="""schema.org (``sc:datePublished``).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'exact_mappings': ['sc:datePublished']} })
    license: str = Field(default=..., description="""License URL or SPDX identifier. schema.org (``sc:license``).""", json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:governance.license'],
         'domain_of': ['Study'],
         'exact_mappings': ['sc:license']} })
    related_publication: Optional[list[Publication]] = Field(default=None, description="""The dataset's publication(s), each a self-contained citation (``doi`` + title + authors + year). schema.org. **RECOMMENDED** — SHOULD be provided whenever the dataset has an associated publication; omitted when there is none.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study'], 'exact_mappings': ['sc:citation'], 'recommended': True} })
    additional_metadata: Optional[list[AdditionalMetadataField]] = Field(default=None, description="""Extra study-provided values not yet defined in this spec. Optional; promotable — once a field here stabilizes it can be elevated to a defined (ultimately required, zarr.json) slot in a future version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study', 'Biosample']} })

    @field_validator('year_imaged')
    def pattern_year_imaged(cls, v):
        pattern=re.compile(r"^(\d{4}|unknown|na)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid year_imaged format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid year_imaged format: {v}"
            raise ValueError(err_msg)
        return v


class Experimenter(ConfiguredBaseModel):
    """
    A person associated with the study. Anchored on OME ``Experimenter``.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Experimenter',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    first_name: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Experimenter'],
         'exact_mappings': ['bioformats:Experimenter.FirstName']} })
    last_name: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Experimenter'],
         'exact_mappings': ['bioformats:Experimenter.LastName']} })
    email: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Experimenter'],
         'exact_mappings': ['bioformats:Experimenter.Email']} })
    institution: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Experimenter'],
         'exact_mappings': ['bioformats:Experimenter.Institution']} })
    orcid: Optional[str] = Field(default=None, description="""ORCID iD. DCA-native (REMBI-derived).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Experimenter']} })


class Publication(ConfiguredBaseModel):
    """
    A related publication (the paper), within ``Study.related_publication``. A self-contained citation: ``doi`` (the machine-readable key) plus REMBI-derived title / authors / year.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    doi: Optional[str] = Field(default=None, description="""DOI of this publication — the citable identifier. Omit within an entry only for a publication that genuinely has no DOI (e.g. a preprint without one).""", json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:doi'],
         'domain_of': ['Publication'],
         'exact_mappings': ['cpg:DOI_to_Cite']} })
    title: Optional[str] = Field(default=None, description="""Title of the related **publication** (the paper). Distinct from ``Study.dataset_name``, which is the dataset's own name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Publication']} })
    authors_name: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Publication']} })
    publication_year: Optional[int] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Publication']} })

    @field_validator('doi')
    def pattern_doi(cls, v):
        pattern=re.compile(r"^10\.\d{4,9}/.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid doi format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid doi format: {v}"
            raise ValueError(err_msg)
        return v


class Biosample(ConfiguredBaseModel):
    """
    Sample-level descriptors. Sample context reuses the CZI cross-modality schema v1.1.0 (CELLxGENE-aligned) as ``ontology_term_id`` + label pairs; cell-line identity uses the CellPainting Gallery + Cellosaurus. Required term-id fields go to ``zarr.json``; labels, cell-line and DCA-native fields to the Parquet.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'rules': [{'description': 'A cell-line sample must name the line. '
                                   '``cell_line_name`` is required when '
                                   '``tissue_type`` is "cell line"; the Cellosaurus '
                                   '``cell_line_id`` is strongly recommended but may '
                                   'be absent for iPSC / primary lines with no '
                                   'accession (record donor / clone ids in '
                                   '``additional_metadata`` instead).',
                    'postconditions': {'slot_conditions': {'cell_line_name': {'name': 'cell_line_name',
                                                                              'required': True}}},
                    'preconditions': {'slot_conditions': {'tissue_type': {'equals_string': 'cell '
                                                                                           'line',
                                                                          'name': 'tissue_type'}}}}]})

    group_id: str = Field(default=..., description="""Row key of the Parquet (``tables/obs``) table, and the key that links this sample's metadata to the image arrays it describes. There is one obs row per image group: a single-image store has one row; an HCS plate has one row per well (the well ``WellPerturbation.plate`` / ``well`` also key on). DCA-native — so sample metadata is not floating free of the data it annotates. Reuses the identifier the ingestor already writes to the obs table rather than minting a separate one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample']} })
    organism: list[OrganismTerm] = Field(default=..., description="""Organism as an ``ontology_term_id`` + ``label`` pair; the id is an NCBITaxon CURIE (e.g. \"NCBITaxon:9606\").""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample']} })
    tissue_type: TissueType = Field(default=..., json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:sample.tissue.type'],
         'domain_of': ['Biosample'],
         'exact_mappings': ['cxmod:tissue_type', 'ops:experiment.tissue_type']} })
    tissue: list[TissueTerm] = Field(default=..., description="""Tissue as an ``ontology_term_id`` + ``label`` pair. The id is a UBERON term (or a ZFA term for zebrafish anatomy); a CL term when ``tissue_type`` is \"cell culture\"; a Cellosaurus ``CVCL:`` term when ``tissue_type`` is \"cell line\" (cross-modality v1.1.0). Cross-modality splits anatomy and cell type across two fields; DCA folds both into this one slot and keys the distinction off ``tissue_type``. For zebrafish that means: in the anatomy role (``tissue_type`` \"tissue\" or \"organoid\") use a descendant of ``ZFA:0100000`` outside the ``ZFA:0009000`` (\"cell\") branch, and in the cell-type role (``tissue_type`` \"cell culture\") use a CL term, or a descendant of ``ZFA:0009000`` where CL has no equivalent. ``ZFA:0001093`` (\"unspecified\") is never valid — use the ``unknown`` sentinel. When ``tissue_type`` is \"organoid\", use the UBERON term for the **tissue of origin** from which the organoid was derived (e.g. ``UBERON:0002108`` for small intestine organoids, ``UBERON:0000948`` for cardiac organoids). All ids are colon CURIEs that expand to a resolvable URL via the ``prefixes`` block. This is the **authoritative** slot for cell-line identity (cross-modality); ``cell_line_id`` is a convenience mirror.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample']} })
    disease: list[DiseaseTerm] = Field(default=..., description="""Disease as an ``ontology_term_id`` + ``label`` pair; the id is a MONDO term, or PATO:0000461 for normal/healthy.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample']} })
    development_stage: list[DevelopmentStageTerm] = Field(default=..., description="""Development stage as an ``ontology_term_id`` + ``label`` pair. Use the stage ontology for the organism — HsapDv (human), MmusDv (mouse), ZFS (zebrafish), FBdv (fly), WBls (worm) — or a UBERON life-cycle term for organisms with no dedicated stage ontology.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample']} })
    cell_line_name: Optional[str] = Field(default=None, description="""Cell line name (CPG harmonized vocabulary). Required when ``tissue_type`` is \"cell line\". Repeats ``tissue``'s ``label`` for a cell-line sample; the duplication is deliberate, for CellPainting Gallery alignment.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['cdp:cell_strain_name'],
         'domain_of': ['Biosample'],
         'exact_mappings': ['cpg:Cell_Line_Name']} })
    cell_line_id: Optional[str] = Field(default=None, description="""Cellosaurus accession as a colon CURIE (e.g. \"CVCL:0030\"), resolvable via the ``prefixes`` block. Convenience mirror of the **authoritative** ``tissue`` term when ``tissue_type`` is \"cell line\", and SHOULD hold the same CVCL accession; nothing checks the two agree, so ``tissue`` wins on conflict. RECOMMENDED rather than required: iPSC and primary lines may have no accession, and SHOULD then record hPSCreg / donor id in ``additional_metadata``.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['CVCL:0000000', 'cdp:cell_strain_id'],
         'domain_of': ['Biosample']} })
    cell_line_type: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample'], 'exact_mappings': ['cpg:Cell_Line_Type']} })
    cell_line_modification: Optional[str] = Field(default=None, description="""Genetic / other modification (e.g. \"Cas9 polyclonal overexpression\").""", json_schema_extra = { "linkml_meta": {'domain_of': ['Biosample'], 'exact_mappings': ['cpg:Cell_Line_Modification']} })
    parent_cell_line_id: Optional[str] = Field(default=None, description="""Cellosaurus accession of the **parental** cell line from which the imaged line was derived, as a colon CURIE (e.g. ``\"CVCL:Y803\"`` for WTC-11 hiPSC). RECOMMENDED when ``tissue_type`` is \"cell line\" and the imaged line was created by gene editing or reprogramming from a registered parental line. ``cell_line_id`` identifies the engineered clone; ``parent_cell_line_id`` links it back to its origin. Use ``additional_metadata`` for donor / hPSCreg identifiers that have no Cellosaurus accession.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['CVCL:0000000'], 'domain_of': ['Biosample']} })
    additional_metadata: Optional[list[AdditionalMetadataField]] = Field(default=None, description="""Extra study-provided values not yet defined in this spec. Optional; promotable — once a field here stabilizes it can be elevated to a defined (ultimately required, zarr.json) slot in a future version.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Study', 'Biosample']} })

    @field_validator('cell_line_id')
    def pattern_cell_line_id(cls, v):
        pattern=re.compile(r"^CVCL:\w+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid cell_line_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid cell_line_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('parent_cell_line_id')
    def pattern_parent_cell_line_id(cls, v):
        pattern=re.compile(r"^CVCL:\w+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid parent_cell_line_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid parent_cell_line_id format: {v}"
            raise ValueError(err_msg)
        return v


class Specimen(ConfiguredBaseModel):
    """
    How the biosample was prepared for imaging (REMBI Specimen module): the preparation state, an optional FBbi sample-preparation method, and the growth conditions. The imaging modality lives on :class:`AcquisitionMetadata`.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    specimen_state: SpecimenState = Field(default=..., description="""Specimen state at imaging time — the coarse, mutually-exclusive live-vs-fixed axis (``live`` / ``fixed`` / ``unknown``). REQUIRED. DCA controlled vocabulary (REMBI-derived): no source ontology structures this state cleanly across specimen types. Specific preparation methods go in ``preparation_method``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Specimen']} })
    preparation_method: Optional[list[PreparationMethodTerm]] = Field(default=None, description="""Sample-preparation method(s) as ``{ontology_term_id, label}`` FBbi terms — descendants of ``FBbi:00000001`` (\"sample preparation method\"), e.g. ``FBbi:00000026`` (sectioned tissue), ``FBbi:00000024`` (whole mounted tissue), ``FBbi:00000025`` (living tissue), or a fixation term such as ``FBbi:00000010`` (formaldehyde fixed tissue). Multivalued: a store may stack methods (e.g. fixation + permeabilization). Use ``unavailable`` with a free-text ``label`` when no FBbi term fits (e.g. optical clearing, expansion microscopy); ``na`` when no preparation applies (e.g. a live sample); ``unknown`` when the source does not record it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['cdp:sample_preparation'], 'domain_of': ['Specimen']} })
    growth_conditions: Optional[str] = Field(default=None, description="""Free-text description of growth media, supplements, and culture conditions. SHOULD include catalog numbers and concentrations for all reagents. Maps to OPS ``cellular.growth_conditions``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Specimen'],
         'exact_mappings': ['ops:cellular.growth_conditions']} })


class ImageData(ConfiguredBaseModel):
    """
    Image-level metadata (REMBI Image Data). For DCA this carries the processing variant that produced the array, i.e. REMBI's \"Type: primary / processed image\" plus the image processing method. The remaining REMBI Image Data fields (format, dimensions, pixel and voxel size, channel information) live in the OME-Zarr arrays and channel metadata, not here. The full derivation (parent dataset, algorithm, parameters) is a Scientific Dataset Catalog ``transformed_from`` Lineage Edge.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    processing_variant: Optional[str] = Field(default=None, description="""Computational processing variant applied to produce this image array. DCA-native. Present when a single acquisition yields multiple differently-processed stores (e.g. ``\"raw\"``, ``\"denoised\"``, ``\"deconvolved\"``). This is the zarr-local mirror of the Scientific Dataset Catalog ``dataset_type`` field (``raw`` / ``processed``); the full derivation relationship — parent dataset id, algorithm, pipeline version, and transformation parameters — is tracked as a catalog ``Lineage Edge`` of type ``transformed_from`` and SHOULD NOT be duplicated here. Bio-Formats / OME has no dedicated field for this concept; it would conventionally appear in ``Image.Name`` or ``Image.Description``.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['bioformats:Image.Description', 'sdc:dataset_type'],
         'domain_of': ['ImageData'],
         'examples': [{'value': 'raw'},
                      {'value': 'denoised'},
                      {'value': 'deconvolved'}]} })


class AdditionalMetadataField(ConfiguredBaseModel):
    """
    A single agent-minted or non-standard metadata value (``name`` + ``value``), used wherever a typed slot does not exist. Its provenance lives in the root ``ExperimentalMetadata.provenance`` map, keyed by ``<block>.additional_metadata.<name>`` (not inline), so all provenance has one home.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    name: str = Field(default=..., description="""Field name (SHOULD be namespaced if targeting a specific spec).""", json_schema_extra = { "linkml_meta": {'domain_of': ['AdditionalMetadataField', 'Perturbation']} })
    value: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['AdditionalMetadataField', 'Concentration']} })


class OntologyTerm(ConfiguredBaseModel):
    """
    An ontology annotation as an ``ontology_term_id`` (CURIE) + human-readable ``label`` pair. The label is derivable from the term id but carried explicitly so a stored record is human-readable without an ontology lookup. Subclasses constrain ``ontology_term_id`` to a specific ontology. Three sentinels stand in for a CURIE, each permitted only where a subclass allows it: ``unknown`` means the value is not available to the curator (absent from the source data); ``na`` means the field does not apply to this experiment or use case; ``unavailable`` means the value is known but no ontology term exists for it. When ``ontology_term_id`` is ``unknown`` or ``na`` the ``label`` repeats the same sentinel; when it is ``unavailable`` the ``label`` carries the free-text value; for a real CURIE the ``label`` is the ontology label.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm']} })

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class OrganismTerm(OntologyTerm):
    """
    Organism ontology annotation (NCBITaxon).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'label': {'close_mappings': ['cdp:organism_name'],
                                  'exact_mappings': ['cxmod:organism',
                                                     'cpg:Cell_Line_Organism',
                                                     'ops:experiment.organism'],
                                  'name': 'label'},
                        'ontology_term_id': {'close_mappings': ['NCBITaxon:0000000',
                                                                'sdc:sample.organism',
                                                                'cdp:organism_taxid'],
                                             'exact_mappings': ['cxmod:organism_ontology_term_id',
                                                                'ops:experiment.organism_ontology_term_id'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(NCBITaxon:\\d+|unknown|na|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['NCBITaxon:0000000',
                            'sdc:sample.organism',
                            'cdp:organism_taxid'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:organism_ontology_term_id',
                            'ops:experiment.organism_ontology_term_id']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['cdp:organism_name'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:organism',
                            'cpg:Cell_Line_Organism',
                            'ops:experiment.organism']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(NCBITaxon:\d+|unknown|na|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class TissueTerm(OntologyTerm):
    """
    Tissue ontology annotation (UBERON / CL / ZFA / Cellosaurus CVCL).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'label': {'close_mappings': ['cdp:tissue_name',
                                                     'cdp:cell_name'],
                                  'exact_mappings': ['cxmod:tissue',
                                                     'ops:experiment.tissue'],
                                  'name': 'label'},
                        'ontology_term_id': {'close_mappings': ['CVCL:0000000',
                                                                'sdc:sample.tissue',
                                                                'cdp:tissue_id',
                                                                'cdp:cell_type_id'],
                                             'exact_mappings': ['cxmod:tissue_ontology_term_id',
                                                                'ops:experiment.tissue_ontology_term_id'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(UBERON:\\d+|CL:\\d+|CVCL:\\w+|ZFA:(?!0001093$)\\d+|unknown|na|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['CVCL:0000000',
                            'sdc:sample.tissue',
                            'cdp:tissue_id',
                            'cdp:cell_type_id'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:tissue_ontology_term_id',
                            'ops:experiment.tissue_ontology_term_id']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['cdp:tissue_name', 'cdp:cell_name'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:tissue', 'ops:experiment.tissue']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(UBERON:\d+|CL:\d+|CVCL:\w+|ZFA:(?!0001093$)\d+|unknown|na|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class DiseaseTerm(OntologyTerm):
    """
    Disease ontology annotation (MONDO, or PATO:0000461 for normal/healthy).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'label': {'exact_mappings': ['cxmod:disease',
                                                     'ops:experiment.disease'],
                                  'name': 'label'},
                        'ontology_term_id': {'close_mappings': ['sdc:sample.disease'],
                                             'exact_mappings': ['cxmod:disease_ontology_term_id',
                                                                'ops:experiment.disease_ontology_term_id'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(MONDO:\\d+|PATO:0000461|unknown|na|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:sample.disease'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:disease_ontology_term_id',
                            'ops:experiment.disease_ontology_term_id']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:disease', 'ops:experiment.disease']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(MONDO:\d+|PATO:0000461|unknown|na|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class DevelopmentStageTerm(OntologyTerm):
    """
    Development-stage ontology annotation. Uses the organism-specific stage ontology the CZI cross-modality schema names for that organism — HsapDv (human), MmusDv (mouse), ZFS (zebrafish), FBdv (fly), WBls (worm) — and a UBERON life-cycle term for organisms with no dedicated stage ontology.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'label': {'close_mappings': ['cdp:development_stage_name'],
                                  'exact_mappings': ['cxmod:development_stage',
                                                     'ops:experiment.development_stage'],
                                  'name': 'label'},
                        'ontology_term_id': {'close_mappings': ['sdc:sample.development_stage',
                                                                'cdp:development_stage_id'],
                                             'exact_mappings': ['cxmod:development_stage_ontology_term_id',
                                                                'ops:experiment.development_stage_ontology_term_id'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(HsapDv:\\d+|MmusDv:\\d+|ZFS:(?!0000000$)\\d+|FBdv:(?!00007012$)\\d+|WBls:\\d+|UBERON:\\d+|unknown|na|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['sdc:sample.development_stage', 'cdp:development_stage_id'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:development_stage_ontology_term_id',
                            'ops:experiment.development_stage_ontology_term_id']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['cdp:development_stage_name'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:development_stage',
                            'ops:experiment.development_stage']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(HsapDv:\d+|MmusDv:\d+|ZFS:(?!0000000$)\d+|FBdv:(?!00007012$)\d+|WBls:\d+|UBERON:\d+|unknown|na|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class AssayTerm(OntologyTerm):
    """
    Assay / experiment-class ontology annotation (EFO only).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'label': {'exact_mappings': ['cxmod:assay',
                                                     'ops:experiment.assay'],
                                  'name': 'label'},
                        'ontology_term_id': {'close_mappings': ['ops:experiment.assay_ontology_term_id',
                                                                'sdc:experiment.assay'],
                                             'exact_mappings': ['cxmod:assay_ontology_term_id'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(EFO:\\d+|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['ops:experiment.assay_ontology_term_id',
                            'sdc:experiment.assay'],
         'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:assay_ontology_term_id']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm'],
         'exact_mappings': ['cxmod:assay', 'ops:experiment.assay']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(EFO:\d+|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class ImagingMethodTerm(OntologyTerm):
    """
    Acquisition-modality ontology annotation (FBbi only).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'ontology_term_id': {'close_mappings': ['FBbi:0000000'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(FBbi:\\d+|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['FBbi:0000000'], 'domain_of': ['OntologyTerm']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(FBbi:\d+|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class PreparationMethodTerm(OntologyTerm):
    """
    Sample-preparation method ontology annotation (FBbi only).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'slot_usage': {'ontology_term_id': {'close_mappings': ['FBbi:0000000'],
                                             'name': 'ontology_term_id',
                                             'pattern': '^(FBbi:\\d+|unknown|na|unavailable)$'}}})

    ontology_term_id: str = Field(default=..., description="""CURIE for the ontology term (e.g. \"NCBITaxon:9606\"), or one of the sentinels (``unknown`` / ``na`` / ``unavailable``) where the subclass permits it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['FBbi:0000000'], 'domain_of': ['OntologyTerm']} })
    label: str = Field(default=..., description="""Human-readable label. For a real term it is the ontology label. When ``ontology_term_id`` is ``unavailable`` it carries the free-text value; when it is ``unknown`` or ``na`` it repeats that sentinel. Must be non-empty.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm']} })

    @field_validator('ontology_term_id')
    def pattern_ontology_term_id(cls, v):
        pattern=re.compile(r"^(FBbi:\d+|unknown|na|unavailable)$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ontology_term_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ontology_term_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('label')
    def pattern_label(cls, v):
        pattern=re.compile(r"^.+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid label format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid label format: {v}"
            raise ValueError(err_msg)
        return v


class Provenance(ConfiguredBaseModel):
    """
    Per-value provenance kept as a parallel layer (keyed by field path) alongside the pure metadata; a deterministic post-ingest step folds it onto the corresponding values in ``zarr.json`` / the Parquet. Lets a downstream consumer decide how much to trust a field without re-reading the source. Blank for manually entered values (``source_kind: human``).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'rules': [{'description': 'An ``agent``-extracted value MUST carry its '
                                   'citation (source_url + quote); provenance is '
                                   'required for agent-curated values, optional for '
                                   '``human`` / ``original_data``.',
                    'postconditions': {'slot_conditions': {'citation': {'name': 'citation',
                                                                        'required': True}}},
                    'preconditions': {'slot_conditions': {'source_kind': {'equals_string': 'agent',
                                                                          'name': 'source_kind'}}}}]})

    citation: Optional[Citation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    grounded: Optional[bool] = Field(default=None, description="""Output of an agent's deterministic grounding gate: ``true`` when the cited ``quote`` was located (exact or fuzzy) in the fetched source text, so the cited evidence is real and not fabricated. Computed, not self-reported — a fabricated quote that is absent from the source grounds to ``false``. Boolean rollup of ``citation.match_status``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    source_kind: ProvenanceSourceKind = Field(default=..., description="""How the value was obtained. REQUIRED on every provenance entry — it is the discriminator that distinguishes an ``agent``-extracted value from a hand-entered (``human``) one from a value taken directly from the source's own metadata (``original_data``). The citation / grounding is what is optional (for ``human`` / ``original_data``); ``source_kind`` itself is always stated.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })


class Citation(ConfiguredBaseModel):
    """
    The supporting quote, the source page it was copied from, and the result of the agent's grounding gate that located it. When a citation is present, ``source_url`` and ``quote`` are both required — a half-empty citation is not allowed.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    source_url: str = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Citation']} })
    quote: str = Field(default=..., description="""Verbatim excerpt from the source supporting the value.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Citation']} })
    match_status: Optional[MatchStatus] = Field(default=None, description="""Outcome of the deterministic grounding gate for this quote (see MatchStatus). ``none`` means the quote did not ground in the cited page.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Citation']} })
    char_start: Optional[int] = Field(default=None, description="""Start offset (inclusive) of the located quote within the cited source page's raw text. Blank when ``match_status`` is ``none``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Citation']} })
    char_end: Optional[int] = Field(default=None, description="""End offset (exclusive) of the located quote within the cited source page's raw text. Blank when ``match_status`` is ``none``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Citation']} })


class FieldProvenance(Provenance):
    """
    One entry in the per-value provenance map (``ExperimentalMetadata.provenance``): the provenance of a single field, keyed by its dotted ``field_path``. Same shape as :class:`Provenance` (``source_kind`` / ``grounded`` / ``citation`` and its grounding rules) plus the path it applies to, so provenance can travel as a parallel layer keyed by field path rather than nested inside each value.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    field_path: str = Field(default=..., description="""Dotted path of the field this provenance applies to, relative to the experimental-metadata root (e.g. ``study.year_imaged``, ``biosample.tissue``). Keys this entry within the map.""", json_schema_extra = { "linkml_meta": {'domain_of': ['FieldProvenance']} })
    citation: Optional[Citation] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    grounded: Optional[bool] = Field(default=None, description="""Output of an agent's deterministic grounding gate: ``true`` when the cited ``quote`` was located (exact or fuzzy) in the fetched source text, so the cited evidence is real and not fabricated. Computed, not self-reported — a fabricated quote that is absent from the source grounds to ``false``. Boolean rollup of ``citation.match_status``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })
    source_kind: ProvenanceSourceKind = Field(default=..., description="""How the value was obtained. REQUIRED on every provenance entry — it is the discriminator that distinguishes an ``agent``-extracted value from a hand-entered (``human``) one from a value taken directly from the source's own metadata (``original_data``). The citation / grounding is what is optional (for ``human`` / ``original_data``); ``source_kind`` itself is always stated.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Provenance']} })


class Perturbation(ConfiguredBaseModel):
    """
    A perturbation definition (the \"codebook\" entry), reusable across wells. Modality-agnostic core (mapped to the OME ``Reagent`` entity) plus an optional chemical or genetic block selected by ``modality``.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'OME:Reagent',
         'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'rules': [{'description': 'A compound perturbation must carry the chemical '
                                   'identifier block.',
                    'postconditions': {'slot_conditions': {'chemical': {'name': 'chemical',
                                                                        'required': True}}},
                    'preconditions': {'slot_conditions': {'modality': {'equals_string': 'compound',
                                                                       'name': 'modality'}}}},
                   {'description': 'A genetic perturbation (crispr / orf / shrna / '
                                   'mirna / sirna) must carry the genetic-target '
                                   'block.',
                    'postconditions': {'slot_conditions': {'genetic': {'name': 'genetic',
                                                                       'required': True}}},
                    'preconditions': {'slot_conditions': {'modality': {'any_of': [{'equals_string': 'crispr'},
                                                                                  {'equals_string': 'orf'},
                                                                                  {'equals_string': 'shrna'},
                                                                                  {'equals_string': 'mirna'},
                                                                                  {'equals_string': 'sirna'}],
                                                                       'name': 'modality'}}}}]})

    perturbation_id: str = Field(default=..., description="""Stable identifier for this perturbation within the dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Perturbation'], 'exact_mappings': ['bioformats:Reagent.ID']} })
    name: Optional[str] = Field(default=None, description="""Human-readable treatment / perturbation name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AdditionalMetadataField', 'Perturbation'],
         'exact_mappings': ['bioformats:Reagent.Name',
                            'cpg:Treatment_Primary_Treatment']} })
    description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Study', 'Perturbation'],
         'exact_mappings': ['bioformats:Reagent.Description']} })
    perturbation_identifier: Optional[str] = Field(default=None, description="""External catalog identifier for the construct. Because the source varies (Broad sample id, vendor id, JCP2022 id), it MUST be namespaced so the value names its database — e.g. ``JCP2022:JCP2022_012345`` or ``Broad:BRD-K12345678``. The canonical, resolvable identity lives in the typed modality block (``chemical.pubchem_cid`` / ``genetic.ncbi_gene_id``); this generic slot records the source's own catalog id.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Perturbation'],
         'exact_mappings': ['bioformats:Reagent.ReagentIdentifier',
                            'cpg:Treatment_Broad_Sample']} })
    modality: PerturbationModality = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Microscope', 'Perturbation'],
         'exact_mappings': ['cpg:Treatment_Category']} })
    mechanism: Optional[str] = Field(default=None, description="""Mechanism of action (free text).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Perturbation'], 'exact_mappings': ['cpg:Treatment_Mechanism']} })
    chemical: Optional[ChemicalPerturbation] = Field(default=None, description="""Chemical identifiers — present when ``modality = compound``.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Perturbation']} })
    genetic: Optional[GeneticPerturbation] = Field(default=None, description="""Genetic-target identifiers — present when ``modality`` is crispr / orf / shrna / mirna / sirna. This is the genetic-perturbation extension (knockout, knockdown, or overexpression of a gene target).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Perturbation']} })


class ChemicalPerturbation(ConfiguredBaseModel):
    """
    Chemical (small-molecule) perturbation identifiers.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    inchikey: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ChemicalPerturbation'],
         'exact_mappings': ['cpg:Treatment_InChIKey']} })
    inchi: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ChemicalPerturbation'],
         'exact_mappings': ['cpg:compound_inchi']} })
    smiles: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ChemicalPerturbation'],
         'exact_mappings': ['cpg:Treatment_SMILES']} })
    pubchem_cid: Optional[str] = Field(default=None, description="""PubChem Compound ID as a colon CURIE (e.g. \"PubChem:2244\"), resolvable via the ``prefixes`` block — the value names its own database.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['PubChem:0'],
         'domain_of': ['ChemicalPerturbation'],
         'exact_mappings': ['cpg:Treatment_PubChem_CID']} })
    chembl_id: Optional[str] = Field(default=None, description="""ChEMBL compound identifier (e.g. \"CHEMBL25\"). Self-identifying — ChEMBL accessions carry the ``CHEMBL`` prefix in the value itself.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ChemicalPerturbation']} })

    @field_validator('pubchem_cid')
    def pattern_pubchem_cid(cls, v):
        pattern=re.compile(r"^PubChem:\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid pubchem_cid format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid pubchem_cid format: {v}"
            raise ValueError(err_msg)
        return v


class GeneticPerturbation(ConfiguredBaseModel):
    """
    Genetic-target identifiers for CRISPR / ORF / shRNA / miRNA perturbations. The gene target is what CellPainting Gallery cannot express (it has only the ``Treatment_Category`` enum); these fields come from the JUMP crispr / orf tables.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    gene_symbol: Optional[str] = Field(default=None, description="""HGNC gene symbol (e.g. \"TP53\") — the human-readable label. ``ncbi_gene_id`` is the canonical, resolvable key; this is the symbol that accompanies it.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['HGNC:0'],
         'domain_of': ['GeneticPerturbation'],
         'exact_mappings': ['cpg:crispr_symbol', 'cpg:orf_symbol']} })
    ncbi_gene_id: Optional[str] = Field(default=None, description="""NCBI Gene ID as a colon CURIE (e.g. \"NCBIGene:7157\") — the canonical, resolvable cross-table gene key in JUMP. The value names its own database via the ``prefixes`` block.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['NCBIGene:0'],
         'domain_of': ['GeneticPerturbation'],
         'exact_mappings': ['cpg:crispr_ncbi_gene_id', 'cpg:orf_ncbi_gene_id']} })
    ensembl_gene_id: Optional[str] = Field(default=None, description="""Ensembl gene id as a colon CURIE (e.g. \"ensembl:ENSG00000141510\"); cross-map, not JUMP-native. Resolvable via the ``prefixes`` block.""", json_schema_extra = { "linkml_meta": {'close_mappings': ['ensembl:0'], 'domain_of': ['GeneticPerturbation']} })
    transcript_refseq: Optional[str] = Field(default=None, description="""NCBI RefSeq transcript id (ORF).""", json_schema_extra = { "linkml_meta": {'domain_of': ['GeneticPerturbation'], 'exact_mappings': ['cpg:orf_transcript']} })
    vector: Optional[str] = Field(default=None, description="""Expression / delivery vector (ORF; sgRNA vector).""", json_schema_extra = { "linkml_meta": {'domain_of': ['GeneticPerturbation'], 'exact_mappings': ['cpg:orf_vector']} })
    insert_length: Optional[int] = Field(default=None, description="""ORF insert length (bp).""", json_schema_extra = { "linkml_meta": {'domain_of': ['GeneticPerturbation'],
         'exact_mappings': ['cpg:orf_insert_length']} })
    gene_description: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['GeneticPerturbation'],
         'exact_mappings': ['cpg:orf_gene_description']} })

    @field_validator('ncbi_gene_id')
    def pattern_ncbi_gene_id(cls, v):
        pattern=re.compile(r"^NCBIGene:\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ncbi_gene_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ncbi_gene_id format: {v}"
            raise ValueError(err_msg)
        return v

    @field_validator('ensembl_gene_id')
    def pattern_ensembl_gene_id(cls, v):
        pattern=re.compile(r"^ensembl:ENSG\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid ensembl_gene_id format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid ensembl_gene_id format: {v}"
            raise ValueError(err_msg)
        return v


class Concentration(ConfiguredBaseModel):
    """
    Perturbation concentration. Structured ``value`` + ``unit`` with a free-text ``text`` fallback for CellPainting Gallery's string ``Treatment_Concentration``.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    value: Optional[float] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['AdditionalMetadataField', 'Concentration']} })
    unit: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['Concentration']} })
    text: Optional[str] = Field(default=None, description="""Free-text concentration as recorded (CPG fallback).""", json_schema_extra = { "linkml_meta": {'domain_of': ['Concentration'],
         'exact_mappings': ['cpg:Treatment_Concentration']} })


class PerturbationAssignment(ConfiguredBaseModel):
    """
    The ASSIGNMENT of a :class:`Perturbation` to imaged material — the dosing / control / timing that turns a reusable codebook entry into \"what was done to this sample\". The entry itself is the ``definition`` slot (plus an optional ``cotreatment``); everything else here is the assignment. Carries no plate / well key, so it is placement-agnostic: it is the ``perturbation`` block of :class:`ExperimentalMetadata` (image level, alongside ``study`` / ``biosample`` / ``specimen``), and it is the body that :class:`WellPerturbation` extends with an HCS ``(plate, well)`` key. Required-when-present: a dataset with perturbations MUST carry the required fields below; a dataset with none simply has no ``perturbation`` block. (Pooled/OPS screens use a deferred per-cell table instead; see the spec doc.)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata',
         'rules': [{'description': 'Treated and control material (treatment / negcon / '
                                   'poscon) must name a perturbation; ``empty`` has '
                                   'none to record, so ``definition`` is not required '
                                   'there.',
                    'postconditions': {'slot_conditions': {'definition': {'name': 'definition',
                                                                          'required': True}}},
                    'preconditions': {'slot_conditions': {'control_class': {'name': 'control_class',
                                                                            'none_of': [{'equals_string': 'empty'}]}}}}]})

    definition: Optional[Perturbation] = Field(default=None, description="""The :class:`Perturbation` definition this assignment applies — the reusable codebook entry (what the perturbagen IS), as distinct from the dosing / control / timing around it (what was DONE to this material), which are the sibling slots here. Required for treated and control material (``treatment`` / ``negcon`` / ``poscon``); omitted when ``control_class`` is ``empty``, which has nothing to record (enforced by the class rule above).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    control_class: PerturbationControlClass = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Control_Class']} })
    concentration: Optional[Concentration] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    solvent: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Solvent']} })
    duration_h: Optional[float] = Field(default=None, description="""Perturbation duration in hours.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    timepoint_primary_h: Optional[float] = Field(default=None, description="""Time of primary treatment (CPG Timepoint_Primary_Treatment).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Timepoint_Primary_Treatment']} })
    timepoint_secondary_h: Optional[float] = Field(default=None, description="""Time of secondary treatment.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Timepoint_Secondary_Treatment']} })
    cotreatment: Optional[Perturbation] = Field(default=None, description="""Optional second :class:`Perturbation` definition applied alongside ``definition`` (a cotreatment).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Secondary_Treatment']} })
    provenance: Optional[Provenance] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata',
                       'AcquisitionMetadata',
                       'PerturbationAssignment']} })


class WellPerturbation(PerturbationAssignment):
    """
    Per-well perturbation assignment for an ARRAYED HCS screen — one row per ``(plate, well)``, written to ``zarr.json`` (well level). A :class:`PerturbationAssignment` plus the ``(plate, well)`` key that resolves it to a well group. This form is DERIVED: ``ExperimentalMetadata.perturbation`` on the image group is authoritative, and this is a plate-shaped index over it that exists so a plate's design can be enumerated without opening every field of view. It is unavailable to a plateless ``image_collection`` store (see the ``perturbation`` block for those). The two MUST agree; because the direction is defined, a disagreement is repaired by re-deriving this row from the image groups, never the other way round. The validator checks this row's own contents when present (dispatched to ``validate_well_perturbation_v0_2``), and separately cross-checks it against the consensus ``perturbation`` value of its own well's field(s) of view, reporting a mismatch as an error (see validator.rst). The non-``empty`` ⇒ ``definition`` rule is inherited from the parent.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://github.com/chanzuckerberg/dynamic-cell-atlas-specs/schema/dca_experimental_metadata'})

    plate: str = Field(default=..., description="""Plate identifier. Together with ``well`` this is the link to the image arrays: it MUST match the OME-NGFF HCS plate name (the ``plate`` group in ``zarr.json``) so a row resolves to a well group in the store.""", json_schema_extra = { "linkml_meta": {'domain_of': ['WellPerturbation']} })
    well: str = Field(default=..., description="""Well identifier in row-letter + column-number form (e.g. \"A03\"), mapping to the OME-NGFF HCS ``row``/``column`` well-group path (``A/3``) under the plate. This is the key tying the assignment to the imaged well.""", json_schema_extra = { "linkml_meta": {'domain_of': ['WellPerturbation']} })
    definition: Optional[Perturbation] = Field(default=None, description="""The :class:`Perturbation` definition this assignment applies — the reusable codebook entry (what the perturbagen IS), as distinct from the dosing / control / timing around it (what was DONE to this material), which are the sibling slots here. Required for treated and control material (``treatment`` / ``negcon`` / ``poscon``); omitted when ``control_class`` is ``empty``, which has nothing to record (enforced by the class rule above).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    control_class: PerturbationControlClass = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Control_Class']} })
    concentration: Optional[Concentration] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    solvent: Optional[str] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Solvent']} })
    duration_h: Optional[float] = Field(default=None, description="""Perturbation duration in hours.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment']} })
    timepoint_primary_h: Optional[float] = Field(default=None, description="""Time of primary treatment (CPG Timepoint_Primary_Treatment).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Timepoint_Primary_Treatment']} })
    timepoint_secondary_h: Optional[float] = Field(default=None, description="""Time of secondary treatment.""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Timepoint_Secondary_Treatment']} })
    cotreatment: Optional[Perturbation] = Field(default=None, description="""Optional second :class:`Perturbation` definition applied alongside ``definition`` (a cotreatment).""", json_schema_extra = { "linkml_meta": {'domain_of': ['PerturbationAssignment'],
         'exact_mappings': ['cpg:Treatment_Secondary_Treatment']} })
    provenance: Optional[Provenance] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ExperimentalMetadata',
                       'AcquisitionMetadata',
                       'PerturbationAssignment']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
ExperimentalMetadata.model_rebuild()
StudyComponent.model_rebuild()
AcquisitionMetadata.model_rebuild()
Microscope.model_rebuild()
Objective.model_rebuild()
Detector.model_rebuild()
ImagingEnvironment.model_rebuild()
Study.model_rebuild()
Experimenter.model_rebuild()
Publication.model_rebuild()
Biosample.model_rebuild()
Specimen.model_rebuild()
ImageData.model_rebuild()
AdditionalMetadataField.model_rebuild()
OntologyTerm.model_rebuild()
OrganismTerm.model_rebuild()
TissueTerm.model_rebuild()
DiseaseTerm.model_rebuild()
DevelopmentStageTerm.model_rebuild()
AssayTerm.model_rebuild()
ImagingMethodTerm.model_rebuild()
PreparationMethodTerm.model_rebuild()
Provenance.model_rebuild()
Citation.model_rebuild()
FieldProvenance.model_rebuild()
Perturbation.model_rebuild()
ChemicalPerturbation.model_rebuild()
GeneticPerturbation.model_rebuild()
Concentration.model_rebuild()
PerturbationAssignment.model_rebuild()
WellPerturbation.model_rebuild()
