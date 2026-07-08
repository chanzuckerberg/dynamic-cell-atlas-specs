# Regenerate the artifacts derived from the LinkML source of truth.
#
# The LinkML schema (schema/v0.2/dca_experimental_metadata.yaml) is the single
# source of truth. Both the JSON Schema and the Pydantic models are generated
# from it and committed; nothing is hand-edited. CI re-runs this and fails if
# the committed artifacts drift from the schema (see
# tests/validation/test_generated_artifacts.py).
#
# The validator enforces the experimental metadata at runtime against the
# generated Pydantic models, which is why they are committed inside the package
# (importable on any install layout). The generated JSON Schema is a test-only
# cross-check (see tests/validation/test_experimental_metadata.py) and an interop
# artifact, not on the runtime path. linkml is a dev-only tool, not a runtime
# dependency of dca_helpers.
#
# Commands mirror docs/v0.2/experimental-metadata.rst § Validation.

SCHEMA       := schema/v0.2/dca_experimental_metadata.yaml
JSON_OUT     := schema/v0.2/generated/dca_experimental_metadata.schema.json
PYDANTIC_OUT := helpers/src/dca_helpers/validation/versions/v0_2/_generated_models.py

# Always regenerate: file-mtime prerequisites are unreliable here because git
# operations (checkout, rebase) can leave the YAML older than the artifacts,
# which would make `make` skip a needed regeneration. The drift-guard test is
# the source of truth for staleness in CI.
.PHONY: schema
schema:
	gen-json-schema --no-metadata $(SCHEMA) > $(JSON_OUT)
	gen-pydantic $(SCHEMA) > $(PYDANTIC_OUT)
