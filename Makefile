# Regenerate the artifacts derived from the LinkML source of truth.
#
# Each version's LinkML schema (schema/v<X.Y>/dca_experimental_metadata.yaml) is the
# single source of truth for that version. Both the JSON Schema and the Pydantic models are generated
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
# Commands mirror docs/v0.3/experimental-metadata.rst § Validation.

VERSIONS := 0.2 0.3

# Always regenerate: file-mtime prerequisites are unreliable here because git
# operations (checkout, rebase) can leave the YAML older than the artifacts,
# which would make `make` skip a needed regeneration. The drift-guard test is
# the source of truth for staleness in CI.
.PHONY: schema
schema:
	@for v in $(VERSIONS); do \
	  vu=$$(echo $$v | tr . _); \
	  echo "schema v$$v"; \
	  gen-json-schema --no-metadata schema/v$$v/dca_experimental_metadata.yaml > schema/v$$v/generated/dca_experimental_metadata.schema.json || exit 1; \
	  gen-pydantic schema/v$$v/dca_experimental_metadata.yaml > helpers/src/dca_helpers/validation/versions/v$$vu/_generated_models.py || exit 1; \
	done
