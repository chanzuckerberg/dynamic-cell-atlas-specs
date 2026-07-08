"""
Unit tests for documentation content validation.
"""

import re

import pytest


@pytest.mark.unit
class TestRSTSyntax:
    """Test reStructuredText syntax validity."""

    def test_rst_files_exist(self, rst_files):
        """Test that expected RST files exist."""
        expected_files = [
            "index.rst",
            "v0.1/index.rst",
            "v0.1/array-standard.rst",
            "v0.1/metadata-standard.rst",
            "v0.2/index.rst",
            "v0.2/array-standard.rst",
            "v0.2/metadata-standard.rst",
            "v0.2/ome-zarr-metadata.rst",
            "v0.2/dca-metadata.rst",
            "v0.2/channel-metadata.rst",
        ]

        file_names = [str(f.relative_to(f.parents[1])) for f in rst_files]

        for expected in expected_files:
            # Handle different path separators
            matching_files = [f for f in file_names if expected in f]
            assert len(matching_files) > 0, f"Missing RST file: {expected}"

    def test_rst_files_not_empty(self, rst_files):
        """Test that RST files are not empty."""
        for rst_file in rst_files:
            assert rst_file.stat().st_size > 0, f"Empty RST file: {rst_file}"

    def test_rst_files_have_titles(self, rst_files):
        """Test that RST files have proper titles."""
        for rst_file in rst_files:
            with open(rst_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for title underlines (=, -, ~, etc.)
            title_pattern = r'^([=\-~`:#"^+*]{4,})\s*$'
            matches = re.findall(title_pattern, content, re.MULTILINE)

            assert len(matches) > 0, f"No title found in {rst_file}"


@pytest.mark.unit
class TestVersionConsistency:
    """Test version number consistency across files."""

    def test_version_consistency(self, docs_dir, sphinx_config):
        """Test that version numbers are consistent across files."""
        current_release = "0.1"

        # Check conf.py points to current release
        assert current_release in sphinx_config.release

        # Check each version directory has an index with a matching title
        for version_dir in sorted(docs_dir.glob("v0.*")):
            version = version_dir.name.removeprefix("v")
            index_file = version_dir / "index.rst"
            assert index_file.exists(), f"Missing index.rst in {version_dir}"
            with open(index_file, "r") as f:
                content = f.read()
            assert (
                f"Version {version}" in content
            ), f"Version {version} not mentioned in {index_file}"


@pytest.mark.unit
class TestRFC2119Compliance:
    """Test RFC 2119 keyword usage in specifications."""

    RFC2119_KEYWORDS = [
        "MUST",
        "MUST NOT",
        "REQUIRED",
        "SHALL",
        "SHALL NOT",
        "SHOULD",
        "SHOULD NOT",
        "RECOMMENDED",
        "NOT RECOMMENDED",
        "MAY",
        "OPTIONAL",
    ]

    def test_rfc2119_keywords_formatted(self, docs_dir):
        """Test that RFC 2119 keywords are properly formatted (bold)."""
        spec_files = [
            docs_dir / "v0.1" / "array-standard.rst",
            docs_dir / "v0.1" / "metadata-standard.rst",
            docs_dir / "v0.2" / "array-standard.rst",
            docs_dir / "v0.2" / "metadata-standard.rst",
            docs_dir / "v0.2" / "ome-zarr-metadata.rst",
            docs_dir / "v0.2" / "channel-metadata.rst",
        ]

        for spec_file in spec_files:
            if not spec_file.exists():
                continue

            with open(spec_file, "r") as f:
                content = f.read()

            # Check for keywords that should be bold (**KEYWORD**)
            for keyword in self.RFC2119_KEYWORDS:
                if keyword in content:
                    # Should be formatted as **KEYWORD** at least once
                    bold_pattern = f"\\*\\*{keyword}\\*\\*"
                    assert re.search(bold_pattern, content), (
                        f"RFC 2119 keyword '{keyword}' not properly formatted in "
                        f"{spec_file}"
                    )

    def test_rfc2119_reference_exists(self, docs_dir):
        """Test that RFC 2119 is referenced in the documentation."""
        index_file = docs_dir / "index.rst"

        with open(index_file, "r") as f:
            content = f.read()

        # Should reference RFC 2119
        assert "2119" in content, "Missing RFC 2119 reference"
        assert (
            "rfc:" in content.lower() or "rfc2119" in content.lower()
        ), "RFC 2119 not properly referenced"


@pytest.mark.unit
class TestSpecificationStructure:
    """Test that specification documents have expected structure."""

    def test_array_standard_sections(self, docs_dir):
        """Test that array standard has required sections."""
        array_standard = docs_dir / "v0.1" / "array-standard.rst"

        with open(array_standard, "r") as f:
            content = f.read()

        required_sections = [
            "Format Requirements",
            "Multiresolution",
            "Dimensions",
            "Data Types",
            "Performance",
            "Compression",
        ]

        for section in required_sections:
            assert section in content, f"Missing required section: {section}"

    def test_specification_status_marked(self, docs_dir):
        """Test that WIP sections are clearly marked."""
        # Check that metadata standard is marked as WIP
        metadata_file = docs_dir / "v0.1" / "metadata-standard.rst"
        with open(metadata_file, "r") as f:
            metadata_content = f.read()

        assert "Work in Progress" in metadata_content or "WIP" in metadata_content

        # Verify array standard exists and has content (it's the complete section)
        array_file = docs_dir / "v0.1" / "array-standard.rst"
        with open(array_file, "r") as f:
            array_content = f.read()

        assert len(array_content) > 1000  # Should have substantial content
        assert "Format Requirements" in array_content  # Check for key section
