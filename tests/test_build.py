"""
Integration tests for documentation build process.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
class TestSphinxBuild:
    """Test Sphinx documentation build process."""

    def test_sphinx_build_html(self, docs_dir):
        """Test that Sphinx can build HTML documentation without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"

            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "html",
                    "-W",  # Treat warnings as errors
                    str(docs_dir),
                    str(build_dir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                "Sphinx HTML build failed:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

            # Check that essential HTML files were generated
            assert (build_dir / "index.html").exists()
            assert (build_dir / "v0.1" / "index.html").exists()
            assert (build_dir / "v0.1" / "array-standard.html").exists()
            assert (build_dir / "v0.2" / "index.html").exists()
            assert (build_dir / "v0.2" / "channel-metadata.html").exists()

    def test_sphinx_build_no_warnings(self, docs_dir):
        """Test that Sphinx build produces no warnings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"

            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "html",
                    "-W",  # Treat warnings as errors
                    "-q",  # Quiet output
                    str(docs_dir),
                    str(build_dir),
                ],
                capture_output=True,
                text=True,
            )

            # If warnings are treated as errors, return code should be 0
            assert result.returncode == 0, (
                "Sphinx build has warnings:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

    @pytest.mark.slow
    def test_sphinx_linkcheck(self, docs_dir):
        """Test that all links in documentation are valid or properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "linkcheck"

            # Configure Sphinx to use shorter timeout and handle private links
            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b",
                    "linkcheck",
                    "-q",  # Quiet output
                    "-D",
                    "linkcheck_timeout=3",
                    str(docs_dir),
                    str(build_dir),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            # Link check should complete but may have warnings for private links
            # Return code 0 = all good, 1 = some broken links (acceptable)
            assert result.returncode in [0, 1], (
                "Link check failed unexpectedly:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

            # Check that linkcheck output was generated and analyze results
            output_file = build_dir / "output.txt"
            if output_file.exists():
                with open(output_file, "r") as f:
                    output = f.read()

                # Count different types of link issues
                broken_count = output.count("broken")
                timeout_count = output.count("timeout")

                # Print summary for debugging
                print(
                    f"Link check summary: {broken_count} broken, "
                    f"{timeout_count} timeouts"
                )

                # Should not have critical infrastructure failures (like our own docs)
                # But private/timeout links are acceptable
                lines = output.split("\n")
                critical_errors = []
                for line in lines:
                    if "broken" in line.lower():
                        # Skip known private links that might be reported as broken
                        if any(
                            pattern in line
                            for pattern in [
                                "github.com/chanzuckerberg",
                                "timeout",
                                "Anchor",  # GitHub anchor links can be fragile
                            ]
                        ):
                            continue
                        critical_errors.append(line)

                assert len(critical_errors) == 0, (
                    "Critical link errors found (excluding private/timeout links):\n"
                    + "\n".join(critical_errors)
                )


@pytest.mark.integration
class TestBuildOutputValidation:
    """Test that build outputs are valid."""

    def test_generated_html_structure(self, docs_dir):
        """Test that generated HTML has expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"

            result = subprocess.run(
                ["sphinx-build", "-b", "html", "-q", str(docs_dir), str(build_dir)],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Check main index file
            index_html = build_dir / "index.html"
            assert index_html.exists()

            with open(index_html, "r") as f:
                content = f.read()

            # Should have title
            assert "Dynamic Cell Atlas Specification" in content

            # Should have navigation to v0.1
            assert "v0.1/index.html" in content

            # Should include CSS
            assert "custom.css" in content or "_static" in content

    def test_static_assets_copied(self, docs_dir):
        """Test that static assets are properly copied."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"

            subprocess.run(
                ["sphinx-build", "-b", "html", "-q", str(docs_dir), str(build_dir)],
                check=True,
            )

            # Check that custom CSS was copied
            static_dir = build_dir / "_static"
            assert static_dir.exists()

            # Custom CSS should exist
            custom_css = static_dir / "css" / "custom.css"
            assert custom_css.exists(), "Custom CSS file not copied to build output"


@pytest.mark.integration
class TestMultipleFormats:
    """Test building documentation in multiple formats."""

    def test_epub_build(self, docs_dir):
        """Test that EPUB format can be built."""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "epub"

            result = subprocess.run(
                ["sphinx-build", "-b", "epub", "-q", str(docs_dir), str(build_dir)],
                capture_output=True,
                text=True,
            )

            # EPUB build might have warnings but should not fail
            assert (
                result.returncode == 0
            ), f"EPUB build failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

            # Should generate EPUB file
            epub_files = list(build_dir.glob("*.epub"))
            assert len(epub_files) > 0, "No EPUB file generated"

    @pytest.mark.slow
    def test_latex_build(self, docs_dir):
        """Test that LaTeX format can be built (if LaTeX is available)."""
        # Check if pdflatex is available
        try:
            subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
            latex_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            latex_available = False

        if not latex_available:
            pytest.skip("LaTeX not available, skipping PDF build test")

        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "latex"

            result = subprocess.run(
                ["sphinx-build", "-b", "latex", "-q", str(docs_dir), str(build_dir)],
                capture_output=True,
                text=True,
            )

            assert (
                result.returncode == 0
            ), f"LaTeX build failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
