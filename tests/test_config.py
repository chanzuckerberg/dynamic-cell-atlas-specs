"""
Unit tests for documentation configuration validation.
"""

import pytest
import yaml


@pytest.mark.unit
class TestSphinxConfiguration:
    """Test Sphinx configuration validity."""

    def test_sphinx_config_loads(self, sphinx_config):
        """Test that Sphinx configuration loads without errors."""
        assert sphinx_config is not None
        assert hasattr(sphinx_config, "project")

    def test_required_config_attributes(self, sphinx_config):
        """Test that required Sphinx configuration attributes exist."""
        required_attrs = [
            "project",
            "copyright",
            "author",
            "release",
            "extensions",
            "html_theme",
        ]

        for attr in required_attrs:
            assert hasattr(sphinx_config, attr), f"Missing required config: {attr}"

    def test_extensions_are_valid(self, sphinx_config):
        """Test that all configured extensions are valid."""
        required_extensions = [
            "sphinx.ext.todo",
            "sphinxcontrib.mermaid",
            "sphinx_reredirects",
        ]

        for ext in required_extensions:
            assert ext in sphinx_config.extensions, f"Missing extension: {ext}"

    def test_theme_configuration(self, sphinx_config):
        """Test that theme is properly configured."""
        assert sphinx_config.html_theme == "pydata_sphinx_theme"
        assert hasattr(sphinx_config, "html_theme_options")
        assert isinstance(sphinx_config.html_theme_options, dict)

    def test_version_format(self, sphinx_config):
        """Test that version follows expected format."""
        version = sphinx_config.release
        assert version.startswith("v"), f"Version should start with 'v': {version}"
        assert version.count(".") >= 1, f"Version should be semantic version: {version}"


@pytest.mark.unit
class TestReadTheDocsConfiguration:
    """Test ReadTheDocs configuration validity."""

    def test_readthedocs_config_exists(self, root_dir):
        """Test that .readthedocs.yaml file exists."""
        rtd_config = root_dir / ".readthedocs.yaml"
        assert rtd_config.exists(), "Missing .readthedocs.yaml file"

    def test_readthedocs_config_valid_yaml(self, root_dir):
        """Test that .readthedocs.yaml is valid YAML."""
        rtd_config = root_dir / ".readthedocs.yaml"

        with open(rtd_config, "r") as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert isinstance(config, dict)

    def test_readthedocs_required_fields(self, root_dir):
        """Test that ReadTheDocs config has required fields."""
        rtd_config = root_dir / ".readthedocs.yaml"

        with open(rtd_config, "r") as f:
            config = yaml.safe_load(f)

        required_fields = ["version", "build", "sphinx", "python"]

        for field in required_fields:
            assert field in config, f"Missing required field: {field}"

        assert config["version"] == 2, "Should use ReadTheDocs version 2"


@pytest.mark.unit
class TestDependencies:
    """Test project dependencies and requirements."""

    def test_requirements_file_exists(self, docs_dir):
        """Test that requirements.txt exists."""
        requirements_file = docs_dir / "requirements.txt"
        assert requirements_file.exists(), "Missing docs/requirements.txt"

    def test_required_packages_listed(self, docs_dir):
        """Test that essential packages are in requirements.txt."""
        requirements_file = docs_dir / "requirements.txt"

        with open(requirements_file, "r") as f:
            requirements = f.read()

        required_packages = [
            "sphinx",
            "pydata-sphinx-theme",
            "sphinxcontrib-mermaid",
            "sphinx-reredirects",
            "sphinx-autobuild",
        ]

        for package in required_packages:
            assert package in requirements, f"Missing required package: {package}"


@pytest.mark.unit
class TestDirectoryStructure:
    """Test repository directory structure."""

    def test_docs_directory_exists(self, docs_dir):
        """Test that docs directory exists with expected structure."""
        assert docs_dir.exists()
        assert docs_dir.is_dir()

    def test_essential_files_exist(self, docs_dir):
        """Test that essential documentation files exist."""
        essential_files = [
            "conf.py",
            "index.rst",
            "requirements.txt",
            "Makefile",
            ".nojekyll",  # GitHub Pages Jekyll prevention
            "_static/css/custom.css",
            "v0.1/index.rst",
            "v0.1/array-standard.rst",
            "v0.1/metadata-standard.rst",
            "v0.2/index.rst",
            "v0.2/array-standard.rst",
            "v0.2/metadata-standard.rst",
            "v0.2/channel-metadata.rst",
            "v0.2/schema.json",
        ]

        for file_path in essential_files:
            full_path = docs_dir / file_path
            assert full_path.exists(), f"Missing essential file: {file_path}"


@pytest.mark.unit
class TestGitHubPagesConfiguration:
    """Test GitHub Pages deployment configuration."""

    def test_pages_workflow_exists(self, root_dir):
        """Test that GitHub Pages deployment workflow exists."""
        workflow_file = root_dir / ".github" / "workflows" / "deploy-pages.yml"
        assert workflow_file.exists(), "Missing GitHub Pages deployment workflow"

    def test_pages_workflow_configuration(self, root_dir):
        """Test that Pages workflow has required configuration."""
        import yaml

        workflow_file = root_dir / ".github" / "workflows" / "deploy-pages.yml"

        with open(workflow_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Check basic workflow structure
        assert "name" in workflow, "Workflow missing name"
        # Note: PyYAML parses 'on' as boolean True, so we check for True key
        assert True in workflow or "on" in workflow, "Workflow missing triggers"
        assert "jobs" in workflow, "Workflow missing jobs"

        # Check permissions for Pages deployment
        assert "permissions" in workflow, "Workflow missing permissions"
        permissions = workflow["permissions"]
        assert "pages" in permissions, "Missing pages permission"
        assert "id-token" in permissions, "Missing id-token permission"
        assert permissions["pages"] == "write", "Pages permission should be write"
        assert permissions["id-token"] == "write", "ID token permission should be write"

        # Check for build and deploy jobs
        jobs = workflow["jobs"]
        assert "build" in jobs, "Missing build job"
        assert "deploy" in jobs, "Missing deploy job"

        # Check deploy job depends on build
        deploy_job = jobs["deploy"]
        assert "needs" in deploy_job, "Deploy job should depend on build job"
        assert "build" in deploy_job["needs"], "Deploy job should need build job"

    def test_nojekyll_file_exists(self, docs_dir):
        """Test that .nojekyll file exists to prevent Jekyll processing."""
        nojekyll_file = docs_dir / ".nojekyll"
        assert nojekyll_file.exists(), "Missing .nojekyll file for GitHub Pages"
