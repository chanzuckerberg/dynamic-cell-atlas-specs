# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
from datetime import datetime
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "Dynamic Cell Atlas Specifications"
copyright = "2026, Biohub"
author = "Benjamin Gallusser, Mikala Caton"
release = "v0.1"

repo_root = Path(__file__).parent.parent


def _git_last_updated(path):
    """Get the last commit date that touched files under path."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci", "--", path],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        date_str = result.stdout.strip()
        if not date_str:
            return "Unknown"
        dt = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return "Unknown"


last_updated_v01 = _git_last_updated("docs/v0.1/")
last_updated_v02 = _git_last_updated("docs/v0.2/")
last_updated_v03 = _git_last_updated("docs/v0.3/")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_reredirects",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings for files that exist but are intentionally not included
suppress_warnings = ["toc.not_included"]

# Make per-version last-updated dates available as RST substitutions
rst_prolog = f"""
.. |last_updated_v01| replace:: {last_updated_v01}
.. |last_updated_v02| replace:: {last_updated_v02}
.. |last_updated_v03| replace:: {last_updated_v03}
"""

# -- Options for HTML output ------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Copy .nojekyll file for GitHub Pages
html_extra_path = [".nojekyll"]

html_logo = "_static/biohub_logo.png"

html_context = {"default_mode": "light"}

# Pygments syntax highlighting — light style matching Biohub brand
pygments_style = "default"


html_theme_options = {
    "show_prev_next": True,
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 3,
    "logo": {
        "text": "DCA Specs",
    },
    # Biohub Brand Configuration
    # CZI Brand Color Configuration
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/chanzuckerberg/dynamic-cell-atlas-specs",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    # PyData theme layout customization
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_persistent": ["search-button"],
}

html_title = "Dynamic Cell Atlas Specification"

# -- Extension configuration -------------------------------------------------

todo_include_todos = True

# Redirects for future version management
redirects: dict[str, str] = {}

# Link checking configuration
linkcheck_ignore = [
    r"http://localhost:\d+/",
    r"https://example\.com/.*",
    # Ignore GitHub repositories — rate-limited from CI runners
    r"https://github\.com/.*",
    # Ignore external databases — slow from CI runners but valid links
    r"https://geneontology\.org/.*",
    r"https://www\.ensembl\.org/.*",
    r"https://www\.genenames\.org/.*",
    r"https://www\.uniprot\.org/.*",
]

# Link check settings
linkcheck_timeout = 3  # 3 second timeout for link checks
linkcheck_workers = 5  # Number of concurrent workers

# Source file suffixes
source_suffix = ".rst"

# The master toctree document
master_doc = "index"
