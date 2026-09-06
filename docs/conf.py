"""Sphinx configuration for pyunwrap's documentation (hosted on Read the Docs).

Build locally with:
    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import sys

# -- Path setup --------------------------------------------------------------
# Make the `pyunwrap` package importable for autodoc without installing it,
# so `sphinx-build` works straight from a fresh clone.
sys.path.insert(0, os.path.abspath(".."))

import shutil
from pathlib import Path


def _sync_notebooks(app) -> None:
    """Copy the example notebooks from the top-level `notebooks/` directory
    into `docs/tutorials/` before Sphinx reads the source tree.

    Sphinx's toctree can only reference documents that live inside `srcdir`
    (`docs/`) -- a relative `../notebooks/foo` toctree entry is silently
    treated as `docs/notebooks/foo` and reported as a missing document, not
    resolved outside the source tree. Rather than maintain two hand-synced
    copies of each notebook (one for `git clone` users, one for the docs
    site), this hook keeps `notebooks/` as the single source of truth and
    copies from it automatically on every build.
    """
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "notebooks"
    dst_dir = Path(app.srcdir) / "tutorials" / "_notebooks"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for notebook in src_dir.glob("*.ipynb"):
        shutil.copy2(notebook, dst_dir / notebook.name)


def setup(app):
    app.connect("builder-inited", _sync_notebooks)

# -- Project information ------------------------------------------------------
project = "pyunwrap"
copyright = "2026, pyunwrap contributors"
author = "pyunwrap contributors"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parses the Google-style docstrings used throughout the codebase
    "sphinx.ext.viewcode",  # adds "[source]" links next to every documented object
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # renders the phase-unwrapping equations on the landing/architecture pages
    "sphinx_autodoc_typehints",  # renders type hints in the signature, not duplicated in the body
    "sphinx_copybutton",  # adds a copy-to-clipboard button on code blocks
    "sphinx_design",  # grid/card layout used on the landing page
    "myst_parser",  # Markdown support, so README/CHANGELOG/CONTRIBUTING can be reused as-is
    "nbsphinx",  # renders the pre-executed example notebooks with their real outputs
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST (Markdown) extensions: enable the handful actually used across the
# included README/CHANGELOG/CONTRIBUTING/architecture pages (fenced code
# blocks with attributes, definition lists, and auto-generated header
# anchors so in-page Markdown links like `#installation` resolve).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

# A small number of relative links inside the shared root-level files
# (CONTRIBUTING.md's `[MIT License](LICENSE)`) are correct on GitHub, where
# LICENSE is a sibling file, but resolve relative to the *including* page
# once pulled into this docs site via MyST's `{include}` directive. Fixing
# this would mean either rewriting a file that must stay correct for GitHub's
# native rendering, or duplicating its content -- both worse than a single,
# explicitly-documented, harmless warning suppression.
suppress_warnings = [
    "myst.xref_missing",
    # sphinx_autodoc_typehints resolves type hints via a path independent of
    # autodoc_mock_imports, so it still inspects the *real* installed jinja2
    # package (needed for real at runtime, so it can't just be uninstalled)
    # and hits a forward-reference issue inside jinja2's OWN annotations --
    # confirmed by the warning's file path pointing into jinja2's own
    # site-packages source, not anything in pyunwrap.
    "sphinx_autodoc_typehints.forward_reference",
]

# -- Napoleon (Google-style docstrings) ---------------------------------------
# napoleon_use_ivar=True renders a docstring's "Attributes:" section as
# inline `:ivar:`/`:vartype:` field-list items rather than standalone
# `.. attribute::` directives. Without this, Napoleon's own attribute
# directives collide with autodoc's independent discovery of the same
# dataclass fields (every `@dataclass` in this codebase documents its
# fields via an Attributes: section), producing a "duplicate object
# description" warning for every single field on every dataclass -- fixed
# by this one setting rather than the ~40 individual `:no-index:`
# annotations Sphinx's own warning text suggests.
napoleon_use_ivar = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Autodoc / autosummary ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autoclass_content = "both"  # merge the class docstring and __init__'s docstring

# Heavy / hard-to-install dependencies are mocked rather than installed, so
# a documentation build never needs a GPU, a compiled GDAL, or a compiled
# SNAPHU binary just to import a module and read its docstrings. Sphinx's
# mock system supports the common patterns used here (subclassing a mocked
# base class, e.g. `class AmbiguityNet(nn.Module)`) without needing the real
# library. Lighter, more portable dependencies (numpy, pandas, matplotlib,
# plotly, folium, jinja2, scikit-image) are installed for real via the
# `docs` extras group instead, since they build quickly and reliably on
# Read the Docs' standard build image.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "rasterio",
    "h5py",
    "onnx",
    "onnxruntime",
    "onnxconverter_common",
    "onnxscript",
    "snaphu",
    "openvino",
    "jinja2",  # lightweight to actually install, but sphinx_autodoc_typehints
    # otherwise tries (and fails) to resolve a forward reference inside
    # jinja2's OWN type annotations, not anything in this codebase.
]

# -- Intersphinx ---------------------------------------------------------------
# Note: in network-restricted build environments (e.g. this project's own
# development sandbox, which allowlists only package-registry domains) these
# inventory fetches fail with a "failed to reach any of the inventories"
# warning. That's an environment limitation, not a configuration error --
# Read the Docs' actual build servers have normal internet access and these
# mappings work as expected there.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- nbsphinx ------------------------------------------------------------------
# The example notebooks are checked into the repo pre-executed with real
# outputs (see notebooks/README or the top-level README's "Notebooks"
# section) specifically so they're readable without rerunning anything.
# `never` here is deliberate, not a default left unset: re-executing them
# during a docs build would require installing the full torch/rasterio/
# onnxruntime/snaphu stack in the docs build environment, defeating the
# point of mocking those imports above.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- HTML output ---------------------------------------------------------------
# sphinx_rtd_theme (not Furo): the classic Read the Docs theme, matching the
# EOCoreINT/pygeofetch sibling project's documentation site for a consistent
# look across the org's projects.
html_theme = "sphinx_rtd_theme"
html_title = f"{project} v{release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon-64.png"
html_logo = "_static/icon.svg"

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    # Match pygeofetch's default RTD accent (the theme's own signature green);
    # override here only if a brand color is specifically wanted instead.
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
