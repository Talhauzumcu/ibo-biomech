"""Sphinx configuration for the ibo-biomech documentation.

Builds an API reference from the library's Google-style docstrings using
autodoc + napoleon, rendered with the Read the Docs theme.
"""
import os
import sys

# Make the package importable for autodoc (also works without `pip install`).
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ibo-biomech"
author = "Talha"
copyright = "2026, Talha"

try:
    from ibo_biomech import __version__ as release
except Exception:  # pragma: no cover - docs should still build
    release = ""
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from the source
    "sphinx.ext.napoleon",     # understand Google-style docstrings
    "sphinx.ext.viewcode",     # add [source] links
    "sphinx.ext.intersphinx",  # cross-link to numpy / python docs
    "myst_parser",             # allow Markdown pages (e.g. the README)
]

# Napoleon: Google style only.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = True

# Autodoc behaviour.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"

# Cross-reference external libraries.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
