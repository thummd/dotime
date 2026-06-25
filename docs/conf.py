# Sphinx configuration for CausalTime documentation.
#
# Build locally:
#     pip install -e ".[docs]"
#     cd docs && make html
#     open _build/html/index.html
#
# Read the Docs picks up this file automatically; pin the install via a
# `.readthedocs.yaml` at the repo root that installs the `docs` extra.

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Make the package importable for autodoc without installing it on RTD's
# build container in editable mode. Adjust the relative path if you move
# this file outside `docs/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import causaltime  # noqa: E402  (must come after sys.path tweak)


# -- Project information -----------------------------------------------------

project = "CausalTime"
author = "Dennis Thumm and contributors"
copyright = f"{datetime.now().year}, {author}"  # noqa: A001
release = causaltime.__version__
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    # Markdown + notebooks
    "myst_parser",
    "nbsphinx",
    # Niceties
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",  # ignored if myst-nb not installed; nbsphinx handles .ipynb
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# Show TODOs only on local / preview builds.
todo_include_todos = os.environ.get("READTHEDOCS", "False").lower() != "true"


# -- MyST (Markdown) options -------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "linkify",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3


# -- nbsphinx (executed notebooks) ------------------------------------------

# "auto"  -> execute only if no outputs are stored
# "never" -> trust whatever is committed (recommended for RTD)
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_timeout = 600


# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
# Document dataclass/class attributes inline (:ivar:) rather than as separate
# object descriptions, avoiding duplicate-object-description warnings under -W.
napoleon_use_ivar = True


# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy", None),
    "torch":      ("https://pytorch.org/docs/stable", None),
    "networkx":   ("https://networkx.org/documentation/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas":     ("https://pandas.pydata.org/docs", None),
}


# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files: list[str] = []  # add overrides under _static/ as you grow

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
}

html_title = f"{project} {version}"
html_short_title = project
html_show_sourcelink = True
html_copy_source = False
html_show_sphinx = False
