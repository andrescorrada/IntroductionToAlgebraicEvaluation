# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
from ntqr import __version__ as ntqr_version


sys.path.insert(0, str(Path("..", "..", "src").resolve()))
# sys.path.insert(0, str(Path("..", "src").resolve()))
# sys.path.insert(0, str(Path("..").resolve()))


# -- Project information -----------------------------------------------------

project = "NTQR"
copyright = "2024, Andrés Corrada-Emmanuel, Walker Lee, Adam Sloat"
authors = ["Andrés Corrada-Emmanuel", "Walker Lee", "Adam Sloat"]

# The full version, including alpha/beta/rc tags
version = ntqr_version
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Using myst-nb automatically imports myst-parser,
    # "myst_nb",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    # "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# List of arguments to be passed to the kernel that executes the notebooks:
# nbsphinx_execute_arguments = [
#     "--InlineBackend.figure_formats={'svg', 'pdf'}",
# ]

# nbsphinx_thumbnails = {
#     "gallery/thumbnail-from-conf-py": "gallery/a-local-file.png",
#     "gallery/*-rst": "images/notebook_icon.png",
#     "orphan": "_static/favicon.svg",
# }

# nb_ipywidgets_js = {
#     "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js": {
#         "integrity": "sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=",
#         "crossorigin": "anonymous",
#     },
#     "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@1.0.6/dist/embed-amd.js": {
#         "data-jupyter-widgets-cdn": "https://cdn.jsdelivr.net/npm/",
#         "crossorigin": "anonymous",
#     },
# }


nbsphinx_assume_equations = True
nbsphinx_execute = "auto"
# nbsphinx_custom_formats = {
#     ".md": ["jupytext.reads", {"fmt": "mystnb"}],
# }
# nb_execution_mode = "auto"

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

autoapi_dirs = [str(Path("..", "..", "src"))]
autoapi_type = "python"
autoapi_generate_api_docs = True
autoapi_keep_files = False
autoapi_template_dir = "_templates/autoapi/python"

napoleon_include_init_with_doc = False


mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
