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
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinx.ext.napoleon",
]

nbsphinx_assume_equations = True
nbsphinx_execute = "never"
autoapi_dirs = [str(Path("..", "..", "src"))]
autoapi_type = "python"
autoapi_generate_api_docs = True
autoapi_keep_files = False
autoapi_template_dir = "_templates/autoapi/python"

napoleon_include_init_with_doc = False

myst_enable_extensions = [
    # "colon_fence",
    # "dollarmath",
    "amsmath",
]

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
