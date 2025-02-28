import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

project = "pangeo-fish"
author = "pangeo-fish developers"
copyright = f"2023-{datetime.datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.doctest",
    # "autoapi.extension",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_remove_toctrees",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

autosummary_generate = True
autosummary_imported_members = False
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]
remove_from_toctrees = ["generated/*"]

html_theme = "sphinx_book_theme"
pygments_style = "sphinx"

html_static_path = []
