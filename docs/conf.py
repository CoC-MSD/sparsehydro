# -*- coding: utf-8 -*-
#
# sparsehydro documentation build configuration file

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from sparsehydro import __version__

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# Mermaid — use CDN so no mermaid CLI install is needed
mermaid_version = "10.9.0"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default', securityLevel:'loose'});"
mermaid_d3_zoom = True

add_function_parentheses = True
add_module_names = False
python_display_short_literal_types = True

toc_object_entries = True
toc_object_entries_show_parents = "hide"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = True
napoleon_preprocess_types = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = False
napoleon_custom_sections = ["Model Attributes", "Class Methods", "Enum Members"]

autodoc_default_options = {
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "groupwise",
}

autodoc_class_signature = "mixed"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {"DataFrame": "pandas.DataFrame", "Series": "pandas.Series"}

autoclass_content = "both"

# Optional third-party dependencies not installed in the ``docs`` build.
# Mock them so autodoc/autosummary can import modules that reference them.
autodoc_mock_imports = ["plotly"]

numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

templates_path = ["_templates"]

autosummary_generate = True
autosummary_generate_overwrite = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST — enable mermaid directive in .md files
myst_fence_as_directive = ["mermaid"]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

master_doc = "index"

project = "sparsehydro"
copyright = "2026 Hazen and Sawyer"
author = "Caleb Buahin"
version = __version__
release = __version__

language = "en"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "rdii_design.md", "unithydrograph_strategy.md", "combined_model.md", "ensemble_visualization_plan.md", "QAQC_dt.md", "parsimonious_integration_plan.md"]

suppress_warnings = [
    "ref.myst",
    "myst.duplicate_def",
    "myst.xref_missing",
    "ref.ref",
    "ref.doc",
    "ref.any",
    "ref.python",
    "autodoc",
]

nitpicky = False

pygments_style = "sphinx"

todo_include_todos = True

# -- HTML output ---------------------------------------------------------------

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    html_theme = "pydata_sphinx_theme"
else:
    html_theme = "default"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MSDGC-SWM/sparsehydro",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": False,
    "primary_sidebar_end": ["indices.html"],
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher.html", "navbar-icon-links.html"],
}

html_static_path = ["_static"]
html_extra_path = [".nojekyll"]

htmlhelp_basename = "sparsehydrodoc"

# -- LaTeX output --------------------------------------------------------------

latex_elements = {
    "printindex": "",
    "sphinxsetup": "hmargin={0.9in,0.9in}, vmargin={0.9in,0.9in}, marginpar=1.0in",
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{tcolorbox}
\definecolor{sphinxnoteBgColor}{RGB}{221,233,239}
\renewenvironment{sphinxnote}[1]
{\begin{tcolorbox}[colback=sphinxnoteBgColor,
colframe=red!35!green!50!blue!87!,
title=\sphinxstrong{#1}]}
{\end{tcolorbox}}
""",
    "figure_align": "htbp",
}

latex_documents = [
    (
        "index",
        "sparsehydro.tex",
        "sparsehydro Documentation",
        "Caleb Buahin",
        "howto",
        "True",
    ),
]

latex_domain_indices = False
latex_use_modindex = False

man_pages = [
    (
        master_doc,
        "sparsehydro",
        "sparsehydro Documentation",
        [author],
        1,
    )
]

texinfo_documents = [
    (
        master_doc,
        "sparsehydro",
        "sparsehydro Documentation",
        author,
        "sparsehydro",
        "Abstract interfaces and utilities for parsimonious hydrological models.",
        "Miscellaneous",
    ),
]


def setup(app):
    app.add_css_file("sparsehydro.css")
