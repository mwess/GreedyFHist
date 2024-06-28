import os
import sphinx_rtd_theme

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GreedyFHist'
copyright = '2024, Maximilian Wess'
author = 'Maximilian Wess'
release = '28/06/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser"]

source_suffix = ['.md']

templates_path = ['_templates']
exclude_patterns = []

master_doc = 'index'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']



# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_context = {
    "sidebar_external_links_caption": "Links",
    "sidebar_external_links": [
        (
            '<i class="fa fa-rss fa-fw"></i> Blog',
            "https://www.poliastro.space",
        ),
        (
            '<i class="fa fa-github fa-fw"></i> Source code',
            "https://github.com/poliastro/poliastro",
        ),
        (
            '<i class="fa fa-bug fa-fw"></i> Issue tracker',
            "https://github.com/poliastro/poliastro/issues",
        ),
        (
            '<i class="fa fa-envelope fa-fw"></i> Mailing list',
            "https://groups.io/g/poliastro-dev",
        ),
        (
            '<i class="fa fa-comments fa-fw"></i> Chat',
            "http://chat.poliastro.space",
        ),
        (
            '<i class="fa fa-file-text fa-fw"></i> Citation',
            "https://doi.org/10.5281/zenodo.593610",
        ),
    ],
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = os.path.join("css", "custom.css")
# html_favicon = os.path.join("_static", "favicon.ico")
html_static_path = ["_static"]
html_js_files = ["language_data.js"]

# html_sidebars = {
#     '**': [
#         'installation.html',
#         'usage/config.html',
#         'usage/pairwise.html',
#         'usage/groupwise.html'
#     ]
# }

html_sidebars = {
    '**': [
        'globaltoc.html'
    ]
}