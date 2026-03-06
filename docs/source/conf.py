# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kszx'
copyright = '2024 Selim, Yurii, Edmond, Alex, Mat, Kendrick, et al'
author = 'Selim, Yurii, Edmond, Alex, Mat, Kendrick, et al'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import kszx

# https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html
extensions = ['sphinx.ext.autosectionlabel']

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
extensions += ['sphinx.ext.autosummary', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon']
autoclass_content = 'both'
autodoc_member_order = 'bysource'

# https://www.sympy.org/sphinx-math-dollar/
extensions += ['sphinx_math_dollar', 'sphinx.ext.mathjax']

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

# linkcode disabled for now -- it didn't work very well out of the box, hope to revisit later
# extensions += ['sphinx.ext.linkcode']

def linkcode_resolve(domain, info):
    """Called by sphinx.ext.linkcode, to return github url.
    FIXME currently only links to the source file, not a line number within the source file"""
    
    assert info['module'] == 'kszx'   # FIXME works for now, but I'll need to revisit later
    fullname = info['fullname']
    module = getattr(kszx,fullname).__module__
    m = module.replace('.','/')
    url = f'https://github.com/kmsmith137/kszx/blob/main/{m}.py'
    # print(f'debug linkcode_resolve: {fullname} -> {module} -> {url}')
    return url

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_static_path = ['_static']

# https://stackoverflow.com/questions/77100412
html_theme_options = {
    'body_max_width' : 'none',
    'page_width': 'auto',
}
