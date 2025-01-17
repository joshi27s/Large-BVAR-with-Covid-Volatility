# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'covbayesvar'
copyright = '2024, Sudiksha Joshi'
author = 'Sudiksha Joshi'
release = '0.1.7'

# -- Path setup --------------------------------------------------------------
# Add the project's root directory to sys.path so Sphinx can find your package
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Go one level up to include the project root

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autosummary_generate = True  # Enable autosummary generation

extensions = [
    'sphinx.ext.autodoc',            # Automatically generate documentation from docstrings
    'sphinx.ext.autosummary',        # Generate summary tables automatically
    'sphinx_autodoc_typehints',      # Type hinting in documentation
    'sphinx.ext.napoleon',           # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.mathjax'             # For LaTeX rendering
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_static_path = ['_static']

