import pkg_resources

project = "Esquilax"
copyright = "2024, zombie-einstein"
author = "zombie-einstein"
release = pkg_resources.get_distribution("esquilax").version

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]

autoapi_dirs = [
    "../../src",
]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_member_order = "alphabetical"
autoapi_own_page_level = "function"
autoapi_python_class_content = "both"
autoapi_template_dir = "_autoapi_templates"
autoapi_python_use_implicit_namespaces = False
autoapi_keep_files = False
autoapi_type = "python"

add_module_names = False
add_package_names = False

autodoc_typehints = "signature"

exclude_patterns = ["_autoapi_templates/**"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
# napoleon_use_rtype = False
napoleon_preprocess_types = True
# napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "chex": ("https://chex.readthedocs.io/en/latest", None),
    "flax": ("https://flax.readthedocs.io/en/latest", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
}

html_title = "Esquilax"
html_theme = "piccolo_theme"
html_static_path = ["_static"]
html_logo = "./_static/images/logo_white.png"
html_favicon = "./_static/images/favicon.png"

html_theme_options = {
    "source_url": "https://github.com/zombie-einstein/esquilax",
    "source_icon": "github",
}
