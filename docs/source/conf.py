import pkg_resources

project = "Esquilax"
copyright = "2024, zombie-einstein"
author = "zombie-einstein"
release = pkg_resources.get_distribution("esquilax").version

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
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

exclude_patterns = ["_autoapi_templates/**"]

add_module_names = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "chex": ("https://chex.readthedocs.io/en/latest", None),
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
