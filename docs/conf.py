# General information about the project.
project = "TileLang MUSA <br>"
author = "Tile Lang Contributors"
copyright = f"2025-2025, {author}"

# Version information.
with open("../VERSION") as f:
    version = f.read().strip()
with open("../VERSION_PATCH") as f:
    release = f"{version}+musa.{f.read().strip()}"

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension",
]

autoapi_type = "python"
autoapi_dirs = ["../tilelang"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_keep_files = False  # Useful for debugging the generated rst files

autoapi_generate_api_docs = True

autodoc_typehints = "description"

autoapi_ignore = ["*language/ast*", "*version*", "*libinfo*", "*parser*"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_title_to_header = True
myst_heading_anchors = 4

redirects = {"get_started/try_out": "../index.html#getting-started"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**/*libinfo*"]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
templates_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]
footer_copyright = "© 2025-2026 TileLang"
footer_note = " "

html_theme_options = {"light_logo": "img/logo-v2.png", "dark_logo": "img/logo-v2.png"}

header_links = [
    ("Home", "https://github.com/tile-ai/tilelang-musa"),
    ("Github", "https://github.com/tile-ai/tilelang-musa"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "tile-ai",
    "github_repo": "tilelang-musa",
    "github_version": f"v{release}/docs/",
    "theme_vcs_pageview_mode": "edit",
}
