import os
import sys
import json
from uuid import uuid4
import shutil
import re
from typing import Dict, Any, List

from pathlib import Path

from urllib.parse import quote

import subprocess
from subprocess import CompletedProcess

from docutils.parsers.rst import directives
from docutils.nodes import SkipNode, Element
from docutils import nodes

from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.fileutil import copy_asset
from sphinx.parsers import RSTParser

from ._try_examples import (
    examples_to_notebook,
    insert_try_examples_directive,
    new_code_cell,
)

import jupytext
import nbformat

try:
    import voici
except ImportError:
    voici = None

HERE = Path(__file__).parent

CONTENT_DIR = "_contents"
JUPYTERLITE_DIR = "lite"


# Used for nodes that do not need to be rendered
def skip(self, node):
    raise SkipNode


# Used to render an element node as HTML
def visit_element_html(self, node):
    self.body.append(node.html())
    raise SkipNode


def _build_options(lite_options: dict[str, str]) -> str:
    """Concatenates options into query parameters, fixing the capitalization
    for parameters where the necessarily lowercase docutils directive value
    needs some uppercase letters in a query parameter."""

    replacements = {
        "showbanner": "showBanner",
    }

    lite_options = (
        (replacements.get(key, key), value) for key, value in lite_options.items()
    )

    return "&".join([f"{key}={quote(value)}" for key, value in lite_options])


class _PromptedIframe(Element):
    def __init__(
        self,
        rawsource="",
        *children,
        iframe_src="",
        width="100%",
        height="100%",
        prompt=False,
        prompt_color=None,
        search_params="false",
        **attributes,
    ):
        super().__init__(
            "",
            iframe_src=iframe_src,
            width=width,
            height=height,
            prompt=prompt,
            prompt_color=prompt_color,
            search_params=search_params,
        )

    def html(self):
        iframe_src = self["iframe_src"]
        search_params = self["search_params"]

        if self["prompt"]:
            prompt = (
                self["prompt"] if isinstance(self["prompt"], str) else "Try It Live!"
            )
            prompt_color = (
                self["prompt_color"] if self["prompt_color"] is not None else "#f7dc1e"
            )

            placeholder_id = uuid4()
            container_style = f'width: {self["width"]}; height: {self["height"]};'

            return f"""
                <div
                    class=\"jupyterlite_sphinx_iframe_container\"
                    style=\"{container_style}\"
                    onclick=\"window.jupyterliteShowIframe(
                        '{placeholder_id}',
                        window.jupyterliteConcatSearchParams('{iframe_src}', {search_params})
                    )\"
                >
                    <div
                        id={placeholder_id}
                        class=\"jupyterlite_sphinx_try_it_button jupyterlite_sphinx_try_it_button_unclicked\"
                        style=\"background-color: {prompt_color};\"
                    >
                    {prompt}
                    </div>
                </div>
            """

        return (
            f'<iframe src="{iframe_src}"'
            f'width="{self["width"]}" height="{self["height"]}" class="jupyterlite_sphinx_raw_iframe"></iframe>'
        )


class _InTab(Element):
    def __init__(
        self,
        rawsource="",
        *children,
        prefix=JUPYTERLITE_DIR,
        notebook=None,
        lite_options={},
        button_text=None,
        **attributes,
    ):
        app_path = self.lite_app
        if notebook is not None:
            lite_options["path"] = notebook
            app_path = f"{self.lite_app}{self.notebooks_path}"

        options = _build_options(lite_options)
        self.lab_src = (
            f'{prefix}/{app_path}{f"index.html?{options}" if options else ""}'
        )

        self.button_text = button_text

        super().__init__(
            rawsource,
            **attributes,
        )

    def html(self):
        return (
            '<button class="try_examples_button" '
            f"onclick=\"window.open('{self.lab_src}')\">"
            f"{self.button_text}</button>"
        )


class _LiteIframe(_PromptedIframe):
    def __init__(
        self,
        rawsource="",
        *children,
        prefix=JUPYTERLITE_DIR,
        content=[],
        notebook=None,
        lite_options={},
        **attributes,
    ):
        if content:
            code_lines = ["" if not line.strip() else line for line in content]
            code = "\n".join(code_lines)

            lite_options["code"] = code

        app_path = self.lite_app
        if notebook is not None:
            lite_options["path"] = notebook
            app_path = f"{self.lite_app}{self.notebooks_path}"

        options = _build_options(lite_options)

        iframe_src = f'{prefix}/{app_path}{f"index.html?{options}" if options else ""}'

        if "iframe_src" in attributes:
            if attributes["iframe_src"] != iframe_src:
                raise ValueError(
                    f'Two different values of iframe_src {attributes["iframe_src"]=},{iframe_src=}, try upgrading sphinx to v 7.2.0 or more recent'
                )
            del attributes["iframe_src"]

        super().__init__(rawsource, *children, iframe_src=iframe_src, **attributes)


class RepliteIframe(_LiteIframe):
    """Appended to the doctree by the RepliteDirective directive

    Renders an iframe that shows a repl with JupyterLite.
    """

    lite_app = "repl/"
    notebooks_path = ""


class JupyterLiteIframe(_LiteIframe):
    """Appended to the doctree by the JupyterliteDirective directive

    Renders an iframe that shows a Notebook with JupyterLite.
    """

    lite_app = "lab/"
    notebooks_path = ""


class BaseNotebookTab(_InTab):
    """Base class for notebook tab implementations. We subclass this
    to create more specific configurations around how tabs are rendered."""

    lite_app = None
    notebooks_path = None
    default_button_text = "Open as a notebook"


class JupyterLiteTab(BaseNotebookTab):
    """Appended to the doctree by the JupyterliteDirective directive

    Renders a button that opens a Notebook with JupyterLite in a new tab.
    """

    lite_app = "lab/"
    notebooks_path = ""


class NotebookLiteTab(BaseNotebookTab):
    """Appended to the doctree by the NotebookliteDirective directive

    Renders a button that opens a Notebook with NotebookLite in a new tab.
    """

    lite_app = "tree/"
    notebooks_path = "../notebooks/"


# We do not inherit from _InTab here because Replite
# has a different URL structure and we need to ensure
# that the code is serialised to be passed to the URL.
class RepliteTab(Element):
    """Appended to the doctree by the RepliteDirective directive

    Renders a button that opens a REPL with JupyterLite in a new tab.
    """

    lite_app = "repl/"
    notebooks_path = ""

    def __init__(
        self,
        rawsource="",
        *children,
        prefix=JUPYTERLITE_DIR,
        content=[],
        notebook=None,
        lite_options={},
        button_text=None,
        **attributes,
    ):
        # For a new-tabbed variant, we need to ensure we process the content
        # into properly encoded code for passing it to the URL.
        if content:
            code_lines: list[str] = [
                "" if not line.strip() else line for line in content
            ]
            code = "\n".join(code_lines)
            lite_options["code"] = code

        if "execute" in lite_options and lite_options["execute"] == "0":
            lite_options["execute"] = "0"

        app_path = self.lite_app
        if notebook is not None:
            lite_options["path"] = notebook
            app_path = f"{self.lite_app}{self.notebooks_path}"

        options = _build_options(lite_options)

        self.lab_src = (
            f'{prefix}/{app_path}{f"index.html?{options}" if options else ""}'
        )

        self.button_text = button_text

        super().__init__(
            rawsource,
            **attributes,
        )

    def html(self):
        return (
            '<button class="try_examples_button" '
            f"onclick=\"window.open('{self.lab_src}')\">"
            f"{self.button_text}</button>"
        )


class NotebookLiteIframe(_LiteIframe):
    """Appended to the doctree by the NotebookliteDirective directive

    Renders an iframe that shows a Notebook with NotebookLite.
    """

    lite_app = "tree/"
    notebooks_path = "../notebooks/"


class VoiciBase:
    """Base class with common Voici application paths and URL structure"""

    lite_app = "voici/"

    @classmethod
    def get_full_path(cls, notebook=None):
        """Get the complete Voici path based on whether a notebook is provided."""
        if notebook is not None:
            # For notebooks, use render path with html extension
            return f"{cls.lite_app}render/{notebook.replace('.ipynb', '.html')}"
        # Default to tree view
        return f"{cls.lite_app}tree"


class VoiciIframe(_PromptedIframe):
    """Appended to the doctree by the VoiciDirective directive

    Renders an iframe that shows a Notebook with Voici.
    """

    def __init__(
        self,
        rawsource="",
        *children,
        prefix=JUPYTERLITE_DIR,
        notebook=None,
        lite_options={},
        **attributes,
    ):
        app_path = VoiciBase.get_full_path(notebook)
        options = _build_options(lite_options)

        # If a notebook is provided, open it in the render view. Else, we default to the tree view.
        iframe_src = f'{prefix}/{app_path}{f"index.html?{options}" if options else ""}'

        super().__init__(rawsource, *children, iframe_src=iframe_src, **attributes)


# We do not inherit from BaseNotebookTab here because
# Voici has a different URL structure.
class VoiciTab(Element):
    """Tabbed implementation for the Voici interface"""

    def __init__(
        self,
        rawsource="",
        *children,
        prefix=JUPYTERLITE_DIR,
        notebook=None,
        lite_options={},
        button_text=None,
        **attributes,
    ):

        self.lab_src = f"{prefix}/"

        app_path = VoiciBase.get_full_path(notebook)
        options = _build_options(lite_options)

        # If a notebook is provided, open it in a new tab. Else, we default to the tree view.
        self.lab_src = f'{prefix}/{app_path}{f"?{options}" if options else ""}'

        self.button_text = button_text

        super().__init__(
            rawsource,
            **attributes,
        )

    def html(self):
        return (
            '<button class="try_examples_button" '
            f"onclick=\"window.open('{self.lab_src}')\">"
            f"{self.button_text}</button>"
        )


class RepliteDirective(SphinxDirective):
    """The ``.. replite::`` directive.

    Adds a replite console to the docs.
    """

    has_content = True
    required_arguments = 0
    option_spec = {
        "width": directives.unchanged,
        "height": directives.unchanged,
        "kernel": directives.unchanged,
        "execute": directives.unchanged,
        "toolbar": directives.unchanged,
        "theme": directives.unchanged,
        "prompt": directives.unchanged,
        "prompt_color": directives.unchanged,
        "search_params": directives.unchanged,
        "new_tab": directives.unchanged,
        "new_tab_button_text": directives.unchanged,
        "showbanner": directives.unchanged,
    }

    def run(self):
        width = self.options.pop("width", "100%")
        height = self.options.pop("height", "100%")

        prompt = self.options.pop("prompt", False)
        prompt_color = self.options.pop("prompt_color", None)

        search_params = search_params_parser(self.options.pop("search_params", False))

        # We first check the global config, and then the per-directive
        # option. It defaults to True for backwards compatibility.
        execute = self.options.pop("execute", str(self.env.config.replite_auto_execute))

        if execute not in ("True", "False"):
            raise ValueError("The :execute: option must be either True or False")

        if execute == "False":
            self.options["execute"] = "0"

        content = self.content

        button_text = None

        prefix = os.path.relpath(
            os.path.join(self.env.app.srcdir, JUPYTERLITE_DIR),
            os.path.dirname(self.get_source_info()[0]),
        )

        new_tab = self.options.pop("new_tab", False)

        if new_tab:
            directive_button_text = self.options.pop("new_tab_button_text", None)
            if directive_button_text is not None:
                button_text = directive_button_text
            else:
                button_text = self.env.config.replite_new_tab_button_text
            return [
                RepliteTab(
                    prefix=prefix,
                    width=width,
                    height=height,
                    prompt=prompt,
                    prompt_color=prompt_color,
                    content=content,
                    search_params=search_params,
                    lite_options=self.options,
                    button_text=button_text,
                )
            ]

        return [
            RepliteIframe(
                prefix=prefix,
                width=width,
                height=height,
                prompt=prompt,
                prompt_color=prompt_color,
                content=content,
                search_params=search_params,
                lite_options=self.options,
            )
        ]


class _LiteDirective(SphinxDirective):
    has_content = False
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "width": directives.unchanged,
        "height": directives.unchanged,
        "theme": directives.unchanged,
        "prompt": directives.unchanged,
        "prompt_color": directives.unchanged,
        "search_params": directives.unchanged,
        "new_tab": directives.unchanged,
        "new_tab_button_text": directives.unchanged,
    }

    def _target_is_stale(self, source_path: Path, target_path: Path) -> bool:
        # Used as a heuristic to determine if a markdown notebook needs to be
        # converted or reconverted to ipynb.
        if not target_path.exists():
            return True

        return source_path.stat().st_mtime > target_path.stat().st_mtime

    # TODO: Jupytext support many more formats for conversion, but we only
    # consider Markdown and IPyNB for now. If we add more formats someday,
    # we should also consider them here.
    def _assert_no_conflicting_nb_names(
        self, source_path: Path, notebooks_dir: Path
    ) -> None:
        """Check for duplicate notebook names in the documentation sources.
        Raises if any notebooks would conflict when converted to IPyNB."""
        target_stem = source_path.stem
        target_ipynb = f"{target_stem}.ipynb"

        # Only look for conflicts in source directories and among referenced notebooks.
        # We do this to prevent conflicts with other files, say, in the "_contents/"
        # directory as a result of a previous failed/interrupted build.
        if source_path.parent != notebooks_dir:

            # We only consider conflicts if notebooks are actually referenced in
            # a directive, to prevent false posiitves from being raised.
            if hasattr(self.env, "jupyterlite_notebooks"):
                for existing_nb in self.env.jupyterlite_notebooks:
                    existing_path = Path(existing_nb)
                    if (
                        existing_path.stem == target_stem
                        and existing_path != source_path
                    ):

                        raise RuntimeError(
                            "All notebooks marked for inclusion with JupyterLite must have a "
                            f"unique file basename. Found conflict between {source_path} and {existing_path}."
                        )

        return target_ipynb

    def _strip_notebook_cells(
        self, nb: nbformat.NotebookNode
    ) -> List[nbformat.NotebookNode]:
        """Strip cells based on the presence of the "jupyterlite_sphinx_strip" tag
        in the metadata. The content meant to be stripped must be inside its own cell
        cell so that the cell itself gets removed from the notebooks. This is so that
        we don't end up removing useful data or directives that are not meant to be
        removed.

        Parameters
        ----------
        nb : nbformat.NotebookNode
            The notebook object to be stripped.

        Returns
        -------
        List[nbformat.NotebookNode]
            A list of cells that are not meant to be stripped.
        """
        return [
            cell
            for cell in nb.cells
            if "jupyterlite_sphinx_strip" not in cell.metadata.get("tags", [])
        ]

    def run(self):
        width = self.options.pop("width", "100%")
        height = self.options.pop("height", "1000px")

        prompt = self.options.pop("prompt", False)
        prompt_color = self.options.pop("prompt_color", None)

        search_params = search_params_parser(self.options.pop("search_params", False))

        new_tab = self.options.pop("new_tab", False)

        button_text = None

        source_location = os.path.dirname(self.get_source_info()[0])

        prefix = os.path.relpath(
            os.path.join(self.env.app.srcdir, JUPYTERLITE_DIR), source_location
        )

        if self.arguments:
            # Keep track of the notebooks we are going through, so that we don't
            # operate on notebooks that are not meant to be included in the built
            # docs, i.e., those that have not been referenced in the docs via our
            # directives anywhere.
            if not hasattr(self.env, "jupyterlite_notebooks"):
                self.env.jupyterlite_notebooks = set()

            # As with other directives like literalinclude, an absolute path is
            # assumed to be relative to the document root, and a relative path
            # is assumed to be relative to the source file
            rel_filename, notebook = self.env.relfn2path(self.arguments[0])
            self.env.note_dependency(rel_filename)

            notebook_path = Path(notebook)

            self.env.jupyterlite_notebooks.add(str(notebook_path))

            notebooks_dir = Path(self.env.app.srcdir) / CONTENT_DIR
            os.makedirs(notebooks_dir, exist_ok=True)

            self._assert_no_conflicting_nb_names(notebook_path, notebooks_dir)
            target_name = f"{notebook_path.stem}.ipynb"
            target_path = notebooks_dir / target_name

            notebook_is_stripped: bool = self.env.config.strip_tagged_cells

            if notebook_path.suffix.lower() == ".md":
                if self._target_is_stale(notebook_path, target_path):
                    nb = jupytext.read(str(notebook_path))
                    if notebook_is_stripped:
                        nb.cells = self._strip_notebook_cells(nb)
                    with open(target_path, "w", encoding="utf-8") as f:
                        nbformat.write(nb, f, version=4)

                notebook = str(target_path)
                notebook_name = target_name
            else:
                notebook_name = notebook_path.name
                target_path = notebooks_dir / notebook_name

                if notebook_is_stripped:
                    nb = nbformat.read(notebook, as_version=4)
                    nb.cells = self._strip_notebook_cells(nb)
                    nbformat.write(nb, target_path, version=4)
                # If notebook_is_stripped is False, then copy the notebook(s) to notebooks_dir.
                # If it is True, then they have already been copied to notebooks_dir by the
                # nbformat.write() function above.
                else:
                    try:
                        shutil.copy(notebook, target_path)
                    except shutil.SameFileError:
                        pass

        else:
            notebook_name = None

        if new_tab:
            directive_button_text = self.options.pop("new_tab_button_text", None)
            if directive_button_text is not None:
                button_text = directive_button_text
            else:
                # If none, we use the appropriate global config based on
                # the type of directive passed.
                if isinstance(self, JupyterLiteDirective):
                    button_text = self.env.config.jupyterlite_new_tab_button_text
                elif isinstance(self, NotebookLiteDirective):
                    button_text = self.env.config.notebooklite_new_tab_button_text
                elif isinstance(self, VoiciDirective):
                    button_text = self.env.config.voici_new_tab_button_text

            return [
                self.newtab_cls(
                    prefix=prefix,
                    notebook=notebook_name,
                    width=width,
                    height=height,
                    prompt=prompt,
                    prompt_color=prompt_color,
                    search_params=search_params,
                    lite_options=self.options,
                    button_text=button_text,
                )
            ]

        return [
            self.iframe_cls(
                prefix=prefix,
                notebook=notebook_name,
                width=width,
                height=height,
                prompt=prompt,
                prompt_color=prompt_color,
                search_params=search_params,
                lite_options=self.options,
            )
        ]


class BaseJupyterViewDirective(_LiteDirective):
    """Base class for jupyterlite-sphinx directives."""

    iframe_cls = None  # to be defined by subclasses
    newtab_cls = None  # to be defined by subclasses

    option_spec = {
        "width": directives.unchanged,
        "height": directives.unchanged,
        "theme": directives.unchanged,
        "prompt": directives.unchanged,
        "prompt_color": directives.unchanged,
        "search_params": directives.unchanged,
        "new_tab": directives.unchanged,
        # "new_tab_button_text" below is useful only if "new_tab" is True, otherwise
        # we have "prompt" and "prompt_color" as options already.
        "new_tab_button_text": directives.unchanged,
    }


class JupyterLiteDirective(BaseJupyterViewDirective):
    """The ``.. jupyterlite::`` directive.

    Renders a Notebook with JupyterLite in the docs.
    """

    iframe_cls = JupyterLiteIframe
    newtab_cls = JupyterLiteTab


class NotebookLiteDirective(BaseJupyterViewDirective):
    """The ``.. notebooklite::`` directive.

    Renders a Notebook with NotebookLite in the docs.
    """

    iframe_cls = NotebookLiteIframe
    newtab_cls = NotebookLiteTab


class VoiciDirective(BaseJupyterViewDirective):
    """The ``.. voici::`` directive.

    Renders a Notebook with Voici in the docs.
    """

    iframe_cls = VoiciIframe
    newtab_cls = VoiciTab

    def run(self):
        if voici is None:
            raise RuntimeError(
                "Voici must be installed if you want to make use of the voici directive: pip install voici"
            )

        return super().run()


class NotebookLiteParser(RSTParser):
    """Sphinx source parser for Jupyter notebooks.

    Shows the Notebook using notebooklite."""

    supported = ("jupyterlite_notebook",)

    def parse(self, inputstring, document):
        title = os.path.splitext(os.path.basename(document.current_source))[0]
        # Make the "absolute" filename relative to the source root
        filename = "/" + os.path.relpath(document.current_source, self.env.app.srcdir)
        super().parse(
            f"{title}\n{'=' * len(title)}\n.. notebooklite:: {filename}",
            document,
        )


class TryExamplesDirective(SphinxDirective):
    """Add button to try doctest examples in Jupyterlite notebook."""

    has_content = True
    required_arguments = 0
    option_spec = {
        "height": directives.unchanged,
        "theme": directives.unchanged,
        "button_text": directives.unchanged,
        "example_class": directives.unchanged,
        "warning_text": directives.unchanged,
    }

    def run(self):
        if "generated_notebooks" not in self.env.temp_data:
            self.env.temp_data["generated_notebooks"] = {}

        directive_key = f"{self.env.docname}-{self.lineno}"
        notebook_unique_name = self.env.temp_data["generated_notebooks"].get(
            directive_key
        )

        # Use global configuration values from conf.py in manually inserted directives
        # if they are provided and the user has not specified a config value in the
        # directive itself.

        default_button_text = self.env.config.try_examples_global_button_text
        if default_button_text is None:
            default_button_text = "Try it with JupyterLite!"
        button_text = self.options.pop("button_text", default_button_text)

        default_warning_text = self.env.config.try_examples_global_warning_text
        warning_text = self.options.pop("warning_text", default_warning_text)

        default_example_class = self.env.config.try_examples_global_theme
        if default_example_class is None:
            default_example_class = ""
        example_class = self.options.pop("example_class", default_example_class)

        # A global height cannot be set in conf.py
        height = self.options.pop("height", None)

        # We need to get the relative path back to the documentation root from
        # whichever file the docstring content is in.
        docname = self.env.docname
        depth = len(docname.split("/")) - 1
        relative_path_to_root = "/".join([".."] * depth)
        prefix = os.path.join(relative_path_to_root, JUPYTERLITE_DIR)

        lite_app = "tree/"
        notebooks_path = "../notebooks/"

        content_container_node = nodes.container(
            classes=["try_examples_outer_container", example_class]
        )
        examples_div_id = uuid4()
        content_container_node["ids"].append(examples_div_id)
        # Parse the original content to create nodes
        content_node = nodes.container()
        content_node["classes"].append("try_examples_content")
        self.state.nested_parse(self.content, self.content_offset, content_node)

        if notebook_unique_name is None:
            nb = examples_to_notebook(self.content, warning_text=warning_text)

            preamble = self.env.config.try_examples_preamble
            if preamble:
                # insert after the "experimental" warning
                nb.cells.insert(1, new_code_cell(preamble))

            self.content = None
            notebooks_dir = Path(self.env.app.srcdir) / CONTENT_DIR
            notebook_unique_name = f"{uuid4()}.ipynb".replace("-", "_")
            self.env.temp_data["generated_notebooks"][
                directive_key
            ] = notebook_unique_name
            # Copy the Notebook for NotebookLite to find
            os.makedirs(notebooks_dir, exist_ok=True)
            with open(
                notebooks_dir / Path(notebook_unique_name), "w", encoding="utf-8"
            ) as f:
                # nbf.write incorrectly formats multiline arrays in output.
                json.dump(nb, f, indent=4, ensure_ascii=False)

        self.options["path"] = notebook_unique_name
        app_path = f"{lite_app}{notebooks_path}"
        options = _build_options(self.options)

        iframe_parent_div_id = uuid4()
        iframe_div_id = uuid4()
        iframe_src = f'{prefix}/{app_path}{f"index.html?{options}" if options else ""}'

        # Parent container (initially hidden)
        iframe_parent_container_div_start = (
            f'<div id="{iframe_parent_div_id}" '
            f'class="try_examples_outer_iframe {example_class} hidden">'
        )

        iframe_parent_container_div_end = "</div>"
        iframe_container_div = (
            f'<div id="{iframe_div_id}" '
            f'class="jupyterlite_sphinx_iframe_container">'
            f"</div>"
        )

        # Button with the onclick event to swap embedded notebook back to examples.
        go_back_button_html = (
            '<button class="try_examples_button" '
            f"onclick=\"window.tryExamplesHideIframe('{examples_div_id}',"
            f"'{iframe_parent_div_id}')\">"
            "Go Back</button>"
        )

        full_screen_button_html = (
            '<button class="try_examples_button" '
            f"onclick=\"window.openInNewTab('{examples_div_id}',"
            f"'{iframe_parent_div_id}')\">"
            "Open In Tab</button>"
        )

        # Button with the onclick event to swap examples with embedded notebook.
        try_it_button_html = (
            '<div class="try_examples_button_container">'
            '<button class="try_examples_button" '
            f"onclick=\"window.tryExamplesShowIframe('{examples_div_id}',"
            f"'{iframe_div_id}','{iframe_parent_div_id}','{iframe_src}',"
            f"'{height}')\">"
            f"{button_text}</button>"
            "</div>"
        )
        try_it_button_node = nodes.raw("", try_it_button_html, format="html")

        # Combine everything
        notebook_container_html = (
            iframe_parent_container_div_start
            + '<div class="try_examples_button_container">'
            + go_back_button_html
            + full_screen_button_html
            + "</div>"
            + iframe_container_div
            + iframe_parent_container_div_end
        )
        content_container_node += try_it_button_node
        content_container_node += content_node

        notebook_container = nodes.raw("", notebook_container_html, format="html")

        # Search config file allowing for config changes without rebuilding docs.
        config_path = os.path.join(relative_path_to_root, "try_examples.json")
        script_html = (
            "<script>"
            'document.addEventListener("DOMContentLoaded", function() {'
            f'window.loadTryExamplesConfig("{config_path}");'
            "});"
            "</script>"
        )
        script_node = nodes.raw("", script_html, format="html")

        return [content_container_node, notebook_container, script_node]


def _process_docstring_examples(app: Sphinx, docname: str, source: List[str]) -> None:
    source_path: os.PathLike = Path(app.env.doc2path(docname))
    if source_path.suffix == ".py":
        source[0] = insert_try_examples_directive(source[0])


def _process_autodoc_docstrings(app, what, name, obj, options, lines):
    try_examples_options = {
        "theme": app.config.try_examples_global_theme,
        "button_text": app.config.try_examples_global_button_text,
        "warning_text": app.config.try_examples_global_warning_text,
    }
    try_examples_options = {
        key: value for key, value in try_examples_options.items() if value is not None
    }
    modified_lines = insert_try_examples_directive(lines, **try_examples_options)
    lines.clear()
    lines.extend(modified_lines)


def conditional_process_examples(app, config):
    if config.global_enable_try_examples:
        app.connect("source-read", _process_docstring_examples)
        app.connect("autodoc-process-docstring", _process_autodoc_docstrings)


def inited(app: Sphinx, config):
    # Create the content dir
    os.makedirs(os.path.join(app.srcdir, CONTENT_DIR), exist_ok=True)

    if (
        config.jupyterlite_bind_ipynb_suffix
        and ".ipynb" not in config.source_suffix
        and ".ipynb" not in app.registry.source_suffix
    ):
        app.add_source_suffix(".ipynb", "jupyterlite_notebook")


def jupyterlite_ignore_contents_args(ignore_contents):
    """Generate `--ignore-contents` argument for each pattern.

    NOTE: Unlike generating `--contents` args, we _do not_ expand globs to generate the
    arguments. We just hand the config off to the JupyterLite build.
    """
    if ignore_contents is None:
        ignore_contents = []
    elif isinstance(ignore_contents, str):
        ignore_contents = [ignore_contents]

    return [
        arg for pattern in ignore_contents for arg in ["--ignore-contents", pattern]
    ]


def jupyterlite_build(app: Sphinx, error):
    if error is not None:
        # Do not build JupyterLite
        return

    if app.builder.format == "html":
        print("[jupyterlite-sphinx] Running JupyterLite build")
        jupyterlite_config = app.env.config.jupyterlite_config
        jupyterlite_overrides = app.env.config.jupyterlite_overrides
        jupyterlite_contents = app.env.config.jupyterlite_contents

        jupyterlite_dir = str(app.env.config.jupyterlite_dir)

        jupyterlite_build_command_options: Dict[str, Any] = (
            app.env.config.jupyterlite_build_command_options
        )

        config = []
        overrides = []
        if jupyterlite_config:
            config = ["--config", jupyterlite_config]

        if jupyterlite_overrides:
            # JupyterLite's build command does not validate the existence
            # of the JSON file, so we do it ourselves.
            # We will raise a FileNotFoundError if the file does not exist
            # in the Sphinx project directory.
            overrides_path = Path(app.srcdir) / jupyterlite_overrides
            if not Path(overrides_path).exists():
                raise FileNotFoundError(
                    f"Overrides file {overrides_path} does not exist. "
                    "Please check your configuration."
                )

            overrides = ["--settings-overrides", jupyterlite_overrides]

        if jupyterlite_contents is None:
            jupyterlite_contents = []
        elif isinstance(jupyterlite_contents, str):
            jupyterlite_contents = [jupyterlite_contents]

        # Expand globs in the contents strings
        contents = []
        for pattern in jupyterlite_contents:
            pattern_path = Path(pattern)

            base_path = (
                pattern_path.parent
                if pattern_path.is_absolute()
                else Path(app.srcdir) / pattern_path.parent
            )
            glob_pattern = pattern_path.name

            matched_paths = base_path.glob(glob_pattern)

            for matched_path in matched_paths:
                # If the matched path is absolute, we keep it as is, and
                # if it is relative, we convert it to a path relative to
                # the documentation source directory.
                contents_path = (
                    str(matched_path)
                    if matched_path.is_absolute()
                    else str(matched_path.relative_to(app.srcdir))
                )

                contents.extend(["--contents", contents_path])

        ignore_contents = jupyterlite_ignore_contents_args(
            app.env.config.jupyterlite_ignore_contents,
        )

        apps_option = []
        for liteapp in ["notebooks", "edit", "lab", "repl", "tree", "consoles"]:
            apps_option.extend(["--apps", liteapp])
        if voici is not None:
            apps_option.extend(["--apps", "voici"])

        command = [
            sys.executable,
            "-m",
            "jupyter",
            "lite",
            "build",
            "--debug",
            *config,
            *overrides,
            *contents,
            "--contents",
            os.path.join(app.srcdir, CONTENT_DIR),
            *ignore_contents,
            "--output-dir",
            os.path.join(app.outdir, JUPYTERLITE_DIR),
            *apps_option,
            "--lite-dir",
            jupyterlite_dir,
        ]

        if jupyterlite_build_command_options is not None:
            for key, value in jupyterlite_build_command_options.items():
                # Check for conflicting options from the default command we use
                # while building. We don't want to allow these to be overridden
                # unless they are explicitly set through Sphinx config.
                if key in ["contents", "output-dir", "lite-dir"]:
                    jupyterlite_command_error_message = f"""
                    Additional option, {key}, passed to `jupyter lite build` through
                    `jupyterlite_build_command_options` in conf.py is already an existing
                    option. "contents", "output_dir", and "lite_dir" can be configured in
                    conf.py as described in the jupyterlite-sphinx documentation:
                    https://jupyterlite-sphinx.readthedocs.io/en/stable/configuration.html
                    """
                    raise RuntimeError(jupyterlite_command_error_message)
                command.extend([f"--{key}", str(value)])

        assert all(
            [isinstance(s, str) for s in command]
        ), f"Expected all commands arguments to be a str, got {command}"

        kwargs: Dict[str, Any] = {}
        if app.env.config.jupyterlite_silence:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.PIPE

        print(f"[jupyterlite-sphinx] Command: {command}")
        completed_process: CompletedProcess[bytes] = subprocess.run(
            command, cwd=app.srcdir, check=True, **kwargs
        )

        if completed_process.returncode != 0:
            if app.env.config.jupyterlite_silence:
                print(
                    "`jupyterlite build` failed but its output has been silenced."
                    " stdout and stderr are reproduced below.\n"
                )
                print("stdout:", completed_process.stdout.decode())
                print("stderr:", completed_process.stderr.decode())

            # Raise the original exception that would have occurred with check=True
            raise subprocess.CalledProcessError(
                returncode=completed_process.returncode,
                cmd=command,
                output=completed_process.stdout,
                stderr=completed_process.stderr,
            )

        print("[jupyterlite-sphinx] JupyterLite build done")

    # Cleanup
    try:
        shutil.rmtree(os.path.join(app.srcdir, CONTENT_DIR))
        os.remove(".jupyterlite.doit.db")
    except FileNotFoundError:
        pass


def setup(app):
    # Initialize NotebookLite parser
    app.add_source_parser(NotebookLiteParser)

    app.connect("config-inited", inited)
    # We need to build JupyterLite at the end, when all the content was created
    app.connect("build-finished", jupyterlite_build)

    # Config options
    app.add_config_value("jupyterlite_config", None, rebuild="html")
    app.add_config_value("jupyterlite_overrides", None, rebuild="html")
    app.add_config_value("jupyterlite_dir", str(app.srcdir), rebuild="html")
    app.add_config_value("jupyterlite_contents", None, rebuild="html")
    app.add_config_value("jupyterlite_ignore_contents", None, rebuild="html")
    app.add_config_value("jupyterlite_bind_ipynb_suffix", True, rebuild="html")
    app.add_config_value("jupyterlite_silence", True, rebuild=True)
    app.add_config_value("strip_tagged_cells", False, rebuild=True)

    # Pass a dictionary of additional options to the JupyterLite build command
    app.add_config_value("jupyterlite_build_command_options", None, rebuild="html")

    app.add_config_value("global_enable_try_examples", default=False, rebuild=True)
    app.add_config_value("try_examples_global_theme", default=None, rebuild=True)
    app.add_config_value("try_examples_global_warning_text", default=None, rebuild=True)
    app.add_config_value(
        "try_examples_global_button_text",
        default=None,
        rebuild="html",
    )
    app.add_config_value("try_examples_preamble", default=None, rebuild="html")

    # Allow customising the button text for each directive (this is useful
    # only when "new_tab" is set to True)
    app.add_config_value(
        "jupyterlite_new_tab_button_text", "Open as a notebook", rebuild="html"
    )
    app.add_config_value(
        "notebooklite_new_tab_button_text", "Open as a notebook", rebuild="html"
    )
    app.add_config_value("voici_new_tab_button_text", "Open with Voici", rebuild="html")
    app.add_config_value(
        "replite_new_tab_button_text", "Open in a REPL", rebuild="html"
    )

    # Initialize NotebookLite and JupyterLite directives
    app.add_node(
        NotebookLiteIframe,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    app.add_directive("notebooklite", NotebookLiteDirective)
    # For backward compatibility
    app.add_directive("retrolite", NotebookLiteDirective)
    app.add_node(
        JupyterLiteIframe,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    for node_class in [NotebookLiteTab, JupyterLiteTab]:
        app.add_node(
            node_class,
            html=(visit_element_html, None),
            latex=(skip, None),
            textinfo=(skip, None),
            text=(skip, None),
            man=(skip, None),
        )
    app.add_directive("jupyterlite", JupyterLiteDirective)

    # Initialize Replite directive and tab
    app.add_node(
        RepliteIframe,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    app.add_node(
        RepliteTab,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    app.add_directive("replite", RepliteDirective)
    app.add_config_value("replite_auto_execute", True, rebuild="html")

    # Initialize Voici directive and tabbed interface
    app.add_node(
        VoiciIframe,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    app.add_node(
        VoiciTab,
        html=(visit_element_html, None),
        latex=(skip, None),
        textinfo=(skip, None),
        text=(skip, None),
        man=(skip, None),
    )
    app.add_directive("voici", VoiciDirective)

    # Initialize TryExamples directive
    app.add_directive("try_examples", TryExamplesDirective)
    app.connect("config-inited", conditional_process_examples)

    # CSS and JS assets
    copy_asset(str(HERE / "jupyterlite_sphinx.css"), str(Path(app.outdir) / "_static"))
    copy_asset(str(HERE / "jupyterlite_sphinx.js"), str(Path(app.outdir) / "_static"))

    app.add_css_file("https://fonts.googleapis.com/css?family=Vibur")
    app.add_css_file("jupyterlite_sphinx.css")

    app.add_js_file("jupyterlite_sphinx.js")

    # Copy optional try examples runtime config if it exists.
    try_examples_config_path = Path(app.srcdir) / "try_examples.json"
    if try_examples_config_path.exists():
        copy_asset(str(try_examples_config_path), app.outdir)

    return {"parallel_read_safe": True}


def search_params_parser(search_params: str) -> str:
    pattern = re.compile(r"^\[(?:\s*[\"']{1}([^=\s\,&=\?\/]+)[\"']{1}\s*\,?)+\]$")
    if not search_params:
        return "false"
    if search_params in ["True", "False"]:
        return search_params.lower()
    elif pattern.match(search_params):
        return search_params.replace('"', "'")
    else:
        raise ValueError(
            'The search_params directive must be either True, False or ["param1", "param2"].\n'
            'The params name shouldn\'t contain any of the following characters ["\\", "\'", """, ",", "?", "=", "&", " ").'
        )
