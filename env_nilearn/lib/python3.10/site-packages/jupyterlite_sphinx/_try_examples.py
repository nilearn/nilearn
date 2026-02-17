import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell
import re


def examples_to_notebook(input_lines, *, warning_text=None):
    """Parse examples section of a docstring and convert to Jupyter notebook.

    Parameters
    ----------
    input_lines : iterable of str.

    warning_text : str[Optional]
        If given, add a markdown cell at the top of the generated notebook
        containing the given text. The cell will be styled to indicate that
        this is a warning.

    Returns
    -------
    dict
        json for a Jupyter Notebook

    Examples
    --------
    >>> from jupyterlite_sphinx._try_examples import examples_to_notebook

    >>> input_lines = [
    >>>            "Add two numbers. This block of text will appear as a\n",
    >>>            "markdown cell. The following block will become a code\n",
    >>>            "cell with the value 4 contained in the output.",
    >>>            "\n",
    >>>            ">>> x = 2\n",
    >>>            ">>> y = 2\n",
    >>>            ">>> x + y\n",
    >>>            "4\n",
    >>>            "\n",
    >>>            "Inline LaTeX like :math:`x + y = 4` will be converted\n",
    >>>            "correctly within markdown cells. As will block LaTeX\n",
    >>>            "such as\n",
    >>>            "\n",
    >>>            ".. math::\n",
    >>>            "\n",
    >>>            "    x = 2,\\;y = 2
    >>>            "\n",
    >>>            "    x + y = 4\n",
    >>>          ]
    >>> notebook = examples_to_notebook(input_lines)
    """
    nb = nbf.v4.new_notebook()

    if warning_text is not None:
        # Two newlines \n\n signal that the inner content should be parsed as
        # markdown.
        warning = f"<div class='alert alert-warning'>\n\n{warning_text}\n\n</div>"
        nb.cells.append(new_markdown_cell(warning))

    code_lines = []
    md_lines = []
    output_lines = []
    inside_multiline_code_block = False

    ignore_directives = [".. plot::", ".. only::"]
    inside_ignore_directive = False

    for line in input_lines:
        line = line.rstrip("\n")

        # Content underneath some directives should be ignored when generating notebook.
        if any(line.startswith(directive) for directive in ignore_directives):
            inside_ignore_directive = True
            continue
        if inside_ignore_directive:
            if line == "" or line[0].isspace():
                continue
            else:
                inside_ignore_directive = False

        if line.startswith(">>>"):  # This is a code line.
            if output_lines:
                # If there are pending output lines, we must be starting a new
                # code block.
                _append_code_cell_and_clear_lines(code_lines, output_lines, nb)
            if inside_multiline_code_block:
                # A multiline codeblock is ending.
                inside_multiline_code_block = False
            # If there is any pending markdown text, add it to the notebook
            if md_lines:
                _append_markdown_cell_and_clear_lines(md_lines, nb)

            # Add line of code, removing '>>> ' prefix
            code_lines.append(line[4:])
        elif line.startswith("...") and code_lines:
            # This is a line of code in a multiline code block.
            inside_multiline_code_block = True
            code_lines.append(line[4:])
        elif line.rstrip("\n") == "" and code_lines:
            # A blank line means a code block has ended.
            _append_code_cell_and_clear_lines(code_lines, output_lines, nb)
        elif code_lines:
            # Non-blank non ">>>" prefixed line must be output of previous code block.
            output_lines.append(line)
        else:
            # Anything else should be treated as markdown.
            md_lines.append(line)

    # After processing all lines, add pending markdown or code to the notebook if
    # any exists.
    if md_lines:
        _append_markdown_cell_and_clear_lines(md_lines, nb)
    if code_lines:
        _append_code_cell_and_clear_lines(code_lines, output_lines, nb)

    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python",
            "language": "python",
            "name": "python",
        },
        "language_info": {
            "name": "python",
        },
    }
    return nb


def _append_code_cell_and_clear_lines(code_lines, output_lines, notebook):
    """Append new code cell to notebook, clearing lines."""
    code_text = "\n".join(code_lines)
    cell = new_code_cell(code_text)
    if output_lines:
        combined_output = "\n".join(output_lines)
        cell.outputs.append(
            nbf.v4.new_output(
                output_type="execute_result",
                data={"text/plain": combined_output},
            ),
        )
    notebook.cells.append(cell)
    output_lines.clear()
    code_lines.clear()


def _append_markdown_cell_and_clear_lines(markdown_lines, notebook):
    """Append new markdown cell to notebook, clearing lines."""
    markdown_text = "\n".join(markdown_lines)
    markdown_text = _process_latex(markdown_text)
    markdown_text = _process_literal_blocks(markdown_text)
    markdown_text = _strip_ref_identifiers(markdown_text)
    markdown_text = _convert_links(markdown_text)
    notebook.cells.append(new_markdown_cell(markdown_text))
    markdown_lines.clear()


_ref_identifier_pattern = re.compile(r"\[R[a-f0-9]+-(?P<ref_num>\d+)\]_")
_link_pattern = re.compile(r"`(?P<link_text>[^`<]+)<(?P<url>[^`>]+)>`_")


def _convert_sphinx_link(match):
    link_text = match.group("link_text").rstrip()
    url = match.group("url")
    return f"[{link_text}]({url})"


def _convert_links(md_text):
    """Convert sphinx style links to markdown style links

    Sphinx style links have the form `link text <url>`_. Converts to
    markdown format [link text](url).
    """
    return _link_pattern.sub(_convert_sphinx_link, md_text)


def _strip_ref_identifiers(md_text):
    """Remove identifiers from references in notebook.

    Each docstring gets a unique identifier in order to have unique internal
    links for each docstring on a page.

    They look like [R4c2dbc17006a-1]_. We strip these out so they don't appear
    in the notebooks. The above would be replaced with [1]_.
    """
    return _ref_identifier_pattern.sub(r"[\g<ref_num>]", md_text)


def _process_latex(md_text):
    # Map rst latex directive to $ so latex renders in notebook.
    md_text = re.sub(
        r":math:\s*`(?P<latex>.*?)`", r"$\g<latex>$", md_text, flags=re.DOTALL
    )

    lines = md_text.split("\n")
    in_math_block = False
    wrapped_lines = []
    equation_lines = []

    for line in lines:
        if line.strip() == ".. math::":
            in_math_block = True
            continue  # Skip the '.. math::' line

        if in_math_block:
            if line.strip() == "":
                if equation_lines:
                    # Join and wrap the equations, then reset
                    wrapped_lines.append(f"$$ {' '.join(equation_lines)} $$")
                    equation_lines = []
            elif line.startswith(" ") or line.startswith("\t"):
                equation_lines.append(line.strip())
        else:
            wrapped_lines.append(line)

        # If you leave the indented block, the math block ends
        if in_math_block and not (
            line.startswith(" ") or line.startswith("\t") or line.strip() == ""
        ):
            in_math_block = False
            if equation_lines:
                wrapped_lines.append(f"$$ {' '.join(equation_lines)} $$")
            equation_lines = []
            wrapped_lines.append(line)

    # Handle the case where the text ends with a math block
    if in_math_block and equation_lines:
        wrapped_lines.append(f"$$ {' '.join(equation_lines)} $$")

    return "\n".join(wrapped_lines)


def _process_literal_blocks(md_text):
    md_lines = md_text.split("\n")
    new_lines = []
    in_literal_block = False
    literal_block_accumulator = []

    for line in md_lines:
        indent_level = len(line) - len(line.lstrip())

        if in_literal_block and (indent_level > 0 or line.strip() == ""):
            literal_block_accumulator.append(line.lstrip())
        elif in_literal_block:
            new_lines.extend(["```"] + literal_block_accumulator + ["```"])
            literal_block_accumulator = []
            if line.endswith("::"):
                # If the line endswith ::, a new literal block is starting.
                line = line[:-2]  # Strip off the :: from the end
                if not line:
                    # If the line contains only ::, we ignore it.
                    continue
            else:
                # Only set in_literal_block to False if not starting new
                # literal block.
                in_literal_block = False
            # We've appended the entire literal block which just ended, but
            # still need to append the current line.
            new_lines.append(line)
        else:
            if line.endswith("::"):
                # A literal block is starting.
                in_literal_block = True
                line = line[:-2]
                if not line:
                    # As above, if the line contains only ::, ignore it.
                    continue
            new_lines.append(line)

    if literal_block_accumulator:
        # Handle case where a literal block ends the markdown cell.
        new_lines.extend(["```"] + literal_block_accumulator + ["```"])

    return "\n".join(new_lines)


# try_examples identifies section headers after processing by numpydoc or
# sphinx.ext.napoleon.
# See https://numpydoc.readthedocs.io/en/stable/format.html for info on numpydoc
# sections and
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#docstring-sections
# for info on sphinx.ext.napoleon sections.

# The patterns below were identified by creating a docstring using all section
# headers and processing it with both numpydoc and sphinx.ext.napoleon.

# Examples section is a rubric for numpydoc and can be configured to be
# either a rubric or admonition in sphinx.ext.napoleon.
_examples_start_pattern = re.compile(r".. (rubric|admonition):: Examples")
_next_section_pattern = re.compile(
    "|".join(
        # Newer versions of numpydoc enforce section order and only Attributes
        # and Methods sections can appear after an Examples section. All potential
        # numpydoc section headers are included here to support older or custom
        # numpydoc versions. e.g. at the time of this comment, SymPy is using a
        # custom version of numpydoc which allows for arbitrary section order.
        # sphinx.ext.napoleon allows for arbitrary section order and all potential
        # headers are included.
        [
            # Notes and References appear as rubrics in numpydoc and
            # can be configured to appear as either a rubric or
            # admonition in sphinx.ext.napoleon.
            r".\.\ rubric:: Notes",
            r".\.\ rubric:: References",
            r".\.\ admonition:: Notes",
            r".\.\ admonition:: References",
            # numpydoc only headers
            r":Attributes:",
            r".\.\ rubric:: Methods",
            r":Other Parameters:",
            r":Parameters:",
            r":Raises:",
            r":Returns:",
            # If examples section is last, processed by numpydoc may appear at end.
            r"\!\! processed by numpydoc \!\!",
            # sphinx.ext.napoleon only headers
            r"\.\. attribute::",
            r"\.\. method::",
            r":param .+:",
            r":raises .+:",
            r":returns:",
            # Headers generated by both extensions
            r".\.\ seealso::",
            r":Yields:",
            r":Warns:",
            # directives which can start a section with sphinx.ext.napoleon
            # with no equivalent when using numpydoc.
            r".\.\ attention::",
            r".\.\ caution::",
            r".\.\ danger::",
            r".\.\ error::",
            r".\.\ hint::",
            r".\.\ important::",
            r".\.\ tip::",
            r".\.\ todo::",
            r".\.\ warning::",
        ]
    )
)


def insert_try_examples_directive(lines, **options):
    """Adds try_examples directive to Examples section of a docstring.

    Hack to allow for a config option to enable try_examples functionality
    in all Examples sections (unless a comment ".. disable_try_examples" is
    added explicitly after the section header.)


    Parameters
    ----------
    docstring : list of str
        Lines of a docstring at time of "autodoc-process-docstring", which has
        been previously processed by numpydoc or sphinx.ext.napoleon.

    Returns
    -------
    list of str
        Updated version of the input docstring which has a try_examples directive
        inserted in the Examples section (if one exists) with all Examples content
        indented beneath it. Does nothing if the comment ".. disable_try_examples"
        is included at the top of the Examples section. Also a no-op if the
        try_examples directive is already included.
    """
    # Search for start of an Examples section
    for left_index, line in enumerate(lines):
        if _examples_start_pattern.search(line):
            break
    else:
        # No Examples section found
        return lines[:]

    # Jump to next line
    left_index += 1
    # Skip empty lines to get to the first content line
    while left_index < len(lines) and not lines[left_index].strip():
        left_index += 1
    if left_index == len(lines):
        # Examples section had no content, no need to insert directive.
        return lines[:]

    # Check for the ".. disable_try_examples" comment.
    if lines[left_index].strip() == ".. disable_try_examples":
        # If so, do not insert directive.
        return lines[:]

    # Check if the ".. try_examples::" directive already exists
    if ".. try_examples::" == lines[left_index].strip():
        # If so, don't need to insert again.
        return lines[:]

    # Find the end of the Examples section
    right_index = left_index
    while right_index < len(lines) and not _next_section_pattern.search(
        lines[right_index]
    ):
        right_index += 1

    # Check if we've reached the end of the docstring
    if right_index < len(lines) and "!! processed by numpydoc !!" in lines[right_index]:
        # Sometimes the .. appears on an earlier line than !! processed by numpydoc !!
        if not re.search(
            r"\.\.\s+\!\! processed by numpy doc \!\!", lines[right_index]
        ):
            while right_index > 0 and lines[right_index].strip() != "..":
                right_index -= 1

    # Add the ".. try_examples::" directive and indent the content of the Examples section
    new_lines = (
        lines[:left_index]
        + [".. try_examples::"]
        + [f"    :{key}: {value}" for key, value in options.items()]
        + [""]
        + ["    " + line for line in lines[left_index:right_index]]
    )

    # Append the remainder of the docstring, if there is any
    if right_index < len(lines):
        new_lines += [""] + lines[right_index:]

    return new_lines
