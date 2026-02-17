"""Parser for reStructuredText files.

The parser simply converts the whole reST file into a single text block.
"""

from pathlib import Path

from .py_source_parser import Block


def split_code_and_text_blocks(source_file, return_node=False):
    """Return list with source file separated into code and text blocks.

    Parameters
    ----------
    source_file : str
        Path to the source file.
    return_node : bool
        If True, return the ast node.

    Returns
    -------
    file_conf : dict
        This is always empty as file config is not defined / supported in ReST files.
    blocks : list
        (type, content, line_number)
        List where each element is a tuple with the label ('text' or 'code'),
        the corresponding content string of block and the leading line number.
    node : None
        Always None.
    """
    file_conf = {}  # not defined in .rst example files
    content = Path(source_file).read_text()
    blocks = [Block("text", content, 1)]
    node = None

    return file_conf, blocks, node


def remove_ignore_blocks(code_block):
    """
    Return the content of *code_block* with ignored areas removed.

    As we don't have ignored areas in ReST, the codeblock is returned
    unchanged.

    Parameters
    ----------
    code_block : str
        A code segment.
    """
    return code_block
