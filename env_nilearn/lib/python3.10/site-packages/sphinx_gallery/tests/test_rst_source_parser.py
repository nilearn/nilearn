import sphinx_gallery.rst_source_parser as rsp

sample_contents = """\
The rst parser
==============

The .rst parser converts the whole file into
a single text block.
"""


def test_split_code_and_text_blocks(tmp_path):
    source_file = tmp_path / "source.rst"
    source_file.write_text(sample_contents)

    file_conf, blocks, node = rsp.split_code_and_text_blocks(source_file)

    assert file_conf == {}
    assert blocks == [rsp.Block("text", sample_contents, 1)]
    assert node is None
