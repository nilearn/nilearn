"""Update AUTHORS and names from CITATION.cff file."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 4096

CORE_DEVS = [
    "Alexis Thual",
    "Bertrand Thirion",
    "Elizabeth DuPre",
    "Hao-Ting Wang",
    "Himanshu Aggarwal",
    "Jerome Dockes",
    "Nicolas Gensollen",
    "Rémi Gau",
    "Taylor Salo",
]


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


def names_rst() -> Path:
    """Return path to names.rst file."""
    return root_dir() / "doc" / "changes" / "names.rst"


def citation_file() -> Path:
    """Return path to CITATIONS.cff file."""
    return root_dir() / "CITATION.cff"


def authors_file() -> Path:
    """Return path to AUTHORS.rst file."""
    return root_dir() / "AUTHORS.rst"


def read_citation_cff() -> dict[str, Any]:
    """Read CITATION.cff file."""
    print(f"Reading file: {citation_file()}")
    with citation_file().open(encoding="utf8") as f:
        citation = yaml.load(f)
    return citation


def write_citation_cff(citation: dict[str, Any]) -> None:
    """Write CITATION.cff file."""
    print(f"Writing file: {citation_file()}")
    with citation_file().open(mode="w", encoding="utf8") as f:
        yaml.dump(citation, f)


def write_names_rst(citation: list[dict[str, str]]) -> None:
    """Write names.rst file."""
    print(f"Writing file: {names_rst()}")
    with names_rst().open(mode="w", encoding="utf8") as f:
        header = """.. This file is automatically generated.
    Do not edit manually.
    If you want to add to add yourself to the list of authors,
    please edit CITATION.cff and run maint_tools/citation_cff_maint.py.

"""
        print(header, file=f)

        for i, author in enumerate(citation["authors"]):
            if "website" in author:
                line = (
                    f'.. _{author["given-names"]} {author["family-names"]}: '
                    f'{author["website"]}'
                )
                print(line, file=f)
            if i < len(citation["authors"]) - 1:
                print("", file=f)


def read_authors_file() -> list[str]:
    """Read AUTHORS.rst file."""
    print(f"Reading file: {authors_file()}")
    with authors_file().open(encoding="utf8") as f:
        authors_file_content = f.readlines()
    return authors_file_content


def write_authors_file(authors: list[dict[str, str]]) -> None:
    """Write AUTHORS.rst file."""
    authors_file_content = read_authors_file()
    print(f"Writing file: {authors_file()}")
    with authors_file().open(mode="w", encoding="utf8") as f:
        writing_team_section = False
        for line in authors_file_content:
            if ".. CORE DEV SECTION STARTS HERE" in line:
                writing_team_section = True
                write_team_section(f, authors)
            if "Funding" in line:
                writing_team_section = False
            if not writing_team_section:
                f.write(line)


def write_team_section(f, authors: list[dict[str, str]]) -> None:
    """Write team section."""
    print(" Updating team section")
    f.write(
        """.. CORE DEV SECTION STARTS HERE
   The Core developers section is added automatically
   and should not be edited manually.

.. _core_devs:

Core developers
...............

The nilearn core developers are:

"""
    )

    write_core_devs(f)

    f.write(
        """.. CORE DEV SECTION ENDS HERE
"""
    )

    f.write(
        """
.. OTHER CONTRIBUTION SECTION STARTS HERE
   The Other contributors section is added automatically
   and should not be edited manually.

Other contributors
..................

Some other past or present contributors are:

"""
    )
    for author_ in authors:
        if "website" in author_:
            f.write(f"* `{author_['given-names']} {author_['family-names']}`_")
        else:
            f.write(f"* {author_['given-names']} {author_['family-names']}")
        if author_.get("affiliation"):
            f.write(f": {author_['affiliation']}")
        f.write("\n")

    f.write(
        """
.. OTHER CONTRIBUTION SECTION ENDS HERE

"""
    )


def write_core_devs(f):
    """Add core devs."""
    for dev in CORE_DEVS:
        f.write(f"* `{dev}`_\n")
    f.write("\n")


def sort_authors(authors: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort authors by given name."""
    print(" Sorting authors by given name")
    authors.sort(key=operator.itemgetter("given-names"))
    return authors


def count_authors() -> int:
    """Count authors in names.rst."""
    n_authors = 0
    with names_rst().open(encoding="utf8") as f:
        # count authors
        lines = f.readlines()
        for line in lines:
            if line.startswith(".. _"):
                n_authors += 1
    return n_authors


def remove_consortium(authors: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove consortium from authors."""
    authors = [
        author
        for author in authors
        if author["family-names"] != "Nilearn contributors"
    ]
    return authors


def add_consortium(authors: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add consortium to authors."""
    return [{"family-names": "Nilearn contributors"}, *authors]


def main():
    """Update names.rst and AUTHORS.rst files."""
    citation = read_citation_cff()
    citation["authors"] = remove_consortium(citation["authors"])
    citation["authors"] = sort_authors(citation["authors"])

    n_authors = count_authors()
    write_names_rst(citation)
    new_n_authors = count_authors()
    # Sanity check to make sure we have not lost anyone
    assert n_authors <= new_n_authors

    write_authors_file(citation["authors"])

    citation["authors"] = add_consortium(citation["authors"])
    write_citation_cff(citation)


if __name__ == "__main__":
    main()
