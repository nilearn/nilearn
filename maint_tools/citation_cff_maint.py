"""Update AUTHORS and names from CITATION.cff file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

CORE_DEVS = [
    "Alexis Thual",
    "Bertrand Thirion",
    "Elizabeth DuPre",
    "Hao-Ting Wang",
    "Jerome Dockes",
    "Nicolas Gensollen",
    "Rémi Gau",
    "Taylor Salo",
    "Yasmin Mzayek",
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
    with open(citation_file(), encoding="utf8") as f:
        citation = yaml.load(f)
    return citation


def write_citation_cff(citation: dict[str, Any]) -> None:
    """Write CITATION.cff file."""
    with open(citation_file(), "w", encoding="utf8") as f:
        yaml.dump(citation, f)


def write_names_rst(citation: list[dict[str, str]]) -> None:
    """Write names.rst file."""
    with open(names_rst(), "w", encoding="utf8") as f:
        for author in citation["authors"]:
            line = (
                f'.. _{author["given-names"]} {author["family-names"]}: '
                f'{author["website"]}'
            )
            print(line, file=f)
            print("", file=f)


def read_authors_file() -> list[str]:
    """Read AUTHORS.rst file."""
    with open(authors_file(), encoding="utf8") as f:
        authors_file_content = f.readlines()
    return authors_file_content


def write_authors_file(authors: list[dict[str, str]]) -> None:
    """Write AUTHORS.rst file."""
    authors_file_content = read_authors_file()
    with open(authors_file(), "w", encoding="utf8") as f:
        writing_team_section = False
        for line in authors_file_content:
            if ".. _core_devs:" in line:
                writing_team_section = True
                write_team_section(f, authors)
            if "Funding" in line:
                writing_team_section = False
            if not writing_team_section:
                f.write(line)


def write_team_section(f, authors: list[dict[str, str]]) -> None:
    """Write team section."""
    f.write(
        """.. The Core developers section is added automatically
   and should not be edited manually.

.. _core_devs:

Core developers
...............

The nilearn core developers are:

"""
    )

    write_core_devs(f)

    f.write(
        """
.. The Other contributors section is added automatically
   and should not be edited manually.

Other contributors
..................

Some other past or present contributors are:

"""
    )
    for author_ in authors:
        f.write(f"* `{author_['given-names']} {author_['family-names']}`_\n")

    f.write("\n")


def write_core_devs(f):
    """Add core devs."""
    for dev in CORE_DEVS:
        f.write(f"* `{dev}`_\n")
    f.write("\n")


def sort_authors(authors: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort authors by given name."""
    authors.sort(key=lambda x: x["given-names"])
    return authors


def count_authors() -> int:
    """Count authors in names.rst."""
    nb_authors = 0
    with open(names_rst(), encoding="utf8") as f:
        # count authors
        lines = f.readlines()
        for line in lines:
            if line.startswith(".. _"):
                nb_authors += 1
    return nb_authors


def main():
    """Update names.rst and AUTHORS.rst files."""
    citation = read_citation_cff()
    citation["authors"] = sort_authors(citation["authors"])

    nb_authors = count_authors()
    write_names_rst(citation)
    new_nb_authors = count_authors()
    # Sanity check to make sure we have not lost anyone
    assert nb_authors <= new_nb_authors

    write_citation_cff(citation)

    write_authors_file(citation["authors"])


if __name__ == "__main__":
    main()
