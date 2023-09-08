"""Read authors.tsv and update citation.cff."""

from pathlib import Path
from typing import Any

import pandas as pd
import ruamel.yaml
from rich import print

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


# read citation.cff
def citation_file() -> Path:
    """Return path to CITATIONS.cff file."""
    return root_dir() / "CITATION.cff"


def read_citation_cff() -> dict[str, Any]:
    """Read CITATION.cff file."""
    print(f"Reading file: {citation_file()}")
    with open(citation_file(), encoding="utf8") as f:
        citation = yaml.load(f)
    return citation


def write_citation_cff(citation: dict[str, Any]) -> None:
    """Write CITATION.cff file."""
    print(f"Writing file: {citation_file()}")
    with open(citation_file(), "w", encoding="utf8") as f:
        yaml.dump(citation, f)


def main():
    """Update CITATION.cff file."""
    authors = pd.read_csv(Path(__file__).parent / "authors.tsv", sep="\t")

    citation = read_citation_cff()

    for row in authors.iterrows():
        # find the author in citation.cff
        for idx, citation_author in enumerate(citation["authors"]):
            if (
                citation_author["given-names"] == row[1]["given-names"]
                and citation_author["family-names"] == row[1]["family-names"]
            ):
                break

        print(citation_author)

        if citation_author.get("website") is None:
            raise ValueError()

        citation_author["website"] = row[1]["website"]

        for key in ["affiliation", "email", "orcid"]:
            if isinstance(row[1][key], str):
                citation_author[key] = row[1][key]

        citation["authors"][idx] = citation_author

    write_citation_cff(citation)

    print(authors)


if __name__ == "__main__":
    main()
