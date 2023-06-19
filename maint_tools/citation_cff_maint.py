"""Update CITATION.cff file."""
from pathlib import Path

import ruamel.yaml
from rich import print

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

root_dir = Path(__file__).parent.parent


def read_citation_cff():
    """Read CITATION.cff file."""
    with open(root_dir / "CITATION.cff", encoding="utf8") as f:
        citation = yaml.load(f)
    return citation


def write_citation_cff(citation):
    """Write CITATION.cff file."""
    with open(root_dir / "CITATION.cff", "w", encoding="utf8") as f:
        yaml.dump(citation, f)


def sort_authors(authors):
    """Sort authors by family name."""
    authors.sort(key=lambda x: x["given-names"])
    return authors


def count_authors():
    """Count authors in names.rst."""
    nb_authors = 0
    with open(
        root_dir / "doc" / "changes" / "names.rst", encoding="utf8"
    ) as f:
        # count authors
        lines = f.readlines()
        for line in lines:
            if line.startswith(".. _"):
                nb_authors += 1
    return nb_authors


def main():
    """Update CITATION.cff file."""
    citation = read_citation_cff()

    citation["authors"] = sort_authors(citation["authors"])

    nb_authors = count_authors()

    print(nb_authors)

    with open(
        root_dir / "doc" / "changes" / "names.rst", "w", encoding="utf8"
    ) as f:
        for author in citation["authors"]:
            line = (
                f'.. _{author["given-names"]} {author["family-names"]}: '
                f'{author["website"]}'
            )
            print(line, file=f)
            print("", file=f)

    new_nb_authors = count_authors()

    print(new_nb_authors)

    assert nb_authors <= new_nb_authors

    write_citation_cff(citation)


if __name__ == "__main__":
    main()
