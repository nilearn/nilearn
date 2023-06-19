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
    print(citation)
    return citation


def write_citation_cff(citation):
    """Write CITATION.cff file."""
    with open(root_dir / "CITATION.cff", "w", encoding="utf8") as f:
        yaml.dump(citation, f)


def sort_authors(authors):
    """Sort authors by family name."""
    authors.sort(key=lambda x: x["family-names"])
    return authors


def main():
    """Update CITATION.cff file."""
    citation = read_citation_cff()

    # get names from doc
    with open(root_dir / "doc" / "changes" / "names.rst") as f:
        # read lines
        lines = f.readlines()

    authors = []
    for line in lines:
        if line.startswith(".. _"):
            line = line.strip(".. _").replace("\n", "")
            website = line.split(": ")[1]
            author = line.split(": ")[0]
            if len(author.split(" ")) == 2:
                this_author = {
                    "given-names": author.split(" ")[0],
                    "family-names": author.split(" ")[1],
                    "website": website,
                }
            elif len(line.split(": ")) > 2:
                this_author = {
                    "given-names": " ".join(author.split(" ")[:1]),
                    "family-names": author.split(" ")[2],
                    "website": website,
                }
            authors.append(this_author)

    for citation_author in citation["authors"]:
        for this_author in authors:
            if (
                citation_author["given-names"] == this_author["given-names"]
                and citation_author["family-names"]
                == this_author["family-names"]
            ):
                citation_author["website"] = this_author["website"]

    citation["authors"] = sort_authors(citation["authors"])
    write_citation_cff(citation)


if __name__ == "__main__":
    main()
