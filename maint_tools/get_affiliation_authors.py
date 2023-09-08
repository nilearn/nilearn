"""Update citation.cff wint info from orcid and github."""
from pathlib import Path
from typing import Any

import requests
import ruamel.yaml
from rich import print

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

TOKEN_FILE = "/home/remi/Documents/tokens/gh_user.txt"
with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()


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


def affiliation_from_orcid(orcid_record: dict[str, Any]) -> str | None:
    """Get affiliation the most recent employment (top of the list)."""
    if employer := (
        orcid_record.get("activities-summary", {})
        .get("employments", {})
        .get("employment-summary", [])
    ):
        return employer[0].get("organization", {}).get("name")
    else:
        return None


def first_name_from_orcid(orcid_record: dict[str, Any]) -> str:
    """Return first name from ORCID record."""
    return (
        orcid_record.get("person", {})
        .get("name", {})
        .get("given-names", {})
        .get("value")
    )


def last_name_from_orcid(orcid_record: dict[str, Any]) -> str:
    """Return last name from ORCID record."""
    return (
        orcid_record.get("person", {})
        .get("name", {})
        .get("family-name", {})
        .get("value")
    )


def get_author_info_from_orcid(orcid: str) -> dict[str, Any]:
    """Get author info from ORCID."""
    orcid = orcid.strip()

    url = f"https://pub.orcid.org/v3.0/{orcid}/record"

    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
        },
    )
    author_info = {}
    if response.status_code == 200:
        record = response.json()
        first_name = first_name_from_orcid(record)
        last_name = last_name_from_orcid(record)
        affiliation = affiliation_from_orcid(record)
        author_info = {
            "firstname": first_name,
            "lastname": last_name,
            "affiliation": affiliation,
            "id": f"ORCID:{orcid}",
        }

    if not author_info:
        print(f"Could not find author info for ORCID: {orcid}")

    return author_info


def get_author_info_from_github(username: str) -> dict[str, Any]:
    """Get author info from GitHub."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
        },
        auth=("RemiGau", TOKEN),
    )
    return response.json()


def main():
    """Update CITATION.cff file."""
    citation = read_citation_cff()

    for idx, citation_author in enumerate(citation["authors"]):
        if citation_author.get("affiliation") is None:
            if citation_author.get("orcid") is not None:
                orcid = citation_author["orcid"].split("orcid.org/")[1]
                author_info = get_author_info_from_orcid(orcid)

                if author_info.get("affiliation") is not None:
                    print(f"Affiliation: {author_info['affiliation']}")

            if "github.com/" in citation_author["website"]:
                username = citation_author["website"].split("github.com/")[1]
                author_info = get_author_info_from_github(username)

                affiliation = []
                if company := author_info.get("company", ""):
                    affiliation.append(company)
                if location := author_info.get("location", ""):
                    affiliation.append(location)
                if affiliation := f"{', '.join(affiliation)}".strip(", "):
                    citation_author["affiliation"] = affiliation

            print(citation_author)

            citation["authors"][idx] = citation_author

    authors_with_no_affiliation = sum(
        citation_author.get("affiliation") is None
        for citation_author in citation["authors"]
    )
    print(
        f"Authors with no affiliation: {authors_with_no_affiliation} "
        f"out of {len(citation['authors'])}"
    )

    write_citation_cff(citation)


if __name__ == "__main__":
    main()
