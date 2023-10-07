"""Small script to check if some words in the documentation \
   are not linked to the glossary."""
from __future__ import annotations

from pathlib import Path


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


def glossary_file() -> Path:
    """Return path to glossary.rst file."""
    return root_dir() / "doc" / "glossary.rst"


def get_terms_in_glossary() -> list[str]:
    """Return list of terms in glossary.rst."""
    terms = []

    track = False
    with open(glossary_file(), encoding="utf-8") as file:
        for line in file:
            if line.startswith(".. glossary::"):
                track = True
                continue

            # keep first word if line starts with 4 spaces a letter
            if track and line.startswith("    ") and line[4].isalpha():
                # remove trailing \n
                line = line.strip()
                terms.append(line)

            if track and line.startswith(".."):
                break

    return terms


def check_files(files, terms: list[str], files_to_skip: list[str]) -> None:
    """Check if terms present in files are not linked to glossary."""
    for file in files:
        if file.name in files_to_skip:
            continue

        with open(file, encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                skip_if = [
                    ":alt:",
                    "<div",
                    ".. ",
                    ":start-after:",
                    ":end-before:",
                    ":ref:",
                ]
                if any(line.strip().startswith(s) for s in skip_if):
                    continue

                for term in terms:
                    if f" {term} " in line and f":term:`{term}`" not in line:
                        print(f"'{term}' in {file} at line {i+1}")
                        break


# check rst files in docs
folders_to_skip = [
    "_build",
    "binder",
    "changes",
    "images",
    "includes",
    "logos",
    "modules",
    "sphinxext",
    "templates",
    "themes",
]

files_to_skip = ["conf.py", "index.rst", "glossary.rst"]

doc_folder = root_dir() / "doc"

terms = get_terms_in_glossary()
print(terms)

files = doc_folder.glob("*.rst")
check_files(files, terms, files_to_skip)

for folder in doc_folder.glob("*"):
    if folder.is_dir() and folder.name not in folders_to_skip:
        files = folder.glob("*.rst")
        check_files(files, terms, files_to_skip)
