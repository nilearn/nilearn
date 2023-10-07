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
    count = 1

    for term in terms:
        for file in files:
            if file is None or file.name in files_to_skip:
                continue
            tmp = check_file_content(file, term)
            count += tmp

    return count


def check_file_content(file, term):
    """Check if term is present in file and not linked to glossary."""
    skip_if = [
        ":alt:",
        "<div",
        ".. ",
        ":start-after:",
        ":end-before:",
        ":ref:",
    ]
    count = 0
    with open(file, encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            if any(line.strip().startswith(s) for s in skip_if):
                continue
            # if file is a python file only check doc strings and comments
            if file.suffix == ".py":
                if not line.startswith("#"):
                    continue
            if f" {term} " in line and f":term:`{term}`" not in line:
                print(f"'{term}' in {file} at line {i+1}")
                count += 1
    return count


terms = get_terms_in_glossary()
print(terms)

print("\n\nCheck rst files in doc\n")

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

files = list(doc_folder.glob("*.rst"))
for folder in doc_folder.glob("*"):
    if folder.is_dir() and folder.name not in folders_to_skip:
        files.extend(f for f in folder.glob("*.rst") if f is not None)

count = check_files(files, terms, files_to_skip)

print(f"\n\nTotal: {count} terms not linked to glossary\n")


print("\n\nCheck py files in examples\n")

example_folder = root_dir() / "examples"

files = []
for folder in example_folder.glob("*"):
    if folder.is_dir():
        files.extend(f for f in folder.glob("*.py") if f is not None)

count = check_files(files, terms, files_to_skip)

print(f"\n\nTotal: {count} terms not linked to glossary\n")
