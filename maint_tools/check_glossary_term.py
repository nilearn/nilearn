"""Small script to check if some words in the documentation \
   are not linked to the glossary."""
from __future__ import annotations

import ast
from pathlib import Path

from docstring_parser import parse
from docstring_parser.common import DocstringStyle
from rich import print


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

            # if file is a python file and only comments
            # TODO: check docstrings
            if file.suffix == ".py":
                if not line.startswith("#"):
                    continue
            if check_string(line, term):
                print(f"'{term}' in {file} at line {i+1}")
                count += 1
    return count


def check_string(string, term):
    """Check a string for a term."""
    if string is None:
        return False
    return f" {term} " in string and f":term:`{term}`" not in string


def check_docstring(docstring, terms):
    """Check docstring content."""
    docstring = parse(docstring, style=DocstringStyle.NUMPYDOC)

    text = check_description(docstring, terms)

    for param in docstring.params:
        if tmp := [
            term for term in terms if check_string(param.description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in param '{param.arg_name}'\n"

    return text


def check_description(docstring, terms):
    """Check docstring description."""
    text = ""
    if docstring.short_description is not None:
        if tmp := [
            term
            for term in terms
            if check_string(docstring.short_description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in short description\n"

    if docstring.long_description is not None:
        if tmp := [
            term
            for term in terms
            if check_string(docstring.long_description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in short description\n"

    return text


def check_functions(body, terms):
    """Check functions of a module or methods of a class."""
    function_definitions = [
        node for node in body if isinstance(node, ast.FunctionDef)
    ]
    for f in function_definitions:
        if f.name.startswith("_"):
            continue

        docstring = ast.get_docstring(f)
        if text := check_docstring(docstring, terms):
            print(f"function '{f.name}' at line {f.lineno}")
            print(text)


def check_doc(terms):
    """Check doc content."""
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

    print("\n\nCheck rst files in doc\n")

    files_to_skip = ["conf.py", "index.rst", "glossary.rst"]

    doc_folder = root_dir() / "doc"

    files = list(doc_folder.glob("*.rst"))
    for folder in doc_folder.glob("*"):
        if folder.is_dir() and folder.name not in folders_to_skip:
            files.extend(f for f in folder.glob("*.rst") if f is not None)

    count = check_files(files, terms, files_to_skip)

    print(f"\n\nTotal: {count} terms not linked to glossary\n")


def check_examples(terms):
    """Check examples content."""
    files_to_skip = []

    print("\n\nCheck py files in examples\n")

    example_folder = root_dir() / "examples"

    files = []
    for folder in example_folder.glob("*"):
        if folder.is_dir():
            files.extend(f for f in folder.glob("*.py") if f is not None)

    count = check_files(files, terms, files_to_skip)

    print(f"\n\nTotal: {count} terms not linked to glossary\n")


def main():
    """Check code and doc for terms not linked to glossary."""
    terms = get_terms_in_glossary()
    print(terms)

    check_doc(terms)

    check_examples(terms)

    modules = (root_dir() / "nilearn").glob("**/*.py")

    files_to_skip = ["test_", "conftest.py", "_"]

    for file in modules:
        if any(file.name.startswith(s) for s in files_to_skip):
            continue

        with open(file) as f:
            module = ast.parse(f.read())

        print(f"{file}")

        check_functions(module.body, terms)

        class_definitions = [
            node for node in module.body if isinstance(node, ast.ClassDef)
        ]

        for class_def in class_definitions:
            docstring = ast.get_docstring(class_def)
            docstring = parse(docstring, style=DocstringStyle.NUMPYDOC)

            if text := check_description(docstring, terms):
                print(f"class '{class_def.name}' at line {class_def.lineno}")
                print(text)
            check_functions(class_def.body, terms)


if __name__ == "__main__":
    main()
