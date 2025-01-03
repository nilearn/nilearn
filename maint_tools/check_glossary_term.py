"""Check if some words are not linked to the glossary.

Check rst files in doc, py files in examples and py files in nilearn.

requirements:

docstring_parser
rich

"""

from __future__ import annotations

import ast
from pathlib import Path

from docstring_parser import parse
from docstring_parser.common import DocstringStyle
from rich import print

SEARCH = []


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


def glossary_file() -> Path:
    """Return path to glossary.rst file."""
    return root_dir() / "doc" / "glossary.rst"


def get_terms_in_glossary() -> list[str]:
    """Return list of terms in glossary.rst."""
    if len(SEARCH) > 0:
        return SEARCH

    terms = []

    track = False
    with glossary_file().open(encoding="utf-8") as file:
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

    # Also take the lower case version of terms
    terms.extend([term.lower() for term in terms])
    terms = list(set(terms))

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
        "#    ",
    ]

    separator_rendered_block = ["# %%", "#############"]

    count = 0

    with file.open(encoding="utf-8") as f:
        if file.suffix == ".py":
            is_rendered_section = False
            is_docstring = False

        for i, line in enumerate(f.readlines()):
            if any(line.strip().startswith(s) for s in skip_if):
                continue

            # Python files
            if file.suffix == ".py":
                if any(
                    line.strip().startswith(s)
                    for s in separator_rendered_block
                ):
                    is_rendered_section = True

                if is_rendered_section and not line.strip().startswith("#"):
                    is_rendered_section = False
                    continue

                if line.strip().startswith('"""'):
                    is_docstring = not is_docstring

                if not is_docstring and not is_rendered_section:
                    continue

                elif check_string(line, term):
                    print(f"'{term}' in {file}:{i + 1}")
                    count += 1

            elif check_string(line, term):
                print(f"'{term}' in {file}:{i + 1}")
                count += 1

    return count


def check_string(string: str, term: str) -> bool:
    """Check a string for a term."""
    if string is None:
        return False
    return (
        f" {term} " in string
        or string.startswith(f"{term} ")
        or string.endswith(f" {term}")
    ) and f":term:`{term}`" not in string


def check_docstring(docstring, terms):
    """Check docstring content."""
    docstring = parse(docstring, style=DocstringStyle.NUMPYDOC)

    text = check_description(docstring, terms)

    for param in docstring.params:
        if tmp := [
            term for term in terms if check_string(param.description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in param '{param.arg_name}'\n"

    for return_arg in docstring.many_returns:
        if tmp := [
            term
            for term in terms
            if check_string(return_arg.description, term)
        ]:
            text += (
                f" terms: '{', '.join(tmp)}' "
                "in return arg param '{return_arg.return_name}'\n"
            )

    return text


def check_description(docstring, terms):
    """Check docstring description."""
    text = ""
    if docstring.short_description is not None:  # noqa: SIM102
        if tmp := [
            term
            for term in terms
            if check_string(docstring.short_description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in short description\n"

    if docstring.long_description is not None:  # noqa: SIM102
        if tmp := [
            term
            for term in terms
            if check_string(docstring.long_description, term)
        ]:
            text += f" terms: '{', '.join(tmp)}' in long description\n"

    return text


def check_functions(body, terms, file):
    """Check functions of a module or methods of a class."""
    function_definitions = [
        node for node in body if isinstance(node, ast.FunctionDef)
    ]
    for f in function_definitions:
        if f.name.startswith("_"):
            continue

        docstring = ast.get_docstring(f)
        if text := check_docstring(docstring, terms):
            print(f"function '{f.name}' in {file}:{f.lineno}")
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
        "description",
    ]

    print("\n\nCheck .rst files in doc\n")

    files_to_skip = ["conf.py", "index.rst", "glossary.rst"]

    doc_folder = root_dir() / "doc"

    print(f"Checking: {doc_folder}")

    files = list(doc_folder.glob("*.rst"))
    for folder in doc_folder.glob("*"):
        if folder.is_dir() and folder.name not in folders_to_skip:
            files.extend(f for f in folder.glob("*.rst") if f is not None)

    files.extend(
        (root_dir() / "nilearn" / "datasets" / "description").glob("*.rst")
    )

    count = check_files(files, terms, files_to_skip)

    print(f"\n\nTotal: {count} terms not linked to glossary\n")


def check_examples(terms):
    """Check examples content."""
    files_to_skip = []

    print("\n\nCheck .py files in examples\n")

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

    print("\n\nCheck .py files in nilearn\n")

    modules = (root_dir() / "nilearn").glob("**/*.py")

    files_to_skip = ["test_", "conftest.py", "_"]

    for file in modules:
        if any(file.name.startswith(s) for s in files_to_skip):
            continue
        if file.parent.name.startswith("_"):
            continue

        with file.open() as f:
            module = ast.parse(f.read())

        check_functions(module.body, terms, file)

        class_definitions = [
            node for node in module.body if isinstance(node, ast.ClassDef)
        ]

        for class_def in class_definitions:
            docstring = ast.get_docstring(class_def)
            docstring = parse(docstring, style=DocstringStyle.NUMPYDOC)

            if text := check_description(docstring, terms):
                print(f"class '{class_def.name}' in {file}:{class_def.lineno}")
                print(text)
            check_functions(class_def.body, terms, file)


if __name__ == "__main__":
    main()
