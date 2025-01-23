"""Update documentation for CI.

Read YAML files from workflows from ".github/workflows",
make sure they have a top comment (whose ends is specified by SEPARATOR),
take the comment and inject into a jinja template
to generate a .rst page for the documentation.
"""

from pathlib import Path

from jinja2 import Template

doc_dir = Path(__file__).parent
root_dir = doc_dir.parent
workflows_dir = root_dir / ".github" / "workflows"
template_file_path = doc_dir / "ci.jinja"
output_file_path = doc_dir / "ci.rst"

SEPARATOR = "###"


def main() -> None:
    """Get top comment of each github action workflows \
        and inject it in the ci.jinja template \
        to create ci.rst file for the for the documentation.
    """
    context = []
    for yml_file in sorted(workflows_dir.glob("*.yml")):
        comment = extract_top_comment(yml_file)
        context.append({"file": yml_file.name, "comment": comment})

    inject_with_jinja(template_file_path, output_file_path, context)


def extract_top_comment(yml_file: Path) -> str:
    """Extract the top comment from a YAML file.

    Collect lines that start with a comment (#) until the `---` SEPARATOR

    Parameters
    ----------
        yml_file : Path to the YAML file.

    Returns
    -------
        str : The extracted comment.

    Raises
    ------
    ValueError
        In case the workflow file has no top comment.
    """
    error_msg = (
        f"yml file '{yml_file.relative_to(root_dir)}' "
        f"should contain a comment before the top {SEPARATOR}."
    )

    with yml_file.open() as file:
        content = file.read()

    if SEPARATOR not in content:
        raise ValueError(error_msg)

    yaml_lines = content.splitlines()

    comments = []
    for line in yaml_lines:
        if line.strip() == SEPARATOR:
            break
        if line.startswith("#"):
            comments.append(line.lstrip("#")[1:])

    if not comments:
        raise ValueError(error_msg)

    return "\n".join(comments)


def inject_with_jinja(
    template_file: Path, output_file: Path, context: list[dict[str, str]]
) -> None:
    """Render Jinja template given context and write it to an output file.

    Parameters
    ----------
    template_file : Path to the Jinja template file.
        output_file : Path to the output file.
        context : The context dictionary to render the template.
    """
    with template_file.open() as file:
        template_content = file.read()

    # Create a Jinja template and render it
    template = Template(template_content)
    rendered_content = template.render(context=context)

    # Write the rendered content to the output file
    with output_file.open("w") as file:
        file.write(rendered_content)

    print("Template rendered and written to", output_file)


if __name__ == "__main__":
    main()
