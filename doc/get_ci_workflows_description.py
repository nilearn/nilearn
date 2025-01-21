"""Update documentation for CI."""

from pathlib import Path

doc_dir = Path(__file__).parent
root_dir = doc_dir.parent
workflows_dir = root_dir / ".github" / "workflows"
template_file_path = doc_dir / "ci.jinja"
output_file_path = doc_dir / "ci.rst"


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

    Collect lines that start with a comment (#) until the `---` separator

    Parameters
    ----------
        yml_file : Path to the YAML file.

    Returns
    -------
        str : The extracted comment or an empty string if no comment is found.
    """
    error_msg = (
        f"yml file '{yml_file.relative_to(root_dir)}' "
        "should contain a comment before the top '---'."
    )

    with yml_file.open() as file:
        content = file.read()

    if "---" not in content:
        raise ValueError(error_msg)

    yaml_lines = content.splitlines()

    comments = []
    for line in yaml_lines:
        if line.startswith("#"):
            comments.append(line.lstrip("#")[1:])
        elif line.strip() == "---":
            break

    if not comments:
        raise ValueError(error_msg)

    return "\n".join(comments)


def inject_with_jinja(
    template_file: Path, output_file: Path, context: list[dict[str, str]]
):
    """Render Jinja template given context and write it to an output file.

    Parameters
    ----------
        template_file (str): Path to the Jinja template file.
        output_file (str): Path to the output file.
        context (dict): The context dictionary to render the template.

    Returns
    -------
        None
    """
    from jinja2 import Template

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
