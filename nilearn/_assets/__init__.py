from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_template(template: str):
    """Set up the jinja Environment."""
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True,
    )
    return env.get_template(template)
