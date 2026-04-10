from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def return_jinja_env() -> Environment:
    """Set up the jinja Environment."""
    return Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True,
    )
