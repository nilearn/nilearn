from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

NIL_ASSETS = Path(__file__).parent


def return_jinja_env() -> Environment:
    """Set up the jinja Environment."""
    return Environment(
        loader=FileSystemLoader(NIL_ASSETS),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True,
    )
