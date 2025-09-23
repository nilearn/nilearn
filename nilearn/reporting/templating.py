from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).parent

TEMPLATES_DIR = ROOT / "templates"


def return_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True,
    )


env = return_jinja_env()

layout = env.get_template("glm_report.jinja")


tmp = layout.render(displayed_runs=[0, 1])

print(tmp)
