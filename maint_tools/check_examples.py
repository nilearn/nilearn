# /// script
# requires-python = ">=3.11"
# ///
"""Check examples toh make sure they contain an admonition for data description
if they use any of the nilearn fetchers.
"""

from utils import root_dir


def _has_admnonition(text: str):
    return ".. admonition:: dataset" in text


examples = (root_dir() / "examples").glob("**/plot*.py")
problematic_examples: list[str] = []
for ex in examples:
    with ex.open(encoding="utf8") as f:
        text = f.read()
        if (
            "from nilearn import datasets" in text
            or "from nilearn.datasets import" in text
        ) and not _has_admnonition(text):
            problematic_examples.append(str(ex))

if problematic_examples:
    raise Exception(
        "The following examples are missing an data description:\n-\t"
        + "\n-\t".join(sorted(problematic_examples))
    )
