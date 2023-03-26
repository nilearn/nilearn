from docutils.nodes import reference
from docutils.parsers.rst.roles import set_classes

# adapted from
# https://github.com/mne-tools/mne-python/blob/main/doc/sphinxext/gh_substitutions.py


def _gh_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to a GitHub issue."""
    try:
        # issue/PR mode (issues/PR-num will redirect to pull/PR-num)
        int(text)
    except ValueError:
        # direct link mode
        slug = text
    else:
        slug = f"issues/{text}"
    text = f"#{text}"
    ref = f"https://github.com/nilearn/nilearn/{slug}"
    set_classes(options)
    node = reference(rawtext, text, refuri=ref, **options)
    return [node], []


def setup(app):
    app.add_role("gh", _gh_role)
