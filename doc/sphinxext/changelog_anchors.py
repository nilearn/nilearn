"""Give changelog subsections stable, version-scoped HTML anchors.

``doc/changes/whats_new.rst`` concatenates every per-release changelog
fragment (``doc/changes/<version>.rst``),
each of which repeats the same subsection titles
(``Fixes``, ``Changes``, ``Enhancements``, ...).
Docutils only gives the first occurrence of a repeated heading a readable id;
every later one falls back to a sequential ``id123``-style id
whose value shifts whenever a new release
is inserted earlier in the document
(see https://github.com/nilearn/nilearn/issues/6456).

This module rewrites the changelog fragments in place,
right before Sphinx reads them,
inserting an explicit ``.. _v<version>-<subsection>:`` label
before each subsection heading.
The label is derived purely from the fragment's own version heading
and the subsection text,
so it never needs to be hand-maintained:
it is recomputed from scratch on every doc build
and is idempotent (re-running it is a no-op).
"""

import re
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

TOP_UNDERLINE = re.compile(r"^=+$")
SUB_UNDERLINE = re.compile(r"^[\-^]{3,}$")
GENERATED_LABEL = re.compile(r"^\.\. _v[0-9][\w.-]*:$")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _version_prefix(title: str) -> str:
    """Turn a release heading into a stable anchor prefix, e.g. ``v0-14-0``.

    The leading "Version" word is dropped, and any trailing "dev" suffix
    is stripped so the prefix stays the same before and after a release
    (when ``latest.rst``, e.g. "Version 0.14.1dev", is renamed to
    ``0.14.1.rst`` with title "Version 0.14.1").
    """
    version = re.sub(r"^version\s+", "", title.strip(), flags=re.IGNORECASE)
    version = re.sub(r"dev$", "", version, flags=re.IGNORECASE)
    return "v" + _slugify(version)


def _strip_generated_labels(lines: list[str]) -> list[str]:
    """Skip eventual generated labels from a previous run.

    This way they are not added twice.
    """
    out = []
    i = 0
    n = len(lines)
    while i < n:
        if (
            GENERATED_LABEL.match(lines[i].strip())
            and i + 1 < n
            and lines[i + 1].strip() == ""
        ):
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def _insert_labels(lines: list[str], path: Path) -> list[str]:
    """Insert a stable anchor label before every subsection heading.

    Each label is scoped to the nearest preceding top-level (version)
    heading, so a file with several version headings (some old changelog
    fragments bundle a release and its release candidates in one file)
    gets a distinct prefix for each one.

    Two subsections under the same version heading with the same title
    (e.g. two ``Fixes`` sections) would otherwise get the same anchor;
    when that happens a numbered suffix is appended to keep anchors
    unique, and a build warning is emitted so the duplicate title can be
    reworded instead.
    """
    out = []
    prefix = None
    used = set()
    i = 0
    n = len(lines)
    while i < n:
        title = lines[i].strip()
        is_heading = i + 1 < n and title

        if is_heading and TOP_UNDERLINE.match(lines[i + 1].strip()):
            # store prefix to use for the following subsections
            prefix = _version_prefix(title)
            used = set()
            out.append(lines[i])
            out.append(lines[i + 1])
            i += 2
            continue

        if is_heading and SUB_UNDERLINE.match(lines[i + 1].strip()) and prefix:
            slug = _slugify(title)
            anchor = f"{prefix}-{slug}"
            if anchor in used:
                logger.warning(
                    "%s:%d: subsection title %r duplicates another one "
                    "under the same release heading; consider renaming "
                    "it so it gets its own stable anchor",
                    path,
                    i + 1,
                    title,
                )
            n_dup = 1
            while anchor in used:
                n_dup += 1
                anchor = f"{prefix}-{slug}-{n_dup}"
            used.add(anchor)
            out.append(f".. _{anchor}:")
            out.append("")

        out.append(lines[i])
        i += 1
    return out


def _process(path: Path) -> None:
    """Update anchors of subsection titles in a single changelog file."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    lines = _strip_generated_labels(lines)
    lines = _insert_labels(lines, path)
    updated = "\n".join(lines) + "\n"
    # Only save the files if it was modified
    if updated != original:
        path.write_text(updated, encoding="utf-8")


def insert_changelog_anchors(app: Sphinx) -> None:
    """Update anchors of subsection titles in all changelog files.

    Connected to the ``builder-inited`` event so it runs, and rewrites
    ``doc/changes/*.rst`` on disk, before Sphinx reads any source file.
    """
    changes_dir = Path(app.srcdir) / "changes"
    if not changes_dir.is_dir():
        return
    for path in sorted(changes_dir.glob("*.rst")):
        if path.name in ("whats_new.rst", "names.rst"):
            continue
        _process(path)


def setup(app: Sphinx) -> None:
    """Register the ``builder-inited`` hook with Sphinx."""
    app.connect("builder-inited", insert_changelog_anchors)
