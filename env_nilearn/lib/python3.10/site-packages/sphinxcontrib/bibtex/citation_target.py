"""Citation keys."""

import re
from typing import Iterable, NamedTuple


class CitationTarget(NamedTuple):
    """Citation key, pre-text, and post-text."""

    key: str
    pre: str
    post: str


_re_citation_target = re.compile(
    r"\s*([{](?P<pre>[^{}]+)[}])?"
    r"\s*(?P<key>[^{}\s,]+)"
    r"\s*([{](?P<post>[^{}]+)[}])?\s*"
)


def parse_citation_targets(targets: str, pos=0) -> Iterable[CitationTarget]:
    """Parse citation target string into a list of citation keys."""
    match = _re_citation_target.match(targets, pos=pos)
    if match is None:
        raise ValueError(f"malformed citation target: {targets}")
    yield CitationTarget(
        key=match.group("key") or "",
        pre=match.group("pre") or "",
        post=match.group("post") or "",
    )
    end = match.end()
    if end < len(targets):
        if targets[end] != ",":
            raise ValueError(f"malformed citation target: {targets}")
        yield from parse_citation_targets(targets, end + 1)
