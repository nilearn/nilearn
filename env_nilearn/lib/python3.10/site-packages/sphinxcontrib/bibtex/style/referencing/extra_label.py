from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List

from sphinxcontrib.bibtex.style.template import entry_label, reference

from . import BaseReferenceStyle, BracketStyle

if TYPE_CHECKING:
    from pybtex.richtext import BaseText
    from pybtex.style.template import Node


@dataclass
class ExtraLabelReferenceStyle(BaseReferenceStyle):
    """Reference just by label."""

    #: Bracket style.
    bracket: BracketStyle = field(default_factory=BracketStyle)

    def role_names(self) -> Iterable[str]:
        return ["label", "labelpar"]

    def outer(self, role_name: str, children: List["BaseText"]) -> "Node":
        return self.bracket.outer(
            children,
            brackets="par" in role_name,
            capfirst=False,
        )

    def inner(self, role_name: str) -> "Node":
        return reference[entry_label]
