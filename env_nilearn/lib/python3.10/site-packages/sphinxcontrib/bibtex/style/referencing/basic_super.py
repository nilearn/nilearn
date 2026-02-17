from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Union

from pybtex.style.template import tag

from sphinxcontrib.bibtex.style.template import (
    entry_label,
    join,
    join2,
    post_text,
    pre_text,
    reference,
)

from . import BaseReferenceStyle, BracketStyle, PersonStyle

if TYPE_CHECKING:
    from pybtex.richtext import BaseText
    from pybtex.style.template import Node


@dataclass
class BasicSuperParentheticalReferenceStyle(BaseReferenceStyle):
    """Reference by label superscript if parenthetical,
    and by author and label superscript if textual.
    """

    #: Bracket style. Left and right brackets are empty by default.
    bracket: BracketStyle = field(
        default_factory=lambda: BracketStyle(left="", right="", sep=",")
    )

    #: Person style.
    person: PersonStyle = field(default_factory=PersonStyle)

    #: Separator between pre-text and citation.
    pre_text_sep: Union["BaseText", str] = " "

    #: Separator between citation and post-text.
    post_text_sep: Union["BaseText", str] = ", "

    def role_names(self) -> Iterable[str]:
        return [
            f"{alt}p{full_author}" for alt in ["", "al"] for full_author in ["", "s"]
        ]

    def outer(self, role_name: str, children: List["BaseText"]) -> "Node":
        if "al" not in role_name:
            return tag("sup")[
                self.bracket.outer(children, brackets=True, capfirst=False)
            ]
        else:
            return self.bracket.outer(children, brackets=False, capfirst=False)

    def inner(self, role_name: str) -> "Node":
        return join2(sep1=self.pre_text_sep, sep2=self.post_text_sep)[
            pre_text,
            reference[entry_label],
            post_text,
        ]


@dataclass
class BasicSuperTextualReferenceStyle(BaseReferenceStyle):
    """Reference by label superscript if parenthetical,
    and by author and label superscript if textual.
    """

    #: Bracket style. Left and right brackets are empty by default.
    bracket: BracketStyle = field(
        default_factory=lambda: BracketStyle(left="", right="", sep=", ")
    )

    #: Person style.
    person: PersonStyle = field(default_factory=PersonStyle)

    #: Separator between text and reference.
    text_reference_sep: Union["BaseText", str] = ""

    #: Separator between pre-text and citation.
    pre_text_sep: Union["BaseText", str] = " "

    #: Separator between citation and post-text.
    post_text_sep: Union["BaseText", str] = ", "

    def role_names(self) -> Iterable[str]:
        return [
            f"{capfirst}t{full_author}"
            for capfirst in ["", "c"]
            for full_author in ["", "s"]
        ]

    def outer(self, role_name: str, children: List["BaseText"]) -> "Node":
        return self.bracket.outer(children, brackets=False, capfirst="c" in role_name)

    def inner(self, role_name: str) -> "Node":
        return join(sep=self.text_reference_sep)[
            self.person.author_or_editor_or_title(full="s" in role_name),
            tag("sup")[
                join[
                    self.bracket.left,
                    join2(sep1=self.pre_text_sep, sep2=self.post_text_sep)[
                        pre_text,
                        reference[entry_label],
                        post_text,
                    ],
                    self.bracket.right,
                ]
            ],
        ]
