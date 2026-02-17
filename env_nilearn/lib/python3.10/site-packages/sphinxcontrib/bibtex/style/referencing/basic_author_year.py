from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Union

from sphinxcontrib.bibtex.style.template import (
    join,
    join2,
    post_text,
    pre_text,
    reference,
    year,
)

from . import BaseReferenceStyle, BracketStyle, PersonStyle

if TYPE_CHECKING:
    from pybtex.richtext import BaseText
    from pybtex.style.template import Node


@dataclass
class BasicAuthorYearParentheticalReferenceStyle(BaseReferenceStyle):
    """Parenthetical reference by author-year."""

    #: Bracket style.
    bracket: BracketStyle = field(default_factory=BracketStyle)

    #: Person style.
    person: PersonStyle = field(default_factory=PersonStyle)

    #: Separator between author and year.
    author_year_sep: Union["BaseText", str] = ", "

    #: Separator between pre-text and citation.
    pre_text_sep: Union["BaseText", str] = " "

    #: Separator between citation and post-text.
    post_text_sep: Union["BaseText", str] = ", "

    def role_names(self) -> Iterable[str]:
        return [
            f"{alt}p{full_author}" for alt in ["", "al"] for full_author in ["", "s"]
        ]

    def outer(self, role_name: str, children: List["BaseText"]) -> "Node":
        return self.bracket.outer(
            children, brackets="al" not in role_name, capfirst=False
        )

    def inner(self, role_name: str) -> "Node":
        return join2(sep1=self.pre_text_sep, sep2=self.post_text_sep)[
            pre_text,
            reference[
                join(sep=self.author_year_sep)[
                    self.person.author_or_editor_or_title(full="s" in role_name),
                    year,
                ]
            ],
            post_text,
        ]


@dataclass
class BasicAuthorYearTextualReferenceStyle(BaseReferenceStyle):
    """Textual reference by author-year."""

    #: Bracket style.
    bracket: BracketStyle = field(default_factory=BracketStyle)

    #: Person style.
    person: PersonStyle = field(default_factory=PersonStyle)

    #: Separator between text and reference.
    text_reference_sep: Union["BaseText", str] = " "

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
            join[
                self.bracket.left,
                join2(sep1=self.pre_text_sep, sep2=self.post_text_sep)[
                    pre_text,
                    reference[year],
                    post_text,
                ],
                self.bracket.right,
            ],
        ]
