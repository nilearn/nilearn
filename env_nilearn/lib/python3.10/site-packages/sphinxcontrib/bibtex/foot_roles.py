"""
.. autoclass:: FootCiteRole
    :show-inheritance:

    .. automethod:: run
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import docutils.nodes
from docutils.nodes import make_id
from pybtex.database import Entry
from pybtex.plugin import find_plugin
from pybtex.style import FormattedEntry
from sphinx.util.docutils import SphinxRole
from sphinx.util.logging import getLogger

from .style.referencing import format_references
from .style.template import FootReferenceInfo
from .transforms import node_text_transform

if TYPE_CHECKING:
    from .domain import BibtexDomain
    from .foot_domain import BibtexFootDomain

logger = getLogger(__name__)


class FootCiteRole(SphinxRole):
    """Class for processing the :rst:role:`footcite` role."""

    def run(
        self,
    ) -> Tuple[List["docutils.nodes.Node"], List["docutils.nodes.system_message"]]:
        """Transform node into footnote references, and
        add footnotes to a node stored in the environment's temporary data
        if they are not yet present.

        .. seealso::

           The node containing all footnotes is inserted into the document by
           :meth:`.foot_directives.FootBibliographyDirective.run`.
        """
        foot_domain = cast("BibtexFootDomain", self.env.get_domain("footcite"))
        keys = [key.strip() for key in self.text.split(",")]
        try:
            foot_bibliography = self.env.temp_data["bibtex_foot_bibliography"]
        except KeyError:
            self.env.temp_data["bibtex_foot_bibliography"] = foot_bibliography = (
                foot_domain.bibliography_header.deepcopy()
            )
        foot_old_refs: set[str] = self.env.temp_data.setdefault(  # type: ignore
            "bibtex_foot_old_refs", set()
        )
        foot_new_refs: set[str] = self.env.temp_data.setdefault(  # type: ignore
            "bibtex_foot_new_refs", set()
        )
        style = find_plugin(
            "pybtex.style.formatting", self.config.bibtex_default_style
        )()
        references: List[Tuple[Entry, FormattedEntry, FootReferenceInfo]] = []
        domain = cast("BibtexDomain", self.env.get_domain("cite"))
        # count only incremented at directive, see foot_directives run method
        footbibliography_count: int = self.env.temp_data.setdefault(  # type: ignore
            "bibtex_footbibliography_count", 0
        )
        footcite_names: dict[str, str] = self.env.temp_data.setdefault(  # type: ignore
            "bibtex_footcite_names", {}
        )
        for key in keys:
            entry: Optional[Entry] = domain.bibdata.data.entries.get(key)
            if entry is not None:
                formatted_entry: FormattedEntry = style.format_entry(
                    label="", entry=entry
                )
                if key not in (foot_old_refs | foot_new_refs):
                    footnote = docutils.nodes.footnote(auto=1)
                    # no automatic ids for footnotes: force non-empty template
                    template: str = (
                        self.env.app.config.bibtex_footcite_id
                        if self.env.app.config.bibtex_footcite_id
                        else "footcite-{key}"
                    )
                    raw_id = template.format(
                        footbibliography_count=footbibliography_count + 1, key=entry.key
                    )
                    # format name with make_id for consistency with cite role
                    name = make_id(raw_id)
                    footnote["names"] += [name]
                    footcite_names[entry.key] = name
                    footnote += domain.backend.paragraph(formatted_entry)
                    self.inliner.document.note_autofootnote(footnote)
                    self.inliner.document.note_explicit_target(footnote, footnote)
                    node_text_transform(footnote)
                    foot_bibliography += footnote
                    foot_new_refs.add(key)
                references.append(
                    (
                        entry,
                        formatted_entry,
                        FootReferenceInfo(
                            key=entry.key,
                            refname=footcite_names[entry.key],
                            document=self.inliner.document,
                        ),
                    )
                )
            else:
                logger.warning(
                    'could not find bibtex key "%s"' % key,
                    location=(self.env.docname, self.lineno),
                    type="bibtex",
                    subtype="key_not_found",
                )
        _, _, role_name = self.name.partition(":")
        ref_nodes = format_references(
            foot_domain.reference_style, role_name or "p", references
        ).render(domain.backend)
        return ref_nodes, []
