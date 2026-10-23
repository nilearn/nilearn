"""Render structured dataset descriptions during the doc build.

Adds the ``nilearn_dataset_content`` and ``nilearn_dataset_license``
directives used in ``nilearn/datasets/description/*.rst``.

Their content comes from
``nilearn.datasets._descriptions.DATASET_DESCRIPTIONS``.
"""

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

from nilearn._utils.docs import content_to_rst, license_to_rst


class _DatasetDescriptionDirective(SphinxDirective):
    required_arguments = 1
    has_content = False

    @staticmethod
    def render(name: str) -> str:
        raise NotImplementedError

    def run(self):
        name = self.arguments[0]
        try:
            rst = self.render(name)
        except KeyError as e:
            raise self.error(
                f"'{name}' not found in "
                "nilearn.datasets._descriptions.DATASET_DESCRIPTIONS"
            ) from e
        container = nodes.container()
        self.state.nested_parse(
            StringList(rst.splitlines(), source=self.get_source_info()[0]),
            self.content_offset,
            container,
        )
        return container.children


class DatasetContentDirective(_DatasetDescriptionDirective):
    """Render the content of a dataset."""

    render = staticmethod(content_to_rst)


class DatasetLicenseDirective(_DatasetDescriptionDirective):
    """Render the license of a dataset."""

    render = staticmethod(license_to_rst)


def setup(app):
    app.add_directive("nilearn_dataset_content", DatasetContentDirective)
    app.add_directive("nilearn_dataset_license", DatasetLicenseDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
