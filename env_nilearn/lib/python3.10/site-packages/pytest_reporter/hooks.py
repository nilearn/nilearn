import pytest


def pytest_reporter_template_dirs(config):
    """Return a list of directories containing templates."""
    pass


def pytest_reporter_loader(dirs, config):
    """Create a template loader, environment, or equivalent.

    The return value is currently not used. The plugin should keep track of it
    itself and may re-use it in other hooks.
    """
    pass


def pytest_reporter_context(context, config):
    """Called when creating a template context.

    The plugin may add keys to the dictionary or modify existing ones.
    """
    pass


@pytest.hookspec(firstresult=True)
def pytest_reporter_render(template_name, dirs, context):
    """Called to render content given a template name and context.

    Should return a string if the template was found.
    """
    pass


def pytest_reporter_save(config):
    """Called to save the report.

    Usually called at end of session, but may for example be called after each
    test in order to update the report during the session.
    """
    pass


def pytest_reporter_finish(path, context, config):
    """Called after a report has been saved.

    May be used for post-processing, saving of additional files, upload et.c.
    """
    pass
