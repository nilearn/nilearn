from pathlib import Path
import logging
import time
import warnings
from itertools import chain

import pytest
import _pytest.hookspec


def pytest_addoption(parser):
    group = parser.getgroup("report generation")
    group.addoption(
        "--report",
        action="append",
        default=[],
        help="path to report output (combined with --template).",
    )
    group.addoption(
        "--template",
        action="append",
        default=[],
        help="name or path to report template relative to --template-dir.",
    )
    group.addoption(
        "--template-dir",
        action="append",
        default=["."],
        help="path to template directory (multiple allowed).",
    )


def pytest_addhooks(pluginmanager):
    from . import hooks

    pluginmanager.add_hookspecs(hooks)


def pytest_configure(config):
    is_worker = hasattr(config, "workerinput")
    config.template_context = {
        "config": config,
        "tests": [],
        "warnings": [],
    }
    if config.getoption("--report") and not is_worker:
        config._reporter = ReportGenerator(config)
        config.pluginmanager.register(config._reporter)


@pytest.hookimpl(hookwrapper=True)
def pytest_collection(session):
    yield
    if not getattr(session, "items", []) and hasattr(session.config, "_reporter"):
        # Collection was skipped (probably due to xdist)
        session.perform_collect()


@pytest.hookimpl(tryfirst=True)
def pytest_reporter_template_dirs(config):
    return config.getoption("--template-dir")


def pytest_reporter_context(context, config):
    """Add status to test runs and phases."""
    nof_sections_per_node = {}
    for test in context["tests"]:
        for phase in test["phases"]:
            # The sections attribute in the TestReport object contains not only the
            # sections captured in this specific call, but also any previous
            # sections captured for this node, including e.g. setup or reruns.
            # Therefore we create a new sections key with only captures that are
            # new for this phase for convenience to the templates.
            nodeid = phase["report"].nodeid
            nof_sections = nof_sections_per_node.get(nodeid, 0)
            phase["sections"] = phase["report"].sections[nof_sections:]
            nof_sections_per_node[nodeid] = nof_sections + len(phase["sections"])

            # Get test status (e.g. passed, failed, error, skipped et.c.) for
            # this report. These will be empty strings except for the phase which
            # determines the status for the whole test.
            category, letter, word = config.hook.pytest_report_teststatus(
                report=phase["report"], config=config
            )
            if isinstance(word, tuple):
                word, style = word
            else:
                style = {}
            phase["status"] = {
                "category": category,
                "letter": letter,
                "word": word,
                "style": style,
            }
            # Set whole test status if this phase determined the outcome
            if letter or word:
                test["status"] = phase["status"]


@pytest.fixture(scope="session")
def template_context(pytestconfig):
    """Report template context for session."""
    return pytestconfig.template_context


class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.context = config.template_context
        self._items = {}
        self._active_tests = {}
        self._loaders = []
        self._log_handler = LogHandler()
        self._reports = set()

    def _get_testrun(self, nodeid: str):
        testrun = self._active_tests.get(nodeid)
        if testrun is None:
            item = self._items.get(nodeid)
            if item is None:
                # pytest-xdist may add node specific suffix
                item = self._items.get(nodeid.split("@", maxsplit=1)[0])
            testrun = {
                "item": item,
                "phases": [],
            }
            self._active_tests[nodeid] = testrun
        return testrun

    def pytest_sessionstart(self, session):
        self.context["session"] = session
        self.context["started"] = time.time()
        logging.getLogger().addHandler(self._log_handler)

    def pytest_report_collectionfinish(self, config, items):
        self._items = {item.nodeid: item for item in items}
        self.context["items"] = self._items

    def pytest_runtest_logstart(self, nodeid):
        testrun = self._get_testrun(nodeid)
        testrun["started"] = time.time()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        testrun = self._get_testrun(item.nodeid)
        phase = {}
        phase["call"] = call
        outcome = yield
        # rerunfailures doesn't always call pytest_runtest_logreport so we collect
        # the report here as well just to be sure
        phase["report"] = outcome.get_result()
        testrun["phases"].append(phase)

    def pytest_runtest_logreport(self, report):
        testrun = self._get_testrun(report.nodeid)
        # Check if there already is an existing phase from makereport
        for phase in testrun["phases"]:
            if phase["report"].when == report.when:
                break
        else:
            phase = {}
            testrun["phases"].append(phase)
        phase["report"] = report
        phase["log_records"] = self._log_handler.pop_records()

    def pytest_runtest_logfinish(self, nodeid):
        testrun = self._get_testrun(nodeid)
        testrun["ended"] = time.time()
        self.context["tests"].append(testrun)
        del self._active_tests[nodeid]

    # the pytest_warning_recorded hook was introduced in pytest 6.0
    if hasattr(_pytest.hookspec, "pytest_warning_recorded"):
        def pytest_warning_recorded(self, warning_message):
            self.context["warnings"].append(warning_message)
    else:
        def pytest_warning_captured(self, warning_message):
            self.context["warnings"].append(warning_message)

    def pytest_sessionfinish(self, session):
        self.context["ended"] = time.time()
        logging.getLogger().removeHandler(self._log_handler)
        self.config.hook.pytest_reporter_save(config=self.config)

    def pytest_reporter_save(self, config):
        # Create a list of all directories that may contain templates
        dirs_list = config.hook.pytest_reporter_template_dirs(config=config)
        dirs = list(chain.from_iterable(dirs_list))
        config.hook.pytest_reporter_loader(dirs=dirs, config=config)
        config.hook.pytest_reporter_context(context=self.context, config=config)
        for name, path in zip(
            config.getoption("--template"), config.getoption("--report")
        ):
            content = config.hook.pytest_reporter_render(
                template_name=name, dirs=dirs, context=self.context
            )
            if content is None:
                warnings.warn("No template found with name '%s'" % name)
                continue
            # Save content to file
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, "utf-8")
            config.hook.pytest_reporter_finish(
                path=target, context=self.context, config=config
            )
            self._reports.add(target)

    def pytest_terminal_summary(self, terminalreporter):
        for report in self._reports:
            terminalreporter.write_sep("-", "generated report: %s" % report.resolve())


class LogHandler(logging.Handler):
    def __init__(self):
        self._buffer = []
        super().__init__()

    def emit(self, record):
        self._buffer.append(record)

    def pop_records(self):
        records = self._buffer
        self._buffer = []
        return records
