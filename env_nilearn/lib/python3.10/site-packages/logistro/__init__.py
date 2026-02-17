"""
Logistro wraps `logging` for added defaults and subprocess logging.

Typical usage:

```python

import logistro
logger = logistro.getLogger(__name__)
logger.debug2("This will be printed more informatively")

# Advanced
pipe, logger = logistro.getPipeLogger(__name__)
# Pipe all stderr to our logger
subprocess.Popen(process_name, stderr=pipe)

# Eventually close the pipe in case other process doesn't
subprocess.wait()
os.close(pipe)
```
"""

import logging

from ._api import (
    DEBUG2,
    betterConfig,
    describe_logging,
    getLogger,
    getPipeLogger,
    human_formatter,
    set_human,
    set_structured,
    structured_formatter,
)
from ._args import parsed, parser, remaining_args

CRITICAL = logging.CRITICAL
"""Equal to logging.CRITICAL level."""

DEBUG = logging.DEBUG
"""Equal to logging.DEBUG level."""

ERROR = logging.ERROR
"""Equal to logging.ERROR level."""

INFO = logging.INFO
"""Equal to logging.INFO level."""

WARNING = logging.WARNING
"""Equal to logging.WARNING level."""

__all__ = [
    "CRITICAL",
    "DEBUG",
    "DEBUG2",
    "ERROR",
    "INFO",
    "WARNING",
    "betterConfig",
    "describe_logging",
    "getLogger",
    "getPipeLogger",
    "human_formatter",
    "parsed",
    "parser",
    "remaining_args",
    "set_human",
    "set_structured",
    "structured_formatter",
]
