import asyncio

import logistro
import pytest

# allows to create a browser pool for tests
pytestmark = pytest.mark.asyncio(loop_scope="function")

_logger = logistro.getLogger(__name__)


async def test_placeholder():
    _logger.info("testing placeholder")
    await asyncio.sleep(0)
    assert True
