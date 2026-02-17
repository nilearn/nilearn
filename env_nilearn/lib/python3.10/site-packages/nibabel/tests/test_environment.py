"""Testing environment settings"""

import os
from os import environ as env
from os.path import abspath
from os.path import join as pjoin

import pytest

from .. import environment as nibe

DATA_KEY = 'NIPY_DATA_PATH'
USER_KEY = 'NIPY_USER_DIR'


@pytest.fixture
def with_environment(request):
    """Setup test environment for some functions that are tested
    in this module. In particular this functions stores attributes
    and other things that we need to stub in some test functions.
    This needs to be done on a function level and not module level because
    each testfunction needs a pristine environment.
    """
    GIVEN_ENV = {}
    GIVEN_ENV['env'] = env.copy()
    yield
    """Restore things that were remembered by the setup_environment function """
    orig_env = GIVEN_ENV['env']
    # Pull keys out into list to avoid altering dictionary during iteration,
    # causing python 3 error
    for key in list(env.keys()):
        if key not in orig_env:
            del env[key]
    env.update(orig_env)


def test_nipy_home():
    # Test logic for nipy home directory
    assert nibe.get_home_dir() == os.path.expanduser('~')


def test_user_dir(with_environment):
    if USER_KEY in env:
        del env[USER_KEY]
    home_dir = nibe.get_home_dir()
    if os.name == 'posix':
        exp = pjoin(home_dir, '.nipy')
    else:
        exp = pjoin(home_dir, '_nipy')
    assert exp == nibe.get_nipy_user_dir()
    env[USER_KEY] = '/a/path'
    assert abspath('/a/path') == nibe.get_nipy_user_dir()


def test_sys_dir():
    sys_dir = nibe.get_nipy_system_dir()
    if os.name == 'nt':
        assert sys_dir == r'C:\etc\nipy'
    elif os.name == 'posix':
        assert sys_dir == r'/etc/nipy'
    else:
        assert sys_dir is None
