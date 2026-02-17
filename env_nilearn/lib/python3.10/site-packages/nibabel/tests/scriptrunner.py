"""Module to help tests check script output

Provides class to be instantiated in tests that check scripts.  Usually works
something like this in a test module::

    from .scriptrunner import ScriptRunner
    runner = ScriptRunner()

Then, in the tests, something like::

    code, stdout, stderr = runner.run_command(['my-script', my_arg])
    assert_equal(code, 0)
    assert_equal(stdout, b'This script ran OK')
"""

import os
import sys
from os.path import dirname, isdir, isfile, pathsep, realpath
from os.path import join as pjoin
from subprocess import PIPE, Popen

MY_PACKAGE = __package__


def local_script_dir(script_sdir):
    """Get local script directory if running in development dir, else None"""
    # Check for presence of scripts in development directory.  ``realpath``
    # allows for the situation where the development directory has been linked
    # into the path.
    package_path = dirname(__import__(MY_PACKAGE).__file__)
    above_us = realpath(pjoin(package_path, '..'))
    devel_script_dir = pjoin(above_us, script_sdir)
    if isfile(pjoin(above_us, 'setup.py')) and isdir(devel_script_dir):
        return devel_script_dir
    return None


def local_module_dir(module_name):
    """Get local module directory if running in development dir, else None"""
    mod = __import__(module_name)
    containing_path = dirname(dirname(realpath(mod.__file__)))
    if containing_path == realpath(os.getcwd()):
        return containing_path
    return None


class ScriptRunner:
    """Class to run scripts and return output

    Finds local scripts and local modules if running in the development
    directory, otherwise finds system scripts and modules.
    """

    def __init__(
        self,
        script_sdir='scripts',
        module_sdir=MY_PACKAGE,
        debug_print_var=None,
        output_processor=lambda x: x,
    ):
        """Init ScriptRunner instance

        Parameters
        ----------
        script_sdir : str, optional
            Name of subdirectory in top-level directory (directory containing
            setup.py), to find scripts in development tree.  Typically
            'scripts', but might be 'bin'.
        module_sdir : str, optional
            Name of subdirectory in top-level directory (directory containing
            setup.py), to find main package directory.
        debug_print_vsr : str, optional
            Name of environment variable that indicates whether to do debug
            printing or no.
        output_processor : callable
            Callable to run on the stdout, stderr outputs before returning
            them.  Use this to convert bytes to unicode, strip whitespace, etc.
        """
        self.local_script_dir = local_script_dir(script_sdir)
        self.local_module_dir = local_module_dir(module_sdir)
        if debug_print_var is None:
            debug_print_var = f'{module_sdir.upper()}_DEBUG_PRINT'
        self.debug_print = os.environ.get(debug_print_var, False)
        self.output_processor = output_processor

    def run_command(self, cmd, check_code=True):
        """Run command sequence `cmd` returning exit code, stdout, stderr

        Parameters
        ----------
        cmd : str or sequence
            string with command name or sequence of strings defining command
        check_code : {True, False}, optional
            If True, raise error for non-zero return code

        Returns
        -------
        returncode : int
            return code from execution of `cmd`
        stdout : bytes
            stdout from `cmd`
        stderr : bytes
            stderr from `cmd`
        """
        if isinstance(cmd, str):
            cmd = [cmd]
        else:
            cmd = list(cmd)
        if not self.local_script_dir is None:
            # Windows can't run script files without extensions natively so we need
            # to run local scripts (no extensions) via the Python interpreter.  On
            # Unix, we might have the wrong incantation for the Python interpreter
            # in the hash bang first line in the source file.  So, either way, run
            # the script through the Python interpreter
            cmd = [sys.executable, pjoin(self.local_script_dir, cmd[0])] + cmd[1:]
        if os.name == 'nt':
            # Quote any arguments with spaces. The quotes delimit the arguments
            # on Windows, and the arguments might be file paths with spaces.
            # On Unix the list elements are each separate arguments.
            cmd = [f'"{c}"' if ' ' in c else c for c in cmd]
        if self.debug_print:
            print(f"Running command '{cmd}'")
        env = os.environ
        if not self.local_module_dir is None:
            # module likely comes from the current working directory. We might need
            # that directory on the path if we're running the scripts from a
            # temporary directory
            env = env.copy()
            pypath = env.get('PYTHONPATH', None)
            if pypath is None:
                env['PYTHONPATH'] = self.local_module_dir
            else:
                env['PYTHONPATH'] = self.local_module_dir + pathsep + pypath
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, env=env)
        stdout, stderr = proc.communicate()
        if proc.poll() is None:
            proc.terminate()
        if check_code and proc.returncode != 0:
            raise RuntimeError(
                f"""Command "{cmd}" failed with
                stdout
                ------
                {stdout}
                stderr
                ------
                {stderr}
                """
            )
        opp = self.output_processor
        return proc.returncode, opp(stdout), opp(stderr)
