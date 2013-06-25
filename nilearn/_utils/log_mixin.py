"""
Mixin for displaying user messages
"""
# Author: Philippe Gervais
# License: simplified BSD

import inspect


class LogMixin(object):
    def log(self, msg, verbose=1):
        """Print a message for the user.

        Signature is that of print(), with an extra keyword argument giving
        the verbosity level above which the message should be printed.
        This method prepends class and method names.
        """
        if self.verbose >= verbose:
            calling_function = (inspect.stack())[1][3]
            prefix = "[%s.%s] " % (self.__class__.__name__, calling_function)
            print(prefix + msg)
