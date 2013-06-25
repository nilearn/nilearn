"""Logging facility for NiLearn"""

# Author: Philippe Gervais
# License: simplified BSD

import inspect
import platform

from sklearn.base import BaseEstimator


def log(msg, verbose=1, object_classes=(BaseEstimator, ),
        stack_level=1, msg_level=1):
    """Display a message to the user, depending on the verbosity level.

    This function does tricky things to ensure that the proper object is
    referenced in the message. If it is called inside a function that is
    called by a method of an object inheriting from any class in
    object_classes, then the name of the object (and the method) will be
    displayed to the user. If several matching objects exist in the call
    stack, the highest one is used (first call chronologically).

    Parameters
    ----------
    msg: str
        message to display

    verbose: int
        current verbosity level. Message is displayed if this value is greater
        or equal to msg_level.

    object_classes: tuple of type
        classes that should appear to emit the message

    stack_level: int
        if no object in the call stack matches object_classes, go back that
        amount in the call stack and display class/function name thereof.

    msg_level: int
        verbosity level at and above which message should be displayed to the
        user. Most of the time this parameter can be left unchanged.
    """

    if verbose >= msg_level:
        # The dirty tricks below work only on CPython
        if platform.python_implementation() != "CPython":
            print(msg)
            return

        object_frame = None
        for n, f in enumerate(inspect.stack()):
            frame = f[0]
            current_self = frame.f_locals.get("self", None)
            if current_self is not None:
                if isinstance(current_self, object_classes):
                    object_frame = frame
                    func_name = f[3]
                    object_self = current_self

        if object_frame is None:  # no object found: use stack_level
            object_frame, _, _, func_name = inspect.stack()[stack_level][:4]
            object_self = object_frame.f_locals.get("self", None)

        prefix = "%s] " % func_name
        if object_self is not None:
            prefix = ("[%s." % object_self.__class__.__name__) + prefix
        else:
            prefix = "[" + prefix

        print(prefix + msg)
