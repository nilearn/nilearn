"""Logging facility for nilearn."""

# Author: Philippe Gervais

import inspect
import traceback

from sklearn.base import BaseEstimator


def _has_rich():
    """Check if rich is installed."""
    try:
        import rich  # noqa: F401

        return True

    except ImportError:
        return False


if _has_rich():
    from rich import print
    from rich.markup import escape


# The technique used in the log() function only applies to CPython, because
# it uses the inspect module to walk the call stack.
def log(
    msg,
    verbose=1,
    object_classes=(BaseEstimator,),
    stack_level=1,
    msg_level=1,
    with_traceback=False,
):
    """Display a message to the user, depending on the verbosity level.

    This function allows to display some information that references an object
    that is significant to the user, instead of a internal function. The goal
    is to make user's code as simple to debug as possible.

    Parameters
    ----------
    msg : str
        Message to display.

    verbose : int, default=1
        Current verbosity level. Message is displayed if this value is greater
        or equal to msg_level.

    object_classes : tuple of type, default=(BaseEstimator, )
        Classes that should appear to emit the message.

    stack_level : int, default=1
        If no object in the call stack matches object_classes, go back that
        amount in the call stack and display class/function name thereof.

    msg_level : int, default=1
        Verbosity level at and above which message should be displayed to the
        user. Most of the time this parameter can be left unchanged.

    Notes
    -----
    This function does tricky things to ensure that the proper object is
    referenced in the message. If it is called e.g. inside a function that is
    called by a method of an object inheriting from any class in
    object_classes, then the name of the object (and the method) will be
    displayed to the user. If several matching objects exist in the call
    stack, the highest one is used (first call chronologically), because this
    is the one which is most likely to have been written in the user's script.

    """
    if verbose < msg_level:
        return
    stack = inspect.stack()
    object_frame = None
    object_self = None
    for f in reversed(stack):
        frame = f[0]
        current_self = frame.f_locals.get("self", None)
        if isinstance(current_self, object_classes):
            object_frame = frame
            func_name = f[3]
            object_self = current_self
            break

    if object_frame is None:  # no object found: use stack_level
        if stack_level >= len(stack):
            func_name = "<top_level>"
        else:
            object_frame, _, _, func_name = stack[stack_level][:4]
            object_self = object_frame.f_locals.get("self", None)

    if object_self is not None:
        func_name = f"{object_self.__class__.__name__}.{func_name}"

    if _has_rich():
        print(f"[blue]\\[{func_name}][/blue] {escape(msg)}")
    else:
        print(f"[{func_name}] {msg}")

    if with_traceback:
        traceback.print_exc()


def compose_err_msg(msg, **kwargs):
    """Append key-value pairs to msg, for display. # noqa: D301.

    Parameters
    ----------
    msg : string
        Arbitrary message.

    kwargs : dict, optional
        Arbitrary dictionary.

    Returns
    -------
    updated_msg : string
        msg, with "key: value" appended. Only string values are appended.

    Example
    -------
    >>> compose_err_msg('Error message with arguments...', arg_num=123, \
        arg_str='filename.nii', arg_bool=True)
    'Error message with arguments...\\narg_str: filename.nii'
    >>>

    """
    updated_msg = msg
    for k, v in sorted(kwargs.items()):
        if isinstance(v, str):  # print only str-like arguments
            updated_msg += f"\n{k}: {v}"

    return updated_msg
