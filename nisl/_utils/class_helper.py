from sets import Set
import inspect


def get_params(_class, _object, ignore=None):

    _ignore = Set(['memory', 'memory_level', 'verbose', 'copy'])
    if ignore is not None:
        _ignore.update(ignore)

    # params is a dictionary
    params = _class.get_params(_object)

    for i in _ignore:
        if i in params:
            params.pop(i)

    for p in params:
        if hasattr(_object, p):
            params[p] = getattr(_object, p)

    return params


def retrieve_scope():
    try:
        caller_frame = inspect.currentframe().f_back.f_back
        if 'self' in caller_frame.f_locals:
            caller_name = caller_frame.f_locals['self'].__class__.__name__
            caller_name = '%s.%s' % (caller_name,
                                    caller_frame.f_code.co_name)
            #caller_name = caller_frame.f_code.co_name
            return caller_name
    except Exception:
        return 'Unknown'
