class BlockWarning(UserWarning):
    """A warning for when block modification operations used on incompatible OS."""


class ChannelClosedError(IOError):
    """An error to throw when the channel has closed from either end or error."""


class JSONError(RuntimeError):
    """Another JSONError."""
