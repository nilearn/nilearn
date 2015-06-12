AuthorizedException = (
        BufferError,
        ArithmeticError,
        AssertionError,
        AttributeError,
        EnvironmentError,
        EOFError,
        LookupError,
        MemoryError,
        ReferenceError,
        RuntimeError,
        SystemError,
        TypeError,
        ValueError
)


class DimensionError(TypeError):
    """Custom error type for dimension checking.

    This error is used in recursive calls in check_niimg to keep track of the
    dimensionality of the data. Its final goal it to generate a user friendly
    message.

    Parameters
    ----------

    dim_required: integer
        Indicates the dimensonality required for the data.

    dim_file: integer
        Dimensionality of the niimg located at the lowest level of the data.

    dim_list: integer
        Dimensions added by the imbrication of the data in lists.
    """
    def __init__(self, dim_required, dim_file, dim_list):
        self.dim_required = dim_required
        self.dim_file = dim_file
        self.dim_list = dim_list

        super(DimensionError, self).__init__()

    def incr(self):
        """Increments the dimension counters

        Typically called when the error is catched and re-raised to count the
        number of recursive calls, ie the number of dimensions added by
        imbrication in lists.
        """
        self.dim_required += 1
        self.dim_list += 1

    def _get_message(self):
        message = (
                "Data must be a %iD Niimg-like object but you provided a "
                "%s%iD image%s. "
                "See http://nilearn.github.io/building_blocks/"
                "manipulating_mr_images.html#niimg." % (
                    self.dim_required,
                    "list of " * self.dim_list,
                    self.dim_file,
                    "s" * (self.dim_list != 0)
                )
        )
        return message

    def _set_message(self, message):
        pass

    def __str__(self):
        return self.message

    message = property(_get_message, _set_message)
