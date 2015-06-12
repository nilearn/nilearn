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
    def __init__(self, dim_required, dim_file, dim_list):
        self.dim_required = dim_required
        self.dim_file = dim_file
        self.dim_list = dim_list

        super(DimensionError, self).__init__()

    def incr(self):
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
