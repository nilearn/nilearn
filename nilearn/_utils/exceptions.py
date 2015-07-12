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

    file_dimension: integer
        Indicates the dimensonality of the bottom-level nifti file

    required_dimension: integer
        The dimension the nifti file should have
    """
    def __init__(self, file_dimension, required_dimension):
        self.file_dimension = file_dimension
        self.required_dimension = required_dimension
        self.stack_counter = 0

        super(DimensionError, self).__init__()

    def increment_stack_counter(self):
        """Increments the counter of recursive calls.

        Called when the error is catched and re-raised to count the
        number of recursive calls, ie the number of dimensions added by
        imbrication in lists.
        """
        self.stack_counter += 1

    @property
    def message(self):
        message = (
                "Data must be a %iD Niimg-like object but you provided a "
                "%s%iD image%s. "
                "See http://nilearn.github.io/building_blocks/"
                "manipulating_mr_images.html#niimg." % (
                    self.required_dimension + self.stack_counter,
                    "list of " * self.stack_counter,
                    self.file_dimension,
                    "s" * (self.stack_counter != 0)
                )
        )
        return message

    def __str__(self):
        return self.message
