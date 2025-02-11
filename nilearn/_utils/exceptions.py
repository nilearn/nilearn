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
    ValueError,
)


class DimensionError(TypeError):
    """Custom error type for dimension checking.

    This error is used in recursive calls in check_niimg to keep track of the
    dimensionality of the data. Its final goal it to generate a user friendly
    message.

    Parameters
    ----------
    file_dimension : integer
        Indicates the dimensionality of the bottom-level nifti file.

    required_dimension : integer
        The dimension the nifti file should have.

    """

    def __init__(self, file_dimension, required_dimension):
        self.file_dimension = file_dimension
        self.required_dimension = required_dimension
        self.stack_counter = 0

        super().__init__()

    def increment_stack_counter(self):
        """Increments the counter of recursive calls.

        Called when the error is caught and re-raised to count the
        number of recursive calls, ie the number of dimensions added by
        imbrication in lists.

        """
        self.stack_counter += 1

    @property
    def message(self):
        """Format error message."""
        expected_dim = self.required_dimension + self.stack_counter
        total_file_dim = f" ({self.file_dimension + self.stack_counter}D)"
        return (
            "Input data has incompatible dimensionality: "
            f"Expected dimension is {expected_dim}D and you provided a "
            f"{'list of ' * self.stack_counter}{self.file_dimension}D "
            f"image{'s' * (self.stack_counter > 0)}"
            f"{total_file_dim * (self.stack_counter > 0)}. "
            "See https://nilearn.github.io/stable/manipulating_images/"
            "input_output.html."
        )

    def __str__(self):
        return self.message


class AllVolumesRemovedError(Exception):
    """Warns the user if no volumes were kept.

    Exception is raised when all volumes are scrubbed, i.e.,
    `sample_mask` is an empty array.
    """

    def __init__(self):
        super().__init__(
            "The size of the sample mask is 0. "
            "All volumes were marked as motion outliers "
            "can not proceed. "
        )

    def __str__(self):
        return f"[AllVolumesRemoved Error] {self.args[0]}"
