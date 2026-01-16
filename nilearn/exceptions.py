"""Custom warnings and errors used across Nilearn."""

__all__ = [
    "AllVolumesRemovedError",
    "DimensionError",
    "MaskWarning",
    "MeshDimensionError",
    "NotImplementedWarning",
]

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


class NotImplementedWarning(UserWarning):
    """Custom warning to warn about not implemented features.

    .. nilearn_versionadded:: 0.13.0
    """


class MaskWarning(UserWarning):
    """Custom warning related to masks.

    .. nilearn_versionadded:: 0.13.0
    """


class DimensionError(TypeError):
    """Custom error type for dimension checking.

    This error is used in recursive calls in check_niimg to keep track of the
    dimensionality of the data. Its final goal it to generate a user friendly
    message.

    Parameters
    ----------
    file_dimension : :obj:`int`
        Indicates the dimensionality of the bottom-level nifti file.

    required_dimension : :obj:`int`
        The dimension the nifti file should have.

    msg_about_samples : :obj:`bool`
        If set to True,
        when the error message will mention the number of expected samples
        rather than dimensions.
        This is useful when working with surface images.

    """

    def __init__(
        self, file_dimension, required_dimension, msg_about_samples=False
    ):
        self.file_dimension = file_dimension
        self.required_dimension = required_dimension
        self.stack_counter = 0
        self.msg_about_samples = msg_about_samples

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

        def unit(n):
            unit = f"{n} sample" if self.msg_about_samples else f"{n}D"
            if self.msg_about_samples and n > 1:
                unit += "s"
            return unit

        expected_dim = self.required_dimension + self.stack_counter
        total_file_dim = f" ({unit(self.file_dimension + self.stack_counter)})"
        return (
            "Input data has incompatible dimensionality: "
            f"Expected{' dimension is' if not self.msg_about_samples else ''} "
            f"{unit(expected_dim)} and you provided a "
            f"{'list of ' * self.stack_counter}"
            f"{unit(self.file_dimension)} "
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


class MeshDimensionError(ValueError):
    """Exception raised when meshes have incompatible dimensions."""

    def __init__(self, msg="Meshes have incompatible dimensions."):
        super().__init__(msg)

    def __str__(self):
        return f"[MeshDimensionError] {self.args[0]}"
