class MissingConfound(Exception):
    """
    Exception raised when failing to find params in the confounds.

    Parameters
    ----------
    params : list of missing params, default=[]
    keywords: list of missing keywords, default=[]
    """

    def __init__(self, params=None, keywords=None):
        """Set missing parameters and keywords."""
        self.params = params or []
        self.keywords = keywords or []

class AllVolumesRemoved(Exception):
    """
    Exception raised when all volumes have either motion outliers or are non-steady-state volumes (`sample_mask` is an empty array).
    """
    def __init__(self):
        super().__init__("All volumes are either outliers or non-steady-state. The size of the sample mask is 0.")
    
    def __str__(self):
        return f"[AllVolumesRemoved Error] {self.args[0]}"
