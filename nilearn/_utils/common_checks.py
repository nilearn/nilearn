"""
Utilities to check type, value or format of common parameters used everywhere
in the codebase.

"""
#Author: Virgile Fritsch, Jun. 2014, <virgile.fritsch@inria.fr>
import sklearn.externals.joblib as joblib


def check_n_jobs(n_jobs):
    """Check and adjust the number of CPUs that can work in parallel.

    Parameters
    ----------
    n_jobs : int,
      Number of parallel workers, specified according to joblib's conventions:
      If 0 is provided, all CPUs are used.
      A negative number indicates that all the CPUs except (|n_jobs| - 1) ones
      will be used.

    Returns
    -------
    n_jobs : int,
      Actual number of CPUs that will be used according to their availability.

    """
    if n_jobs == 0:  # invalid according to joblib's conventions
        raise ValueError("'n_jobs == 0' is not a valid choice. "
                         "Please provide a positive number of CPUs, or -1 "
                         "for all CPUs, or a negative number (-i) for "
                         "'all but (i-1)' CPUs (joblib conventions).")
    elif n_jobs < 0:
        n_jobs = max(1, joblib.cpu_count() + n_jobs + 1)
    else:
        n_jobs = min(n_jobs, joblib.cpu_count())

    return n_jobs
