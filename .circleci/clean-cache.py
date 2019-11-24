#! /usr/bin/env python
"""
Flushes the cached docs for next CircleCI build.
"""
import os
from datetime import datetime as dt


def update_cache_timestamp(timestamp_filename):
    """ Updates the contents of the docs-cache-timestamp file
    with current timestamp.

    Returns
    -------
    None
    """
    timestamp_dirpath = os.path.dirname(__file__)
    timestamp_filepath = os.path.join(timestamp_dirpath, timestamp_filename)
    utc_now_timestamp = dt.utcnow()
    with open(timestamp_filepath, 'w') as write_obj:
        write_obj.write(str(utc_now_timestamp))


if __name__ == '__main__':
    update_cache_timestamp('docs-cache-timestamp')
    update_cache_timestamp('packages-cache-timestamp')
