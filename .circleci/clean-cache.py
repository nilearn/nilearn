#! /usr/bin/env python

import os
from datetime import datetime as dt


'nilearn/.circleci' in __file__
timestamp_dirpath = os.path.dirname(__file__)
timestamp_filename = 'manual-cache-timestamp'
timestamp_filepath = os.path.join(timestamp_dirpath, timestamp_filename)
utc_now_timestamp = dt.utcnow()
with open(timestamp_filepath, 'w') as write_obj:
    write_obj.write(str(utc_now_timestamp))
