CircleCI is used to build our documentations and tutorial examples.
CircleCI's cache stores previously built documentation and 
only rebuild any changes, instead of rebuilding the documentation from scratch. 
This saves a lot of time.

Occasionally, some changes necessitate rebuilding the documentation from scratch,
either to see the full effect of the changes 
or because the cached builds are raising some error.

To run a new CircleCI build from the beginning, without using the cache:

1. Run the script `clean-cache.py`.
2. Commit the change (with a clear message).
3. Push the commit.
