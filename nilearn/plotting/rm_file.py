"""
Remove a file after a certain time. This is run in a subprocess
by nilearn.plotting.html_surface.SurfaceView to remove the temporary
file it uses to open a plot in a web browser.

"""
import os
import time
import warnings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('n_seconds', type=float)
    args = parser.parse_args()

    time.sleep(args.n_seconds)
    if os.path.isfile(args.file_name):
        try:
            os.remove(args.file_name)
        except Exception as e:
            warnings.warn('failed to remove {}:\n{}'.format(args.file_name, e))
