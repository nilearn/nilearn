from __future__ import print_function
import sys

DEPENDENCIES = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'nibabel']


def print_package_version(package_name, indent='  '):
    try:
        package = __import__(package_name)
        version = package.__version__
    except ImportError:
        version = 'not installed'
    except AttributeError:
        version = None

    print('{0}{1}: {2}'.format(indent, package_name, version))

if __name__ == '__main__':
    print('=' * 80)
    print('Python', sys.version, '\n')

    print('Dependencies versions')
    for package_name in DEPENDENCIES:
        print_package_version(package_name)
    print('=' * 80)
