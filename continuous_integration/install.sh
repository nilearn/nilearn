#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to be in control of the numpy version
    # we are using for example through apt-get install
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose pytest
}

echo_requirements_string() {
    # Echo a requirement string for example
    # "pip nose python='2.7.3 scikit-learn=*". It has a hardcoded
    # list of possible packages to install and looks at _VERSION
    # environment variables to know whether to install a given package and
    # if yes which version to install. For example:
    #   - for numpy, NUMPY_VERSION is used
    #   - for scikit-learn, SCIKIT_LEARN_VERSION is used
    TO_INSTALL_ALWAYS="pip nose pytest"
    REQUIREMENTS="$TO_INSTALL_ALWAYS"
    TO_INSTALL_MAYBE="numpy scipy matplotlib scikit-learn pandas flake8 lxml joblib"
    for PACKAGE in $TO_INSTALL_MAYBE; do
        # Capitalize package name and add _VERSION
        PACKAGE_VERSION_VARNAME="${PACKAGE^^}_VERSION"
        # replace - by _, needed for scikit-learn for example
        PACKAGE_VERSION_VARNAME="${PACKAGE_VERSION_VARNAME//-/_}"
        # dereference $PACKAGE_VERSION_VARNAME to figure out the
        # version to install
        PACKAGE_VERSION="${!PACKAGE_VERSION_VARNAME}"
        if [[ -n "$PACKAGE_VERSION" ]]; then
            if [[ "$PACKAGE_VERSION" == "*" ]]; then
                REQUIREMENTS="$REQUIREMENTS $PACKAGE"
            else
                REQUIREMENTS="$REQUIREMENTS $PACKAGE==$PACKAGE_VERSION"
            fi
        fi
    done
    echo $REQUIREMENTS
}

create_new_travisci_env() {
    REQUIREMENTS=$(echo_requirements_string)
    pip install $PIP_FLAGS ${REQUIREMENTS}
    pip install pytest pytest-cov

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        pip install mkl
    fi
}

if [[ "$DISTRIB" == "neurodebian" ]]; then
    create_new_venv
    pip install nose-timer
    bash <(wget -q -O- http://neuro.debian.net/_files/neurodebian-travis.sh)
    sudo apt-get install -qq python-scipy python-nose python-nibabel python-sklearn python-joblib

elif [[ "$DISTRIB" == "travisci" ]]; then
    create_new_travisci_env
    pip install nose-timer
    # Note: nibabel is in setup.py install_requires so nibabel will
    # always be installed eventually. Defining NIBABEL_VERSION is only
    # useful if you happen to want a specific nibabel version rather
    # than the latest available one.
    if [[ -n "$NIBABEL_VERSION" ]]; then
        pip install nibabel=="$NIBABEL_VERSION"
    fi

else
    echo "Unrecognized distribution ($DISTRIB); cannot setup CI environment."
    exit 1
fi

pip install psutil memory_profiler

if [[ "$COVERAGE" == "true" ]]; then
    pip install codecov
fi

# numpy not installed when skipping the tests so we do not want to run
# setup.py install
if [[ "$SKIP_TESTS" != "true" ]]; then
    pip install $PIP_FLAGS .
fi
