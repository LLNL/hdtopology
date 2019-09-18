#!/bin/bash

# very verbose
set -x

# stop on error
set -e

# stop of the first command that fails with pipe (|)
set -o pipefail

# cmake flags
export PYTHON_VERSION=${PYTHON_VERSION:-3.7.4}

# assume miniconda installed here
export MINICONDA_ROOT=${HOME}/miniconda${PYTHON_VERSION:0:1}

if [ "$(uname)" == "Darwin" ]; then
    OSX=1
fi

# INSTALL_CONDA=1

# install conda
if (( INSTALL_CONDA == 1 )) ; then
    # install Miniconda
    if [ ! -d  ${MINICONDA_ROOT} ]; then
        pushd ${HOME}
        if (( OSX == 1 )) ; then
            __name__=MacOSX-x86_64
        else
            __name__=Linux-x86_64
        fi

        url="https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-${__name__}.sh"
        curl -fsSL --insecure ${url} -o miniconda.sh
        chmod +x miniconda.sh
        ./miniconda.sh -b
        popd
    fi

    export PATH=${MINICONDA_ROOT}/bin:$PATH
    conda config  --set changeps1 no --set anaconda_upload no
    conda update  --yes conda
    conda install --yes -q conda-build anaconda-client

fi

# activate
export PATH=${MINICONDA_ROOT}/bin:$PATH
conda create  --yes --force --name hdtopology-conda python=${PYTHON_VERSION}
source ${MINICONDA_ROOT}/etc/profile.d/conda.sh
# see https://github.com/conda/conda/issues/8072
eval "$(conda shell.bash hook)"
conda activate hdtopology-conda
conda install --yes numpy

# see conda/hdtopology/build.sh
if (( 1 == 1 )) ; then
    # pushd ./conda
    conda-build --python=${PYTHON_VERSION} -q hdtopology
    conda install --yes -q --use-local hdtopology
    # popd
fi

# test
if (( 1 == 1 )) ; then
    pushd $(python -m hdtopology dirname)
    python utilities/testWriteSummaryTopology.py
    python utilities/testReadSummaryTopology.py
    popd
fi

# the deploy happens here for the top-level build
if [[ "${TRAVIS_TAG}" != "" ]] ; then
    CONDA_BUILD_FILENAME=$(find ${MINICONDA_ROOT}/conda-bld -iname "hdtopology*.tar.bz2")
    echo "Doing deploy to anaconda ${CONDA_BUILD_FILENAME}..."
    anaconda -t ${ANACONDA_TOKEN} upload "${CONDA_BUILD_FILENAME}"
fi

echo "hdtopology build_conda.sh finished"
