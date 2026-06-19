#!/usr/bin/env bash
# Phase-0 direct build of the muse pybind11 module (CMake packaging is Phase 1).
set -e
cd "$(dirname "$0")/.."
VENV=python/.venv/bin
PYINC=$($VENV/python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND=$($VENV/python -c "import pybind11; print(pybind11.get_include())")
NPINC=$($VENV/python -c "import numpy; print(numpy.get_include())")
EXT=$($VENV/python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT=python/src/muse/_musecore$EXT
g++ -O2 -std=gnu++17 -shared -fPIC -fvisibility=hidden \
    -DMUSE_PYTHON_BUILD -DARMA_DONT_USE_WRAPPER \
    -I"$PYINC" -I"$PYBIND" -I"$NPINC" -Isrc \
    src/python/musecpp2py.cpp -o "$OUT" \
    -llapack -lblas
echo "BUILT $OUT"
