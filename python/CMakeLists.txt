cmake_minimum_required(VERSION 3.16)
project(muse_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# The shared C++ engine lives in the R package's src/ (one directory up from
# python/).  The Python binding includes those headers verbatim -- the same
# files the R front-end compiles.
set(MUSE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../src")

pybind11_add_module(_musecore "${MUSE_SRC}/python/musecpp2py.cpp")

target_include_directories(_musecore PRIVATE "${MUSE_SRC}")
target_compile_definitions(_musecore PRIVATE MUSE_PYTHON_BUILD ARMA_DONT_USE_WRAPPER)

# ==============================================================================
# Armadillo headers (header-only; ARMA_DONT_USE_WRAPPER)
# ==============================================================================
# Linux/macOS build against the system package (apt/EPEL -> /usr/include;
# Homebrew -> /opt/homebrew or /usr/local, outside the default search path).
# Windows has no system Armadillo and we deliberately do NOT use vcpkg (its
# openblas + lapack-reference chain fails to build under vcpkg-gfortran /
# LLVMFlang), so fall back to fetching the header-only source.
if(APPLE)
  find_program(BREW_EXECUTABLE brew)
  if(BREW_EXECUTABLE)
    execute_process(COMMAND ${BREW_EXECUTABLE} --prefix
                    OUTPUT_VARIABLE HOMEBREW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
  endif()
endif()

find_path(ARMADILLO_INCLUDE_DIR
          NAMES armadillo
          PATHS "${HOMEBREW_PREFIX}/include" "/opt/homebrew/include"
                "/usr/local/include" "/usr/include")

if(ARMADILLO_INCLUDE_DIR)
  message(STATUS "Found system Armadillo headers: ${ARMADILLO_INCLUDE_DIR}")
else()
  message(STATUS "System Armadillo not found; fetching header-only source")
  include(FetchContent)
  # SOURCE_SUBDIR points at include/ (no CMakeLists.txt there), so
  # FetchContent_MakeAvailable downloads + extracts but skips the
  # add_subdirectory step -- we only want the headers, not Armadillo's
  # own CMake (which would try to find/build a BLAS/LAPACK wrapper).
  FetchContent_Declare(
    armadillo_src
    URL https://gitlab.com/conradsnicta/armadillo-code/-/archive/15.2.x/armadillo-code-15.2.x.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(armadillo_src)
  set(ARMADILLO_INCLUDE_DIR "${armadillo_src_SOURCE_DIR}/include")
  # The pristine source config.hpp keeps the wrapper on and LAPACK/BLAS
  # commented out; we use the headers directly, so enable BLAS/LAPACK calls
  # explicitly (the system packages above ship these already enabled).
  target_compile_definitions(_musecore PRIVATE ARMA_USE_BLAS ARMA_USE_LAPACK)
endif()
target_include_directories(_musecore PRIVATE "${ARMADILLO_INCLUDE_DIR}")

# ==============================================================================
# BLAS / LAPACK
# ==============================================================================
# Windows: consume the OpenMathLib pre-built OpenBLAS release
# (libopenblas.dll ships BLAS *and* LAPACK in one library, no Fortran compiler
# needed at our build time).  The CI step extracts the zip and sets
# -DOPENBLAS_HOME=C:/openblas; point the link at the MinGW import lib
# (libopenblas.dll.a -- MSVC's linker accepts COFF import libs).  delvewheel
# then bundles libopenblas.dll into the wheel.  Linux/macOS use module-mode
# find_package against the system openblas/lapack.
if(WIN32)
  if(NOT DEFINED OPENBLAS_HOME AND DEFINED ENV{OPENBLAS_HOME})
    set(OPENBLAS_HOME $ENV{OPENBLAS_HOME})
  endif()
  if(OPENBLAS_HOME)
    find_library(OPENBLAS_IMPLIB
                 NAMES libopenblas.dll.a libopenblas.lib openblas
                 PATHS "${OPENBLAS_HOME}/lib" NO_DEFAULT_PATH)
    find_path(OPENBLAS_INCLUDE_DIR
              NAMES openblas_config.h cblas.h
              PATHS "${OPENBLAS_HOME}/include" NO_DEFAULT_PATH)
    if(OPENBLAS_IMPLIB)
      set(BLAS_FOUND TRUE)
      set(LAPACK_FOUND TRUE)
      set(BLAS_LIBRARIES   ${OPENBLAS_IMPLIB})
      set(LAPACK_LIBRARIES ${OPENBLAS_IMPLIB})
      if(OPENBLAS_INCLUDE_DIR)
        target_include_directories(_musecore PRIVATE "${OPENBLAS_INCLUDE_DIR}")
      endif()
      message(STATUS "Found OpenMathLib OpenBLAS (BLAS + LAPACK): ${OPENBLAS_IMPLIB}")
    endif()
  endif()
  if(NOT BLAS_FOUND)
    message(FATAL_ERROR
      "OpenBLAS not found on Windows.  The wheel build expects the OpenMathLib "
      "pre-built OpenBLAS zip extracted to OPENBLAS_HOME -- see "
      "CIBW_BEFORE_ALL_WINDOWS in .github/workflows/build-release-wheels.yml.")
  endif()
endif()

if(NOT BLAS_FOUND)
  find_package(BLAS)
  find_package(LAPACK)
endif()

if(LAPACK_FOUND AND BLAS_FOUND)
  target_link_libraries(_musecore PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message(STATUS "BLAS/LAPACK not found via find_package; linking -llapack -lblas")
  target_link_libraries(_musecore PRIVATE lapack blas)
endif()

install(TARGETS _musecore DESTINATION muse)
