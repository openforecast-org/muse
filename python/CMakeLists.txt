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

# Homebrew installs Armadillo's headers outside the compiler's default search
# path (/opt/homebrew on Apple Silicon, /usr/local on Intel), so <armadillo>
# is not found out of the box on macOS.  Add Homebrew's prefix.  (On Linux the
# Debian/EPEL packages land in /usr/include, already on the default path.)
if(APPLE)
  find_program(BREW_EXECUTABLE brew)
  if(BREW_EXECUTABLE)
    execute_process(COMMAND ${BREW_EXECUTABLE} --prefix
                    OUTPUT_VARIABLE HOMEBREW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
  endif()
  foreach(prefix "${HOMEBREW_PREFIX}" "/opt/homebrew" "/usr/local")
    if(EXISTS "${prefix}/include/armadillo")
      message(STATUS "Found Armadillo headers under ${prefix}/include")
      target_include_directories(_musecore PRIVATE "${prefix}/include")
      break()
    endif()
  endforeach()
endif()

# Armadillo is used header-only (system <armadillo>); it needs LAPACK + BLAS
# at link time (same as the R package's src/Makevars).
find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
  target_link_libraries(_musecore PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
else()
  message(STATUS "BLAS/LAPACK not found via find_package; linking -llapack -lblas")
  target_link_libraries(_musecore PRIVATE lapack blas)
endif()

install(TARGETS _musecore DESTINATION muse)
