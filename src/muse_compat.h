// muse_compat.h - cross-front-end compatibility shims.
//
// The muse engine (musecore.h and below) is written against Armadillo +
// STL and is intended to compile under either the R front-end
// (musecpp2R.cpp, where the R C-API supplies Rprintf via RcppArmadillo)
// or a Python front-end (musecpp2py.cpp, built with -DMUSE_PYTHON_BUILD).
//
// The only R C-API symbol the engine relies on is Rprintf (verbose / ident
// output).  When building for Python we provide a stdout-backed shim so the
// engine headers link without R.  Under the R build this header is a no-op
// and Rprintf resolves to R's own implementation.
#ifndef MUSE_COMPAT_H
#define MUSE_COMPAT_H

#ifdef MUSE_PYTHON_BUILD
#include <cstdio>
#include <cstdarg>

inline int Rprintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int r = std::vfprintf(stdout, fmt, args);
    va_end(args);
    return r;
}
#endif  // MUSE_PYTHON_BUILD

#endif  // MUSE_COMPAT_H
