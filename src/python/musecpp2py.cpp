// musecpp2py.cpp - Python (pybind11) binding for the muse engine.
//
// Mirror of musecpp2R.cpp: it is the *only* Python-side file that knows
// about pybind11.  It converts numpy arrays / scalars into a MuseInputs
// struct, calls runMuseCommand(), and packs the resulting MuseOutputs into
// a Python dict.  All dispatch logic lives in musecore.h and is shared
// verbatim with the R front-end.
//
// Built with -DMUSE_PYTHON_BUILD so muse_compat.h supplies the Rprintf
// shim (the engine's only R C-API dependency).

#ifndef MUSE_PYTHON_BUILD
#define MUSE_PYTHON_BUILD 1
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <armadillo>
#include <string>
#include <vector>

#include "../musecore.h"

namespace py = pybind11;

// ---- numpy <-> Armadillo helpers (copying; sizes are small) -------------
static arma::vec np_to_vec(const py::array_t<double>& a) {
    auto buf = a.request();
    arma::vec v(static_cast<arma::uword>(buf.size));
    auto r = a.unchecked<1>();
    for (py::ssize_t i = 0; i < buf.size; ++i) v(i) = r(i);
    return v;
}

// u arrives as (k rows x n cols), matching the engine's orientation.
static arma::mat np_to_mat(const py::array_t<double>& a) {
    auto buf = a.request();
    if (buf.ndim != 2)
        throw std::runtime_error("u must be a 2-D array (k x n)");
    arma::uword nr = static_cast<arma::uword>(buf.shape[0]);
    arma::uword nc = static_cast<arma::uword>(buf.shape[1]);
    arma::mat m(nr, nc);
    auto r = a.unchecked<2>();
    for (arma::uword i = 0; i < nr; ++i)
        for (arma::uword j = 0; j < nc; ++j)
            m(i, j) = r(i, j);
    return m;
}

static py::array_t<double> vec_to_np(const arma::vec& v) {
    py::array_t<double> a(static_cast<py::ssize_t>(v.n_elem));
    auto r = a.mutable_unchecked<1>();
    for (arma::uword i = 0; i < v.n_elem; ++i) r(i) = v(i);
    return a;
}

static py::array_t<double> mat_to_np(const arma::mat& m) {
    py::array_t<double> a({static_cast<py::ssize_t>(m.n_rows),
                           static_cast<py::ssize_t>(m.n_cols)});
    auto r = a.mutable_unchecked<2>();
    for (arma::uword i = 0; i < m.n_rows; ++i)
        for (arma::uword j = 0; j < m.n_cols; ++j)
            r(i, j) = m(i, j);
    return a;
}

// ---- the single entry point, mirroring UCompC ---------------------------
static py::dict ucomp(std::string command,
                      py::array_t<double> y,
                      py::array_t<double> u,
                      std::string model,
                      int h,
                      double lambda,
                      double outlier,
                      bool tTest,
                      std::string criterion,
                      py::array_t<double> periods,
                      py::array_t<double> rhos,
                      bool verbose,
                      bool stepwise,
                      py::array_t<double> p0,
                      bool armaFlag,
                      py::array_t<double> TVP,
                      double seas,
                      std::string trendOptions,
                      std::string seasonalOptions,
                      std::string irregularOptions,
                      int nsim,
                      unsigned seed,
                      double lambdaLower) {
    MuseInputs in;
    in.command          = command;
    in.y                = np_to_vec(y);
    in.u                = np_to_mat(u);
    in.model            = model;
    in.h                = h;
    in.lambda           = lambda;
    in.outlier          = outlier;
    in.tTest            = tTest;
    in.criterion        = criterion;
    in.periods          = np_to_vec(periods);
    in.rhos             = np_to_vec(rhos);
    in.verbose          = verbose;
    in.stepwise         = stepwise;
    in.p0               = np_to_vec(p0);
    in.armaFlag         = armaFlag;
    in.TVP              = np_to_vec(TVP);
    in.seas             = seas;
    in.trendOptions     = trendOptions;
    in.seasonalOptions  = seasonalOptions;
    in.irregularOptions = irregularOptions;
    in.nsim             = nsim;
    in.seed             = seed;
    in.lambdaLower      = lambdaLower;

    MuseOutputs out;
    runMuseCommand(in, out);

    py::dict d;
    if (out.isError) {
        d["model"] = std::string("error");
        return d;
    }

    // Always populated
    d["model"]           = out.model;
    d["yFor"]            = vec_to_np(out.yFor);
    d["h"]               = out.h;
    d["yForV"]           = vec_to_np(out.yForV);
    d["estimOk"]         = out.estimOk;
    d["lambda"]          = out.lambda;
    d["lambdaEstimated"] = out.lambdaEstimated;
    d["objFunValue"]     = out.objFunValue;
    d["periods"]         = vec_to_np(out.periods);
    d["rhos"]            = vec_to_np(out.rhos);
    d["p"]               = vec_to_np(out.p);
    d["p0"]              = vec_to_np(out.p0Return);
    d["parNames"]        = out.parNames;     // std::vector<std::string>
    d["ns"]              = out.ns;
    d["criteria"]        = vec_to_np(out.criteria);

    if (out.hasValidate) {
        d["table"]        = out.table;       // std::vector<std::string>
        d["v"]            = vec_to_np(out.v);
        d["covp"]         = mat_to_np(out.covp);
        d["coef"]         = vec_to_np(out.coef);
        d["typeOutliers"] = mat_to_np(out.typeOutliers);
    }
    if (out.hasComponents) {
        d["comp"]      = mat_to_np(out.comp);
        d["m"]         = out.m;
        d["compNames"] = out.compNames;
    }
    if (out.hasSimulate) {
        d["simPaths"] = mat_to_np(out.simPaths);
    }
    return d;
}

PYBIND11_MODULE(_musecore, mod) {
    mod.doc() = "muse C++ engine (PTS state-space) - pybind11 binding";
    mod.def("ucomp", &ucomp,
            "Run the muse engine. Mirrors the R-side .UCompC entry point.",
            py::arg("command"), py::arg("y"), py::arg("u"), py::arg("model"),
            py::arg("h"), py::arg("lambda"), py::arg("outlier"),
            py::arg("tTest"), py::arg("criterion"), py::arg("periods"),
            py::arg("rhos"), py::arg("verbose"), py::arg("stepwise"),
            py::arg("p0"), py::arg("armaFlag"), py::arg("TVP"),
            py::arg("seas"), py::arg("trendOptions"),
            py::arg("seasonalOptions"), py::arg("irregularOptions"),
            py::arg("nsim"), py::arg("seed"), py::arg("lambdaLower"));
}
