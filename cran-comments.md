---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "26 June 2026"
output: html_document
---

## Update

This update addresses the two issues flagged in the previous CRAN review:

* **Missing `\value`** -- the documentation for the exported S3 methods
  (`man/pts-methods.Rd`) now includes a `\value` section describing what each
  method returns.

* **Modifying the global environment** -- the simulation code no longer leaves
  the user's `.Random.seed` altered.  When a `seed` is supplied, the previous
  value of `.Random.seed` is saved and restored on exit, so `.GlobalEnv` is
  left unchanged.

## Submission

This is the initial CRAN submission of `muse` (version 0.1.0).

`muse` implements the PTS (Power / Trend / Seasonal) multiple-source-of-error
state-space model for time series analysis and forecasting.  The estimation
engine is written in C++ (via Rcpp / RcppArmadillo).

## Test environments

* local: Ubuntu Linux, R 4.5.2
* (please add win-builder / R-hub results here before submitting)

## R CMD check results

`R CMD check --as-cran` produces no ERRORs or WARNINGs.

Remaining NOTEs:

* **New submission** -- this is the first release of the package.

* **Installed size** -- the installed package is larger than 5 MB, almost
  entirely in `libs/` (the compiled shared object).  This is inherent to the
  templated Armadillo-based C++ state-space engine.  Debug symbols are already
  stripped on Linux via `src/Makevars` to keep the size down; the remaining
  size is the compiled engine itself.

All examples, tests, and vignettes run successfully.

## Github actions
Successful checks for:

- Windows latest release with latest R
- MacOS 15.7.3 with latest R
- Ubuntu 24.04.4 LTS with latest R

## R-hub
Successful checks for:

- Windows Server 2022 x64 (build 26100), R 4.5.0
- MacOS macOS Sequoia 15.7.7, R 4.5.0
- Ubuntu 24.04.4 LTS, R 4.5.0


## Downstream dependencies

There are currently no reverse dependencies.
