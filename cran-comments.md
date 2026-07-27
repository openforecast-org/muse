---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "27 July 2026"
output: html_document
---

## Submission

This is the CRAN submission of `muse` (version 0.1.1).

## Test environments

* local: Ubuntu Linux, R 4.6.1
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

- Windows Server 2022 x64 (build 26100), R 4.6.1
- MacOS macOS Sequoia 15.7.7, R 4.6.1
- Ubuntu 24.04.4 LTS, R 4.6.1


## Downstream dependencies

There are currently no reverse dependencies.
