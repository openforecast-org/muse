# Box-Cox lambda selection in `pts()` — decomposition + Guerrero

This document describes how `pts()` chooses the Box-Cox power parameter
`lambda` when the user leaves it unspecified (the `"Z"` power slot, e.g.
`model = "ZZZ"`), why it works that way, and the exact constraints that keep it
numerically safe — in particular the zero-series floor
`lambda >= log(2) / log(max(y))`, which is **muse-specific** and derived below.

The implementation lives in:

| | |
|---|---|
| R | `R/pts-internals.R :: .pts_guerrero_decomp_lambda()`, called from `R/pts.R` |
| Python | `python/src/muse/core/lambda_screen.py :: guerrero_decomp_lambda()` |

The two are line-for-line equivalent and are verified to return the same
`lambda` on the same series. **Any change to the screen must be made in both.**


## 1. What it does and where it sits

When the power slot is `"Z"`, `pts()` does **not** estimate `lambda` jointly
with the structural parameters by maximum likelihood. Instead it runs a fast,
model-free **screen** once, up front, fixes `lambda` to the chosen value, and
then runs the structural identification / estimation at that fixed `lambda`.
`nParam` is incremented by one to charge a degree of freedom for the screened
`lambda`.

Rationale: a joint BFGS over `(lambda, structural params)` is slow and the
`lambda`-profile is often flat or multimodal; a variance-stabilisation screen
on a cheap decomposition of the series is robust and ~free.

The Box-Cox transform used throughout (`src/boxcox.h`, `src/bcnorm.h`) is

```
            | log(y)                 lambda == 0      (singular point: log)
  g(y) =    | y                      lambda == 1      (singular point: identity)
            | (y^lambda - 1)/lambda  otherwise        (general)
```

Note the `lambda == 1` branch is the **identity** `g(y) = y`, not the textbook
`(y - 1)`. The exact-equality singular points keep the transform (and hence
AIC/BIC) continuous in `lambda`; do not reintroduce threshold shortcuts.


## 2. The Guerrero idea (variance stabilisation)

The goal of a power transform here is **variance stabilisation**: pick
`lambda` so the transformed series has roughly constant variance across the
range of its level (homoscedastic), which is the regime the Gaussian
state-space model assumes.

Guerrero (1993) formalises this. Suppose the standard deviation of `y` scales
with its mean as a power of the level. Split the series into groups; for group
`i` with mean `m_i` and standard deviation `s_i`, the delta method gives the
standard deviation of the transformed group as

```
  SD( g(y) )  ~=  |g'(m_i)| * s_i  =  m_i^(lambda - 1) * s_i .
```

Variance is stabilised when this is **constant across groups**. So the optimal
`lambda` is the one that minimises the dispersion of
`{ s_i * m_i^(lambda - 1) }` across groups, measured scale-free by the
**coefficient of variation**

```
  CV(lambda)  =  sd_i( s_i * m_i^(lambda-1) )  /  mean_i( s_i * m_i^(lambda-1) ) .
```

`lambda = argmin CV(lambda)` is the Guerrero estimate.


## 3. The muse recipe: decomposition-based groups

Classical Guerrero uses contiguous blocks and the raw block mean/SD. muse uses
a seasonal decomposition to define the groups and to separate "level" from
"dispersion", which behaves better on seasonal data. The procedure
(`.pts_guerrero_decomp_lambda`):

1. **Decompose.** `smooth::msdecompose(y, lags = m, type = "additive",
   smoother = "ma")` with `m` = the (largest) structural seasonal period. This
   fits a centred moving average of order `m` for the trend and an averaged
   seasonal pattern.

2. **Level.** Take the smoothed trend as the level: `mu_t = states[, 1]`.

3. **Blocks.** Form `R = floor(n / m)` non-overlapping blocks of length `m`.
   For block `i`:
   - `mu_b[i] = mean(mu_t over block i)` — the block-average **level**.
   - `sd_b[i] = sd( (y - mu_t) over block i )` — the within-block
     **dispersion**. Note `y - mu_t` removes the trend but **retains the
     seasonal swing**: for multiplicative seasonality the swing grows with the
     level, which is exactly the level-dependence Guerrero needs. (Subtracting
     the additive seasonal component would erase that signal.)

4. **Minimise.** `lambda = argmin CV(lambda)` with
   `CV(lambda) = sd(sd_b * mu_b^(lambda-1)) / mean(sd_b * mu_b^(lambda-1))`,
   via `stats::optimize` over the clipped range `[lower, upper]`.

Only blocks with finite, **strictly positive** `mu_b` and `sd_b > 0` are kept;
the screen needs at least 2 such blocks.


## 4. The clipped range `[0, 2]`

- **Lower bound 0.** Clipping at 0 forbids negative `lambda`. The inverse
  transform has a `-1/lambda` vertical asymptote that, for negative `lambda`,
  lets back-transformed forecasts run to `Inf` on outlier-contaminated series.
- **Upper bound 2.** The FPP-standard generous cap. Above 1 the inverse
  Box-Cox is sub-linear, so it never explodes; 2 is plenty of head-room.

The caller passes `lower = max(0, lambdaLower)`, where `lambdaLower` is the
R-side domain guard (see §6).


## 5. Fallback to `lambda = 1` (no transform)

The screen returns `1` (identity / raw scale) without running when it can't
produce a trustworthy estimate:

- the series is too short relative to the season (`n < 2m`), or `m < 2`;
- fewer than 4 finite observations;
- **any negative value** (`y^lambda` is complex for `y < 0` at fractional
  `lambda`) — see §6;
- `msdecompose` errors, or fewer than 2 usable blocks;
- the optimiser fails or returns a non-finite objective.


## 6. Zeros: domain, the relaxation, and the floor

### 6.1 Zeros are in the domain for `lambda > 0`

`g(y)` is finite at `y = 0` whenever `lambda > 0`:

```
  g(0) = (0^lambda - 1)/lambda = -1/lambda     (lambda > 0)
```

e.g. `sqrt(0) = 0`. The transform is undefined at `y = 0` **only** for
`lambda <= 0` (`log 0` at `lambda = 0`; `0^negative = Inf` for `lambda < 0`).
Negative `y` is undefined for any fractional `lambda` (complex result).

Accordingly:

- The **BCnorm density** (`bcnormLogDensityScalar`, `src/bcnorm.h`) rejects
  only the genuinely-undefined corner `y <= 0 && lambda <= 0`. `y == 0` with
  `lambda > 0` is a normal, finite observation. (`lambda == 1` admits all real
  `y`, which is needed for raw-scale models with negative data.)
- The **screen** disqualifies only on **negative** values, not zeros.
- The **R-side `lambdaLower` guard** (`R/pts.R`) is `1e-10` when the series
  contains zeros (so the joint path can never reach `lambda <= 0`), and `0`
  when it contains negatives (forcing identity).

A power transform in `(0, 1)` is in fact *ideal* for intermittent / zero-heavy
series: it both stabilises the variance of the spikes and — because the inverse
Box-Cox of a real number is non-negative — **guarantees non-negative
forecasts**. (On the raw scale, `lambda = 1`, an additive trend + seasonal
routinely forecasts below zero, which is nonsensical for counts.)

### 6.2 Why a floor is needed

The Guerrero CV in §3 is built **only from the positive block levels**
(`mu_b > 0`); the zeros never enter the objective. So the CV has no idea zeros
exist and, on a zero-heavy series, can happily drive `lambda` toward 0 to
maximally compress the spikes. But as `lambda -> 0`,

```
  g(0) = -1/lambda  ->  -Inf ,
```

so **every zero becomes an enormous negative outlier** that dominates the
transformed variance and collapses the fit. (Observed in practice: eurostat
tourism series 267 screened to `lambda = 4.8e-5`, mapping each zero to
`g(0) = -20670`, and the fit degenerated to an all-zero / divergent forecast.)

The optimiser is doing exactly what it was asked — it just isn't told that the
unmodelled zeros punish small `lambda`. The floor supplies that information.

### 6.3 Deriving the floor `lambda >= log(2) / log(max(y))`

We require the transformed zero to be **no more extreme than the transformed
maximum**:

```
  |g(0)|  <=  |g(y_max)| .
```

With `y_max > 1` (so `g(y_max) > 0`) and `lambda > 0`:

```
  |g(0)|      = 1 / lambda
  |g(y_max)|  = (y_max^lambda - 1) / lambda

  1/lambda  <=  (y_max^lambda - 1)/lambda          (the requirement)
  1         <=  y_max^lambda - 1                    (multiply by lambda > 0)
  y_max^lambda  >=  2
  lambda * log(y_max)  >=  log 2
  lambda  >=  log(2) / log(y_max) .
```

So the floor is **exactly** the `lambda` at which the zero and the maximum are
equally far from the centre of the transformed range. At the boundary
`lambda = log2/log(max)` we have `y_max^lambda = 2`, hence
`g(y_max) = (2-1)/lambda = 1/lambda = |g(0)|`: the transformed series spans a
symmetric `[-1/lambda, +1/lambda]` from the zeros up to the largest spike.
Intuitively: *don't compress so hard that the zeros stick out further than the
biggest observation.*

Implementation detail: the floor is applied only when zeros are present, and is
**capped at 1** — if `max(y) <= 2` (`log2/log(max) >= 1`) the series is
small-count and just stays on the raw scale. When the floor exceeds the
caller's `lambda_lower`, it raises it, so `optimize` searches `[floor, 2]`.

### 6.4 Status of the floor — this is a muse heuristic, not a cited method

**The floor is our own construction, not from the literature.** It is a
pragmatic, scale-free, data-driven bound chosen to keep the existing
Box-Cox + Guerrero path stable on exact zeros. The specific yardstick
("the maximum", the constant `2` from requiring `y_max^lambda >= 2`) is a
defensible choice, not a unique or canonical one — using the second-largest
value, a high quantile, or a different threshold than `2` would give a
similar-spirited but different bound.

The **textbook** ways to admit zeros (and negatives) into a power transform are
*different mechanisms* — they change the transform family rather than bounding
`lambda`:

- **Yeo & Johnson (2000)** — a power-transform family defined for all real `y`
  (including 0 and negatives) by construction, so no floor is needed.
- **Shifted Box-Cox** — transform `y + c` for a shift `c > 0` (discussed in
  Box & Cox 1964), making zeros strictly positive.

If muse later wants a properly-pedigreed zero treatment, switching the
zero-case to Yeo-Johnson or a shift would supersede this floor.


## 7. References

- Guerrero, V. M. (1993). *Time-series analysis supported by power
  transformations.* Journal of Forecasting, 12(1), 37-48. — the CV-minimisation
  screen of §2-§3 (assumes strictly positive data).
- Box, G. E. P., & Cox, D. R. (1964). *An analysis of transformations.* JRSS B,
  26(2), 211-252. — the Box-Cox transform and the shift variant.
- Yeo, I.-K., & Johnson, R. A. (2000). *A new family of power transformations to
  improve normality or symmetry.* Biometrika, 87(4), 954-959. — power transform
  for all real `y`.
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*
  (FPP). — the `[0, 2]` lambda convention.

The zero floor `lambda >= log(2)/log(max(y))` (§6.3) has **no reference**; it is
specific to muse.


## 8. Worked example — eurostat tourism series 267

Intermittent series (27 leading zeros, mostly zeros, rare spikes up to 7081):

| | model | lambda | logLik | forecast |
|---|---|---|---|---|
| before the zero fixes | `PTS(1,G,D)` | 1 (forced) | -430 | `[-101, 1906]` (negative) |
| after (holdout) | `PTS(0.13,N,D)` | 0.13 | -254 | `[0, 15]` |

The screen, freed to consider `lambda > 0`, picks a strong variance-stabilising
transform; the floor keeps it off the `lambda -> 0` cliff (it would otherwise
have gone to `0.078`'s neighbourhood or below on the full series); and the
square-root-ish inverse transform makes the forecast non-negative.
