#!/usr/bin/env Rscript
# Loss-surface exploration for PTS(1, G, T) on AirPassengers.
#
# Goal: answer "is the pathological fit at sigma2_irreg ~ 5e-5 the
# global ML maximum, or one of several local basins?" via multi-start
# optimisation with diverse initial conditions.
#
# Run from package root:
#   Rscript experiments/loss_surface_1gt.R
#
# Outputs land in experiments/results/.

.libPaths(c("/home/config/Misc/R/x86_64-pc-linux-gnu-library/4.5", .libPaths()))
suppressMessages({
    library(smooth)
    library(muse)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

OUTDIR    <- "experiments/results"
PLOTSDIR  <- file.path(OUTDIR, "plots")
dir.create(OUTDIR,   showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTSDIR, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Baseline fit -----------------------------------------------

cat("[1/4] baseline pts(AirPassengers, model = '1GT')...\n")
m_star <- pts(AirPassengers, model = "1GT", h = 12, holdout = TRUE)
coef_star <- coef(m_star)
loss_star <- as.numeric(-2 * logLik(m_star))   # -2 * LLIK = the loss
cat(sprintf("    converged loss (-2 LLIK): %.4f\n", loss_star))
cat("    coef:\n");  print(round(coef_star, 8))

# ---- 2. Multi-start --------------------------------------------------

# Engine takes p0 on NATURAL scale (positive variances).  We sample
# uniformly in log-space across many orders of magnitude so we span
# from "essentially zero" (1e-9) through "huge" (1e+3).  Three
# variances: Level, Seas(All), Irregular.

N <- 500
log_var_range <- c(-9, 3)   # 1e-9 ... 1e3

set.seed(1L)
init_mat <- matrix(NA_real_, nrow = N, ncol = 3,
                   dimnames = list(NULL, c("Level", "Seas", "Irreg")))
final_mat <- init_mat
final_loss <- numeric(N)
final_conv <- character(N)

cat(sprintf("[2/4] multi-start N=%d... (~%d s)\n", N, round(N * 0.25)))
t0 <- proc.time()[3]
for (i in seq_len(N)){
    p_init <- 10 ^ runif(3, log_var_range[1], log_var_range[2])
    init_mat[i, ] <- p_init
    m_i <- try(pts(AirPassengers, model = "1GT", h = 12, holdout = TRUE,
                   B = p_init), silent = TRUE)
    if (inherits(m_i, "try-error")){
        if (i <= 3) cat("    [run", i, "] error:",
                        substr(attr(m_i, "condition")$message, 1, 200), "\n")
        final_loss[i] <- NA_real_
        final_conv[i] <- "error"
        next
    }
    final_loss[i]  <- as.numeric(-2 * logLik(m_i))
    final_mat[i, ] <- coef(m_i)
    final_conv[i]  <- m_i$estimOk %||% "ok"
    if (i %% 50 == 0)
        cat(sprintf("    %d / %d (elapsed: %.1f s)\n",
                    i, N, proc.time()[3] - t0))
}
cat(sprintf("    multi-start done in %.1f s\n", proc.time()[3] - t0))

# ---- 3. Cluster + report --------------------------------------------

cat("[3/4] clustering by final loss...\n")
ok        <- !is.na(final_loss)
n_failed  <- sum(!ok)
loss_ok   <- final_loss[ok]
final_ok  <- final_mat[ok, , drop = FALSE]
init_ok   <- init_mat[ok, , drop = FALSE]
conv_ok   <- final_conv[ok]

# Cluster: round losses to 0.01.  Each unique rounded value ~ one basin.
loss_round <- round(loss_ok, 2)
clusters   <- sort(unique(loss_round))
cluster_id <- match(loss_round, clusters)

cluster_summary <- data.frame(
    cluster      = seq_along(clusters),
    loss         = clusters,
    AIC          = clusters + 2 * 3,    # k = 3 for 1GT (Level, Seas, Irreg)
    n_runs       = tabulate(cluster_id),
    pct          = round(100 * tabulate(cluster_id) / sum(ok), 1)
)
# Representative final params per cluster
cluster_centroid <- t(sapply(seq_along(clusters), function(k){
    sel <- cluster_id == k
    apply(final_ok[sel, , drop = FALSE], 2, median)
}))
cluster_summary <- cbind(cluster_summary, cluster_centroid)
cluster_summary <- cluster_summary[order(cluster_summary$loss), ]

write.csv(data.frame(init = init_mat, final = final_mat,
                     loss = final_loss, conv = final_conv),
          file.path(OUTDIR, "multistart.csv"), row.names = FALSE)
write.csv(cluster_summary,
          file.path(OUTDIR, "clusters.csv"), row.names = FALSE)
cat(sprintf("    %d distinct loss clusters across %d successful runs (%d failed)\n",
            length(clusters), sum(ok), n_failed))

# ---- 4. Plots -------------------------------------------------------

cat("[4/4] plots...\n")

png(file.path(PLOTSDIR, "multistart_loss_hist.png"),
    width = 900, height = 500, res = 100)
op <- par(mar = c(5, 4, 4, 2))
hist(final_loss, breaks = 80, main = "Multi-start final loss (-2 LLIK)",
     xlab = "Loss", col = "lightgrey", border = "white")
abline(v = loss_star, col = "red", lwd = 2)
mtext(sprintf("Red line: default-init loss = %.2f", loss_star), side = 3,
      adj = 0, line = 0.2)
mtext(sprintf("%d clusters across %d runs (%d failed)",
              length(clusters), sum(ok), n_failed), side = 3,
      adj = 1, line = 0.2)
par(op)
dev.off()

# Scatter: final (log10 Level, log10 Irreg), coloured by cluster
png(file.path(PLOTSDIR, "multistart_basins.png"),
    width = 900, height = 700, res = 100)
op <- par(mar = c(5, 5, 4, 2))
cols <- c("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
          "#7f7f7f", "#bcbd22", "#e377c2", "#8c564b")
cluster_col <- cols[((cluster_id - 1L) %% length(cols)) + 1L]
plot(log10(final_ok[, "Level"]), log10(final_ok[, "Irreg"]),
     col = cluster_col, pch = 19, cex = 0.7,
     xlab = "log10 sigma2_Level (converged)",
     ylab = "log10 sigma2_Irregular (converged)",
     main = "Multi-start basins on (Level, Irregular)")
points(log10(coef_star["Level"]), log10(coef_star["Irregular"]),
       pch = 4, lwd = 3, cex = 2, col = "black")
legend("topright",
       legend = sprintf("loss=%.2f  (n=%d)",
                        cluster_summary$loss[seq_len(min(5, nrow(cluster_summary)))],
                        cluster_summary$n_runs[seq_len(min(5, nrow(cluster_summary)))]),
       col = cols[seq_len(min(5, nrow(cluster_summary)))],
       pch = 19, cex = 0.8, bg = "white")
par(op)
dev.off()

# ---- 5. REPORT.md ---------------------------------------------------

global_loss   <- min(loss_ok)
global_cluster <- cluster_summary[which.min(cluster_summary$loss), ]
default_global <- abs(loss_star - global_loss) < 0.05
top_clusters   <- head(cluster_summary, 20)
findings <- if (length(clusters) == 1L){
    sprintf("**Single cluster found.** The pathological fit at loss = %.2f is the unique ML maximum on this (data, model) pair.  No initialisation can escape it; fixes must change the objective (irregular floor, REML, penalisation).", global_loss)
} else if (default_global){
    sprintf("**Multi-modal landscape (%d distinct local minima).**  The pathological fit at loss = %.2f IS the global ML maximum -- no cluster has lower loss.  %d of %d successful runs (%.0f%%) reach it.  ML is doing its job correctly; the result is the genuine ML answer, and the fix has to come from changing the objective rather than the optimiser or its initialisation.", length(clusters), global_loss, global_cluster$n_runs, sum(ok), 100*global_cluster$n_runs/sum(ok))
} else {
    sprintf("**Multi-modal landscape (%d distinct local minima)** and the default-init optimum (loss = %.2f) is NOT the global.  Global is loss = %.2f, reached from %d runs.  Better initialisation could fix this inside ML.", length(clusters), loss_star, global_loss, global_cluster$n_runs)
}

REPORT <- sprintf('# Loss-surface exploration for PTS(1, G, T) on AirPassengers

## Findings

%s

## Setup

- Data: `AirPassengers` (monthly, 1949–1960), `h = 12`, `holdout = TRUE`.
- Model: `PTS(1, G, T)` = lambda = 1 (no Box-Cox), td trend (damped),
  equal seasonal (one variance for all harmonics), arma(0,0)
  irregular.
- Multi-start: %d random initial natural-scale variance vectors
  sampled log-uniformly in [10^%d, 10^%d] for Level, Seas, Irregular.

## Default-init optimum

| param      | value         |
|------------|---------------|
| Level      | %.4e |
| Seas(All)  | %.4e |
| Irregular  | %.4e |
| loss (-2 LLIK) | %.4f      |
| AIC        | %.4f      |

This is the "pathological" fit: residuals collapse to a near-constant
offset, sigma2_Irregular is ~5 orders of magnitude below sigma2_Level.

## Multi-start outcome

- **%d successful runs, %d failed**.
- **%d distinct loss clusters** (rounded to 0.01).
- Distribution of final losses:

%s

## Diagnosis

The number of distinct clusters answers the question "is the
pathological fit the global ML maximum or one of several local
basins?":

- If **1 cluster** at the pathological loss: the basin is the unique
  ML maximum on this (data, model) pair.  No initialisation strategy
  inside ML can escape it; the only fixes are an irregular-variance
  floor, REML, or some other change to the objective.
- If **several clusters** with the pathological as global: ML is
  multi-modal and the pathological is still the worst-case answer,
  but multi-start with diverse initials would surface alternative
  fits the user might prefer.
- If **several clusters** with a non-pathological global: the
  current single-start initialisation is the bug; better initials
  would suffice.

See `clusters.csv` for full per-cluster centroids.  See
`plots/multistart_basins.png` for the scatter of converged points in
the (sigma2_Level, sigma2_Irregular) plane.

## Files

- `multistart.csv`  - (init, final, loss, convergence flag) per run.
- `clusters.csv`    - cluster summary (loss, AIC, n_runs, centroid).
- `plots/multistart_loss_hist.png` - histogram of final losses with
  the default-init loss marked.
- `plots/multistart_basins.png`    - scatter of converged points
  coloured by cluster, default-init marked with X.
',
    findings,
    N, log_var_range[1], log_var_range[2],
    coef_star["Level"], coef_star["Seas(All)"], coef_star["Irregular"],
    loss_star, AIC(m_star),
    sum(ok), n_failed,
    length(clusters),
    paste(
        sprintf("  - loss = %.4f (AIC = %.4f), %d runs (%.1f%%)",
                top_clusters$loss, top_clusters$AIC,
                top_clusters$n_runs, top_clusters$pct),
        collapse = "\n")
)
writeLines(REPORT, file.path(OUTDIR, "REPORT.md"))

cat("\nDone.  Outputs in", OUTDIR, "\n")
cat("  - REPORT.md            (summary)\n")
cat("  - multistart.csv       (raw runs)\n")
cat("  - clusters.csv         (basin summary)\n")
cat("  - plots/*.png          (visualisations)\n")
