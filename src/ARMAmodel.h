/*************************
 Stationary ARMA models with zero mean
Needs Armadillo
Needs SSpace.h
***************************/
struct ARMAinputs{
  // Free coef counts (= sum of arOrders / maOrders entries).
  int ar, ma;
  // Expanded polynomial degrees (= Σ arOrders[i] · armaLags[i]).
  // Size the companion state block; when arOrders / armaLags are empty
  // (legacy / non-seasonal callers) arDeg defaults to ar, maDeg to ma.
  int arDeg = 0, maDeg = 0;
  // Per-lag breakdown.  Length 1 for non-seasonal arma(p,q); length 2
  // for SARMA(p,q)(P,Q)_s with armaLags = [1, s].
  arma::ivec arOrders, maOrders, armaLags;
};
/**************************
 * Model CLASS stationary ARMA
 ***************************/
class ARMAmodel : public SSmodel{
  private:
    ARMAinputs dataARMA;
    int ns;
  public:
    ARMAmodel(SSinputs, int, int);
    ARMAmodel(SSinputs, arma::ivec arOrders, arma::ivec maOrders,
              arma::ivec armaLags);
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Convert non-invertible ma polynomial into invertible
void maInvert(vec&);
// Returns the parameters of an AR model from the PACF
void pacfToAr(vec&);
// Returns the PACF from the parameters of an AR model
void arToPacf(vec&);
// Returns stationary polynomial from an arbitrary one
void polyStationary(vec&);
// Inverse of polyStationary
void InvPolyStationary(vec&);
// Initialising matrices
void initMatricesArma(int, int, int&, SSmatrix&);
// Filling changing matrices
void armaMatrices(vec, SSmatrix*, void*);
// Filling changing matrices with true parameters
void armaMatricesTrue(vec, SSmatrix*, void*);
//#include ARMAmodel.cpp
/****************************************************
 // ARMA implementations for stationary ARMA models
 ****************************************************/
// Constructors
ARMAmodel::ARMAmodel(SSinputs data, int ar, int ma) : SSmodel(data){
  //int ns;

  // Initialising matrices
  initMatricesArma(ar, ma, ns, data.system);
  // Storing information
  this->inputs.system = data.system;
  this->dataARMA.ar = ar;
  this->dataARMA.ma = ma;
  this->dataARMA.arDeg = ar;
  this->dataARMA.maDeg = ma;
  if (ar == 0){
    this->inputs.exact = true;
  } else {
    this->inputs.exact = false;
  }
  this->inputs.userInputs = &this->dataARMA;
  // User function to fill the changing matrices
  this->inputs.userModel = armaMatrices;
  // Initializing parameters of ARMA model
  this->inputs.p0.zeros(ar + ma + 1);
  this->inputs.p0(0) = -1;
}
// Per-lag SARMA constructor.  arOrders / maOrders / armaLags are paired
// position-wise: arOrders[i] AR coefs at lag armaLags[i], similarly for MA.
// armaMatrices() consumes the same per-lag layout via the polyMult-based
// buildARMApoly().
ARMAmodel::ARMAmodel(SSinputs data,
                     arma::ivec arOrders, arma::ivec maOrders,
                     arma::ivec armaLags) : SSmodel(data){
  int arFree = (arOrders.n_elem > 0) ? (int)arma::sum(arOrders) : 0;
  int maFree = (maOrders.n_elem > 0) ? (int)arma::sum(maOrders) : 0;
  int arDeg  = 0, maDeg = 0;
  for (arma::uword b = 0; b < arOrders.n_elem; ++b)
    arDeg += arOrders(b) * armaLags(b);
  for (arma::uword b = 0; b < maOrders.n_elem; ++b)
    maDeg += maOrders(b) * armaLags(b);
  initMatricesArma(arDeg, maDeg, ns, data.system);
  this->inputs.system    = data.system;
  this->dataARMA.ar      = arFree;
  this->dataARMA.ma      = maFree;
  this->dataARMA.arDeg   = arDeg;
  this->dataARMA.maDeg   = maDeg;
  this->dataARMA.arOrders = arOrders;
  this->dataARMA.maOrders = maOrders;
  this->dataARMA.armaLags = armaLags;
  this->inputs.exact     = (arFree == 0);
  this->inputs.userInputs = &this->dataARMA;
  this->inputs.userModel  = armaMatrices;
  this->inputs.p0.zeros(arFree + maFree + 1);
  this->inputs.p0(0) = -1;
}
/*************************************************************
 * Implementation of auxiliar functions
 ************************************************************/
// Convert non-invertible ma polynomial into invertible
void maInvert(vec& maPoly){
  vec ma(maPoly.n_elem + 1);
  ma.row(0) = 1;
  ma.rows(1, maPoly.n_elem) = maPoly;
  unsigned int q = max(find(ma != 0));
  cx_vec maRoots;
  cx_double iRoot;
  ma = ma.rows(0, q);
  maRoots = roots(flipud(ma));
  uvec ind = find(abs(maRoots) < 1);
  cx_vec poly = zeros<cx_vec>(q + 1);
  poly.row(0) = 1;
  if (ind.n_elem > 0){
    // Some roots are not invertible
    maRoots(ind) = 1 / maRoots(ind);
    for (unsigned i = 0; i < q; i++){
      iRoot = as_scalar(maRoots.row(i));
      poly.rows(0, i + 1) -= join_vert(poly.rows(i + 1, i + 1), poly.rows(0, i) / iRoot);
    }
    maPoly = real(poly.rows(1, poly.n_elem - 1));
  }
}
// Returns the parameters of an AR model from the PACF
void pacfToAr(vec& PAR){
  // y(t) = PAR(1) * y(t - 1) + PAR(2) * y(t - 2) + ...
  // Monahan, JF (1984), A note on enforcing stationarity in ARMA models,
  // Biometrika, 71, 2, 403-404.
  vec par0 = PAR;
  for (unsigned int i = 0; i < PAR.n_elem - 1; i++){
    PAR(i + 1) = par0(i + 1);
    PAR.rows(0, i) = (PAR.rows(0, i) - PAR(i + 1) * flipud(PAR.rows(0, i)));
  }
}
// sampleACF: empirical autocorrelations rho_1..rho_K of a vector y.
// Uses the standard ratio of mean-centred autocovariances; the lag-0 term
// is divided out so rho_0 = 1 and is omitted from the return vector.
// Skips non-finite entries (the engine's BC-scale y may carry early NaNs
// from filter warm-up).
inline arma::vec sampleACF(const arma::vec& y, int maxLag){
    using arma::uword;
    arma::uvec ok = arma::find_finite(y);
    if (ok.n_elem < 2 || maxLag < 1)
        return arma::vec(std::max(maxLag, 0), arma::fill::zeros);
    arma::vec yc = y(ok) - arma::mean(y(ok));
    double gamma0 = arma::dot(yc, yc);
    if (!std::isfinite(gamma0) || gamma0 <= 0.0)
        return arma::vec(maxLag, arma::fill::zeros);
    int K = std::min<int>(maxLag, (int)yc.n_elem - 1);
    arma::vec rho(maxLag, arma::fill::zeros);
    for (int k = 1; k <= K; ++k){
        arma::vec a = yc.rows(k, yc.n_elem - 1);
        arma::vec b = yc.rows(0, yc.n_elem - 1 - k);
        rho(k - 1) = arma::dot(a, b) / gamma0;
    }
    return rho;
}

// sampleYWpacf: empirical partial autocorrelations phi_kk for k = 1..maxLag.
// Durbin-Levinson recursion run on the sample ACF — gives the Yule-Walker
// AR(k) lead coefficient at each level k.  This is what stats::pacf
// computes via ar.yw, just inlined so the engine doesn't need to call out.
inline arma::vec sampleYWpacf(const arma::vec& y, int maxLag){
    using arma::uword;
    if (maxLag < 1) return arma::vec();
    arma::vec rho = sampleACF(y, maxLag);
    arma::vec phi(maxLag, arma::fill::zeros);
    if (rho.n_elem == 0 || arma::any(arma::abs(rho) >= 1.0))
        return phi;
    // Durbin-Levinson on standardised autocovariances (= autocorrelations).
    arma::vec aPrev(maxLag, arma::fill::zeros);
    double v = 1.0;          // variance of order-0 process (normalised)
    for (int k = 0; k < maxLag; ++k){
        double num = rho(k);
        for (int j = 0; j < k; ++j) num -= aPrev(j) * rho(k - 1 - j);
        if (!std::isfinite(v) || std::abs(v) < 1e-12){
            phi.rows(k, maxLag - 1).zeros();
            break;
        }
        double phi_kk = num / v;
        phi(k) = phi_kk;
        arma::vec aNew(maxLag, arma::fill::zeros);
        for (int j = 0; j < k; ++j)
            aNew(j) = aPrev(j) - phi_kk * aPrev(k - 1 - j);
        aNew(k) = phi_kk;
        aPrev = aNew;
        v = v * (1.0 - phi_kk * phi_kk);
    }
    return phi;
}

// Returns the PACF from the parameters of an AR model
void arToPacf(vec& PAR){
  // y(t) = PAR(1) * y(t - 1) + PAR(2) * y(t - 2) + ...
  // Monahan, JF (1984), A note on enforcing stationarity in ARMA models,
  // Biometrika, 71, 2, 403-404.
  int j;
  for (int i = PAR.n_elem - 1; i > 0; i--){
    j = i - 1;
    PAR.rows(0, j) = (PAR.rows(0, j) + PAR(i) * flipud(PAR.rows(0, j)))
      / (1 - PAR(i) * PAR(i));
  }
}
// Returns stationary polynomial from an arbitrary one
void polyStationary(vec& PAR){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  vec limits(2);
  limits(0) = -0.98;
  limits(1) = 0.98;
  constrain(PAR, limits);
  pacfToAr(PAR);
  PAR = -PAR;
}
// Inverse of polyStationary
void invPolyStationary(vec& PAR){
  // (1 + PAR(1) * B + PAR(2) *B^2 + ...) y(t) = a(t)
  mat limits(PAR.n_elem, 2);
  limits.col(0).fill(-0.98);
  limits.col(1).fill(0.98);
  PAR = -PAR;
  arToPacf(PAR);
  unconstrain(PAR, limits);
}
// Initialising matrices.  The state-block size matches the EXPANDED polynomial
// degree — for non-seasonal ARMA that's just max(ar, ma + 1); for SARMA
// callers pass the convolved-polynomial degrees through arDeg / maDeg.
void initMatricesArma(int arDeg, int maDeg, int& ns, SSmatrix& model){
  ns = std::max(arDeg, maDeg + 1);
  model.T.zeros(ns, ns);
  if (ns > 1)
    model.T.diag(1) += 1;
  model.Gam = model.D = model.H = model.C = 0.0;
  model.Z.zeros(1, ns);
  model.Z(0, 0) = 1.0;
  model.R.zeros(ns, 1);
  model.R(0) = 1;
  model.Q = 0.0;
}
// Polynomial multiplication helper.  Lifted from
// smooth/src/headers/ssGeneral.h:56-70.  Operates on coefficient vectors that
// INCLUDE the leading 1 (so poly1[0] = poly2[0] = 1 for AR / MA polys).
// Returns the convolved polynomial, also leading-1.
inline arma::vec polyMult(const arma::vec& poly1, const arma::vec& poly2){
  int n1 = (int)poly1.n_elem - 1;
  int n2 = (int)poly2.n_elem - 1;
  arma::vec poly3(n1 + n2 + 1, arma::fill::zeros);
  for (int i = 0; i <= n1; ++i)
    for (int j = 0; j <= n2; ++j)
      poly3(i + j) += poly1(i) * poly2(j);
  return poly3;
}
// Build the convolved polynomial (1 + c_1·B + c_2·B^2 + …) from per-lag
// blocks.  Each block contributes (1 + φ_{i,1}·B^{L_i} + … + φ_{i,p_i}·B^{p_i·L_i}).
// `coefs` holds the FREE coefficients laid out per-block back-to-back.  When
// `applyStationary == true`, each block is passed through polyStationary()
// before being embedded (joint-BFGS path); when false (forecast-only / true
// param path) the raw coefficients are used directly.
//
// Returns a leading-1 polynomial of length (1 + Σ p_i·L_i).
inline arma::vec buildARMApoly(const arma::vec& coefs,
                                const arma::ivec& orders,
                                const arma::ivec& lags,
                                bool applyStationary){
  arma::vec full(1, arma::fill::ones);
  int offset = 0;
  for (arma::uword i = 0; i < orders.n_elem; ++i){
    int pi = orders(i);
    int Li = lags(i);
    if (pi <= 0) continue;
    arma::vec block = coefs.rows(offset, offset + pi - 1);
    if (applyStationary)
      polyStationary(block);
    arma::vec blockPoly(1 + pi * Li, arma::fill::zeros);
    blockPoly(0) = 1.0;
    for (int j = 0; j < pi; ++j)
      blockPoly((j + 1) * Li) = block(j);
    full = polyMult(full, blockPoly);
    offset += pi;
  }
  return full;
}
// fillARMAcolumn — write the per-block stationary AR / MA polynomial of one
// side (AR or MA) directly into a length-`deg` coefficient vector, skipping
// the intermediate convolution.  For SARMA(p, q)(P, Q)_s the convolved
// polynomial has only ~ p + P + p·P + 1 non-zero entries out of (1 + p + P·s)
// positions, and the structure is closed-form: enumerate the (i, j·s) and
// (i + j·s) cross-terms by hand.  Generalises naturally to L > 2 lag blocks
// through repeated nested loops, but the common cases L = 1 (non-seasonal)
// and L = 2 (SARMA) are specialised since they dominate.
//
// `coefs` holds the FREE coefficients laid out per-block back-to-back.
// `applyStationary` runs polyStationary block-by-block (BFGS path); when
// false the raw coefficients are placed directly (forecast-only path).
//
// The output `out` is sized to deg = Σ orders[i]·lags[i] (caller-provided)
// and filled with the coefficients (1 + c_1·B + … + c_deg·B^deg)[1..deg].
inline void fillSARMAcoefs(const arma::vec& coefs,
                            const arma::ivec& orders,
                            const arma::ivec& lags,
                            bool applyStationary,
                            arma::vec& out){
  out.zeros();
  if (orders.n_elem == 0) return;
  // Block 1: non-seasonal — fill positions 1..p with -φ_i (sign matches the
  // pre-flip polynomial (1 - φ_1·B - …) after polyStationary returns -PACF).
  int p1   = orders(0);
  int L1   = lags(0);
  arma::vec block1;
  if (p1 > 0){
    block1 = coefs.rows(0, p1 - 1);
    if (applyStationary) polyStationary(block1);
    for (int i = 0; i < p1; ++i)
      out((i + 1) * L1 - 1) = block1(i);
  }
  if (orders.n_elem == 1) return;
  // Block 2: seasonal block — fill positions {j·s, j·s + 1, …, j·s + p}.
  int p2   = orders(1);
  int L2   = lags(1);
  int off2 = (p1 > 0 ? p1 : 0);
  arma::vec block2;
  if (p2 > 0){
    block2 = coefs.rows(off2, off2 + p2 - 1);
    if (applyStationary) polyStationary(block2);
    for (int j = 0; j < p2; ++j){
      int basePos = (j + 1) * L2;
      // Seasonal-only term: position basePos.
      out(basePos - 1) = block2(j);
      // Cross terms with block 1: positions basePos + i·L1 for i = 1..p1.
      // The product (1 + c₁B^L1)(1 + c₂B^L2) picks up c₁·c₂ at position
      // L1 + L2 — same sign as the convolved buildARMApoly path.
      if (p1 > 0){
        for (int i = 0; i < p1; ++i)
          out(basePos + (i + 1) * L1 - 1) = block1(i) * block2(j);
      }
    }
  }
  // L > 2 (multiple seasonal lags) is not yet exposed through pts(); when
  // it lands the natural extension is to fall back to the legacy
  // buildARMApoly path via the comment-restored armaMatrices below.
}

// New armaMatrices — direct in-place fill of T's first column (AR) and R's
// first column (MA), skipping the intermediate convolved-polynomial vector
// allocation that the legacy implementation produced.  Functionally
// identical to the legacy version (commented out below) for L = 1 and L = 2;
// kept around for L > 2 as a future fallback.
void armaMatrices(vec p, SSmatrix* model, void* userInputs){
  ARMAinputs* inp = (ARMAinputs*)userInputs;
  model->Q(0, 0) = exp(2 * p(0));
  // AR.
  if (inp->ar > 0){
    arma::ivec orders, lags;
    if (inp->arOrders.n_elem > 0){
      orders = inp->arOrders;
      lags   = inp->armaLags;
    } else {
      orders = arma::ivec(1); orders(0) = inp->ar;
      lags   = arma::ivec(1); lags(0)   = 1;
    }
    int arDeg = inp->arDeg > 0 ? inp->arDeg : (int)arma::sum(orders % lags);
    arma::vec col(arDeg, arma::fill::zeros);
    fillSARMAcoefs(p.rows(1, inp->ar), orders, lags, true, col);
    model->T.submat(0, 0, arDeg - 1, 0) = -col;
  }
  // MA.
  if (inp->ma > 0){
    arma::ivec orders, lags;
    if (inp->maOrders.n_elem > 0){
      orders = inp->maOrders;
      lags   = inp->armaLags;
    } else {
      orders = arma::ivec(1); orders(0) = inp->ma;
      lags   = arma::ivec(1); lags(0)   = 1;
    }
    int maDeg = inp->maDeg > 0 ? inp->maDeg : (int)arma::sum(orders % lags);
    arma::vec col(maDeg, arma::fill::zeros);
    fillSARMAcoefs(p.rows(inp->ar + 1, inp->ar + inp->ma),
                   orders, lags, true, col);
    model->R.submat(1, 0, maDeg, 0) = col;
  }
}
// armaMatricesTrue — forecast-only variant.  Variance comes in absolute
// scale (no exp(2·)) and per-block coefficients aren't passed through
// polyStationary.  Direct-fill mirror of armaMatrices.
void armaMatricesTrue(vec p, SSmatrix* model, void* userInputs){
  ARMAinputs* inp = (ARMAinputs*)userInputs;
  model->Q(0, 0) = p(0);
  if (inp->ar > 0){
    arma::ivec orders, lags;
    if (inp->arOrders.n_elem > 0){
      orders = inp->arOrders;
      lags   = inp->armaLags;
    } else {
      orders = arma::ivec(1); orders(0) = inp->ar;
      lags   = arma::ivec(1); lags(0)   = 1;
    }
    int arDeg = inp->arDeg > 0 ? inp->arDeg : (int)arma::sum(orders % lags);
    arma::vec col(arDeg, arma::fill::zeros);
    fillSARMAcoefs(p.rows(1, inp->ar), orders, lags, false, col);
    model->T.submat(0, 0, arDeg - 1, 0) = -col;
  }
  if (inp->ma > 0){
    arma::ivec orders, lags;
    if (inp->maOrders.n_elem > 0){
      orders = inp->maOrders;
      lags   = inp->armaLags;
    } else {
      orders = arma::ivec(1); orders(0) = inp->ma;
      lags   = arma::ivec(1); lags(0)   = 1;
    }
    int maDeg = inp->maDeg > 0 ? inp->maDeg : (int)arma::sum(orders % lags);
    arma::vec col(maDeg, arma::fill::zeros);
    fillSARMAcoefs(p.rows(inp->ar + 1, inp->ar + inp->ma),
                   orders, lags, false, col);
    model->R.submat(1, 0, maDeg, 0) = col;
  }
}

/* ---------------------------------------------------------------------------
 * Legacy armaMatrices / armaMatricesTrue — kept here for reference and as a
 * fallback for L > 2 lag-block configurations (none today; SARMA always uses
 * L = 1 or L = 2 from pts()).  The new functions above do the direct in-
 * place fill that's exactly equivalent for L ≤ 2.
 *
 * void armaMatrices_legacy(vec p, SSmatrix* model, void* userInputs){
 *   ARMAinputs* inp = (ARMAinputs*)userInputs;
 *   model->Q(0, 0) = exp(2 * p(0));
 *   if (inp->ar > 0){
 *     arma::ivec orders, lags;
 *     if (inp->arOrders.n_elem > 0){
 *       orders = inp->arOrders;
 *       lags   = inp->armaLags;
 *     } else {
 *       orders = arma::ivec(1); orders(0) = inp->ar;
 *       lags   = arma::ivec(1); lags(0)   = 1;
 *     }
 *     arma::vec full = buildARMApoly(p.rows(1, inp->ar), orders, lags, true);
 *     int arDeg = (int)full.n_elem - 1;
 *     model->T.submat(0, 0, arDeg - 1, 0) = -full.rows(1, arDeg);
 *   }
 *   if (inp->ma > 0){
 *     arma::ivec orders, lags;
 *     if (inp->maOrders.n_elem > 0){
 *       orders = inp->maOrders;
 *       lags   = inp->armaLags;
 *     } else {
 *       orders = arma::ivec(1); orders(0) = inp->ma;
 *       lags   = arma::ivec(1); lags(0)   = 1;
 *     }
 *     arma::vec full = buildARMApoly(p.rows(inp->ar + 1, inp->ar + inp->ma),
 *                                    orders, lags, true);
 *     int maDeg = (int)full.n_elem - 1;
 *     model->R.submat(1, 0, maDeg, 0) = full.rows(1, maDeg);
 *   }
 * }
 * void armaMatricesTrue_legacy(vec p, SSmatrix* model, void* userInputs){
 *   ARMAinputs* inp = (ARMAinputs*)userInputs;
 *   model->Q(0, 0) = p(0);
 *   if (inp->ar > 0){
 *     arma::ivec orders, lags;
 *     if (inp->arOrders.n_elem > 0){
 *       orders = inp->arOrders;
 *       lags   = inp->armaLags;
 *     } else {
 *       orders = arma::ivec(1); orders(0) = inp->ar;
 *       lags   = arma::ivec(1); lags(0)   = 1;
 *     }
 *     arma::vec full = buildARMApoly(p.rows(1, inp->ar), orders, lags, false);
 *     int arDeg = (int)full.n_elem - 1;
 *     model->T.submat(0, 0, arDeg - 1, 0) = -full.rows(1, arDeg);
 *   }
 *   if (inp->ma > 0){
 *     arma::ivec orders, lags;
 *     if (inp->maOrders.n_elem > 0){
 *       orders = inp->maOrders;
 *       lags   = inp->armaLags;
 *     } else {
 *       orders = arma::ivec(1); orders(0) = inp->ma;
 *       lags   = arma::ivec(1); lags(0)   = 1;
 *     }
 *     arma::vec full = buildARMApoly(p.rows(inp->ar + 1, inp->ar + inp->ma),
 *                                    orders, lags, false);
 *     int maDeg = (int)full.n_elem - 1;
 *     model->R.submat(1, 0, maDeg, 0) = full.rows(1, maDeg);
 *   }
 * }
 * ------------------------------------------------------------------------ */
