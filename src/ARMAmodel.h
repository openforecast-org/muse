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
// Filling changing matrices.  Handles seasonal SARMA by convolving per-lag
// blocks (see buildARMApoly).  For backwards compatibility a caller that
// leaves arOrders / armaLags empty falls back to a single non-seasonal block
// of size ar / ma at lag 1.
void armaMatrices(vec p, SSmatrix* model, void* userInputs){
  ARMAinputs* inp = (ARMAinputs*)userInputs;
  // Variance.
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
    arma::vec full = buildARMApoly(p.rows(1, inp->ar), orders, lags, true);
    int arDeg = (int)full.n_elem - 1;
    model->T.submat(0, 0, arDeg - 1, 0) = -full.rows(1, arDeg);
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
    arma::vec full = buildARMApoly(p.rows(inp->ar + 1, inp->ar + inp->ma),
                                    orders, lags, true);
    int maDeg = (int)full.n_elem - 1;
    model->R.submat(1, 0, maDeg, 0) = full.rows(1, maDeg);
  }
}
// Filling changing matrices with true parameters (forecast-only).  Same shape
// as armaMatrices but without polyStationary on the inner coefficients and
// without log-stddev → variance translation.
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
    arma::vec full = buildARMApoly(p.rows(1, inp->ar), orders, lags, false);
    int arDeg = (int)full.n_elem - 1;
    model->T.submat(0, 0, arDeg - 1, 0) = -full.rows(1, arDeg);
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
    arma::vec full = buildARMApoly(p.rows(inp->ar + 1, inp->ar + inp->ma),
                                    orders, lags, false);
    int maDeg = (int)full.n_elem - 1;
    model->R.submat(1, 0, maDeg, 0) = full.rows(1, maDeg);
  }
}
