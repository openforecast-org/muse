/**************************************
 State Space systems class
 Needs Armadillo and many others
***************************************/
#include "bcnorm.h"
/***************************************************
  * Data structures
****************************************************/
// Model matrices
struct SSmatrix{
   // system matrices of a general State Space model
   // y(t)   = Z a(t) + D   u(t) + C eps(t)
   // a(t+1) = T a(t) + Gam u(t) + R eta(t)
   // Var(eps(t)) = H;  Var(eta(t)) = Q; Cov(eta(t), eps(t)) = S
   mat T, Gam, R, Q, Z, D, C, H, S;
};
// Model structure
struct SSinputs{
   // Inputs
   vec y,                 // output data
       p,                 // vector of parameter values
       pTransform,        // un-concentrated transformed parameters
       p0,                // vector of initial values for parameters
       stdP;              // standard errors of parameters
   mat u;                 // input data
   int h = 24;            // forecast horizon
   bool cLlik = true;    // concentrated log-likelihood on / off
   // user function implementing the model
   std::function <void (vec, SSmatrix*, void*)> userModel;
   void* userInputs;      // inputs needed by the user model
   // Outputs
   vec v,                 // innovations
       yFit,              // fitted values
       F,                 // Variance of fitted values
       yFor,              // output forecasts
       FFor,              // Variance of forecasts
       betaAug,           // betas of augmented KF (including initial states)
       betaAugVar,        // variances of betaAug (idag(iSn))
       criteria,          // identification criterion
       coef;              // Coefs for coef function
   mat a,                 // estimated states
       P,                 // variances of states
       eta,               // estimates of transition perturbations
       betaAugVarMat,     // full covariance matrix of initial states
       covp;              // Covariance of parameters
   SSmatrix system;       // system matrices
   double objFunValue,    // value of objective function at optimum
          outlier;        // critical value for outlier detection
   string estimOk;        // type of estimation convergence
   vector<string> table;  // output table from evaluate()
   // Needed for other purposes
   vec Finf,              // innovation variance before colapsing
       aEnd,              // final state vector estimated
       iF,                // inverse of F
       grad,              // gradient at optimum
       rNrOut;            // Needed for outlier detection
   mat K,                 // Kalman gain for smoothing
       Kinf,              // Kalman gain before colapsing
       PEnd,              // final P estimated
       rOut;              // Needed for outlier detection
   cube NOut;             // Needed for outlier detection
   int d_t = 0,           // colapsing observation
       nonStationaryTerms, // number of non stationary terms in state vector
       flag,              // output of optimization algorithm
       Iter;              // Number of iterations in estimation
   double innVariance;    // innovations variance
   bool exact = true,     // exact or numerical gradient
        verbose,          // intermediate output verbose on / off
        cleanInnovations = false, // cleaning innovations on/off
        augmented = false, // Augmented KF estimation on / off
        estimateLambda = false, // true when lambda is last element of p
        stateOnly = false; // fast state smoother: skip the O(m^3) backward Nt
                           // recursion + smoothed-variance work (data.P/F,
                           // disturbance, outlier) when only smoothed STATES
                           // (data.a) are consumed -- e.g. components().
   // Lower bound on Box-Cox lambda during joint-BFGS estimation.  In
   // llik() the value pulled from p.back() is clamped to >= lambdaLower
   // before computing BoxCox(y_raw, lam); the clamp creates a flat
   // region below the bound so the numerical gradient pushes BFGS back
   // into the feasible side.  Default = -inf (no bound).  Set from R
   // to 1e-10 when y contains zeros so log(0) / 0^negative can't appear.
   double lambdaLower = -arma::datum::inf;
   double lambda = 1.0;  // Box-Cox lambda; kept in sync with BSMmodel::lambda
   double logJac  = 0.0; // BCnorm Jacobian  Σ log|g'(y_t)|; computed in llik()
   vec y_raw;             // original (untransformed) y
   std::function <double (vec&, void*)> llikFUN; // LogLik to select llik or llikAug
};
/****************************************************
 * Defining SSmodel class
 ****************************************************/
// SS system class
class SSmodel{
  protected:
    SSinputs inputs;
  public:
    // Constructor declarations
    SSmodel(){}
    SSmodel(SSinputs);
    SSmodel(SSinputs, SSmatrix);
    // Destructor
    ~SSmodel();
    // Estimate by Maximum-Likelihood
    void estim();
    void estim(vec);
    // Forecasting system
    void forecast();
    // Kalman filter pass
    void filter();
    void filter(unsigned int);
    // Smoothing pass
    void smooth(bool);
    // Disturbance pass
    void disturb();
    // Validation
    void validate(bool, double);
    // // Getters and setters
    // Get inputs
    SSinputs getInputs(){
      return inputs;
    }
    // Set inputs
    void setInputs(SSinputs inputs){
      this->inputs = inputs;
    }
    // Set system matrices
    void setSystemMatrices(){
      inputs.userModel(inputs.p, &inputs.system, inputs.userInputs);
    }
    // Get Objective function (after estimation)
    double getObjFunValue(){
      return inputs.objFunValue;
    }
    // Print inputs on screen
     void print();
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Check stationarity of transition matrix (KFinit)
void isStationary(mat&, uvec&);
// Initialize Kalman Filter (llik)
void KFinit(mat&, mat&, uword, vec&, mat&, mat&);
// Matrix operations in KF (llik)
void MFK(mat&, vec&, mat, vec&, mat&, vec&);
void aP(vec& at, mat& Pt, vec& Kt, vec& vt, vec& Mt);
// Correction step of KF (llik)
void KFcorrection(bool, bool, bool, bool, SSinputs*, mat, mat&, vec&,
                  mat&, vec&, double, mat&, mat&, mat&, vec&, uword,
                  vec&, mat&, mat);
// Auxiliar function for computing llik in KF (llik)
void llikCompute(bool, mat, mat, mat, mat, mat&, mat&, mat&);
// Prediction stage in KF (llik)
void KFprediction(bool steadyState, bool colapsed, mat& T, mat& RQRt, vec& at, mat& Pt, mat& Pinft);
void KFprediction(bool steadyState, bool colapsed, const arma::sp_mat& T, mat& RQRt, vec& at, mat& Pt, mat& Pinft);
// Compute log-likelihood
double llik(vec&, void*);
// Compute log-likelihood with eXogenous inputs
double llikAug(vec&, void*);
// Select differentials (increments)
vec differential(vec p);
// Analytic and numeric gradient of log-likelihood
vec gradLlik(vec&, void*, double, int&);
// Llik hessian (for parameter covariances)
mat hessLlik(void*);
// True filter/smooth/disturb function
void auxFilter(unsigned int, SSinputs&);
// Calculating innovations from very beginning
vec KFinnovations(SSinputs&);
// solution to lyapunto equation P = Phi * P * Phi' + Q
mat dlyap(mat Phi, mat Q);
/****************************************************
 // SS implementations for univariate SS systems
 ****************************************************/
// Constructors with inputs
SSmodel::SSmodel(SSinputs inputs){
  this->inputs = inputs;
}
SSmodel::SSmodel(SSinputs inputs, SSmatrix system){
  this->inputs = inputs;
  this->inputs.system = system;
}
// Destructor
SSmodel::~SSmodel(){}
// Estimation by Maximum-Likelihood
void SSmodel::estim(){
  SSmodel::estim(inputs.p0);
}
void SSmodel::estim(vec p){
  double objFunValue;
  vec grad;
  mat iHess;
  this->inputs.p0 = p;

  wall_clock timer;
  timer.tic();
  int flag = quasiNewton(inputs.llikFUN, gradLlik, p, &inputs, objFunValue, grad, iHess, inputs.verbose);
  // Information criteria
  uvec indNan = find_nonfinite(inputs.y);
  int nNan = inputs.y.n_elem - indNan.n_elem;
  double LLIK, AIC, BIC, AICc, BICc;
  LLIK = -0.5 * nNan * (log(2*datum::pi) + objFunValue);
  infoCriteria(LLIK, p.n_elem + inputs.nonStationaryTerms, nNan,
               AIC, BIC, AICc, BICc);
  vec criteria(5);
  criteria(0) = LLIK;
  criteria(1) = AIC;
  criteria(2) = BIC;
  criteria(3) = AICc;
  criteria(4) = BICc;
  this->inputs.criteria = criteria;
  if (!isfinite(objFunValue))
      flag = 0;
  // Printing results
  if (flag == 1) {
    this->inputs.estimOk = "Q-Newton: Gradient convergence.\n";
  } else if (flag == 2){
    this->inputs.estimOk = "Q-Newton: Function convergence.\n";
  } else if (flag == 3){
      this->inputs.estimOk = "Q-Newton: Parameter convergence.\n";
  } else if (flag == 4){
      this->inputs.estimOk = "Q-Newton: Maximum number of iterations reached.\n";
  } else if (flag == 5){
      this->inputs.estimOk = "Q-Newton: Maximum number of Function evaluations.\n";
  } else if (flag == 6){
      this->inputs.estimOk = "Q-Newton: Unable to decrease objective function.\n";
  } else if (flag == 7){
      this->inputs.estimOk = "Q-Newton: Objective function returns nan.\n";
  } else {
      this->inputs.estimOk = "Q-Newton: No convergence!!\n";
  }
  if (inputs.verbose){
    double nSeconds = timer.toc();
    Rprintf("%s", this->inputs.estimOk.c_str());
    Rprintf("Elapsed time: %10.5f seconds\n", nSeconds);
  }
  this->inputs.p = p;
  this->inputs.objFunValue = objFunValue;
  this->inputs.grad = grad;
  this->inputs.flag = flag;
  this->inputs.v.reset();
}
// Forecasting system
void SSmodel::forecast(){
  mat RQRt = inputs.system.R * inputs.system.Q * inputs.system.R.t(),
      CHCt = inputs.system.C * inputs.system.H * inputs.system.C.t();
  int n = SSmodel::inputs.y.n_elem, k = SSmodel::inputs.u.n_rows;
  inputs.yFor.zeros(inputs.h);
  inputs.FFor.zeros(inputs.h);
  vec at = inputs.aEnd;
  mat Pt;
  if (at.has_nan()){
    filter();
    inputs.yFor = SSmodel::inputs.yFit.tail_rows(SSmodel::inputs.h);
    inputs.FFor = SSmodel::inputs.F.tail_rows(SSmodel::inputs.h);
  } else {
    // if (abs(inputs.innVariance - 1) > 1e-4){
      Pt = inputs.PEnd * inputs.innVariance;
    // } else {
    //   Pt = inputs.PEnd; // * inputs.innVariance;
    // }
    mat P0 = Pt;
    mat Z = inputs.system.Z.row(0);
    arma::sp_mat Tsp(inputs.system.T);   // sparse view for the h-step recursion
    uword t = inputs.y.n_elem;
    bool TVP = (inputs.system.Z.n_rows > 1);
    if (k > 0 && inputs.system.Z.n_rows == 1){
        int npar = inputs.betaAug.n_elem;
        inputs.system.D = inputs.betaAug.rows(npar - k, npar - 1).t();
    }
    if (k == 0) {
        inputs.system.D = {0};
    }
    for (int i = 0; i < inputs.h; i++){
      if (TVP)
          Z = inputs.system.Z.row(t + i);
      inputs.yFor.row(i) = Z * at;
      if (k == 0){
          inputs.yFor.row(i) += inputs.system.D;
      } else if (!TVP) {
          inputs.yFor.row(i) += inputs.system.D * SSmodel::inputs.u.col(n + i);
      }
      inputs.FFor.row(i) = Z * Pt * Z.t() + CHCt;
      KFprediction(false, true, Tsp, RQRt, at, Pt, P0);
    }
    // if (abs(inputs.innVariance - 1) < 1e-4){
    //   inputs.FFor *= inputs.innVariance;
    // }
  }
}
// Kalman filter pass
void SSmodel::filter(){
  SSmodel::filter(0);
}
void SSmodel::filter(unsigned int smooth){
    auxFilter(smooth, inputs);
}
// Smoothing pass
void SSmodel::smooth(bool outlier){
  if (outlier){
    SSmodel::filter(3);
  } else {
    SSmodel::filter(1);
  }
}
// Disturbance pass
void SSmodel::disturb(){
  SSmodel::filter(2);
}
// Validation
void SSmodel::validate(bool estimateHess, double nPar){
  // Input is inverse of Hessian. Calculated if empty
  uvec auxx;
  // Inverse of hessian and covariance of parameters
  int k = inputs.p.n_elem;
  uvec nn = find_finite(inputs.y);
  mat hess = eye(k, k);
  mat iHess = hess, table0;
  vec t, pValue(k);
//  if (k > 0){
      if (estimateHess){
          hess = hessLlik(&inputs) * 0.5 * nn.n_elem;
          iHess.fill(datum::nan);
          if (hess.is_finite()){
              iHess = pinv(hess);
              iHess.diag() = abs(iHess.diag());
          }
      }
      inputs.stdP = sqrt(iHess.diag());
      t = abs(inputs.p / inputs.stdP);
      pValue = 2 * (1- tCdf(t, nn.n_elem - k));
      uvec aux = find(t > 1000);
      if (aux.n_elem > 0){
        t(aux).fill(datum::inf);
        pValue(aux).fill(0);
      }
      table0 = join_horiz(join_horiz(join_horiz(inputs.p, inputs.stdP), t), pValue);
//  }
  // First part of table
  char str[70];
  inputs.table.clear();
  inputs.table.push_back("-------------------------------------------------------------\n");
  snprintf(str, 70, " %s", inputs.estimOk.c_str());
  inputs.table.push_back(str);
  inputs.table.push_back("-------------------------------------------------------------\n");
  inputs.table.push_back("            Param       S.E.        |T|    P-value     |Grad| \n");
  inputs.table.push_back("-------------------------------------------------------------\n");
  for (unsigned int i = 0; i < nPar; i++){
    snprintf(str, 70, "       %10.4f %10.4f %10.4f %10.4f %10.6f\n", table0(i, 0), table0(i, 1), table0(i, 2), table0(i, 3), abs(inputs.grad(i)));
    inputs.table.push_back(str);
  }
  // Adding inputs betas
  int nu = inputs.u.n_rows;
  if (nu > 0){
    vec betas, stdBetas;
    if (inputs.system.Z.n_rows > 1){
        filter();
        betas = inputs.aEnd.rows(inputs.aEnd.n_rows - nu, inputs.aEnd.n_rows - 1);
        vec dPEnd = inputs.PEnd.diag();
        stdBetas = sqrt(dPEnd.rows(inputs.aEnd.n_rows - nu, inputs.aEnd.n_rows - 1));
    } else {
        int ind = inputs.betaAug.n_elem - nu;
        betas = inputs.betaAug.rows(ind, ind + nu - 1);
        stdBetas = sqrt(inputs.betaAugVar.rows(ind, ind + nu - 1));
    }
    vec tBetas = betas / stdBetas;
    vec pValueBetas = 2 * (1- tCdf(tBetas, nn.n_elem - k));
    for (int i = 0; i < nu; i++){
      snprintf(str, 70, "       %10.4f %10.4f %10.4f %10.4f %10.6f\n", betas(i),
              stdBetas(i), tBetas(i), pValueBetas(i), datum::nan);
      inputs.table.push_back(str);
    }
  }
  uvec ind = find_finite(inputs.y);
  inputs.table.push_back("-------------------------------------------------------------\n");
  snprintf(str, 70, "  AIC: %12.4f   BIC: %12.4f   AICc: %12.4f\n", inputs.criteria(1), inputs.criteria(2), inputs.criteria(3));
  inputs.table.push_back(str);
  snprintf(str, 70, "           Log-Likelihood: %12.4f\n", inputs.criteria(0));
  inputs.table.push_back(str);
  inputs.table.push_back("-------------------------------------------------------------\n");
  // Recovering innovations for tests
  // if (inputs.augmented)
  //   llikAug(inputs.p, &inputs);
  // else
  //   llik(inputs.p, &inputs);
  // filter();
  inputs.cleanInnovations = true;
  vec inn = KFinnovations(inputs);
  filter();
  inputs.cleanInnovations = false;
  inputs.v.rows(0, inn.n_elem - 1) = inn;
  //Second part of table
  inputs.table.push_back("   Summary statistics:\n");
  inputs.table.push_back("-------------------------------------------------------------\n");
  auxx = find_finite(inputs.v);
  if (auxx.n_elem < 5){
    inputs.table.push_back("  All innovations are NaN!!\n");
  } else {
    outputTable(inputs.v, inputs.table);
  }
  inputs.table.push_back("-------------------------------------------------------------\n");
  // Show Table
  // if (show){
      // // for (auto i = inputs.table.begin(); i != inputs.table.end(); i++){
      // //   cout << *i << " ";
      // // }
      // for (unsigned int i = 0; i < inputs.table.size(); i++){
      //   Rprintf("%s ", inputs.table[i].c_str());
      // }
  // }
}
/*************************************************************
 * Implementation of auxiliar functions
 ************************************************************/
// Initializing Kalman Filter
void KFinit(mat& T, mat& RQRt, uword ns, vec& at, mat& Pt, mat& Pinft){
  at.zeros(ns);
  Pt.zeros(ns, ns);
  vec Pinfdiag; Pinfdiag.ones(ns);
  uvec stat;
  isStationary(T, stat);
  if (!stat.is_empty()){
    // int Ns = stat.n_elem;
    Pinfdiag.elem(stat).zeros();
    // Lyapunov for stationary elements
    mat q = RQRt(stat, stat);
    mat t = T(stat, stat);
    mat P2 = dlyap(t, q);
    // int Ns = stat.n_elem, Ns2 = Ns * Ns; mat P2bis = reshape(pinv(eye(Ns2, Ns2) - kron(t, t)) * vectorise(q), Ns, Ns);
    Pt(stat, stat) = P2;
  }
  Pinft = diagmat(Pinfdiag);
}
// Check stationarity of transition matrix (Kfinit)
void isStationary(mat& T, uvec& stat){
  int n = T.n_rows;
  cx_vec eigval(n);
  vec nons, nonstat;
  cx_mat V(n, n);
  double tol = 0.99;
  nons.zeros(n);
  nonstat = nons;
  eig_gen(eigval, V, T);
  nons.elem(find(abs(eigval) >= tol)).ones();
  nonstat.elem(find(abs(V) * nons > 0)).ones();
  stat = find(1 - nonstat);
}
// Update of Mt, Ft and Kt
void MFK(mat& Pt, mat& Z, mat CHCt, vec& Mt, mat& Ft, vec& Kt){
  Mt = Pt * Z.t();
  Ft = Z * Mt + CHCt;
  Kt = Mt / Ft(0, 0);
}
// Update of at and Pt
void aP(vec& at, mat& Pt, vec& Kt, vec& vt, vec& Mt){
  at = at + Kt * vt;
  Pt = Pt - Kt * Mt.t();
}
// Correction step in Kalman Filtering for every t
void KFcorrection(bool miss, bool colapsed, bool steadyState, bool smooth,
                  SSinputs* data, mat CHCt,
                  mat& Finft, vec& vt, double Dt, mat& Ft, mat& iFt, vec& at, mat& Pt,
                  mat& Pinft, vec& Kt, uword t, vec& auxFinf, mat& auxKinf, mat Z){
  vec Mt, Minft, Kinft;
  mat KK;
  rowvec yt = data->y.row(t);
  mat iFinft;
  if (miss){
        vt.fill(datum::nan);
        Kt.fill(0);
        Ft = Z * Pt * Z.t() + CHCt;
    } else {
      vt = yt - Z * at - Dt;
      if (steadyState){
        at = at + Kt * vt;
      } else if(colapsed) {
        MFK(Pt, Z, CHCt, Mt, Ft, Kt);
        aP(at, Pt, Kt, vt, Mt);
        iFt = 1 / Ft(0, 0);
      } else {
        Minft = Pinft * Z.t();
        Finft = Z * Minft;
        if (data->exact || smooth) auxFinf.row(t) = Finft;
        if (Finft(0, 0) > 1e-8){
          Mt = Pt * Z.t();
          Ft = Z * Mt + CHCt;
          iFinft = 1 / Finft(0, 0);
          iFt = iFinft;
          Kinft = Minft * iFinft;
          if (data->exact || smooth) auxKinf.col(t) = Kinft;
          Kt = (Mt - Kinft * Ft) * iFinft;
          aP(at, Pinft, Kinft, vt, Minft);
          KK = Mt * Kinft.t();
          Pt = Pt + Kinft * Ft * Kinft.t() - (KK + KK.t());
        } else {
          MFK(Pt, Z, CHCt, Mt, Ft, Kt);
          aP(at, Pt, Kt, vt, Mt);
          iFt = 1 / Ft(0, 0);
        }
      }
    }
}
// Llik computation inside llik
void llikCompute(bool colapsed, mat Finft, mat vt, mat Ft, mat iFt,
                 mat& v2F, mat& logF, mat& llikValue){
  if (colapsed || Finft(0, 0) < 1e-8){
    v2F  += vt * iFt * vt;
    logF += log(Ft);
  } else {
    llikValue += log(Finft);
  }
}
// KF Prediction step in Kalman filtering for every t
void KFprediction(bool steadyState, bool colapsed, mat& T, mat& RQRt, vec& at, mat& Pt, mat& Pinft){
  at = T * at;
  if (!steadyState){
    Pt = T * Pt * T.t() + RQRt;
  }
  if (!colapsed){
    Pinft = T * Pinft * T.t();
  }
}
// Sparse-T overload (Phase 1 of the sparse-matrix refactor).  T is the
// transition matrix, which is ~98% zeros at high lags (block-diagonal 2x2
// rotation blocks / companion sub-diagonals).  With T as a sp_mat, T*Pt and
// Pt*T.t() are sp*dense / dense*sp -> dense, O(nnz(T)*m) = O(m^2) instead of
// the dense O(m^3).  Pt/Pinft stay DENSE (they fill in during filtering);
// triple products are split into explicit binary products to guarantee the
// dense-result path.  RQRt stays dense in this phase.
void KFprediction(bool steadyState, bool colapsed, const arma::sp_mat& T, mat& RQRt, vec& at, mat& Pt, mat& Pinft){
  at = T * at;
  if (!steadyState){
    mat TPt = T * Pt;
    Pt = TPt * T.t();
    Pt += RQRt;
  }
  if (!colapsed){
    mat TPinf = T * Pinft;
    Pinft = TPinf * T.t();
  }
}
// Compute log-likelihood
double llik(vec& p, void* opt_data){
  // Converting void* to SSinputs*
  SSinputs* data = (SSinputs*)opt_data;
  // Running user function model (builds Q, H from structural params).
  // Lambda is the LAST element of p when jointly estimated; bsmMatrices
  // only reads structural positions so the extra element is silently ignored.
  data->userModel(p, &data->system, data->userInputs);
  // Joint-lambda: re-apply BoxCox at the current lambda (p.back()) so
  // the KF always runs on the correctly transformed series.
  if (data->estimateLambda) {
    double lam = p(p.n_elem - 1);
    // Clamp lambda to the user-supplied lower bound (default -inf =
    // no bound).  Required when y has zeros so BoxCox(0, lam<=0) =
    // +/-Inf doesn't poison the KF.  The unclamped p.back() stays as
    // the BFGS internal parameter; we just propagate the clamped lam
    // to data->lambda (read by downstream consumers) and to BoxCox.
    if (lam < data->lambdaLower) lam = data->lambdaLower;
    data->lambda = lam;
    data->y = BoxCox(data->y_raw, lam);
  }
  double tolsta = 1e-19;
  uword n,
        ns = data->system.T.n_rows,
        nMiss = 0;
  mat RQRt = data->system.R * data->system.Q * data->system.R.t(),
      CHCt = data->system.C * data->system.H * data->system.C.t(),
      Pt,
      Pinft,
      Ft(1, 1),
      Finft(1, 1),
      iFt(1, 1),
      oldPt,
      llikValue(1, 1),
      logF(1, 1),
      v2F(1, 1),
      auxKinf;
  vec at,
      Kt(ns),
      vt(1),
      auxFinf;
  bool colapsed = false,
       steadyState = false,
       miss = false;
  data->innVariance = 1;
  // Initializing variables
  llikValue.fill(0);
  logF.fill(0);
  v2F.fill(0);
  n = data->y.n_rows;
  data->d_t = n;
  // Per-observation mask: 1 where llikCompute() takes the v^2/F (non-diffuse)
  // branch, 0 where it takes the diffuse log|Finf| branch (or the obs is
  // missing).  Captured here so the central BCnorm likelihood below splits the
  // observations exactly as llikCompute() did.
  vec nonDiffuseMask = zeros(n);
  // Kfinit
  KFinit(data->system.T, RQRt, ns, at, Pt, Pinft);
  oldPt.zeros(ns, ns);
  data->nonStationaryTerms = sum(Pinft.diag());
  data->v = zeros(n);
  data->F = data->v;
  data->iF = data->v;
  if (data->exact){
    data->K = zeros(ns, n);
    auxFinf = data->v;
    auxKinf = data->K;
  }
  mat Z = data->system.Z.row(0);
  bool TVP = false;
  if (data->system.Z.n_rows > 1)
      TVP = true;
  // Phase-1 sparse PoC: build a sparse view of the (dense-stored) transition
  // matrix once per likelihood evaluation; the per-timestep prediction below
  // then uses the O(m^2) sparse path.  KFinit above keeps the dense T (its
  // eig_gen/schur stationarity machinery is sparse-hostile).
  arma::sp_mat Tsp(data->system.T);
  // KF loop
  for (uword t = 0; t < n; t++){
    // Data missing
    // if (!is_finite(data->y.row(t))){
    if (!std::isfinite(data->y(t))) {
      steadyState = false;
      miss = true;
      nMiss += 1;
    } else {
      miss = false;
    }
    if (TVP)
        Z = Z = data->system.Z.row(t);
    // Correction
    KFcorrection(miss, colapsed, steadyState, data->exact, data, CHCt,
                 Finft, vt, data->system.D(0, 0), Ft, iFt, at, Pt, Pinft,
                 Kt, t, auxFinf, auxKinf, Z);
    // llik calculation
    if (!miss && t < n){
      llikCompute(colapsed, Finft, vt, Ft, iFt, v2F, logF, llikValue);
      // Same branch condition as llikCompute(): non-diffuse iff collapsed or
      // the diffuse innovation variance is already negligible.
      if (colapsed || Finft(0, 0) < 1e-8)
        nonDiffuseMask(t) = 1.0;
    }
    // Prediction (sparse-T path)
    KFprediction(steadyState, colapsed, Tsp, RQRt, at, Pt, Pinft);
    // Storing final state and covariance for forecasting
   if (t == n - 1){
     data->PEnd = Pt;
     data->aEnd = at;
   }
    // Checking colapsed
    if (!colapsed){
      if (all(all(abs(Pinft) < 1e-6))){
        colapsed = true;
        data->d_t = t;
        if (data->exact){
          data->Finf = auxFinf.rows(0, t);
          data->Kinf = auxKinf.cols(0, t);
        }
      }
    }
    // Checking steady state
    if (!steadyState){
      if (colapsed && all(all(abs(Pt - oldPt) < tolsta))){
        steadyState = true;
      } else {
        oldPt = Pt;
      }
    }
    // Storing for analytical derivatives
    data->v.row(t)= vt;
    data->F.row(t)= Ft;
    data->iF.row(t)= iFt;
    if (data->exact){
      data->K.col(t)= Kt;
    }
  }
  // System did not colapsed
  if (data->exact && !colapsed){
    data->Finf = auxFinf;
    data->Kinf = auxKinf;
  }
  // Computing llik value
  // MLE form: σ̂² = SSR / n_finite (not REML n-k).  Both the σ̂² estimator
  // and the BoxCox Jacobian are summed over the SAME n_finite observations,
  // so the BCnorm marginal log-likelihood is internally consistent and the
  // BoxCox MLE in lambda lands at the right value.
  //
  // Resulting objFunValue is set so that
  //   LL_BCnorm = -0.5 · (n_finite · log(2π) + n_finite · objFunValue)
  // which expands to
  //   LL_BCnorm = -n/2 · log(2π σ̂²) - n/2 - 1/2 · Σ log F + logJac
  // i.e. the closed-form equivalent of Σ_t bcnormLogDensityScalar(y_t, μ_t,
  // sqrt(σ̂²·F_t), λ), with the diffuse-phase log|Finf| terms accumulated
  // inside llikValue acting as the log-determinant correction for the
  // implicit initial-state integration.
  int nFinite = (int)n - (int)nMiss;
  if (nFinite < 1) nFinite = 1;  // guard against pathological all-NA
  // BoxCox Jacobian — Σ log|g'(y_raw_t)| over the same n_finite observations
  // as the Gaussian density.  Identity at λ=1 (Jacobian=0) and log at λ=0.
  {
      double logJac = 0.0;
      if (data->y_raw.n_elem == n){
          for (uword t = 0; t < n; t++){
              if (!std::isfinite(data->y(t))) continue;  // skip missing
              logJac += bcnormLogJac(data->y_raw(t), data->lambda);
          }
      }
      data->logJac = logJac;
  }
  if (data->cLlik){         // Concentrated Likelihood (MLE)
      data->innVariance = v2F(0, 0) / nFinite;
      const double s2 = data->innVariance;
      // Objective = the BCnorm log-likelihood (the C++ analogue of adam()'s
      // loss = "likelihood": the concentrated scale s2 is plugged straight into
      // the per-observation predictive sd).  The reported logLik / IC derive
      // from this same objFunValue (PTSmodel.h::computeLLIK), so bcnorm is the
      // single source.
      const double diffuseTerm = llikValue(0, 0);   // Sum_{diffuse} log Finf
      const bool haveRaw = (data->y_raw.n_elem == n);
      vec yr     = haveRaw ? data->y_raw : data->y;
      double lam = haveRaw ? data->lambda : 1.0;    // !haveRaw => identity
      uvec finiteIdx = find_finite(data->y);
      vec maskFin = nonDiffuseMask.elem(finiteIdx);
      // Non-diffuse observations: full BCnorm data density in ONE vectorised
      // call, each with its own predictive sd sqrt(s2 * F_t).
      uvec ndIdx = finiteIdx.elem(find(maskFin > 0.5));
      double LL = sum(bcnormLogDensity(
                          yr.elem(ndIdx),
                          data->y.elem(ndIdx) - data->v.elem(ndIdx),
                          sqrt(s2 * data->F.elem(ndIdx)), lam));
      // Diffuse observations are consumed estimating the diffuse initial state:
      // they carry no data-fit term, only the log|Finf| determinant (==
      // diffuseTerm) plus their BoxCox Jacobian.  The determinant is NOT scaled
      // by the concentrated variance s2 -- the exact-diffuse likelihood
      // convention (Durbin-Koopman): only the n - d_t non-diffuse observations
      // carry the s2 scale.  The previous form added -0.5*nDiff*log(s2), which
      // is harmless near s2 ~ 1 but, when the optimiser drives s2 to an extreme
      // corner (a degenerate trend whose observation variance has collapsed),
      // injected a large spurious +likelihood -- the +986 "positive logLik" on
      // strongly seasonal series.
      uvec dIdx = finiteIdx.elem(find(maskFin < 0.5));
      if (!dIdx.is_empty()){
          double nDiff = (double)dIdx.n_elem;
          LL += sum(bcnormLogJac(yr.elem(dIdx), lam))
                - 0.5 * nDiff * std::log(2.0 * datum::pi)
                - 0.5 * diffuseTerm;
      }
      llikValue(0, 0) = -2.0 * LL / nFinite - std::log(2.0 * datum::pi);
  } else {                  // Crude Likelihood (forecast-only path)
      llikValue = (llikValue + v2F + logF - 2.0 * data->logJac) / nFinite;
  }
  data->objFunValue = llikValue(0, 0);
  // System colapsed
  if ((uword)data->d_t < n){
      data->v.rows(0, data->d_t).fill(datum::nan);
  }
  return data->objFunValue;
}
// Compute log-likelihood for model with eXogenous inputs
double llikAug(vec& p, void* opt_data){
  // Augmented Kalman Filter
  SSinputs* data = (SSinputs*)opt_data;
  // Running user function model (setting system matrices)
  data->userModel(p, &data->system, data->userInputs);
  uword ns = data->system.T.n_rows,
        nMiss = 0,
        n = data->y.n_rows,
        nu = data->u.n_rows,
        k = nu + ns;
  double tolsta = 1e-19;
  mat RQRt = data->system.R * data->system.Q * data->system.R.t(),
      CHCt = data->system.C * data->system.H * data->system.C.t(),
      Pt(ns, ns),
      oldPt(ns, ns),
      Ft(1, 1),
      FEnd(1, 1),
      llikValue(1, 1),
      At(ns, k),
      Sn(k, k),
      iSn(k, k),
      AtiSn(ns, k),
      VtiSn(1, k),
      PEndZ(ns ,1); //, W(ns, nu);
  vec at(ns),
      vt(1),
      vEnd(1),
      sn(k),
      beta(k),
      Kt(ns),
      iFt(1),
      viFt(1),
      logF(1),
      v2F(1);
  rowvec Vt(k),
         Xt(k);
  // Per-step cache for a cancellation-free residual sum of squares.  beta_hat
  // is only known after the full pass, so we store the innovation v_t, the
  // augmented sensitivity V_t and the weight 1/F_t at every step and form the
  // GLS residual r_t = v_t - V_t·beta_hat afterwards.  iFtStore is left 0 at
  // missing / steady-state-skipped steps so they contribute nothing.
  mat Vstore(k, n, fill::zeros);
  vec vStore(n, fill::zeros),
      iFtStore(n, fill::zeros);
  bool miss = false,
       steadyState = false;
  at.fill(0);
  Pt = RQRt;
  oldPt.fill(-10);
  At.fill(0);  // At = -Wt;
  At.submat(0, 0, ns - 1, ns - 1) = -eye(ns, ns);
  sn.fill(0);
  Sn.fill(0);
  iFt.fill(1);
  viFt.fill(0);
  logF.fill(0);
  v2F.fill(0);
  Xt.fill(0);
  uvec aux = regspace<uvec>(ns, k - 1);
  // KF loop
  for (uword t = 0; t < n; t++){
    if (nu > 0){
      Xt(aux) = data->u.col(t).t();
    }
    // miss = !is_finite(data->y.row(t));
    miss = !std::isfinite(data->y(t));
    // Checking for missing
    if (miss){
      nMiss++;
      vt.fill(0);
      Kt.fill(0);
      steadyState = false;
    }
    // Main calculations
    if (steadyState){
      if (!miss){
        vt = data->y.row(t) - data->system.Z * at;
      }
    } else {
      Ft = data->system.Z * Pt * data->system.Z.t() + CHCt;
      iFt = 1 / Ft;
      if (!miss){
        vt = data->y.row(t) - data->system.Z * at;
        Kt = data->system.T * Pt * data->system.Z.t() * iFt;
      }
      Pt = data->system.T * Pt * data->system.T.t() + RQRt - Kt * Ft * Kt.t();
    }
    at = data->system.T * at + Kt * vt;
    // Augmented part
    Vt = Xt - data->system.Z * At;
    At = data->system.T * At + Kt * Vt; // + Wt;
    if (!miss){
      viFt = vt * iFt;
      sn += Vt.t() * viFt;
      Sn += Vt.t() * iFt * Vt;
      v2F += vt * viFt;
      logF += log(Ft);
      // Cache for the cancellation-free RSS reduction below.
      vStore(t) = vt(0);
      iFtStore(t) = iFt(0);
      Vstore.col(t) = Vt.t();
    }
    // Checking steady state
    if (!steadyState && t > ns){
      if (all(all(abs(Pt - oldPt) < tolsta))){
        steadyState = true;
      } else {
        oldPt = Pt;
      }
    }
  }
  if (Sn.has_nan() || Sn.has_inf()){
    // Algorithm blew up
    llikValue(0, 0) = datum::nan;
  } else {
    iSn = pinv(Sn);
    // Storing final state and covariance for forecasting
    AtiSn = At * iSn;
    data->aEnd = at - AtiSn * sn;
    data->PEnd = Pt + AtiSn * At.t();
    beta = iSn * sn;
    // MLE form: σ̂² = (SSR − sn'·iSn·sn) / n_finite, integrating β with flat
    // prior.  Identical normalisation as llik() so both paths produce LLs on
    // the same scale and the BoxCox MLE is consistent across model types.
    //
    //   LL_BCnorm = -n/2·log(2π σ̂²) - n/2 - 1/2·log det Sn - 1/2·Σ log F
    //               + logJac
    //
    // Packaged into objFunValue so that
    //   LL_BCnorm = -0.5·(n_finite·log(2π) + n_finite·objFunValue).
    int nFinite = (int)n - (int)nMiss;
    if (nFinite < 1) nFinite = 1;
    // BoxCox Jacobian over the same n_finite observations.
    double logJac = 0.0;
    if (data->y_raw.n_elem == n){
        for (uword t = 0; t < n; t++){
            if (!std::isfinite(data->y(t))) continue;
            logJac += bcnormLogJac(data->y_raw(t), data->lambda);
        }
    }
    data->logJac = logJac;
    // Concentrated variance = residual sum of squares / n.  The textbook form
    //   RSS = v2F - sn'beta = Sum v_t^2/F_t - sn'Sn^{-1}sn
    // is the projection identity (total weighted SS minus the part explained by
    // the augmented states = diffuse initial states + regressors).  It is a sum
    // of squares, hence >= 0 -- but computing it as that subtraction is the
    // classic unstable move: at a near-interpolating fit v2F and sn'beta are
    // huge and nearly equal, so every significant digit cancels and the result can
    // come out negative, poisoning innVariance, the displayed variances and the
    // parameter covariance.
    //
    // Instead form the residual explicitly: r_t = v_t - V_t·beta_hat, and
    //   RSS = Sum r_t^2 / F_t,
    // a sum of non-negative terms.  This is structurally >= 0 in floating point
    // (no subtraction of large near-equal numbers) so the negative-variance
    // artifact cannot occur, however degenerate the fit.  (A genuinely
    // interpolating fit still gives RSS ~ 0; that is the separate degeneracy
    // handled by the disturbance-variance bound, not a numerical defect here.)
    vec r = vStore - Vstore.t() * beta;     // GLS residual r_t = v_t - V_t·beta
    double rss = dot(square(r), iFtStore);  // Sum r_t^2 / F_t (iFtStore==0 at
                                            // missing / steady-skipped steps)
    // Guard only against log(0) at an exactly-interpolating fit (RSS == 0); this
    // is the unavoidable log-domain guard, not a sign fix.
    double rssFloor = arma::datum::eps * v2F(0, 0);
    if (rss < rssFloor) rss = rssFloor;
    data->innVariance = rss / nFinite;
    // Objective via the BCnorm density (single source, exactly as llik()).  The
    // GLS residual r_t has predictive sd sqrt(innVar * F_t); summing
    // bcnormLogDensity over the observations gives the data likelihood, and the
    // augmented-state determinant -0.5*log|Sn| (the diffuse initial states +
    // regressors integrated under a flat prior) is the single correction term --
    // the augmented analogue of llik()'s log|Finf| diffuse correction.  Packed
    // so LL_BCnorm = -0.5*(nFinite*log(2pi) + nFinite*objFunValue), from which
    // computeLLIK() derives the reported logLik / IC.
    const bool haveRaw = (data->y_raw.n_elem == n);
    vec yr     = haveRaw ? data->y_raw : data->y;
    double lam = haveRaw ? data->lambda : 1.0;   // !haveRaw => identity
    vec mu = data->y - r;                         // g(y_t) - r_t
    vec sd = sqrt(data->innVariance / iFtStore);  // sqrt(innVar * F_t)
    uvec finiteIdx = find_finite(data->y);
    double LL = sum(bcnormLogDensity(yr.elem(finiteIdx), mu.elem(finiteIdx),
                                     sd.elem(finiteIdx), lam))
                - 0.5 * std::log(det(Sn));
    llikValue(0, 0) = -2.0 * LL / nFinite - std::log(2.0 * datum::pi);
    data->objFunValue = llikValue(0, 0);
    data->betaAug = beta;
    data->betaAugVarMat = data->innVariance * iSn;
    data->betaAugVar = data->betaAugVarMat.diag();
  }
  return llikValue(0, 0);
}
// Select differentials (increments)
vec differential(vec p){
  vec signP = sign(p);
  signP(find(signP == 0)).fill(1);
  return max(join_horiz(abs(p), ones(p.n_elem, 1)), 1) % signP * 1e-8;
}
// Analytic and numeric gradient of log-likelihood
vec gradLlik(vec& p, void* opt_data, double llikValue, int& nFuns){
  int nPar = p.n_elem;
  vec grad(nPar),
      p0 = p,
      inc;
  SSinputs* data = (SSinputs*)opt_data;
  nFuns = 0;
  inc = differential(p);
  if (p.has_nan()){
    grad.fill(datum::nan);
    return grad;
  }
  if (data->exact){  // Analytical derivative
    int ns = data->system.T.n_rows,
        n = data->y.n_elem,
        cQ,
        nMiss = 0;
    mat GammaQ(ns, ns),
        Nt(ns, ns),
        RR(ns, ns),
        sysmatQ,
        sysmatR,
        Z = data->system.Z,
        Gamma(ns + 1, ns + 1),
        Qt,
        dQt,
        dRQRt(ns, ns),
        Inew(ns, ns),
        Lt(ns, ns);
    vec rt(ns),
        vt(1),
        Kt(ns),
        GammaD(1),
        iFt(1),
        e(1),
        D(1),
        Kinft(ns),
        Z_Ft(ns);
    double Finft = 0.0;
    bool colapsed = true;
    // Initialising variables
    GammaQ.fill(0);
    GammaD.fill(0);
    Gamma.fill(0);
    cQ = data->system.Q.n_cols;
    sysmatQ = zeros(cQ + 1, cQ + 1);
    sysmatR = zeros(ns + 1, cQ + 1);
    Qt = zeros(cQ + 1, cQ + 1);
    dQt = sysmatQ;
    e.fill(0);
    D.fill(0);
    Nt = GammaQ;
    rt.fill(0);
    Inew.eye();
    // Sparse view of T for the backward recursion (T'*rt, T'*Nt*T).
    arma::sp_mat Tsp(data->system.T);
    arma::sp_mat TspT = Tsp.t();
    // Main Loop
    for (int t = n - 1; t >= 0; t--){
      if (t <= data->d_t){
        colapsed = false;
      }
      vt = data->v.row(t);
      Kt = data->K.col(t);
      iFt = data->iF.row(t) / data->innVariance;
      if (!colapsed){
        Finft = data->Finf(t); // * data->innVariance;
        Kinft = data->Kinf.col(t);
      }
      // if (!is_finite(data->y.row(t))){
      if (!std::isfinite(data->y(t))){
        e.fill(0);
        D.fill(0);
        nMiss += 1;
      } else if (colapsed || Finft< 1e-8) {
        // Lt = I - Kt*Z is identity minus rank-1, so Lt'*Nt*Lt and Lt'*rt
        // collapse to O(m^2)/O(m) rank updates (no dense triple product):
        //   Lt'*Nt*Lt = Nt - w z' - z w' + (k'w) z z'   with w = Nt*Kt
        //   Lt'*rt    = rt - z (Kt'*rt)
        vec w = Nt * Kt;                       // O(m^2)
        double kw  = dot(Kt, w);
        double ktr = dot(Kt, rt);
        double if0 = iFt(0, 0);
        vec zc = Z.t();
        e(0) = if0 * vt(0) - ktr;
        D(0) = if0 + kw;
        rt += zc * e(0);                       // = z*(iFt*vt) + (rt - z*Kt'rt)
        Nt += (if0 + kw) * (zc * zc.t()) - w * zc.t() - zc * w.t();
      } else {
        vec w = Nt * Kinft;
        double kr = dot(Kinft, rt);
        e(0) = -kr;
        D(0) = dot(Kinft, w);
        if (Finft >= 1e-8){   // Finf not singular
          vec zc = Z.t();
          rt -= zc * kr;
          Nt += D(0) * (zc * zc.t()) - w * zc.t() - zc * w.t();
        }
      }
      GammaD += e * e - D;
      RR = rt * rt.t() - Nt;
      GammaQ += RR;
      rt = TspT * rt;
      mat NtT = TspT * Nt;   // sp*dense -> dense
      Nt = NtT * Tsp;        // dense*sp -> dense
    }
    // Derivatives of RQRt and CHCt.  The smoother accumulants (Gamma) are in
    // ABSOLUTE units (iFt was divided by innVariance above), so the baseline
    // system matrices must also be built at the point p where the gradient is
    // evaluated -- NOT left at the stale ratio-scale Q/H from the last objFun()
    // call.  Otherwise dQt = (Qt - sysmatQ)/inc mixes absolute and ratio units
    // and blows up by ~1/innVariance when the concentrated variance is small.
    data->userModel(p, &data->system, data->userInputs);
    sysmatQ.submat(0, 0, cQ - 1, cQ - 1) = data->system.Q;
    sysmatQ.submat(cQ, cQ, cQ, cQ) = data->system.H;
    sysmatR.submat(0, 0, ns - 1, cQ - 1) = data->system.R;
    sysmatR.submat(ns, cQ, ns, cQ) = data->system.C;
    Gamma.submat(0, 0, ns - 1, ns - 1) = GammaQ;
    Gamma.submat(ns, ns, ns, ns) = GammaD;
    // Normalise by the same sample size objFun()/llik() averages over
    // (nFinite = n - nMiss); the old "- d_t - 1" diffuse subtraction made the
    // gradient ~2-3% too large on seasonal models (large d_t) and broke the
    // match with the finite-difference reference.
    int nn = n - nMiss;
    for (int i = 0; i < nPar; i++){
      // Lambda is the last element when jointly estimated.  It affects y
      // via BoxCox, not Q/H, so the analytic disturbance-smoother formula
      // gives zero.  Use a numerical step through the full llik() instead.
      if (data->estimateLambda && i == nPar - 1){
        p0 = p;
        p0.row(i) += inc(i);
        double f1 = data->llikFUN(p0, opt_data);
        grad.row(i) = (f1 - llikValue) / inc(i);
        data->userModel(p, &data->system, data->userInputs);  // restore Q/H
        nFuns += 1;
        continue;
      }
      p0 = p;
      p0.row(i) += inc(i);
      data->userModel(p0, &data->system, data->userInputs);
      Qt.submat(0, 0, cQ - 1, cQ - 1) = data->system.Q;
      Qt.submat(cQ, cQ, cQ, cQ) = data->system.H;
      dQt= (Qt - sysmatQ) / inc(i);
      dRQRt = sysmatR * dQt * sysmatR.t();
      grad.row(i) = -trace(Gamma * dRQRt) / nn;
    }
    nFuns += 1;
  } else {          // Numerical derivative
    vec F1 = p;
    for (int i = 0; i < nPar; i++){
        p0 = p;
        p0(i) += inc(i);
        F1(i) = data->llikFUN(p0, opt_data);
    }
    grad = (F1 - llikValue) / inc;
    nFuns += nPar;
  }
  return grad;
}
// Llik hessian (for parameter covariances).
//
// Central second differences with a per-parameter, magnitude-scaled step.
// The previous scheme used ONE-SIDED forward differences with a fixed
// absolute step (1e-5) for every parameter: truncation error O(h) (biased),
// and for weakly-identified directions the tiny fixed step caused
// catastrophic cancellation (rounding floor ~eps*|llik|/h^2 ~ 1e-2 per
// entry), which on a near-singular Hessian flipped small eigenvalues and
// produced indefinite, non-reproducible covariances.
//
// Central differences are unbiased to O(h^2); the step h_i = eps^(1/4) *
// max(|p_i|, 1) (~1.22e-4 at unit scale) is the standard near-optimal choice
// for 3-/4-point second derivatives, balancing truncation against rounding
// and scaling with the parameter so flat directions are differenced cleanly.
mat hessLlik(void* optData){
  SSinputs* inputs = (SSinputs*)optData;
  uword nPar = inputs->p.n_elem;
  vec p0 = inputs->p;
  mat Hess(nPar, nPar, fill::zeros);
  // Per-parameter step, scaled by magnitude (relative for |p|>1, absolute floor 1).
  const double h0 = std::pow(datum::eps, 0.25);   // ~1.22e-4
  vec h(nPar);
  for (uword i = 0; i < nPar; i++)
    h(i) = h0 * std::max(std::abs(p0(i)), 1.0);
  // Evaluate the (negative, averaged) log-likelihood at an arbitrary point.
  auto f = [&](vec pp)->double {
    return inputs->augmented ? llikAug(pp, inputs) : llik(pp, inputs);
  };
  double f0 = f(p0);
  // Diagonal: [f(p+h) - 2 f(p) + f(p-h)] / h^2
  for (uword i = 0; i < nPar; i++){
    vec pp = p0; pp(i) += h(i);
    vec pm = p0; pm(i) -= h(i);
    Hess(i, i) = (f(pp) - 2.0 * f0 + f(pm)) / (h(i) * h(i));
  }
  // Off-diagonal: [f(++) - f(+-) - f(-+) + f(--)] / (4 h_i h_j)
  for (uword i = 0; i < nPar; i++){
    for (uword j = i + 1; j < nPar; j++){
      vec ppp = p0; ppp(i) += h(i); ppp(j) += h(j);
      vec ppm = p0; ppm(i) += h(i); ppm(j) -= h(j);
      vec pmp = p0; pmp(i) -= h(i); pmp(j) += h(j);
      vec pmm = p0; pmm(i) -= h(i); pmm(j) -= h(j);
      double val = (f(ppp) - f(ppm) - f(pmp) + f(pmm)) / (4.0 * h(i) * h(j));
      Hess(i, j) = val;
      Hess(j, i) = val;
    }
  }
  // Leave inputs->v / inputs->F at the solution for downstream consumers
  // (OPG fallback in parCov reads them).
  f(p0);
  return Hess;
}
// True filter/smooth/disturb function
void auxFilter(unsigned int smooth, SSinputs& data){
  // smooth (0: filter, 1: smooth, 2: disturb)
  // double tolsta = 0; //1e-7;
  uword n,
        ns,
        nMiss = 0;
  mat RQRt,
      CHCt(1, 1),
      Pt,
      Pinft,
      Ft(1, 1),
      Finft(1, 1),
      v2F(1, 1),
      iFt(1, 1); //, oldPt;  //, auxKinf;
  vec at,
      Kt,
      vt(1),
      data_F;  //, auxFinf;
  bool colapsed = false,
       steadyState = false,
       miss = false;
  cube cP,
       Pinf;
  // Initialising variables
  uword ny = data.y.n_elem;
  int k = data.u.n_rows;
  vec Nans(data.h); Nans.fill(datum::nan);
  data.y = join_vert(data.y, Nans);
  n = data.y.n_elem;
  data.d_t = n;
  ns = data.system.T.n_rows;
  RQRt = data.system.R * data.system.Q * data.system.R.t();
  CHCt = data.system.C * data.system.H * data.system.C.t();
  // Inputs part
  rowvec Dt(n, fill::zeros);
  if (k > 0 && data.system.Z.n_rows == 1){
    int nn = data.betaAug.n_rows;
    data.system.D = data.betaAug.rows(nn - k, nn - 1).t();
    Dt = data.system.D * data.u;
  }
  KFinit(data.system.T, RQRt, ns, at, Pt, Pinft);
  data.v = zeros(n);
  data.a = zeros(ns, n);
  if  (smooth > 0){
    cP = zeros(ns, ns, n);
    Pinf = zeros(ns, ns, data.d_t + 1);
  }
  data.P = zeros(ns, n);
  data.yFit = data.v;
  data.K = zeros(ns, n);
  data_F = data.v;
  data.iF = data.v;
  data.Finf = zeros(data.d_t + 1);
  data.Kinf = zeros(ns, data.d_t + 1);
  v2F.fill(0);
  Kt.resize(ns);
  mat Z = data.system.Z.row(0);
  bool TVP = false;
  if (data.system.Z.n_rows > 1)
      TVP = true;
  arma::sp_mat Tsp(data.system.T);   // sparse view for the forward recursion
  // KF loop
  for (uword t = 0; t < n; t++){
    if (TVP)
        Z = data.system.Z.row(t);
    if (!colapsed && smooth > 0){
      Pinf.slice(t) = Pinft;
    }
    data.yFit.row(t) = Z * at + Dt(t);
    // Storing for smoothing/disturbing
    data.v.row(t) = data.y.row(t) - data.yFit.row(t);
    if (smooth > 0){
      data.a.col(t) = at;
      cP.slice(t) = Pt;
      data.P.col(t) = Pt.diag();
      if (steadyState){
        data_F.row(t) = data_F.row(t - 1);
      } else {
        data_F.row(t) = Z * Pt * Z.t() + CHCt;
      }
    }
    // Data missing
    if (!std::isfinite(data.y(t))){
      steadyState = false;
      miss = true;
      nMiss++;
    } else {
      miss = false;
    }
    // Correction
    KFcorrection(miss, colapsed, steadyState, smooth, &data, CHCt,
                 Finft, vt, Dt(t), Ft, iFt, at, Pt, Pinft, Kt, t,
                 data.Finf, data.Kinf, Z);
    if (!miss && t < n){
      if (colapsed || Finft(0, 0) < 1e-8){
        v2F  += vt * iFt * vt;
      }
    }
    // Storing information
    data.v.row(t) = vt;
    // añadido copilot
    // if (miss) {
    //     data.v(t) = datum::nan;

    // } else if (!colapsed && Finft(0,0) > 1e-8) {
    //     // ---- FASE DIFUSA (correcta) ----
    //     data.v(t) = vt(0) / std::sqrt(Finft(0,0));

    // } else {
    //     // ---- FASE ESTÁNDAR ----
    //     data.v(t) = vt(0) / std::sqrt(Ft(0,0));
    // }

    data_F.row(t) = Ft;
    data.iF.row(t) = iFt;
    if (smooth > 0){
      data.K.col(t) = Kt;
    } else {
      data.yFit.row(t) = Z * at + Dt(t);
      data.a.col(t) = at;
      data.P.col(t) = Pt.diag();
    }
    // Prediction (sparse-T path)
    KFprediction(steadyState, colapsed, Tsp, RQRt, at, Pt, Pinft);
    // Checking colapsed
    if (!colapsed && all(all(abs(Pinft) < 1e-6))){
        colapsed = true;
        data.d_t = t;
    }
    // Storing final state and covariance for forecasting
    if (t == ny - 1){
      data.PEnd = Pt;
      data.aEnd = at;
    }
  }
  // Smoothing loop
  data.F = data_F;   // For final normalization of innovations
  if (smooth > 0){
    mat Nt(ns, ns),
        Ninfti(ns, ns),
        N2t(ns, ns),
        PPinf(ns, ns); //, RR(ns, ns), sysmatQ, sysmatR, Z = data->system.Z;
    mat Inew(ns, ns),
        Lt(ns, ns),
        Linft(ns, ns),
        LinftNt(ns, ns),
        Ninft(ns, ns);
    vec rt(ns),
        rinft(ns),
        Kinft(ns),
        Z_Ft(ns),
        Z_Finft(ns); //, eta; //, vt(1), Kt(ns), Ft(1), GammaD(1);
      mat QRt,
          Veta,
          pinvVeta;
    bool colapsed = true;
    if (smooth == 2){   // Disturbance
      int rQ = data.system.Q.n_rows; //, cR = data.system.R.n_cols;
      QRt = data.system.Q * data.system.R.t();
      data.eta.zeros(rQ, n - data.h);
      Veta.zeros(rQ, rQ);
    }
    // Storing in case of outlier detection
    if (smooth == 3){
      data.rNrOut = zeros(n);
      data.rOut = zeros(ns, n);
      data.NOut = zeros(ns, ns, n);
    }
    // Initialising variables
    Nt.fill(0);
    Ninft = Nt;
    N2t = Nt;
    rt.fill(0);
    rinft = rt;
    Inew.eye();
    // Fast state smoother: when only smoothed STATES (data.a) are consumed
    // (e.g. components()), skip the entire backward Nt recursion and the
    // O(m^3) smoothed-variance work (data.P/F, disturbance, outlier) -- those
    // outputs are not surfaced.  The state recursion (rt/rinft) is independent
    // of Nt, so smoothed states stay identical.
    bool needNt = !data.stateOnly;
    // Main Loop
    int intN = n;
    for (int t = n - 1; t >= 0; t--){
      if (TVP)
        Z = data.system.Z.row(t);
      if (t <= data.d_t){
        colapsed = false;
      }
      vt = data.v.row(t);
      Kt = data.K.col(t);
      iFt = data.iF.row(t);
      Ft = data.F.row(t);
      if (!colapsed){
        Finft = data.Finf.row(t);
        Kinft = data.Kinf.col(t);
      }
      // if (is_finite(data.y.row(t))){
      if (std::isfinite(data.y(t))){
        miss = false;
        if (colapsed || Finft(0, 0) < 1e-8) {
          Lt = data.system.T - data.system.T * Kt * Z;
          Z_Ft = Z.t() * iFt(0);
          rt = Z_Ft * vt + Lt.t() * rt;
          if (needNt) Nt = Z_Ft * Z + Lt.t() * Nt * Lt;
          if (!colapsed){
              rinft = data.system.T.t() * rinft;
              if (needNt){
                N2t = data.system.T.t() * N2t * data.system.T;
                Ninft = data.system.T.t() * Ninft * Lt;
              }
          }
        } else if (Finft(0, 0) >= 1e-8) {
            Lt = data.system.T - data.system.T * Kinft * Z;
            Linft = -data.system.T * Kt * Z;
            Z_Finft = Z.t() * iFt(0, 0);  //  / Finft(0, 0);
            rinft = Z_Finft * vt + Lt.t() * rinft + Linft.t() * rt;
            rt = Lt.t() * rt;
            if (needNt){
              LinftNt = Lt.t() * Ninft * Linft;
              N2t = -Z_Finft * Z_Finft.t() * Ft(0, 0) + Linft.t() * Nt * Linft +
                  LinftNt + LinftNt.t() + Lt.t() * N2t * Lt;
              Ninft = Z_Finft * Z + Lt.t() * Ninft * Lt + Linft.t() * Nt * Lt;
              Nt = Lt.t() * Nt * Lt;
            }
        }
      } else {
        miss = true;
        // No information at t: r_{t-1} = T' r_t, N_{t-1} = T' N_t T (and
        // likewise for the diffuse quantities).  Must happen here, before
        // data.a.col(t) is computed below, so the smoothed state at this
        // missing observation uses r_{t-1} rather than the stale r_t.
        rt = data.system.T.t() * rt;
        if (!colapsed)
          rinft = data.system.T.t() * rinft;
        if (needNt){
          Nt = data.system.T.t() * Nt * data.system.T;
          if (!colapsed){
            Ninft = data.system.T.t() * Ninft * data.system.T;
            N2t = data.system.T.t() * N2t * data.system.T;
          }
        }
      }
      Pt = cP.slice(t);
      data.a.col(t) += Pt * rt;
      if (!colapsed){
        Pinft = Pinf.slice(t);
        data.a.col(t) += Pinft * rinft;
      }
      data.yFit.row(t) = Z * data.a.col(t) + Dt(t); // + data.system.D;
      if (needNt){
        Pt -= Pt * Nt * Pt;
        if (!colapsed){
          PPinf = Pinft * Ninft * Pt;
          Pt = Pt - PPinf - PPinf.t() - Pinft * N2t * Pinft;
        }
        data.F.row(t) = Z * Pt * Z.t() + CHCt;
        data.P.col(t) = Pt.diag();
        //Disturbance smoother
        if (smooth == 2 && t < intN - data.h){
          // Raw smoothed disturbance E[eta_t | y] = Q R' r_t.  The
          // theoretical per-step standardisation by sqrt(diag(Veta)) with
          // Veta = Q - Q R' N_t R Q is SINGULAR when a disturbance variance
          // collapses (Veta.diag -> 0, e.g. an estimated variance near zero):
          // it produces inf/nan that the empirical re-standardisation below
          // then discards, ZEROING the very component carrying an outlier.
          // Leave the disturbance raw here; the cov/pinv block after the loop
          // standardises each component to a finite, unit-variance auxiliary
          // residual, so a spike survives as a large t-stat regardless of the
          // variance regime.
          data.eta.col(t) = QRt * rt;
        }
        // Storing for outlier detection
        if (smooth == 3){
          data.rNrOut.row(t) = rt.t() * pinv(Nt) * rt;
          data.rOut.col(t) = rt;
          data.NOut.slice(t) = Nt;
        }
      }
    }
  }
  // Post-processing outputs
  // MLE σ̂² = SSR / n_finite, matching llik() / llikAug().
  int nFinite = (int)n - (int)nMiss;
  if (nFinite < 1) nFinite = 1;
  double innVar = data.innVariance;
  if (data.cLlik){         // Concentrated Likelihood (MLE)
    innVar = v2F(0, 0) / nFinite;
  }
  data.y = data.y.rows(0, ny - 1);
  data_F *= innVar;
  data.P *= innVar;
  data.FFor *= innVar; // * scale;
  // Cleaning innovations
  // if ((uword)data.d_t < n - 10){
  //   data.v.rows(0, data.d_t).fill(datum::nan);
  // } else {
  //   data.v.rows(0, sum(ns) + 1).fill(datum::nan);
  // }

  if (!data.cleanInnovations) {
      if ((uword)data.d_t < n - 10){
          data.v.rows(0, data.d_t).fill(datum::nan);
      } else {
          data.v.rows(0, sum(ns) + 1).fill(datum::nan);
      }
  }

  uvec ind = find_finite(data.v);

  // Mantener tamaño completo (NO recortar, evita el shift con NaNs iniciales)
  data.F = data_F;

  // Normalizar solo los valores válidos
  if (smooth != 3 && !data.cleanInnovations) {
      for (uword i = 0; i < ind.n_elem; i++) {
          uword t = ind(i);
          data.v(t) = data.v(t) / std::sqrt(data.F(t));
      }
  }

  // uvec ind = find_finite(data.v);
  // if (ind.n_elem < 5){
  //     ind = regspace<uvec>(sum(ns), data.v.n_elem - data.h - 1);
  // }
  // if (smooth == 3){
  //   data.F = data_F.rows(0, max(ind)); // * data.innVariance;
  //   data.v = data.v.rows(0, max(ind));
  // } else {
  //   data.F = data_F.rows(min(ind), max(ind));
  //   data.v = data.v.rows(min(ind), max(ind));
  //   data.v = data.v / sqrt(data.F);
  // }
  // Disturbances: standardise each component to a unit-variance auxiliary
  // residual (Harvey-Koopman diagnostic).  Do it PER COLUMN, dividing by the
  // component's own empirical SD -- a single matrix pinv() applies one
  // tolerance across all components and silently zeroes any whose variance is
  // far below the largest (e.g. a near-zero observation/level variance when
  // another component dominates), which destroys the outlier signal there.
  if (smooth == 2){
    data.eta = data.eta.t();   // n x rQ
    for (uword j = 0; j < data.eta.n_cols; j++){
      double s = stddev(data.eta.col(j));
      if (s > 0 && std::isfinite(s)) data.eta.col(j) /= s;
      else                           data.eta.col(j).zeros();
    }
    data.eta = data.eta.t();
    data.eta.replace(datum::nan, 0);
    data.eta.replace(datum::inf, 0);
  }
}
// Calculating innovations from very beginning
vec KFinnovations(SSinputs& data) {
    // 1. Ejecutar KF aumentado para obtener estados iniciales óptimos
    vec p = data.p;
    llikAug(p, &data);

    // Dimensiones
    uword n  = data.y.n_elem;
    uword ns = data.system.T.n_rows;
    uword top_t = std::max((int)data.d_t, (int)ns) + 1;

    vec inn(top_t);

    // 2. Inicialización desde betaAug
    vec at = data.betaAug.rows(0, ns - 1);
    mat Pt = data.betaAugVarMat.submat(0, 0, ns - 1, ns - 1);

    // Sin fase difusa
    mat Pinft(ns, ns, fill::zeros);

    // Precalcular matrices
    mat RQRt = data.system.R * data.system.Q * data.system.R.t();
    mat CHCt = data.system.C * data.system.H * data.system.C.t();

    // Salidas
    data.v.zeros(n);
    data.F.zeros(n);
    data.iF.zeros(n);

    vec vt(1), Kt(ns);
    mat Ft(1,1), iFt(1,1), Finft(1,1);

    bool miss = false;
    bool steadyState = false;
    bool colapsed = true;   // clave: sin fase difusa

    mat Z = data.system.Z.row(0);
    bool TVP = (data.system.Z.n_rows > 1);
    arma::sp_mat Tsp(data.system.T);   // sparse view for the forward recursion

    // Inputs exógenos
    int k = data.u.n_rows;
    rowvec Dt(n, fill::zeros);
    if (k > 0 && data.system.Z.n_rows == 1) {
        int nn = data.betaAug.n_rows;
        data.system.D = data.betaAug.rows(nn - k, nn - 1).t();
        Dt = data.system.D * data.u;
    }

    // ---- LOOP KF ----
    for (uword t = 0; t < top_t; t++) {

        if (TVP)
            Z = data.system.Z.row(t);

        // Missing
        if (!std::isfinite(data.y(t))) {
            miss = true;
        } else {
            miss = false;
        }

        // ----- CORRECCIÓN -----
        KFcorrection(miss,
                     colapsed,
                     steadyState,
                     false,
                     &data,
                     CHCt,
                     Finft,
                     vt,
                     Dt(t),
                     Ft,
                     iFt,
                     at,
                     Pt,
                     Pinft,
                     Kt,
                     t,
                     data.Finf,
                     data.Kinf,
                     Z);

        // Guardar innovación SIN eliminar nada
        if (!miss) {
            inn(t) = vt(0);
            data.F(t) = Ft(0,0);
            data.iF(t) = iFt(0,0);
        } else {
            inn(t) = datum::nan;
        }

        // ----- PREDICCIÓN -----
        KFprediction(steadyState,
                     colapsed,
                     Tsp,
                     RQRt,
                     at,
                     Pt,
                     Pinft);
    }

    // Normalización (opcional, como en tu framework)
    // uvec ind = find_finite(data.v);
    // for (uword i = 0; i < ind.n_elem; i++) {
    //     uword t = ind(i);
    //     data.v(t) = data.v(t) / std::sqrt(data.F(t));
    // }
    // uvec ind = find_finite(data.v);
    // if (!data.cleanInnovations) {
    //     for (uword t = 0; t < data.y.n_elem; t++) {
    //          data.v(t) = data.v(t) * std::sqrt(data.iF(t));
    //     }
    // }
    // Standardising
    if (!data.cleanInnovations)
        inn = inn * sqrt(data.iF.rows(0, inn.n_elem - 1));
    return inn;
}
// solution to lyapunov equation P = Phi * P * Phi' + Q
mat dlyap(mat T, mat Q){
    uword n = T.n_cols;
    mat Ur, Phir;
    schur(Ur, Phir, T);  // U * S * U' = T
    mat ceros = zeros(n, n);
    cx_mat U = cx_mat(Ur, ceros), Phi = cx_mat(Phir, ceros);
    // Pass schur to complex schur
    uvec k(2);
    cx_vec mu(2), r(1), c(1), s(1);
    cx_mat G(2, 2);
    for (uword m = n - 1; m >= 1; --m) {
        if (norm(Phi(m, m - 1)) != 0.0) {
            k(0) = m - 1; k(1) = m;  //regspace<uvec>(m - 1, m);
            mu = eig_gen(Phi(k, k)) - Phi(m, m) * ones<cx_vec>(2);
            // r = std::hypot(mu(0), Phir(m, m - 1));
            r = sqrt(norm(mu(0)) + norm(Phi(m, m - 1)));
            c = mu(0) / r;
            s = Phi(m, m - 1) / r;
            G = join_rows(join_cols(c.t(), -s), join_cols(s, c));
            Phi.submat(m - 1, m - 1, m, n - 1) = G * Phi.submat(m - 1, m - 1, m, n - 1);
            Phi.submat(0, m - 1, m, m) = Phi.submat(0, m - 1, m, m) * G.t();
            U.cols(k) = U.cols(k) * G.t();
            Phi(m, m - 1) = cx_double(0.0, 0.0);
        }
    }
    // finding matrix P
    cx_mat Qc = U.t() * Q * U, P = zeros<cx_mat>(n, n);
    c = cx_double(1.0, 0.0);
    for (int j = n-1; j >= 0; --j) {
        for (int i = n-1; i >= 0; --i) {
            r = c - Phi(i,i) * Phi.submat(j, j, j, j).t();
            s = Phi.submat(i, i, i, n - 1) * P.submat(i, j, n - 1, n - 1) *
                    Phi.submat(j, j, j, n - 1).t() + Qc.submat(i, j, i, j);
            if (norm(s) < pow(2, -52)) {
                P(i, j) = cx_double(0.0, 0.0);
            } else if (norm(r) < pow(2, -52)) {
                throw std::runtime_error("SSpace: Lyapunov equation with no solution!!!");
            } else {
                P(i, j) = s(0) / r(0);
            }
        }
    }
    return real(U * P * U.t());
}

