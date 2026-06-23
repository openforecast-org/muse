/*
 Basic Structural models
 Additional inputs for R, MATLAB, python:
 TVP
 trendOptions
 seasonalOptions
 irregularOptions
 */
// #include <iostream>
// #include <math.h>
// #include <string.h>
// #include <armadillo>
// using namespace arma;
// using namespace std;
#include <iostream>
#include <math.h>
#include <string.h>
#include <armadillo>
using namespace arma;
using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
#include "boxcox.h"
#include "bcnorm.h"
#include "SSpace.h"
#include "ARMAmodel.h"


struct BSMmodel{
        // INPUTS:
        string model = "llt/none/equal/arma(0,0)",  // model to fit
                criterion = "aic";                      // identification criterion
        bool stepwise,              // stepwise identification or brute one
        tTest,                 // unit roots test or not for identification
        arma;                  // check arma models for irregular component
        vec periods,                // vector of periods for harmonics
        TVP;                    // vector of zeros and ones to allocate position of TVP parameters
        bool MSOE = false,          // MSOE model or BSM
                PTSnames = false,      // PTS model names or UC
                succeed = true;        // identification sucseed or not
        // OUTPUTS:
        string trend,               // type of trend
        cycle,                  // type of cycle
        seasonal,               // type of seasonal component
        irregular,              // type of irregular
        cycle0,                 // type of cycle without numbers
        compNames = "",         // components names
        trendOptions = "none/rw/llt/td",          // trend options to select amongst (none/rw/llt/td/dt/srw)
        seasonalOptions = "none/linear/equal", // seasonal options to select amongst (none/equal/different/linear)
        irregularOptions = "arma(0,0)";      // irregular components (none/arma(p,q))
        int ar = 0, ma = 0;         // number of FREE AR / MA coefficients
                                    // (= sum of arOrders / maOrders entries)
        int arDeg = 0, maDeg = 0;   // EXPANDED polynomial degree
                                    // (= sum_i arOrders[i] * armaLags[i] ...)
                                    // — sizes the ARMA state block.
        ivec arOrders,              // per-lag AR block orders, e.g. [p, P]
             maOrders,              // per-lag MA block orders, e.g. [q, Q]
             armaLags;              // matching lags, e.g. [1, s].  Length 1
                                    // for non-seasonal arma(p,q); length 2
                                    // for SARMA(p,q)(P,Q)_s.
        double seas,                // seasonal period
        lambda = 1.0;              // Box-Cox transformation parameter
        vec rhos,                   // vector indicating whether period is cyclical or seasonal
        ns,                     // number of states in components (trend, cycle, seasonal, irregular)
        nPar,                   // number of parameters in components (trend, cycle, seasonal, irregular)
        p0Return,               // initial parameters user understandable
        typePar,                // type of parameter (0: variance;
        //         -1: damped of trend;
        //          1: cycle rhos;
        //          2: cycle periods;
        //          3: ARMA;
        //          4: inputs)
        eps,                    // observation perturbation
        beta0ARMA,              // initial estimates of ARMA model (without variance)
        constPar;               // constrained parameters (0: not constrained;
        //                         1: concentrated-out;
        //                         2: variance constrained to 0;
        //                         3: alpha constrained to 0 or 1)
        uvec harmonics;             // vector with the indices of harmonics selected
        mat comp,                   // estimated components
        compV,                  // variance of components
        typeOutliers,           // Matrix with type of outliers and sample of each outlier
        cycleLimits;            // limits for period of cycle estimation
        bool pureARMA = false,      // Pure ARMA model flag
                Drift = false,         // trend with drift
                estimateLambda = false, // joint-BFGS includes lambda as last
                                        // element of `p` when true
                profileLambda = false,  // enable joint-BFGS lambda estimation
                                        // inside ident() / estimUCs() — name
                                        // is a vestige of an earlier Brent
                                        // implementation (now removed)
                lambdaEstimated = false;// engine's authoritative DoF flag for lambda
                                        // (true => lambda jointly estimated, +1 DoF;
                                        //  false => lambda held fixed)
        vector<string> parNames;    // Parameter names
};
/**************************
 * Model CLASS BSM
 ***************************/
class BSMclass : public SSmodel{
private:
        BSMmodel inputs;
        // Set model
        void setModel(string, vec, vec, bool);
        // Count states and parameters of BSM model
        void countStates(vec, string, string, string, string);
        // Initializing parameters of BSM model
        void initParBsm();
        // Optimization routine
        int quasiNewtonBSM(std::function <double (vec&, void*)>,
                           std::function <vec (vec&, void*, double, int&)>,
                           vec&, void*, double&, vec&, mat&, bool);
        // Estimation of a family of UC models
        void estimUCs(vector<string>, uvec, double&, bool, double, int);
public:
        // Constructors
        BSMclass();
        BSMclass(SSinputs, BSMmodel);
        // Fix matrices in standard BSM models (all except variances)
        void initMatricesBsm(vec, vec, string, string, string, string);
        // Convert to MSOE
        void bsm2msoe();
        // Parameter names
        void parLabels();
        // Interpolation of initial NaN values
        void interpolate(int);
        //Estimation
        void estim(bool);
        void estim(vec, bool);
        // Plug in a previously-estimated parameter vector and warm up the
        // filter so forecast() can run without re-optimisation.
        void setEstimatedParams(vec);
        // Evaluate the loss (concentrated objective) at a user-supplied
        // Identification
        void ident(string, bool);
        // Outlier detection
        void estimOutlier(vec, bool);
        // Check whether re-estimation is necessary
        void checkModel(uvec);
        // Components
        void components();
        // Covariance of parameters (inverse of hessian)
        mat parCov(vec&);
        // Finding true parameter values out of transformed parameters
        vec parameterValues(vec);
        // Validation of BSM models
        void validate(bool);
        // Disturbance smoother (to recover just trend and epsilons)
        void disturb();
        // Get data
        BSMmodel getInputs(){
                parLabels();
                return inputs;
        }
        // Set data
        void setInputs(BSMmodel inputs){
                this->inputs = inputs;
        }
        //Print inputs on screen
        // void showInputs();
};
/***************************************************
 * Auxiliar function declarations
 ****************************************************/
// Convert UC model to PTS
string UC2PTS(string, double);
// States names for filtering and smoothing
string stateNames(BSMmodel);
// Pre-processing
bool preProcess(vec, mat&, string&, int&, double&, string&,
                vec, vec, int&, string, string, string, vec&, double&);
// Variance matrices in standard BSM
void bsmMatrices(vec, SSmatrix*, void*);
// Variance matrices in standard BSM for true parameters
void bsmMatricesTrue(vec, SSmatrix*, void*);
// Extract trend seasonal and irregular of model in a string
void splitModel(string, string&, string&, string&, string&);
// SS form of trend models
void trend2ss(int, mat*, mat*);
// SS form of seasonal models
void bsm2ss(int, int, vec, vec, mat*, mat*);
// Remove elements of vector in n adjacent points
uvec selectOutliers(vec&, int, float);
// Create dummy variable for outliers 0: AO, 1: LS, 2: SC
void dummy(uword, uword, rowvec&);
// combining UC models
void findUCmodels(string, string, string, string, vector<string>&);
// Find first observation of n non-nan contiguous values
int findFirst(vec, int);
// Show SS model
void showSS(SSmatrix);
// Show BSMmodel
void showBSM(BSMmodel);

/****************************************************
 // BSM implementations for univariate UC models
 ****************************************************/
// Constructors
BSMclass::BSMclass(){
}
BSMclass::BSMclass(SSinputs data, BSMmodel inputs) : SSmodel(data){
        SSmodel::inputs = data;
        this->inputs = inputs;
        this->inputs.rhos = ones(size(inputs.periods));
        this->inputs.cycleLimits.resize(1, 1);
        this->inputs.cycleLimits(0, 0) = datum::nan;
        vec reserve = inputs.constPar;
        setModel(inputs.model, inputs.periods, inputs.rhos, true);
        if (!reserve.has_nan() && reserve.n_elem > 0)
                this->inputs.constPar = reserve;
        this->inputs.harmonics = regspace<uvec>(0, inputs.periods.n_elem - 1);
}
// Pre-processing
bool preProcess(vec y, mat& u, string& model, int& h, double& outlier,
                string& criterion, vec periods, vec p0, int& iniObs,
                string trendOptions, string seasonalOptions, string irregularOptions,
                vec& TVP, double& lambda){
        // Looking for first observation for estimation
        // if (y.has_nan()){
        //     uword nskip = 2 * periods.n_elem + 3;
        //     uvec indFinite, poly(3, fill::ones), aux3;
        //     indFinite = find_finite(y);
        //     aux3 = conv(diff(indFinite), poly);
        //     //iniObs = indFinite(min(find(aux3.rows(2, aux3.n_elem - 1) == 3))) + iMin;
        //     iniObs = indFinite(min(find(aux3.rows(nskip - 1, aux3.n_elem - 1) == nskip))) + iMin;
        // } else {
        //     iniObs = 0;
        // }
        // vec p(1);
        // p.fill(datum::nan);
        // Correcting inputs by user
        // Checking for constant values
        double vary = var(y.elem(find_finite(y)));
        if (vary == 0.0) {
            u.reset();
            model = "none/none/none/arma(0,0)";
            outlier = 0.0;
            lambda = 1.0;
        }
        iniObs = findFirst(y, iniObs);
        lower(criterion);
        deblank(model);
        lower(model);
        lower(trendOptions);
        lower(irregularOptions);
        lower(seasonalOptions);
        // checking trendOptions
        vector<string> aux;
        bool alright = false; uword ind = 0;
        chopString(trendOptions, "/", aux);
        do{
                alright = false;
                if (aux[ind] == "none")
                        alright = true;
                else if (aux[ind] == "rw")
                        alright = true;
                else if (aux[ind] == "irw")
                        alright = true;
                else if (aux[ind] == "llt")
                        alright = true;
                else if (aux[ind] == "dt")
                        alright = true;
                else if (aux[ind] == "srw")   // Hyndman damped
                        alright = true;
                else if (aux[ind] == "td")
                        alright = true;
                ind++;
        } while(alright == true && ind < aux.size());
        if (alright == false){
                Rprintf("%s", "ERROR: trendOptions wrongly set (none/rw/irw/llt/dt/td/hd)!!!\n");
                return true;
        }
        // checking seasonalOptions
        ind = 0;
        if (max(periods) == 1)
                seasonalOptions = "none";
        chopString(seasonalOptions, "/", aux);
        do{
                alright = false;
                if (aux[ind] == "none")
                        alright = true;
                else if (aux[ind] == "equal")
                        alright = true;
                else if (aux[ind] == "different")
                        alright = true;
                else if (aux[ind] == "linear")
                        alright = true;
                ind++;
        } while(alright == true && ind < aux.size());
        if (alright == false){
                Rprintf("%s", "ERROR: seasonalOptions wrongly set (none/equal/different/linear)!!!\n");
                return true;
        }
        // checking irregularOptions
        alright = false; ind = 0;
        chopString(irregularOptions, "/", aux);
        do{
                alright = false;
                if (aux[ind] == "none")
                        alright = true;
                else if (aux[ind].find("arma(") != std::string::npos)
                        alright = true;
                ind++;
        } while(alright == true && ind < aux.size());
        if (alright == false){
                Rprintf("%s", "ERROR: irregularOptions wrongly set (none/arma(p,q))!!!\n");
                return true;
        }
        // Checking TVP
        if (any((TVP > -999) && (TVP < 0))){
                Rprintf("%s", "ERROR: TVP vector should be positive!!!\n");
                return true;
        }
        if (u.n_rows > 0){
            TVP = sort(TVP);
            if (TVP.n_elem == 0)
                TVP.zeros(u.n_rows);
            else if (TVP.n_elem < u.n_rows){
                vec aux2(u.n_rows - TVP.n_elem, fill::zeros);
                TVP = join_vert(TVP, aux2);
            } else if (TVP.n_elem > u.n_rows){
                TVP = TVP.rows(0, u.n_rows - 1);
            }
        } else {
            TVP = {};
        }
        // Clamping user-supplied fixed lambda to the global BC range
        // (LAMBDA_BOUND_LOWER, LAMBDA_BOUND_UPPER, defined in boxcox.h).
        // The previous cap was |lambda| > 1 -> sign(lambda), which
        // silently rewrote any user-supplied lambda outside [-1, 1] to
        // +/- 1.  That made `2LD` and `1LD` indistinguishable (the
        // engine ran both at lambda = 1) and was asymmetric with the
        // joint-BFGS path, which freely converges into [-2, 2] and snaps
        // to the same anchor set.
        if (lambda != 9999.9){
                if (lambda < LAMBDA_BOUND_LOWER) lambda = LAMBDA_BOUND_LOWER;
                else if (lambda > LAMBDA_BOUND_UPPER) lambda = LAMBDA_BOUND_UPPER;
        }
        // Checking forecasting horizon
        int initPer;
        initPer = max(periods);
        h = abs(h);
        if (h == 9999){
                if (initPer == 1)
                        initPer = 5;
                h = 2 * initPer;
        }
        if (outlier == -9999)
                outlier = 0.0;
        // Correcting h in case there are inputs
        if (u.n_rows > 0){
                h = u.n_cols - y.n_elem;
                if (h < 0){
                        Rprintf("%s", "ERROR: Inputs should be at least as long as the ouptut!!!\n");
                        return true;
                }
        }
        // Checking periods
        if (min(periods) < 1){
                Rprintf("%s", "ERROR: All periods should be higher or equal to zero!!!\n");
                return true;
        }
        // Removing nans at beginning or end
        /*
         if (!y.row(0).is_finite() || !y.row(y.n_elem - 1).is_finite()){
         uvec ind = find_finite(y);
         int minInd = min(ind), maxInd = max(ind);
         y = y.rows(minInd, maxInd);
         if (u.n_rows > 0){
         u = u.cols(minInd, maxInd);
         }
         }
         */
        // Checking periods
        //    if (is.ts(y) && is.na(periods) && frequency(y) > 1){
        //        periods = frequency(y) / (1 : floor(frequency(y) / 2))
        //    } else if (is.ts(y) && is.na(periods)){
        //        periods = 1
        //    } else if (!is.ts(y) && is.na(periods)){
        //        stop("Input \"periods\" should be supplied!!")
        //    }
        if (model.find("/") == string::npos){
                Rprintf("%s", "ERROR: Incorrect number of components (trend/seasonal/irregular)!!!\n");
                return true;
        }
        vector<string> comps;
        chopString(model, "/", comps);
        // Number of components
        if (comps.size() < 3 || comps.size() > 4){
                Rprintf("%s", "ERROR: Incorrect number of components (trend/seasonal/irregular)!!!\n");
                return true;
        }
        // Adding cycle in case of T/S/I model specification
        if (comps.size() == 3){
                model = comps[0] + "/none/" + comps[1] + "/" + comps[2];
                chopString(model, "/", comps);
        }
        // Setting seasonal to none for annual data
        if (max(periods) == 1){
                if (comps.size() == 3){
                        comps[1] = "none";
                        model = comps[0] + "/" + comps[1] + "/" + comps[2];
                }
                if (comps.size() == 4){
                        comps[2] = "none";
                        model = comps[0] + "/" + comps[1] + "/" + comps[2] + "/" + comps[3];
                }
        }
        // Checking model
        if (comps[1][0] != 'n' && comps[2][0] == 'l'){
                Rprintf("%s", "ERROR: Cycle can estimated only with trigonometric seasonal components!!!\n");
                return true;
        }
        if (comps[0][0] == 'n' && comps[1][0] == 'n' && comps[2][0] == 'n' && comps[3][0] == 'n'){
                Rprintf("%s", "ERROR: No correct model specified!!!\n");
                return true;
        }
        if (model.find("?") == string::npos && p0(0) != -9999.9){
                p0.set_size(1);
                p0(0) = -9999.9;
        }
        if (model.find("?") == string::npos && isnan(p0(0))){
                p0.set_size(1);
                p0(0) = datum::nan;
        }
        if (model.find("arma") != string::npos && model.find("(") == string::npos){
                model = model + "(0,0)";
                chopString(model, "/", comps);
        }
        if (model.find("arma") != string::npos && model[model.length() - 1] != ')'){
                model = model + ")";
                chopString(model, "/", comps);
        }
        // Checking components
        string opt = "?nritlds";
        if (opt.find(comps[0][0]) == string::npos){
                Rprintf("%s", "ERROR: Incorrect TREND model (? / none / rw / irw / td / llt / dt / srw)!!!\n");
                return true;
        }
        opt = "?n+-0123456789";
        if (opt.find(comps[1][0]) == string::npos){
                Rprintf("%s", "ERROR: Incorrect CYCLE model (? / none / +-integer)!!!\n");
                return true;
        }
        opt = "?nedl";
        if (opt.find(comps[2][0]) == string::npos){
                Rprintf("%s", "ERROR: Incorrect SEASONAL model (? / none / equal / different / linear)!!!\n");
                return true;
        }
        if (comps[3][0] == 'a' && model.find("arma") == string::npos){
                Rprintf("%s", "ERROR: Incorrect IRREGULAR model (? / none / arma(p, q))!!!\n");
                return true;
        }
        opt = "?na";
        if (opt.find(comps[3][0]) == string::npos){
                Rprintf("%s", "ERROR: Incorrect IRREGULAR model (? / none / arma(p, q))!!!\n");
                return true;
        }
        // Set rhos
        //vec rhos(periods.n_elem, fill::ones);
        // Checking cycle
        if (comps[1][0] == '?'){
                initPer *= -4;
                if (initPer == -4)
                        initPer = -8;
                comps[1] = to_string(initPer) + '?';
        } else if (comps[1][0] != '+' && comps[1][0] != '-' && comps[1][0] != 'n')
                comps[1] = '+' + comps[1];
        model = comps[0] + "/" + comps[1] + "/" + comps[2] + "/" + comps[3];
        // Checking criterion
        if (criterion != "aic" && criterion != "bic" &&
            criterion != "aicc" && criterion != "bicc"){
                criterion = "aic";
        }
        // Building structures
        // SSinputs inputsSS;
        // BSMmodel inputsBSM;
        // inputsSS.y = y;
        // inputsSS.u = u;   // Inputs data (mat)
        // inputsBSM.model = model;
        // inputsSS.h = h;
        // inputsBSM.tTest = tTest;
        // inputsBSM.criterion = criterion;
        // inputsSS.outlier = outlier;
        // vec aux(1); aux(0) = outlier;
        // if (aux.has_nan()){
        //     outlier = 0;
        // }
        return false;
        // inputsBSM.periods = periods;
        // inputsSS.verbose = verbose;
        // inputsBSM.stepwise = stepwise;
        // inputsSS.p0 = p0;
        // inputsBSM.arma = arma;
        //inputsBSM.errorExit = errorExit;
        //inputsBSM.rhos = rhos;

        //vec VOID(1); VOID.fill(datum::nan);
        // inputsSS.p = VOID;
        // inputsSS.grad = VOID;
        // inputsSS.criteria = VOID;
        // inputsSS.d_t = 0;
        // inputsSS.innVariance = VOID(0);
        // inputsSS.objFunValue = VOID(0);
        // inputsSS.cLlik = true;
        // inputsSS.Iter = 0;
        // inputsSS.nonStationaryTerms = VOID(0);
        // inputsBSM.ns = VOID;
        // inputsBSM.nPar = VOID;
        // if (harmonics.has_nan()){
        //     inputsBSM.harmonics.resize(1);
        //     inputsBSM.harmonics(0) = 0;
        // } else {
        //     inputsBSM.harmonics = conv_to<uvec>::from(harmonics);
        // }
        // inputsBSM.constPar = VOID;
        // inputsBSM.typePar = VOID;
        // inputsBSM.typeOutliers = {-1, -1};
        // inputsBSM.cycleLimits = VOID;

        //inputsBSM.seas = max(periods);
        //inputsSS.p = {datum::nan};        // Estimated parameters (vec)
        // inputsSS.v = {datum::nan};        // Estimated innovations (vec)
        // inputsSS.yFit = {datum::nan};     // Fitted values (vec)
        // inputsSS.yFor = {datum::nan};     // Point forecasts (vec)
        // inputsSS.F = {datum::nan};        // Innovations variance (vec)
        // inputsSS.FFor = {datum::nan};     // Variance of forecasts (vec)
        // inputsBSM.comp = {datum::nan};    // Estimated components (mat)
        // inputsBSM.compV = {datum::nan};   // Variance of estimated components (mat)
        // inputsSS.a = {datum::nan};        // Estimated states (mat)
        // inputsSS.P = {datum::nan};        // Estimated variance of states (mat)
        // inputsSS.eta = {datum::nan};      // Estimated transition perturbations (mat)
        // inputsBSM.eps = {datum::nan};     // Estimated observed perturbations (vec)
        // inputsSS.criteria = {datum::nan}; // Likelihood and information criteria at optimum (vec)
        // inputsBSM.cycleLimits = {datum::nan};
        // inputsBSM.rhos = inputsBSM.periods; inputsBSM.rhos.fill(1);

        //BSMclass m(inputsSS, inputsBSM);
        //return m;
        //if (errorExit)
        //    return m;
        //   if (y.has_nan())
        //        m.interpolate();
        //return m;

        // Building UComp system
        //BSMclass sysBSM(inputsSS, inputsBSM);
}
// Set model (part of constructor)
void BSMclass::setModel(string model, vec periods, vec rhos, bool runFromConstructor){
        string trend, cycle, seasonal, irregular;
        vec ns(7), nPar(7), typePar, noVar, constPar;
        splitModel(model, trend, cycle, seasonal, irregular);
        // (PTS never uses stochastic cycles -- cycle is always "none" -- so
        // the cycle-string correction and cycle-period limit computation that
        // used to sit here were unreachable and have been removed.)
        this->inputs.trend = trend;
        this->inputs.cycle = cycle;
        this->inputs.seasonal = seasonal;
        this->inputs.irregular = irregular;
        // Checking for arma identification
        if (irregular != "?"){
                inputs.arma = 0;
        }
        // Checking for constant input from user and removing in that case
        if (SSmodel::inputs.u.n_rows > 0){
                uvec rowCnt = find(sum(SSmodel::inputs.u - 1, 1) == 0);
                SSmodel::inputs.u.shed_rows(rowCnt);
        }
        // Initializing matrices
        if (trend != "?" && cycle != "?" && seasonal != "?" && irregular != "?"){  // One model
                initMatricesBsm(periods, rhos, trend, cycle, seasonal, irregular);
                this->inputs.model = model;
                this->SSmodel::inputs.userInputs = &this->inputs;
                // User function to fill the changing matrices
                this->SSmodel::inputs.userModel = bsmMatrices;
                // Initializing parameters of BSM model
                if (!runFromConstructor)
                        SSmodel::inputs.p0(0) = -9999.9;
                typePar = this->inputs.typePar;
                // inputs.beta0ARMA.reset();
                initParBsm();
                // Convert to MSOE form
                if (inputs.MSOE)
                        bsm2msoe();
                // Keep SSinputs::lambda in sync so llik() can use it for the
                // BCnorm Jacobian without needing to cast userInputs.
                SSmodel::inputs.lambda = inputs.lambda;
        }
        // Making coherent h and size(u)
        if (SSmodel::inputs.u.n_elem > 0){
                SSmodel::inputs.h =  SSmodel::inputs.u.n_cols - SSmodel::inputs.y.n_elem;
        }
}
// Convert bsm system to msoe
void BSMclass::bsm2msoe(){
        bool seas = false;
        if (inputs.seasonal[0] != 'n')
                seas = true;
        uword nsAdd = seas + 1, ns = SSmodel::inputs.system.T.n_rows;
        // Change R
        SSmodel::inputs.system.R = join_vert(SSmodel::inputs.system.R, zeros(nsAdd, SSmodel::inputs.system.R.n_cols));
        // Change T and Z
        mat aux(ns + nsAdd, ns + nsAdd, fill::zeros);
        aux.submat(0, 0, ns - 1, ns - 1) = SSmodel::inputs.system.T;
        SSmodel::inputs.system.T = aux;
        SSmodel::inputs.system.T.row(ns) = SSmodel::inputs.system.T.row(0);
        SSmodel::inputs.system.Z = join_horiz(SSmodel::inputs.system.Z, zeros(1, nsAdd));
        SSmodel::inputs.system.Z(ns) = 1.0;
        if (seas){
                if (inputs.seasonal[0] == 'l'){
                        SSmodel::inputs.system.T.row(ns + 1) = SSmodel::inputs.system.T.row(inputs.ns(0));
                        SSmodel::inputs.system.Z.cols(0, inputs.ns(0) + inputs.ns(1) + inputs.ns(2) - 1).fill(0.0);
                        SSmodel::inputs.system.Z(ns + 1) = 1.0;
                } else {
                        SSmodel::inputs.system.Z.cols(0, inputs.ns(0) - 1).fill(0.0);
                        SSmodel::inputs.system.T.submat(ns + 1, inputs.ns(0), ns + 1, ns - 1) = SSmodel::inputs.system.Z.cols(inputs.ns(0), ns - 1);
                }
        } else {
                SSmodel::inputs.system.Z.cols(0, inputs.ns(0) - 1).fill(0.0);
        }
        //    showSS(SSmodel::inputs.system);
        //    cout << "model: " << inputs.model << endl;
        //    cout << "here" << endl;
}
// Interpolation of initial NaN values
void BSMclass::interpolate(int iniObs){
        BSMmodel sysCopy = inputs;
        SSinputs ssCopy = SSmodel::inputs;
        SSmodel::inputs.h = 0;
        uvec missing = find_nonfinite(SSmodel::inputs.y.rows(0, iniObs));
        SSmodel::inputs.y = reverse(SSmodel::inputs.y);
        if (SSmodel::inputs.u.n_rows > 0)
                SSmodel::inputs.u = reverse(SSmodel::inputs.u, 1);
        int lastObs = findFirst(SSmodel::inputs.y, sum(inputs.ns));
        SSmodel::inputs.y = SSmodel::inputs.y.rows(lastObs, SSmodel::inputs.y.n_elem - 1);
        if (SSmodel::inputs.u.n_rows > 0)
                SSmodel::inputs.u = SSmodel::inputs.u.cols(lastObs, SSmodel::inputs.u.n_cols - 1);
        estim(false);
        SSmodel::smooth(false);
        vec yFit = reverse(SSmodel::inputs.yFit);
        inputs = sysCopy;
        SSmodel::inputs = ssCopy;
        SSmodel::inputs.y(missing) = yFit(missing);
        // BSMmodel sysCopy = inputs;
        // SSinputs ssCopy = SSmodel::inputs;
        // SSmodel::inputs.h = 0;
        // SSmodel::inputs.y = reverse(SSmodel::inputs.y);
        // if (SSmodel::inputs.u.n_rows > 0)
        //     SSmodel::inputs.u = reverse(SSmodel::inputs.u, 1);
        // int lastObs = findFirst(SSmodel::inputs.y, sum(inputs.ns));
        // SSmodel::inputs.y = SSmodel::inputs.y.rows(lastObs, SSmodel::inputs.y.n_elem - 1);
        // estim(false);
        // SSmodel::smooth(false);
        // vec yFit = reverse(SSmodel::inputs.yFit);
        // SSmodel::inputs.y = reverse(SSmodel::inputs.y);
        // uvec missing = find_nonfinite(SSmodel::inputs.y.rows(0, iniObs));
        // inputs = sysCopy;
        // SSmodel::inputs = ssCopy;
        // SSmodel::inputs.y(missing) = yFit(missing + lastObs);

}
// Estimation: runs estim(p) or ident()
void BSMclass::estim(bool VERBOSE){
        bool verboseCopy = SSmodel::inputs.verbose;
        SSmodel::inputs.verbose = VERBOSE;
        if (inputs.trend != "?" && inputs.cycle != "?" && inputs.seasonal != "?" && inputs.irregular != "?"){
                // Particular model
                if (SSmodel::inputs.outlier == 0){
                        // Without outlier detection
                        uvec harmonics = inputs.harmonics;
                        SSmodel::inputs.p.reset();
                        estim(SSmodel::inputs.p0, VERBOSE);
                        // checkModel(harmonics);
                } else {
                        // With outlier detection
                        estimOutlier(SSmodel::inputs.p0, VERBOSE);
                }
        } else {
                // Some or all the components to identify
                string cycle = inputs.cycle;
                string cycle0 = inputs.cycle0;
                size_t found = cycle.find('?');
                if (found != string::npos && inputs.arma){  // cycle has ?
                        BSMmodel old = inputs;
                        SSinputs oldSS = SSmodel::inputs;
                        // First estimation with cycle
                        inputs.cycle = inputs.cycle0;
                        ident("head", VERBOSE);
                        SSinputs bestSS = SSmodel::inputs;
                        BSMmodel bestBSM = inputs;
                        inputs = old;
                        SSmodel::inputs = oldSS;
                        // Second estimation without cycle
                        inputs.cycle = "none";
                        strReplace("?", "", inputs.cycle0);
                        ident("tail", VERBOSE);
                        // Now decide which is best
                        int crit = 1;  // "aic" default
                        if (inputs.criterion == "bic"){
                                crit = 2;
                        } else if (inputs.criterion == "aicc"){
                                crit = 3;
                        } else if (inputs.criterion == "bicc"){
                                crit = 4;
                        }
                        if (SSmodel::inputs.criteria(crit) > bestSS.criteria(crit)){
                                SSmodel::inputs = bestSS;
                                inputs = bestBSM;
                        }
                        inputs.cycle = cycle;
                        inputs.cycle0 = cycle0;
                } else {
                        // Estimation as is
                        ident("both", VERBOSE);
                }
        }
        SSmodel::inputs.verbose = verboseCopy;
}
// Check whether re-estimation is necessary
void BSMclass::checkModel(uvec harmonics){
        // Repeat estimation of one model in case of anomalies
        string ok = SSmodel::inputs.estimOk;
        bool add = (inputs.model[0] == 'd');
        bool printed = false;
        // If no convergence and llt or dt trend model, then slope p0 more rigid
        if ((ok[10] == 'M' || ok[10] == 'U' || ok[10] == 'O' || ok[10] == 'N') &&
            (inputs.model[0] == 'l' || inputs.model[0] == 'd')){
                // Next 5 lines in every exception
                if (SSmodel::inputs.verbose){
                        Rprintf("    --\n");
                        Rprintf("    Estimation problems, trying again...\n");
                        Rprintf("    --\n");
                        printed = true;
                }
                SSinputs old = SSmodel::inputs;
                setModel(inputs.model, inputs.periods(harmonics), inputs.rhos(harmonics), false);
                bool VERBOSE = old.verbose;
                SSmodel::inputs.verbose = false;
                SSmodel::inputs.p0(1 + add) = -6.2325;
                // Estimation of particular model
                if (SSmodel::inputs.outlier == 0){
                        // Without outlier detection
                        estim(SSmodel::inputs.p0, VERBOSE);
                } else {
                        // With outlier detection
                        estimOutlier(SSmodel::inputs.p0, VERBOSE);
                }
                if (!old.criteria.has_nan() &&
                    (old.criteria(1) < SSmodel::inputs.criteria(1))){
                        SSmodel::inputs = old;
                        SSmodel::inputs.verbose = VERBOSE;
                }
        }
        // Repeat estimation of one model in case of anomalies
        ok = SSmodel::inputs.estimOk;
        //add = (inputs.model[0] == 'd');
        // If no convergence and llt or dt trend model, then level p0 more rigid
        if ((ok[10] == 'M' || ok[10] == 'U' || ok[10] == 'O' || ok[10] == 'N') &&
            (inputs.model[0] == 'l' || inputs.model[0] == 'd')){
                // Next 5 lines in every exception
                if (SSmodel::inputs.verbose && !printed){
                        Rprintf("    --\n");
                        Rprintf("    Estimation problems, trying again...\n");
                        Rprintf("    --\n");
                        printed = true;
                }
                SSinputs old = SSmodel::inputs;
                setModel(inputs.model, inputs.periods(harmonics), inputs.rhos(harmonics), false);
                bool VERBOSE = old.verbose;
                SSmodel::inputs.verbose = false;
                SSmodel::inputs.p0(0 + add) = -6.2325;
                // Estimation of particular model
                if (SSmodel::inputs.outlier == 0){
                        // Without outlier detection
                        estim(SSmodel::inputs.p0, VERBOSE);
                } else {
                        // With outlier detection
                        estimOutlier(SSmodel::inputs.p0, VERBOSE);
                }
                if (!old.criteria.has_nan() &&
                    (old.criteria(1) < SSmodel::inputs.criteria(1))){
                        SSmodel::inputs = old;
                        SSmodel::inputs.verbose = VERBOSE;
                }
        }
        inputs.harmonics = harmonics;
}
void BSMclass::estim(vec p, bool VERBOSE){
        bool verboseCopy = SSmodel::inputs.verbose;
        SSmodel::inputs.verbose = VERBOSE;
        double objFunValue;
        vec grad;
        mat iHess;
        int flag; //, nPar, k;
        SSmodel::inputs.p0 = p;
        wall_clock timer;
        timer.tic();
        if (SSmodel::inputs.augmented){
                SSmodel::inputs.llikFUN = llikAug;
        } else {
                SSmodel::inputs.llikFUN = llik;
        }
        // This next line is a bit dangerous
        SSmodel::inputs.userInputs = &inputs;
        flag = quasiNewtonBSM(SSmodel::inputs.llikFUN, gradLlik, p, &(SSmodel::inputs),
                              objFunValue, grad, iHess, SSmodel::inputs.verbose);
        uvec indNan = find_nonfinite(SSmodel::inputs.y);
        int nNan2pi = SSmodel::inputs.y.n_elem - indNan.n_elem;
        // Track non-stationary state count (used downstream by validate() to
        // size the Hessian penalty, NOT by the LL formula — LL now uses
        // nNan2pi as the MLE sample size).
        if (SSmodel::inputs.augmented){
                uvec stat;
                isStationary(SSmodel::inputs.system.T, stat);
                SSmodel::inputs.nonStationaryTerms = SSmodel::inputs.system.T.n_rows - stat.n_elem;
        }
        bool lambdaWasEstimated = SSmodel::inputs.estimateLambda;
        // Capture final lambda.  After quasiNewtonBSM, the converged value
        // sits in p.back() (BFGS optimised it in place).  llik() also wrote
        // it to SSmodel::inputs.lambda on the last call, but BSMmodel's
        // inputs.lambda is never touched by llik -- it still holds the
        // estimUCs warm-start.  Reading p.back() is unambiguous.
        // Apply the same lower bound that llik()'s clamp uses, so the
        // reported lambda matches the value actually fed into BoxCox()
        // during the final iterations -- otherwise BFGS can leave
        // p.back() below SSinputs.lambdaLower (the clamp is a "shadow"
        // that bounds the *used* value, not the parameter slot), and
        // downstream consumers see an unclamped value (e.g. lambda=0
        // when y has zeros, which then crashes the snap-anchor / trig
        // seasonal init at lambda=0 -> log(0) = -Inf).
        double lambdaStar = lambdaWasEstimated
                            ? std::max(LAMBDA_BOUND_LOWER,
                                       std::min(LAMBDA_BOUND_UPPER,
                                                p(p.n_elem - 1)))
                            : inputs.lambda;
        if (lambdaWasEstimated &&
            std::isfinite(SSmodel::inputs.lambdaLower) &&
            lambdaStar < SSmodel::inputs.lambdaLower){
                lambdaStar = SSmodel::inputs.lambdaLower;
                p(p.n_elem - 1) = lambdaStar;  // keep BFGS slot consistent
        }
        if (lambdaWasEstimated){
                inputs.lambda          = lambdaStar;
                SSmodel::inputs.lambda = lambdaStar;
        }
        double LLIK; //, AIC, BIC, AICc, BICc;
        // flag=7 means "line search proposed a NaN at some point and
        // BFGS reverted xNew/objNew to the last good iterate" (see the
        // isnan(objNew) revert in quasiNewtonBSM around line 2946).
        // After that revert objFunValue IS finite -- the NaN lived in
        // dobj (= objOld - objNew where objOld was set to NaN to flag
        // the revert), not in objFunValue itself.  Only blank to NaN
        // when objFunValue actually came back non-finite (i.e. flag=7
        // was raised before any good iterate was recorded).
        if (flag > 6 && !std::isfinite(objFunValue)){
                objFunValue = datum::nan;
        }
        // ----------------------------------------------------------------
        // Single source of truth for LLIK and IC formulas.
        //
        // computeLLIK: convert the BFGS objective into the BCnorm marginal
        //   log-likelihood on the original response scale.  llik()/llikAug()
        //   now always fold the BoxCox Jacobian into objFunValue and use
        //   the MLE σ̂² = SSR/n_finite (not the REML n-k divisor), so the
        //   LL is directly available with one consistent formula.
        //
        // computeCriteria: write LLIK / AIC / BIC / AICc / BICc into the
        //   shared 5-vector with k matching R's nparam(m) -- p.n_elem
        //   (concentrated variance counts, alm/adam convention) + outlier
        //   dummies + lambda when free.
        auto computeLLIK = [&](double obj) -> double {
                if (!std::isfinite(obj)) return datum::nan;
                return -0.5 * (log(2*datum::pi) * nNan2pi + nNan2pi * obj);
        };
        auto computeCriteria = [&](double llik, int k) -> vec {
                vec ic(5);
                ic(0) = llik;
                if (std::isfinite(llik)){
                        infoCriteria(llik, k, nNan2pi,
                                     ic(1), ic(2), ic(3), ic(4));
                } else {
                        ic(1) = ic(2) = ic(3) = ic(4) = datum::nan;
                }
                return ic;
        };
        // k convention: optimised parameters (p.n_elem already counts the
        // concentrated variance), outlier dummies, plus lambda when free,
        // plus the deterministic-trend drift slope (G / td) which is
        // concentrated out as a regressor and absent from pSize.
        auto kFor = [&](uword pSize) -> int {
                return static_cast<int>(pSize + SSmodel::inputs.u.n_rows
                                              + (inputs.lambdaEstimated ? 1 : 0)
                                              + (inputs.Drift ? 1 : 0));
        };

        LLIK = computeLLIK(objFunValue);
        // ----------------------------------------------------------------
        // Anchor snap: run a second BFGS at the nearest anchor in
        // {-2,-1,-0.5,0,0.5,1,2} and compare IC.  The snap saves one DoF
        // (lambda fixed → k stays k instead of k+1); it wins on AIC ties.
        // Both runs are cheap — one BFGS each, warm-started from psi*.
        if (lambdaWasEstimated && std::isfinite(LLIK)){
                // Save optimal state before modifying anything
                double LLIK_star      = LLIK;
                double objFun_star    = objFunValue;
                vec    grad_star      = grad;
                vec    p_opt_struct   = p.rows(0, p.n_elem - 2);  // structural only
                vec    typePar_struct = inputs.typePar.rows(0, inputs.typePar.n_elem - 2);
                vec    constPar_struct= inputs.constPar.rows(0, inputs.constPar.n_elem - 2);

                // Find nearest anchor within the valid lambda domain.
                // lambdaMin combines two sources: (a) yMin=0 forbids
                // lambda <= 0 (log/negative powers of 0 are +/-Inf), and
                // (b) the user-supplied SSinputs.lambdaLower (1e-10 when R
                // detects zeros) takes precedence over (a) so the snap
                // never picks the anchor 0 either, which would still feed
                // log(0) = -Inf into the KF and crash the trig seasonal
                // initialiser.
                double yMin = SSmodel::inputs.y_raw.n_elem > 0
                              ? SSmodel::inputs.y_raw.elem(find_finite(SSmodel::inputs.y_raw)).min()
                              : 1.0;
                double lambdaMin = (yMin > 0.0) ? LAMBDA_BOUND_LOWER : 0.0;
                if (std::isfinite(SSmodel::inputs.lambdaLower) &&
                    SSmodel::inputs.lambdaLower > lambdaMin)
                        lambdaMin = SSmodel::inputs.lambdaLower;
                const double allAnchors[] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
                double lambdaSnap = lambdaStar;
                double dBest = datum::inf;
                for (double aA : allAnchors){
                        // Strict lower-bound check (no tolerance) so a
                        // tiny positive lambdaMin (e.g. 1e-10) still
                        // correctly excludes the anchor 0.
                        if (aA < lambdaMin ||
                            aA > LAMBDA_BOUND_UPPER + 1e-6) continue;
                        double dist = std::abs(aA - lambdaStar);
                        if (dist < dBest){ dBest = dist; lambdaSnap = aA; }
                }

                // Snap BFGS: fixed lambda at anchor, warm-started from psi*
                SSmodel::inputs.estimateLambda = false;
                inputs.estimateLambda          = false;
                inputs.typePar  = typePar_struct;
                inputs.constPar = constPar_struct;
                inputs.lambda          = lambdaSnap;
                SSmodel::inputs.lambda = lambdaSnap;
                SSmodel::inputs.y = BoxCox(SSmodel::inputs.y_raw, lambdaSnap);
                vec    p_snap = p_opt_struct;   // warm-start; modified in place
                double snapObjFun; vec snapGrad; mat snapHess;
                quasiNewtonBSM(SSmodel::inputs.llikFUN, gradLlik, p_snap,
                               &(SSmodel::inputs), snapObjFun, snapGrad, snapHess, false);

                const double LLIK_snap = computeLLIK(snapObjFun);

                // IC comparison: opt path keeps lambda (+1 DoF); snap path drops it.
                inputs.lambdaEstimated = true;
                vec ic_opt  = computeCriteria(LLIK_star, kFor(p_opt_struct.n_elem));
                inputs.lambdaEstimated = false;
                vec ic_snap = computeCriteria(LLIK_snap, kFor(p_snap.n_elem));
                int critIdx = (inputs.criterion == "bic")  ? 2 :
                              (inputs.criterion == "aicc") ? 3 :
                              (inputs.criterion == "bicc") ? 4 : 1;
                double crit_opt  = ic_opt(critIdx);
                double crit_snap = ic_snap(critIdx);

                if (std::isfinite(crit_snap) && crit_snap <= crit_opt){
                        // Snap wins: keep snap-BFGS state
                        p = p_snap;
                        grad = snapGrad;
                        objFunValue = snapObjFun;
                        LLIK = LLIK_snap;
                        inputs.lambda = lambdaSnap;
                        inputs.lambdaEstimated = false;
                        // SSmodel::inputs.y, .lambda, aEnd, PEnd already at snap
                } else {
                        // Optimal wins: restore optimal-BFGS state
                        p = p_opt_struct;
                        grad = grad_star;
                        objFunValue = objFun_star;
                        LLIK = LLIK_star;
                        inputs.lambda = lambdaStar;
                        inputs.lambdaEstimated = true;
                        // Restore y and KF state for subsequent forecast()
                        SSmodel::inputs.lambda = lambdaStar;
                        SSmodel::inputs.y = BoxCox(SSmodel::inputs.y_raw, lambdaStar);
                        inputs.typePar  = typePar_struct;
                        inputs.constPar = constPar_struct;
                        // One extra llik() call to repopulate aEnd/PEnd/v/F/iF
                        SSmodel::inputs.llikFUN(p, &(SSmodel::inputs));
                }
        } else if (lambdaWasEstimated){
                // Estimation failed (non-finite LLIK): just strip lambda cleanly
                p.shed_row(p.n_elem - 1);
                if (grad.n_elem > 0) grad.shed_row(grad.n_elem - 1);
                inputs.typePar.shed_row(inputs.typePar.n_elem - 1);
                inputs.constPar.shed_row(inputs.constPar.n_elem - 1);
                inputs.lambdaEstimated = false;
                SSmodel::inputs.y = BoxCox(SSmodel::inputs.y_raw, lambdaStar);
                SSmodel::inputs.lambda = lambdaStar;
        }
        // Clear transient estimateLambda flags: y is now correctly set for
        // the selected lambda; subsequent validate()/parCov()/forecast() must
        // NOT try to re-BoxCox.
        SSmodel::inputs.estimateLambda = false;
        inputs.estimateLambda          = false;

        SSmodel::inputs.criteria = computeCriteria(LLIK, kFor(p.n_elem));
        // AIC  = SSmodel::inputs.criteria(1);
        // BIC  = SSmodel::inputs.criteria(2);
        // AICc = SSmodel::inputs.criteria(3);
        // BICc = SSmodel::inputs.criteria(4);
        if (flag == 1) {
                SSmodel::inputs.estimOk = "Q-Newton: Gradient convergence\n";
        } else if (flag == 2){
                SSmodel::inputs.estimOk = "Q-Newton: Function convergence\n";
        } else if (flag == 3){
                SSmodel::inputs.estimOk = "Q-Newton: Parameter convergence\n";
        } else if (flag == 4){
                SSmodel::inputs.estimOk = "Q-Newton: Maximum Number of iterations reached\n";
        } else if (flag == 5){
                SSmodel::inputs.estimOk = "Q-Newton: Maximum Number of function evaluations\n";
        } else if (flag == 6){
                SSmodel::inputs.estimOk = "Q-Newton: Unable to decrease objective function\n";
        } else if (flag == 7){
                SSmodel::inputs.estimOk = "Q-Newton: Objective function returns nan\n";
                objFunValue = datum::nan;
        } else {
                SSmodel::inputs.estimOk = "Q-Newton: No convergence!!\n";
        }
        if (SSmodel::inputs.verbose){
                double nSeconds = timer.toc();
                Rprintf("%s", SSmodel::inputs.estimOk.c_str());
                Rprintf("Elapsed time: %10.5f seconds\n", nSeconds);
        }
        SSmodel::inputs.p = p;
        SSmodel::inputs.objFunValue = objFunValue;
        SSmodel::inputs.grad = grad;
        // Eliminating cycle periods
        uvec aux = find(inputs.rhos > 0);
        inputs.rhos = inputs.rhos(aux);
        inputs.periods = inputs.periods(aux);
        SSmodel::inputs.v.reset();
        inputs.harmonics = regspace<uvec>(0, inputs.periods.n_elem - 1);
        SSmodel::inputs.verbose = verboseCopy;
}
// Forecast-only path: skip optimisation and run one Kalman pass with the
// user-supplied (natural-scale) parameters so that aEnd / PEnd are
// populated.  A subsequent forecast() then projects forward without
// invoking the optimiser.
//
// We use bsmMatricesTrue (which takes absolute variances directly) and
// disable cLlik, so that:
//   * the system matrices are built in absolute scale;
//   * innVariance stays at 1 (line 552 of SSpace.h);
//   * forecast()'s  Pt = PEnd * 1  and  CHCt = H  are both in absolute
//     scale and match what the normal estim()+forecast() path produces.
//
// The caller must pass p0 in the constructor as a no-op sentinel
// (e.g. -9999.9) so that initParBsm() takes the default-init branch and
// does not try to re-transform user values (which would crash on zero
// variances or on irregular variances < 1e-6).  The actual user-supplied
// parameters are then handed to this method.
void BSMclass::setEstimatedParams(vec userParams){
        SSmodel::inputs.userModel  = bsmMatricesTrue;
        SSmodel::inputs.cLlik      = false;
        SSmodel::inputs.userInputs = &inputs;
        SSmodel::inputs.p   = userParams;
        SSmodel::inputs.p0  = userParams;
        if (SSmodel::inputs.augmented){
                SSmodel::inputs.llikFUN = llikAug;
        } else {
                SSmodel::inputs.llikFUN = llik;
        }
        if (SSmodel::inputs.aEnd.n_elem > 0){
                // Fast path: a cached terminal state (aEnd / PEnd / innVariance
                // and betaAug) was supplied for a fitted object.  Just build the
                // system matrices from p and reuse the cached state -- skipping
                // the O(n * m^2) full-series re-filter.  betaAug carries the
                // augmented-KF state (xreg coefs + initial states), so xreg
                // models forecast from the cache too (adam-style).
                SSmodel::setSystemMatrices();
        } else {
                // One llik call: bsmMatricesTrue builds the system from p in
                // absolute scale, KF populates aEnd / PEnd as a side effect.
                SSmodel::inputs.llikFUN(SSmodel::inputs.p, &(SSmodel::inputs));
        }
        SSmodel::inputs.estimOk = "Q-Newton: Skipped (forecast-only).\n";
}
// Estimation of a family of UC models
void BSMclass::estimUCs(vector <string> allUCModels, uvec harmonics,
                        double& minCrit, bool VERBOSE,
                        double oldMinCrit, int nuInit){
        // Estim a number of UC models and select the best according to minCrit
        //       The best is compared to oldMinCrit that is the current best system
        //       and the overall best is put into SSmodel::inputs and inputs
        //       If there is no previous model to compare to set oldMinCrit to 1e12
        double curCrit,
        AIC,
        BIC,
        AICc,
        BICc;
        SSinputs bestSS = SSmodel::inputs;
        BSMmodel bestBSM = inputs;
        if (isnan(oldMinCrit)){
                oldMinCrit = 1e12;
        }
        mat uCopy = SSmodel::inputs.u;
        minCrit = oldMinCrit;
        bool inputsArma = inputs.arma;
        vec PERIODS = inputs.periods, RHOS = inputs.rhos;
        // Snapshot the testBoxCox warm-start before the loop so every
        // candidate's joint-lambda BFGS starts from the SAME neutral seed.
        // Otherwise estim()'s snap test overwrites inputs.lambda with the
        // previous candidate's snap anchor and the next candidate starts on
        // the boundary basin of that anchor, killing the per-model lambda
        // exploration the table is meant to display.
        double lambdaSeed = inputs.lambda;
        for (unsigned int i = 0; i < allUCModels.size(); i++){
                SSmodel::inputs.p0 = -9999.9;
                int wide = 30;
                if (inputs.PTSnames)
                        wide = 8;
                bool arma = inputs.arma;
                inputs.periods = PERIODS;
                inputs.rhos = RHOS;
                SSmodel::inputs.u = uCopy;
                inputs.typeOutliers.resize(0);
                // For joint-lambda: reset y=y_raw + lambda=seed every iteration
                // so each candidate optimises lambda from the same neutral
                // starting point.  estim() clears estimateLambda at the end of
                // each candidate, so it must be re-asserted here.
                if (inputs.profileLambda){
                        inputs.lambda                  = lambdaSeed;
                        SSmodel::inputs.lambda         = lambdaSeed;
                        SSmodel::inputs.y              = SSmodel::inputs.y_raw;
                        SSmodel::inputs.estimateLambda = true;
                        inputs.estimateLambda          = true;
                }
                setModel(allUCModels[i], inputs.periods(harmonics), inputs.rhos(harmonics), false);
                // setModel(allUCModels[i], PERIODS(harmonics), RHOS(harmonics), false);
                inputs.arma = arma;
                // Cleaning variables for outliers starting anew
                // if (SSmodel::inputs.u.n_elem > 0){
                //         if (nuInit > 0){
                //                 SSmodel::inputs.u = SSmodel::inputs.u.rows(0, nuInit - 1);
                //         } else {
                //                 SSmodel::inputs.u.resize(0);
                //         }
                // }
                SSmodel::inputs.p.reset();
                estim(SSmodel::inputs.verbose);
                inputs.periods = PERIODS;
                inputs.rhos = RHOS;
                AIC  = SSmodel::inputs.criteria(1);
                BIC  = SSmodel::inputs.criteria(2);
                AICc = SSmodel::inputs.criteria(3);
                BICc = SSmodel::inputs.criteria(4);
                // Avoid selecting a model with problems
                if (AIC == -datum::inf || AIC == datum::inf){
                        AIC = BIC = AICc = BICc = datum::nan;
                }
                if (VERBOSE){
                        // The displayed ICs are the same values selection
                        // uses (estim() now populates criteria(1..4) with
                        // the alm/adam k convention), so just print them.
                        string MODEL = allUCModels[i];
                        char lambdaTag[16];
                        snprintf(lambdaTag, sizeof(lambdaTag), "%5.2f%s",
                                 inputs.lambda,
                                 inputs.lambdaEstimated ? " " : "*");
                        if (inputs.PTSnames){
                                MODEL = UC2PTS(allUCModels[i], inputs.lambda);
                                Rprintf(" %*s  %6s: %13.4f %13.4f %13.4f %13.4f\n",
                                       wide, MODEL.c_str(), lambdaTag,
                                       AIC, AICc, BIC, BICc);
                        } else {
                                Rprintf(" %*s  %6s: %8.4f %8.4f %8.4f %8.4f\n",
                                       wide, MODEL.c_str(), lambdaTag,
                                       AIC, AICc, BIC, BICc);
                        }
                }
                if (inputs.criterion == "bic"){
                        curCrit = BIC;
                } else if (inputs.criterion == "aicc"){
                        curCrit = AICc;
                } else if (inputs.criterion == "bicc"){
                        curCrit = BICc;
                } else {  // "aic" (default)
                        curCrit = AIC;
                }
                if ((curCrit < minCrit && !isnan(curCrit))){  // || i == 0){
                        minCrit = curCrit;
                        bestSS = SSmodel::inputs;
                        bestBSM = inputs;
                }
        }
        SSmodel::inputs = bestSS;
        inputs = bestBSM;
        inputs.arma = inputsArma;
}
// Identification
// ident: choose a UC model when one or more of trend/cycle/seasonal/
// irregular were left as "?".  ~344 lines.  Outline:
//
//   1. Setup / copy inputs, suppress outliers during identification.
//   2. Trend ADF test (when stepwise + tTest are on).
//   3. Seasonal / harmonics selection.
//   4. Verbose header (only printed when show=="head"|"both").
//   5. Build the candidate model list (runAll vs stepwise branches).
//   6. estimUCs(...) -> picks the best of the candidates.
//   7. Outlier re-estimation pass (if outlier detection was requested).
//
// A future split would lift sections 2-3 into stepwiseChecks(),
// section 5 into buildCandidates(), and section 7 into rerunWithOutliers().
void BSMclass::ident(string show, bool VERBOSE){
        bool verboseCopy = SSmodel::inputs.verbose;
        SSmodel::inputs.verbose = VERBOSE;
        wall_clock timer;
        timer.tic();
        // Interpolation in case of missing
        // if (inputs.missing.n_elem > 0)
        //     interpolate();
        double season,
        maxLag,
        outlierCopy = SSmodel::inputs.outlier;
        string inputTrend = inputs.trend,
                inputCycle = inputs.cycle,
                inputSeasonal = inputs.seasonal,
                inputIrregular = inputs.irregular,
                model,
                trendTypes,
                cycTypes,
                seasTypes,
                irrTypes,
                restRW;
        int trueTrend,
        nuInit = SSmodel::inputs.u.n_rows;
        //bool VERBOSE = SSmodel::inputs.verbose;
        // Controling estimation with or without outliers
        if (outlierCopy > 0){
                SSmodel::inputs.outlier = 0;
        }
        // Controlling verbose output
        SSmodel::inputs.verbose = false;
        vec periods = inputs.periods(find(inputs.rhos > 0));
        season = max(periods);
        if (season == 1){
                inputSeasonal = "none";
                inputs.seasonal = "none";
        }
        maxLag = floor(season / 2);
        // Trend tests
        if (SSmodel::inputs.y.n_rows < 15){
                inputs.tTest = false;
        }
        if (inputs.stepwise && inputTrend == "?" && inputs.tTest){
                vec lagsAdf(2);
                double lagsAdfMax;
                lagsAdf(0) = 2 * season + 2;
                lagsAdf(1) = 10;
                lagsAdfMax = max(lagsAdf);
                if (lagsAdfMax < SSmodel::inputs.y.n_rows / 2){
                        inputs.tTest = false;
                } else {
                        trueTrend = adfTests(SSmodel::inputs.y, max(lagsAdf), "bic");
                        if (trueTrend == 0){    // No trend detected
                                inputTrend = "none";
                                trendTypes = "none";
                        } else if (trueTrend == 1){
                                inputTrend = "SOME";
                                trendTypes = "rw";
                        }
                }
        }
        // Seasonal test
        string isSeasonal;
        if (inputSeasonal[0] == 'n'){
                inputSeasonal = "none";
                isSeasonal = "none";
        } else {
                isSeasonal = "true";
        }
        // Selecting harmonics.
        // selectHarmonics pre-filters which harmonics to include before
        // building the candidate list.  We skip it when the seasonal type
        // is either "equal" ('e') or unknown ('?') because for "equal"
        // all harmonics share ONE variance — dropping one doesn't reduce
        // the parameter count, only the state count, and this causes an
        // inconsistency with direct estimation (e.g. "ZGT" vs "ZZZ"):
        // the ident sweep would drop the π-harmonic while the direct run
        // keeps it, giving different T matrices and different optima for
        // what the user sees as the same model.  For "linear" ('l') the
        // original code already skipped this; for unknown '?' the final
        // seasonal type is not yet known and "equal" is always a candidate.
        uvec harmonics = regspace<uvec>(0, periods.n_elem - 1);
        uvec harmonics0 = harmonics;
        if (inputSeasonal[0] != 'n' && inputSeasonal[0] != 'l' &&
            inputSeasonal[0] != 'e' && inputSeasonal[0] != '?' &&
            !inputs.MSOE && periods.n_elem > 1){
                vec betaHR;
                selectHarmonics(SSmodel::inputs.y, SSmodel::inputs.u, periods, harmonics, betaHR, isSeasonal);
                if (harmonics.n_rows == 0){
                        inputSeasonal = "none";
                        harmonics = harmonics0;
                        periods = inputs.periods;
                }
        }
        if (season == 4 && harmonics.n_rows > 0){
                harmonics.reset();
                harmonics = regspace<uvec>(0, 1);
        }
        // 18/07/2025 Correction
        uvec rhosm1 = find(inputs.rhos == -1);
        harmonics = harmonics + rhosm1.n_rows;
        ///
        inputs.harmonics = harmonics;
        // UC identification
        double minCrit; // = 1e12, minCrit1;
        if (VERBOSE && (show == "head" || show == "both")){
                Rprintf("-----------------------------------------------------------------------------------\n");
                if (SSmodel::inputs.outlier < 0){
                        Rprintf(" Identification started WITH outlier detection\n");
                } else {
                        if (inputs.PTSnames)
                                Rprintf(" Identification of PTS models:\n");
                        else
                                Rprintf(" Identification started WITHOUT outlier detection\n");
                }
                Rprintf("-----------------------------------------------------------------------------------\n");
                if (inputs.PTSnames)
                        Rprintf("    Model          Lambda           AIC          AICc           BIC          BICc\n");
                else
                        Rprintf("          Model                     Lambda      AIC     AICc      BIC     BICc\n");
                Rprintf("           (Lambda values marked with '*' were held fixed, not estimated.)\n");
                Rprintf("-----------------------------------------------------------------------------------\n");
        }
        // Finding models to identify
        vector<string> allUCModels;
        size_t pos;
        bool runAll = !inputs.stepwise;
        if (season == 1 || inputSeasonal != "?"){
                runAll = true;
        }
        if (!runAll){
                if (isSeasonal[0] == 't'){
                        seasTypes = "equal/different";
                } else if (isSeasonal[0] == 'd'){
                        seasTypes = "none/equal/different";
                } else if (isSeasonal[0] == 'f'){
                        seasTypes = "none";
                }
        }
        if (inputIrregular == "?"){
                irrTypes = "none/arma(0,0)";
        } else {          // Model with one irregular
                irrTypes = inputIrregular;
                runAll = true;
        }
        if (inputCycle == "?"){
                cycTypes = "none/" + inputs.cycle0;
        } else {
                cycTypes = inputCycle;
        }
        if (inputTrend == "?"){
                trendTypes = "none/rw";
        } else if (inputTrend[0] != 'S'){
                trendTypes = inputTrend;
                runAll = true;
        } else if (inputTrend[0] == 'S'){
                runAll = false;
        }
        if (runAll){    // no stepwise
                if (inputTrend == "?"){
                        // PTS structural part comes first; ARMA refines it.
                        // The historical strip of `srw` / `dt` when an ARMA
                        // term was present was over-broad — it killed every
                        // damped/Hyndman trend rather than just the
                        // (trend, ARMA) combos that genuinely collide.
                        // Let every trend × irregular combination through;
                        // ICs flag the ill-identified ones via degraded LL.
                        trendTypes = inputs.trendOptions;
                }
                if (inputSeasonal == "?")
                        seasTypes = inputs.seasonalOptions;
                else
                        seasTypes = inputSeasonal;
                if (inputIrregular == "?")
                        irrTypes = inputs.irregularOptions;
                else
                        irrTypes = inputIrregular;
                findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
        } else {                  // stepwise
                if (inputSeasonal[0] == 'n'){
                        // Annual or non seasonal data
                        if (inputTrend == "?" || inputTrend == "SOME")
                                trendTypes = trendTypes + "/llt/dt";
                        seasTypes = "none";
                        findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
                        estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
                } else {
                        // Seasonal data
                        if (inputSeasonal != "?"){
                                //   seasTypes = "none/equal/different";
                                // } else {
                                seasTypes = inputSeasonal;
                        }
                        // Best of rw or none trends
                        findUCmodels(trendTypes, cycTypes, seasTypes, irrTypes, allUCModels);
                        estimUCs(allUCModels, harmonics, minCrit, VERBOSE, 1e12, nuInit);
                        allUCModels.clear();
                        if (inputs.model.substr(0, 1) == "n"){
                                // case if best trend is none
                                findUCmodels("dt", cycTypes, seasTypes, "arma(0,0)", allUCModels);
                                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
                        } else {
                                // case rw or llt is best: estimate some dts
                                pos = inputs.model.find("/", 0);
                                // Extract non trend part of the best model so far
                                restRW = inputs.model.substr(pos, inputs.model.size() - pos);
                                // Estimate the best model changing the trend to LLT
                                findUCmodels("llt", cycTypes, seasTypes, irrTypes, allUCModels);
                                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
                                // Now take the non trend part of the best model so far and use the DT trend instead
                                pos = inputs.model.find("/", 0);
                                restRW = inputs.model.substr(pos, inputs.model.size() - pos);
                                allUCModels.clear();
                                allUCModels.push_back("dt" + restRW);
                                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
                        }
                }
        }
        // Checking for identification total failure
        bool succeed = true;
        if (SSmodel::inputs.p.has_nan() || minCrit > 1e10){
                inputs.succeed = false;
                succeed = false;
                // Setting up default model
                setModel("rw/none/none/none", inputs.periods(harmonics), inputs.rhos(harmonics), false);
                SSmodel::inputs.p.set_size(1);
                SSmodel::inputs.p.fill(0);
                llikAug(SSmodel::inputs.p, &(SSmodel::inputs));
        }
        if (VERBOSE && !succeed){
                Rprintf("                      Identification failed!!\n");
                Rprintf("              Unable to find a proper model!!\n");
        }
        // Selecting best ARMA
        if (inputs.arma && succeed){
                string armaModel, modelNew;
                vec beta0, orders(2);
                orders.fill(0);
                if (inputIrregular == "?" && succeed){
                        string inputIrregular2;
                        splitModel(inputs.model, inputTrend, inputCycle, inputSeasonal, inputIrregular2);
                        if (inputIrregular == "?" && SSmodel::inputs.y.n_elem > 30){
                                int maxSearch = season + 4;
                                if (maxSearch > 28){
                                        maxSearch = 28;
                                }
                                maxLag = 5;
                                if (season == 1){
                                        maxSearch = 8;
                                }
                                if ((float)SSmodel::inputs.y.n_elem - (float)SSmodel::inputs.system.T.n_rows - 3 - (float)maxLag - (float)maxSearch > 3 * (float)season){
                                        filter();
                                        uvec ind = find_finite(SSmodel::inputs.v);
                                        // if ((float)SSmodel::inputs.v.n_elem - (float)SSmodel::inputs.system.T.n_rows - 2 > 3 * (float)season){
                                        if ((float)ind.n_elem - (float)SSmodel::inputs.system.T.n_rows - 2 > 3 * (float)season){
                                                selectARMA(SSmodel::inputs.v.rows(SSmodel::inputs.system.T.n_rows + 2, SSmodel::inputs.v.n_elem - 1),
                                                           maxLag, maxSearch, "bic", orders, beta0);
                                                inputs.beta0ARMA = beta0;
                                        }
                                }
                        }
                        // Model with ARMA
                        if (sum(orders) > 0){    // ARMA identified
                                armaModel.append(to_string((int)orders(0))).append(",").append(to_string((int)orders(1)));
                                // Reformulating the irregular model
                                string tModel, sModel, cModel, iModel;
                                splitModel(inputs.model, tModel, cModel, sModel, iModel);
                                // if (pureARMA)
                                //   tModel = "none";
                                modelNew.append(tModel).append("/").append(cModel).append("/").append(sModel).append("/").append("arma(").append(armaModel).append(")");
                                allUCModels.clear();
                                allUCModels.push_back(modelNew);
                                // Estimating potential best model
                                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
                        }
                }
                // Selecting best pure ARMA (in case best model is trend + noise of any kind so far)
                pos = inputs.model.find("/none/none/");
                if (VERBOSE && outlierCopy > 0 && outlierCopy < 1000){
                        Rprintf("------------------------------------------------------------\n");
                        Rprintf(" Final model WITH outlier detection\n");
                        Rprintf("------------------------------------------------------------\n");
                }
        }
        // bool correct = true;
        if (outlierCopy > 0 && outlierCopy < 1000){
                SSmodel::inputs.outlier = -abs(outlierCopy);
                allUCModels.clear();
                allUCModels.push_back(inputs.model);
                estimUCs(allUCModels, harmonics, minCrit, VERBOSE, minCrit, nuInit);
        }
        if (VERBOSE && (show == "tail" || show == "both")){
                double nSeconds = timer.toc();
                Rprintf("-----------------------------------------------------------------------------------\n");
                Rprintf("  Identification time: %10.5f seconds\n", nSeconds);
                Rprintf("-----------------------------------------------------------------------------------\n");
        }
        // Final estimation (genunine with nans in case of missing data)
        //if (inputs.missing.n_elem > 0){
        //     SSmodel::inputs.y(inputs.missing).fill(datum::nan);
        //     estim();
        //}
        SSmodel::inputs.verbose = VERBOSE;
        // Updating inputs
        if (inputSeasonal[0] == 'n'){
                inputs.periods = {1};
                harmonics = {0};
                inputs.rhos = {1};
        }
        inputs.harmonics = harmonics;
        inputs.rhos = inputs.rhos(harmonics);
        SSmodel::inputs.outlier = -abs(outlierCopy);
        if (harmonics.n_elem > 0){
                inputs.periods = inputs.periods(harmonics);
        } else {
                inputs.periods.resize(1);
                inputs.periods.fill(1);
        }
        SSmodel::inputs.verbose = verboseCopy;
}
// Outlier detection a la Harvey and Koopman
// estimOutlier: identify additive outliers / level shifts / slope changes
// and re-estimate after each one is inserted.  ~223 lines.  Outline:
//
//   1. Setup: starting params, save originals.
//   2. Run a baseline estim() with no outliers as the reference fit.
//   3. Iterative detection loop:
//        a. Standardise innovations, find peaks above the threshold.
//        b. For each peak, fit AO / LS / SC dummy and pick the best type.
//        c. Insert the chosen dummy into u, re-estimate, update criteria.
//        d. Stop when no peak survives the threshold.
//   4. Final clean-up (sort outliers, restore verbose, etc.).
//
// Natural extraction points: detectOutliers() for the peak-finding pass
// and applyOutlier() for the per-peak fit-and-insert step.
void BSMclass::estimOutlier(vec p0, bool VERBOSE){
        bool verboseCopy = SSmodel::inputs.verbose;
        mat uCopy = SSmodel::inputs.u;
        SSmodel::inputs.verbose = VERBOSE;
        // Havey, A.C. and Koopman, S.J. (1992), Diagnostic checking of unobserved
        //        components time series models , JBES, 10, 377-389.
        int n = SSmodel::inputs.y.n_elem - 1, //nNan,
            nu = SSmodel::inputs.u.n_rows,
            lu = 0;
        //bool VERBOSE = SSmodel::inputs.verbose;
        SSmodel::inputs.verbose = false;
        vec periodsCopy = inputs.periods,
            rhosCopy = inputs.rhos;
        // Length of u's
        if (nu == 0){
            lu = n + SSmodel::inputs.h + 1;
        } else {
            lu = SSmodel::inputs.u.n_cols;
        }
        wall_clock timer;
        timer.tic();
        // Initial estimation without checking oultiers
        SSmodel::inputs.p0 = p0;
        estim(SSmodel::inputs.p0, VERBOSE);
        inputs.periods = periodsCopy;
        inputs.rhos = rhosCopy;
        // Storing initial model clean
        SSinputs bestSS = SSmodel::inputs;
        BSMmodel bestBSM = inputs;
        // if (inputs.model == "llt/none/different/arma(0,0)") {
        //     SSmodel::inputs.u.print("ind 1703 BSMmodel");
        // }
        // mat bestU = SSmodel::inputs.u;
        // Forward Addition loop
        // Disturbances estimation
        disturb();
        // AO
        vec eps;
        if (inputs.nPar(3) == 1){
                eps = abs(inputs.eps);
        } else {
                eps = join_vert(zeros<vec>(n - SSmodel::inputs.v.n_elem + 1),
                                abs(SSmodel::inputs.v / sqrt(SSmodel::inputs.F)));
                eps.replace(datum::nan, 0);
        }
        // Per-type thresholds scale with the user-supplied z-score
        // (SSmodel::inputs.outlier).  The historical defaults — AO 2.3,
        // LS 2.5, SC 3.0 — set the relative stiffness; level=0.99 →
        // z≈2.576 → AO 2.576 / LS 2.80 / SC 3.36.  The engine flags the
        // "final fit with detection" path via a *negative* outlier
        // value (PTSmodel.h:1874), so take std::abs to recover the
        // user's z-score in either sign.
        const double zUser = std::abs(SSmodel::inputs.outlier);
        const double thrAO = zUser;
        const double thrLS = zUser * (2.5 / 2.3);
        const double thrSC = zUser * (3.0 / 2.3);
        uvec indAO = find(eps > thrAO);
        vec  valAO = eps(indAO);
        uvec sortInd;
        // Correction in case there are too many AO's
        if (valAO.n_elem > 10){
                sortInd = sort_index(valAO, "descend");
                sortInd = sortInd.rows(0, 9);
                indAO = indAO(sortInd);
                valAO = valAO(sortInd);
        }
        // LS
        uvec indLS;
        vec  valLS;
        if (inputs.nPar(0) > 0){ // && SSmodel::inputs.eta.row(0).max() > 2.5){
                valLS = abs(SSmodel::inputs.eta.row(0).t());
                indLS = selectOutliers(valLS, 3, thrLS);
                eps = abs(SSmodel::inputs.eta.row(0).t());
                valLS = eps(indLS);
        }
        // SC
        uvec indSC;
        vec  valSC;
        if (inputs.ns(0) > 1 && inputs.nPar(0) > 0){ // && SSmodel::inputs.eta.row(1).max() > 3){
                valSC = abs(SSmodel::inputs.eta.row(1).t());
                indSC = selectOutliers(valSC, 3, thrSC);
                eps = abs(SSmodel::inputs.eta.row(1).t());
                valSC = eps(indSC);
        }
        // All outliers together
        inputs.typeOutliers = join_vert(join_vert(zeros(size(indAO)), ones(size(indLS))), 2 * ones(size(indSC)));
        uvec ind = join_vert(join_vert(indAO, indLS), indSC);
        vec  val = join_vert(join_vert(valAO, valLS), valSC);
        // Sorting vectors
        // if (inputs.model == "llt/none/different/arma(0,0)") {
        //     ind.t().print("ind 1703 BSMmodel");
        // }
        if (ind.n_elem > 0){
                // Sorting and removing less significant in case of many outliers
                sortInd = sort_index(val, "descend");
                val = val(sortInd);
                ind = ind(sortInd);
                inputs.typeOutliers = inputs.typeOutliers(sortInd);
                if (ind.n_elem > 20){
                        val = val.rows(0, 19);
                        ind = ind.rows(0, 19);
                        inputs.typeOutliers = inputs.typeOutliers.rows(0, 19);
                }
                // Sorting now by date
                sortInd = sort_index(ind);
                val = val(sortInd);
                ind = ind(sortInd);
                inputs.typeOutliers = inputs.typeOutliers(sortInd);
        }
        // Removing duplicated outliers of different types
        vec uniqueInd = unique(conv_to<mat>::from(ind));
        if (uniqueInd.n_elem < ind.n_elem){
                uvec indAux(uniqueInd.n_elem);
                vec  valAux(uniqueInd.n_elem);
                mat  outlAux(uniqueInd.n_elem, 1);
                uvec ii;
                int j = 0;
                for (uword i = 0; i < uniqueInd.n_elem; i++){
                        ii = find(uniqueInd(i) == ind);
                        if (ii.n_elem > 1){
                                j = ii(val(ii).index_max());
                        } else {
                                j = ii(0);
                        }
                        valAux(i) = val(j);
                        outlAux(i, 0) = inputs.typeOutliers(j);
                        indAux(i) = ind(j);
                }
                ind = indAux;
                inputs.typeOutliers = outlAux;
                val = valAux;
        }
        // done
        bool cLlikCopy = SSmodel::inputs.cLlik,
                augmentedCopy = SSmodel::inputs.augmented,
                exactCopy = SSmodel::inputs.exact;
        if (ind.n_elem > 0){
                // matrix of potential inputs
                mat uNew(ind.n_elem, lu);
                uNew.fill(0);
                rowvec ui(lu);
                ui.fill(0);
                for (unsigned int i = 0; i < uNew.n_rows; i++){
                        dummy(ind(i), inputs.typeOutliers(i), ui);
                        uNew.row(i) = ui;
                }
                if (nu > 0){  ///////////////////////////////
                        // SSmodel::inputs.u = join_vert(SSmodel::inputs.u, uNew);
                        SSmodel::inputs.u = join_vert(uCopy, uNew);
                } else {
                        SSmodel::inputs.u = uNew;
                }
                // Re-estimation with inputs and all outliers in model
                SSmodel::inputs.cLlik = true;
                SSmodel::inputs.augmented = true;
                SSmodel::inputs.exact = false;
                if (VERBOSE){
                        SSmodel::inputs.verbose = true;
                }
                estim(SSmodel::inputs.p0, VERBOSE);
                // bestU = SSmodel::inputs.u;
                inputs.periods = periodsCopy;
                inputs.rhos = rhosCopy;
                vec obj(1); obj(0) = SSmodel::inputs.objFunValue;
                if (obj.is_finite()){
                        // Model with all initial outliers converged
                        SSmodel::inputs.verbose = false;
                        // Backward deletion step
                        uvec remove;
                        int ns = SSmodel::inputs.system.T.n_rows,
                                nuAll;
                        int count = 0;
                        do{
                                nuAll = nu + ind.n_elem;
                                vec t = abs(SSmodel::inputs.betaAug.rows(ns + nu, ns + nuAll - 1) /
                                        sqrt(SSmodel::inputs.betaAugVar.rows(ns + nu, ns + nuAll - 1)));
                                remove = find(t < abs(SSmodel::inputs.outlier));
                                if (remove.n_elem > 0){
                                        // Removing inputs
                                        SSmodel::inputs.u.shed_rows(nu + remove);
                                        inputs.typeOutliers.shed_rows(remove);
                                        ind.shed_rows(remove);
                                        // if (SSmodel::inputs.u.n_rows == 0 && inputs.model[0] != 'd'){
                                        //   SSmodel::inputs.augmented = false;
                                        //   SSmodel::inputs.exact = true;
                                        // }
             /////////////////          // if (SSmodel::inputs.u.n_rows == 0){
                                        if (ind.n_elem == 0) {
                                                SSmodel::inputs = bestSS;
                                                inputs = bestBSM;
                                                // bestU = uCopy;
                                        } else {
                                                // Final estimation
                                                estim(SSmodel::inputs.p0, VERBOSE);
                                                inputs.periods = periodsCopy;
                                                inputs.rhos = rhosCopy;
                                                // bestU = SSmodel::inputs.u;
                                        }
                                }
                                count++;
                        } while (count < 4 && ind.n_elem > 0 && remove.n_elem > 0);
                }
                if (ind.n_elem > 0){
                        inputs.typeOutliers.insert_cols(1, conv_to<mat>::from(ind));
                }
                // Final check
                vec best(1);
                if (inputs.criterion == "bic"){
                        obj(0) = SSmodel::inputs.criteria(2);
                        best(0) = bestSS.criteria(2);
                } else if (inputs.criterion == "aicc"){
                        obj(0) = SSmodel::inputs.criteria(3);
                        best(0) = bestSS.criteria(3);
                } else if (inputs.criterion == "bicc"){
                        obj(0) = SSmodel::inputs.criteria(4);
                        best(0) = bestSS.criteria(4);
                } else {  // "aic" (default)
                        obj(0) = SSmodel::inputs.criteria(1);
                        best(0) = bestSS.criteria(1);
                }
                // Keep the outlier model unless it failed to converge.  The
                // forward/backward step already retained only outliers whose
                // augmented-KF coefficient is statistically significant (t >
                // z-threshold), so the surviving set is justified on its own
                // terms.  The earlier "IC worse than baseline -> revert" gate is
                // unreliable: when the structural baseline overfits into the
                // unbounded variance->0 likelihood singularity (flexible model
                // on a short series), its IC is artificially good and no genuine
                // outlier model can beat it, so real, highly-significant
                // outliers were silently dropped.
                if (!obj.is_finite()){
                        // Model with outliers did not converge
                        SSmodel::inputs = bestSS;
                        inputs = bestBSM;
                }
        }
        // Restoring initial values
        SSmodel::inputs.verbose = VERBOSE;
        SSmodel::inputs.cLlik = cLlikCopy;
        SSmodel::inputs.augmented = augmentedCopy;
        SSmodel::inputs.exact = exactCopy;
        uvec aux = find(inputs.rhos > 0);
        inputs.rhos = inputs.rhos(aux);
        inputs.periods = inputs.periods(aux);
        SSmodel::inputs.verbose = verboseCopy;
        // SSmodel::inputs.u = bestU;
}
// Components
void BSMclass::components(){
        // components consumes only smoothed states (inputs.a); inputs.P feeds
        // the dead compV.  Request the fast state smoother (skips the O(m^3)
        // backward Nt recursion + smoothed-variance work).
        SSmodel::inputs.stateOnly = true;
        SSmodel::smooth(true);
        SSmodel::inputs.stateOnly = false;
        int nCycles = sum(inputs.rhos < 0), k = SSmodel::inputs.u.n_rows;
        inputs.comp.set_size(4 + nCycles + k, SSmodel::inputs.yFit.n_rows);
        inputs.comp.fill(datum::nan);
        inputs.compV = inputs.comp;
        uword ny = SSmodel::inputs.y.n_elem - 1;
        vec nsCum = cumsum(inputs.ns);
        uvec remove(inputs.comp.n_rows, fill::zeros);
        inputs.compNames = "";
        uword ns = sum(inputs.ns), ind = 0;
        // Level
        if (inputs.ns(0) > 0 && SSmodel::inputs.system.T(0, 0) != 0){
                if (SSmodel::inputs.a.n_rows > ns)
                        ind = ns;
                inputs.comp.row(0) = SSmodel::inputs.a.row(ind);
                inputs.compV.row(0) = SSmodel::inputs.P.row(ind);
                //if (sum(abs(inputs.comp.row(0).cols(0, ny))) > 1e-7)
                inputs.compNames += "Level/";
        } else
                remove(0) = 1;
        // Slope
        if (inputs.ns(0) > 1){
                inputs.comp.row(1) = SSmodel::inputs.a.row(1);
                inputs.compV.row(1) = SSmodel::inputs.P.row(1);
                //if (sum(abs(inputs.comp.row(1).cols(0, ny))) > 1e-7)
                inputs.compNames += "Slope/";
                //else
                //    remove(1) = 1;
        } else
                remove(1) = 1;
        // Seasonal
        if (inputs.ns(2) > 0){
                if (inputs.seasonal[0] != 'l'){
                        urowvec ind = regspace<urowvec>(nsCum(1), 2, nsCum(2) - 1);
                        inputs.comp.row(2) = sum(SSmodel::inputs.a.rows(ind));
                        inputs.compV.row(2) = sum(SSmodel::inputs.P.rows(ind));
                } else {
                        // Linear (dummy) seasonal: state[i1] = gamma_t under
                        // the convention set by initMatricesBsm's T block.
                        // The else-branch handles the augmented (xreg / TVP)
                        // case where extra states are appended; the seasonal
                        // there ends up at a.n_rows - 1.  Both branches now
                        // agree with the Z fix at initMatricesBsm.
                        ind = nsCum(1);
                        if (SSmodel::inputs.a.n_rows > ns)
                                ind = SSmodel::inputs.a.n_rows - 1;
                        inputs.comp.row(2) = SSmodel::inputs.a.row(ind);
                        inputs.compV.row(2) = SSmodel::inputs.P.row(ind);
                }
                inputs.compNames += "Seasonal/";
        } else
                remove(2) = 1;
        // Irregular
        if (inputs.ns(3) == 0){     // White noise
                inputs.comp.row(3).cols(0, ny) = SSmodel::inputs.y.t() - SSmodel::inputs.yFit.rows(0, ny).t();
                //vec F = SSmodel::inputs.y;
                // uvec ind = find_finite(SSmodel::inputs.y);
                // inputs.compV()
                // inputs.compV.row(3).cols(0, ny) = SSmodel::inputs.F.rows(0, ny).t();
                //
                //
                //
                // inputs.compV.row(3).cols(0, ny) = SSmodel::inputs.F.rows(0, ny).t();
                rowvec v = inputs.comp.row(3).cols(0, ny);
                uvec aux = find_finite(v);
                //inputs.comp.submat(3, 0, 3, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
                if (sum(abs(v(aux))) > 1e-7)
                        inputs.compNames += "Irregular/";
                else
                        remove(3) = 1;
        } else {                    // ARMA
                inputs.comp.row(3) = SSmodel::inputs.a.row(nsCum(2));
                // inputs.compV.row(3) = SSmodel::inputs.P.row(nsCum(2));
                rowvec v = inputs.comp.row(3).cols(0, ny);
                uvec aux = find_finite(v);
                //inputs.comp.submat(3, 0, 3, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
                if (sum(abs(aux)) > 1e-7){
                        if (inputs.armaLags.n_elem == 2 &&
                            (inputs.arOrders(1) > 0 || inputs.maOrders(1) > 0)){
                                inputs.compNames += "SARMA(" +
                                        to_string(inputs.arOrders(0)) + "," +
                                        to_string(inputs.maOrders(0)) + ")(" +
                                        to_string(inputs.arOrders(1)) + "," +
                                        to_string(inputs.maOrders(1)) + ")_" +
                                        to_string(inputs.armaLags(1)) + "/";
                        } else {
                                inputs.compNames += "ARMA(" +
                                        to_string(inputs.ar) + "," +
                                        to_string(inputs.ma) + ")/";
                        }
                } else
                        remove(3) = 1;
        }
        // NaNs in irregular
        //if (inputs.missing.n_elem > 0){
        //    vec aux = inputs.comp.row(3);
        //    aux(inputs.missing).fill(datum::nan);
        //    inputs.comp.row(3) = aux;
        //}
        // Cycle
        if (inputs.ns(1) > 0){
                for (int i = 0; i < nCycles; i++){
                        inputs.comp.row(4 + i) = SSmodel::inputs.a.row(nsCum(0) + 2 * i);
                        inputs.compV.row(4 + i) = SSmodel::inputs.P.row(nsCum(0) + 2 * i);
                        //if (sum(abs(inputs.comp.row(4 + i).cols(0, ny))) > 1e-7)
                        inputs.compNames += "Cycle" + to_string(i + 1) + "/";
                        //else
                        //    remove(4 + i) = 1;
                }
        }
        // Inputs
        if (k > 0){
                if (SSmodel::inputs.system.Z.n_rows == 1){
                        for (int i = 0; i < k; i++){
                                inputs.comp.row(4 + nCycles + i) = SSmodel::inputs.system.D(i) *
                                        SSmodel::inputs.u.submat(i, 0, i, SSmodel::inputs.u.n_cols - 1);
                        }
                } else {
                        for (int i = 0; i < k; i++){
                                inputs.comp.row(4 + nCycles + i) =
                                        SSmodel::inputs.a.row(nsCum(5) + i) % SSmodel::inputs.u.row(i);
                        }
                }
                // Components names
                int nOut = 0;
                if (inputs.typeOutliers.n_rows > 0 && inputs.typeOutliers(0, 1) != -1)
                        nOut = inputs.typeOutliers.n_rows;
                int nU = k - nOut;
                if (nU > 0){
                        for (int i = 0; i < nU; i++)
                                inputs.compNames += "Reg(" + to_string(i + 1) + ")/";
                }
                if(nOut > 0){
                        string namei;
                        for (int i = 0; i < nOut; i++){
                                namei = "AO";
                                if (inputs.typeOutliers(i, 0) == 1){
                                        namei = "LS";
                                } else if(inputs.typeOutliers(i, 0) == 2){
                                        namei = "SC";
                                }
                                inputs.compNames += namei + to_string((uword)(inputs.typeOutliers(i, 1) + 1)) + "/";
                        }
                }
        }
        //inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::nan, 0);
        //inputs.comp.submat(0, 0, inputs.comp.n_rows - 1, SSmodel::inputs.y.n_elem - 1).replace(datum::inf, 0);
        if (sum(remove) > 0){
                inputs.comp.shed_rows(find(remove == 1));
                inputs.compV.shed_rows(find(remove == 1));
        }
        // Components that are (numerically) constant throughout the sample
        // -- e.g. the slope of a global/deterministic trend -- are forced
        // to be strictly constant, taking the last estimated value.
        for (uword i = 0; i < inputs.comp.n_rows; i++){
                rowvec r = inputs.comp.row(i);
                uvec fin = find_finite(r);
                if (fin.n_elem > 1 && r(fin).max() - r(fin).min() < 1e-10){
                        inputs.comp.row(i).fill(r(fin(fin.n_elem - 1)));
                        rowvec rv = inputs.compV.row(i);
                        uvec finV = find_finite(rv);
                        if (finV.n_elem > 0)
                                inputs.compV.row(i).fill(rv(finV(finV.n_elem - 1)));
                }
        }
        /*
         // Correcting for lambda
         mat aux = sqrt(compV);
         aux = inputs.comp + aux;
         inputs.compPlus = invBoxCoxMat(inputs.comp + aux, inputs.lambda);
         inputs.compMinus = invBoxCoxMat(inputs.comp - aux, inputs.lambda);
         inputs.comp = invBoxCoxMat(inputs.comp, inputs.lambda);
         inputs.compNames = inputs.compNames.substr(0, inputs.compNames.length() - 1);
         */
}
// Covariance of parameters (inverse of hessian)
mat BSMclass::parCov(vec& returnP){
        vec reserveP = SSmodel::inputs.p;
        // Finding true parameter values
        SSmodel::inputs.p = parameterValues(SSmodel::inputs.p);
        // Hessian and covariance of parameters
        int k = SSmodel::inputs.p.n_elem;
        uvec nn = find_finite(SSmodel::inputs.y);
        bool reserveCLLIK = SSmodel::inputs.cLlik;
        uvec isVar = find(inputs.typePar == 0); //, nonConstrained = find(inputs.constPar == 0);
        if (reserveCLLIK){
                SSmodel::inputs.userModel = bsmMatricesTrue;
                SSmodel::inputs.cLlik = false;
                //    SSmodel::inputs.p(isVar) =log(exp(2 * SSmodel::inputs.p(isVar)) * SSmodel::inputs.innVariance) / 2;
        }
        returnP = SSmodel::inputs.p;
        mat hess = hessLlik(&(SSmodel::inputs));
        hess *= 0.5 * (nn.n_elem);   // - SSmodel::inputs.nonStationaryTerms - reserveP.n_elem + 1);
        // Index set that actually gets inverted: the concentrated-likelihood
        // path inverts the unconstrained submatrix, the crude path the whole
        // matrix.  (Mirrors the original cLlik branch.)
        mat iHess(k, k);
        iHess.fill(datum::nan);
        uvec indHess = find(inputs.constPar < 1);
        uvec ind = reserveCLLIK ? indHess : regspace<uvec>(0, k - 1);
        if (ind.n_elem > 0){
                // OPG / BHHH information over the requested parameter indices.
                // Built from per-observation scores of the (absolute-scale)
                //   nll_t = 0.5 (log F_t + v_t^2 / F_t) - logJac_t
                // (cLlik off and bsmMatricesTrue active here), central-
                // differenced from the stored innovations.  PSD by construction
                // and -- crucially -- finite even when the second-difference
                // Hessian is not (single, small perturbations, no inversion of a
                // near-singular matrix).  Returns an empty matrix if it cannot be
                // formed (too few usable obs, or non-finite scores).
                auto tryOPG = [&]()->mat {
                        uword n = SSmodel::inputs.y.n_rows;
                        vec p0 = SSmodel::inputs.p;
                        auto perObsNLL = [&](vec pp)->vec {
                                llik(pp, &(SSmodel::inputs));
                                vec vv = SSmodel::inputs.v, FF = SSmodel::inputs.F;
                                double lam = SSmodel::inputs.lambda;
                                bool haveRaw = (SSmodel::inputs.y_raw.n_elem == n);
                                vec c(n); c.fill(datum::nan);
                                for (uword t = 0; t < n; t++){
                                        if (!std::isfinite(vv(t)) || !std::isfinite(FF(t)) || FF(t) <= 0)
                                                continue;
                                        double val = 0.5 * (std::log(FF(t)) + vv(t) * vv(t) / FF(t));
                                        if (haveRaw)
                                                val -= bcnormLogJac(SSmodel::inputs.y_raw(t), lam);
                                        c(t) = val;
                                }
                                return c;
                        };
                        mat S(n, ind.n_elem, fill::zeros);
                        for (uword c = 0; c < ind.n_elem; c++){
                                uword idx = ind(c);
                                double hh = std::pow(datum::eps, 1.0 / 3.0)
                                            * std::max(std::abs(p0(idx)), 1.0);
                                vec pp = p0; pp(idx) += hh;
                                vec pm = p0; pm(idx) -= hh;
                                S.col(c) = (perObsNLL(pp) - perObsNLL(pm)) / (2.0 * hh);
                        }
                        SSmodel::inputs.p = p0;
                        llik(p0, &(SSmodel::inputs));        // restore v/F at the solution
                        uvec good = find_finite(sum(S, 1));  // drop diffuse/missing obs
                        if (good.n_elem < ind.n_elem) return mat();
                        mat Sg = S.rows(good);
                        mat opg = Sg.t() * Sg;
                        if (!opg.is_finite()) return mat();
                        return pinv(opg);
                };
                // Prefer the observed-information Hessian when it is finite,
                // positive definite and not numerically singular.  Otherwise --
                // indefinite (a boundary / weakly-identified variance), ill-
                // conditioned, or outright non-finite (a degenerate optimum where
                // the second differences blow up) -- fall back to OPG.  The
                // augmented (xreg) path stores no per-obs v/F, so OPG is
                // unavailable there; it keeps the Hessian (or NaN if non-finite).
                mat H;
                bool hessFinite = hess.is_finite();
                bool hessUsable = false;
                if (hessFinite){
                        H = hess.submat(ind, ind);
                        H = 0.5 * (H + H.t());           // symmetrise
                        vec eval;
                        bool eigok = eig_sym(eval, H);
                        double emax = eval.n_elem ? eval.max() : 0.0;
                        double emin = eval.n_elem ? eval.min() : 0.0;
                        const double condTol = 1e-8;     // emin <= condTol*emax covers emin<=0
                        hessUsable = eigok && (emin > condTol * std::max(emax, 1e-300));
                }
                mat covSub;
                bool haveCov = false;
                if (hessUsable || SSmodel::inputs.augmented){
                        if (hessFinite){ covSub = pinv(H); haveCov = true; }
                } else {
                        covSub = tryOPG();
                        if (!covSub.is_empty()){
                                haveCov = true;
                        } else if (hessFinite){
                                covSub = pinv(H);        // last resort
                                haveCov = true;
                        }
                }
                if (haveCov){
                        iHess.submat(ind, ind) = covSub;
                        iHess.diag() = abs(iHess.diag());
                }
        }
        // Restore the estimation-scale configuration.
        SSmodel::inputs.cLlik = reserveCLLIK;
        SSmodel::inputs.p = reserveP;
        SSmodel::inputs.userModel = bsmMatrices;
        return iHess;
}
// Finding true parameter values out of transformed parameters
vec BSMclass::parameterValues(vec p){
        vec nparCum = cumsum(inputs.nPar);
        int nCycles = sum(inputs.rhos < 0);
        // Transforming all variances
        vec parValues(p.n_elem);
        vec isVar(p.n_elem);
        uvec aux; aux = find(inputs.typePar == 0);
        parValues(aux) = exp(2 * p(aux));
        isVar.fill(0);
        isVar(aux).fill(1);
        // Transforming the rest of parameters
        // Trend
        if (inputs.nPar(0) == 3){                     // Damped trend
                double alpha = p(0);
                constrain(alpha, regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
                parValues(0) = alpha;
        }
        //Cycle
        if (inputs.nPar(1) > 0){
                // Rhos
                int pos;
                aux = regspace<uvec>(nparCum(0), nparCum(1) - 1);
                vec pCycle = SSmodel::inputs.p(aux);
                vec pp = pCycle(arma::span(0, nCycles - 1));
                constrain(pp, regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
                pos = nparCum(0) + nCycles;
                parValues(arma::span(nparCum(0), pos - 1)) = pp;
                // Periods
                int nn = sum(inputs.periods < 0);
                if (nn > 0){
                        aux = find(inputs.periods < 0);
                        pp = pCycle(arma::span(nCycles, nCycles - 1 + nn));
                        constrain(pp, inputs.cycleLimits.rows(aux)); //exp(p(0)) / (1+ exp(p(0)));
                        parValues(arma::span(pos, pos - 1 + nn)) = pp;
                        pos = pos + nn;
                }
                // Variances
                // parValues(arma::span(pos, nparCum(1) - 1)) = exp(2 * pCycle(arma::span(nCycles + nn, 2 * nCycles + nn - 1)));
        }
        // ARMA — apply polyStationary per-lag block so seasonal coefficients
        // round-trip correctly (the convolution lives in armaMatrices and
        // must not be re-applied here; what user code sees are the raw per-
        // block stationary coefficients).
        if (inputs.ar > 0 || inputs.ma > 0) {  // ARMA model
                uvec ind;
                vec polyAux;
                isVar(arma::span(nparCum(2) + 1, isVar.n_elem - 1)).fill(0);
                uword offset = nparCum(2) + 1;
                if (inputs.ar > 0){
                        if (inputs.arOrders.n_elem > 0){
                                uword pos = offset;
                                for (uword b = 0; b < inputs.arOrders.n_elem; ++b){
                                        int pi = inputs.arOrders(b);
                                        if (pi == 0) continue;
                                        ind = regspace<uvec>(pos, pos + pi - 1);
                                        polyAux = p(ind);
                                        polyStationary(polyAux);
                                        parValues(ind) = polyAux;
                                        pos += pi;
                                }
                        } else {
                                ind = regspace<uvec>(offset, offset + inputs.ar - 1);
                                polyAux = p(ind);
                                polyStationary(polyAux);
                                parValues(ind) = polyAux;
                        }
                }
                if (inputs.ma > 0){
                        uword startMA = offset + inputs.ar;
                        if (inputs.maOrders.n_elem > 0){
                                uword pos = startMA;
                                for (uword b = 0; b < inputs.maOrders.n_elem; ++b){
                                        int qi = inputs.maOrders(b);
                                        if (qi == 0) continue;
                                        ind = regspace<uvec>(pos, pos + qi - 1);
                                        polyAux = p(ind);
                                        polyStationary(polyAux);
                                        parValues(ind) = polyAux;
                                        pos += qi;
                                }
                        } else {
                                ind = regspace<uvec>(startMA, startMA + inputs.ma - 1);
                                polyAux = p(ind);
                                polyStationary(polyAux);
                                parValues(ind) = polyAux;
                        }
                }
        }
        // Inputs
        //if (nu > 0){
        //    ind1 = SSmodel::inputs.betaAug.n_elem - nu;
        //    parValues.rows(nparCum(3), nparCum(3) + nu - 1) = SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1);
        //}
        // pureARMA
        if (inputs.pureARMA){
                // setting constant value
                parValues(nparCum(5) - 1) = SSmodel::inputs.p(nparCum(5) - 1);
        }
        if (SSmodel::inputs.cLlik){
                parValues(find(isVar)) *= SSmodel::inputs.innVariance;
        }
        return parValues;
}
// Parameter names
void BSMclass::parLabels(){
        inputs.parNames.clear();
        // Trend
        if (inputs.trend[0] == 'd' || inputs.trend[0] == 's'){
                inputs.parNames.push_back("Damping");
        }
        if (inputs.trend[0] != 'n' && inputs.trend[0] != 'i')
                inputs.parNames.push_back("Level");
        if (inputs.trend[0] != 'n' && inputs.trend[0] != 'r' && inputs.trend[0] != 't'){
                inputs.parNames.push_back("Slope");
        }
        vec nsCum = cumsum(inputs.ns);
        vec nparCum = cumsum(inputs.nPar);
        // Cycle
        int nCycles = sum(inputs.rhos < 0);
        uvec aux;
        vec pCycle, typeParC;
        //if (inputs.cycle[0] != 'n'){
        if (nCycles > 0 && SSmodel::inputs.p.n_elem > 0){
                aux = regspace<uvec>(nparCum(0), nparCum(1) - 1);
                pCycle = SSmodel::inputs.p(aux);
                typeParC = inputs.typePar(aux);
                int count;
                char name[20];
                for (count = 0; count < nCycles; count++){
                        snprintf(name, 20, "Rho(%d)", count + 1);
                        inputs.parNames.push_back(name);
                }
                for (count = 0; count < nCycles; count++){
                        if (inputs.periods(count) < 0){
                                snprintf(name, 20, "Period(%d)", count + 1);
                                inputs.parNames.push_back(name);
                        }
                }
                for (count = 0; count < nCycles; count++){
                        snprintf(name, 20, "Var(%d)", count + 1);
                        inputs.parNames.push_back(name);
                }
        }
        // Seasonal
        if (inputs.seasonal[0] == 'l')
                inputs.parNames.push_back("Seas");
        if (inputs.seasonal[0] == 'e')
                inputs.parNames.push_back("Seas(All)");
        if (inputs.seasonal[0] == 'd'){
                char seasNames[20];
                for (unsigned int i = sum(inputs.rhos < 0); i < inputs.periods.n_elem; i++){
                        snprintf(seasNames, 20, "Seas(%1.1f)", inputs.periods(i));
                        inputs.parNames.push_back(seasNames);
                }
        }
        // Irregular
        if (inputs.irregular[0] != 'n')
                inputs.parNames.push_back("Irregular");
        if ((inputs.ar > 0 || inputs.ma > 0) && inputs.irregular[0] != 'n'){
                char arNames[20];
                // Emit per-lag AR labels.  Block 0 (lag = 1) is "AR(i)";
                // block 1 (lag = s) is "SAR(i)" — the R-side methods in
                // R/methods.R and R/pts-summary.R already pattern-match on
                // "^AR(" / "^MA(" so the leading "S" stays compatible.
                if (inputs.arOrders.n_elem > 0){
                        for (arma::uword b = 0; b < inputs.arOrders.n_elem; ++b){
                                const char* prefix = (b == 0) ? "AR" : "SAR";
                                for (int i = 0; i < inputs.arOrders(b); ++i){
                                        snprintf(arNames, 20, "%s(%d)",
                                                 prefix, i + 1);
                                        inputs.parNames.push_back(arNames);
                                }
                        }
                } else {
                        for (int i = 0; i < inputs.ar; i++){
                                snprintf(arNames, 20, "AR(%d)", i + 1);
                                inputs.parNames.push_back(arNames);
                        }
                }
                if (inputs.maOrders.n_elem > 0){
                        for (arma::uword b = 0; b < inputs.maOrders.n_elem; ++b){
                                const char* prefix = (b == 0) ? "MA" : "SMA";
                                for (int i = 0; i < inputs.maOrders(b); ++i){
                                        snprintf(arNames, 20, "%s(%d)",
                                                 prefix, i + 1);
                                        inputs.parNames.push_back(arNames);
                                }
                        }
                } else {
                        for (int i = 0; i < inputs.ma; i++){
                                snprintf(arNames, 20, "MA(%d)", i + 1);
                                inputs.parNames.push_back(arNames);
                        }
                }
        }
        // Inputs
        int nOut = inputs.typeOutliers.n_rows;
        int nu = SSmodel::inputs.u.n_rows;
        if (sum(inputs.TVP) > 0){
                char betas[20];
                uvec ind = find(inputs.TVP);
                /////////////////****
                vec uniqueTVP = unique(inputs.TVP);
                for (uword i = 0; i < sum(uniqueTVP > 0); i++){
                        snprintf(betas, 20, "TVP(%d)", (int)ind(i) + 1);
                        inputs.parNames.push_back(betas);
                }
                for (int i = 0; i < nu - nOut; i++){
                        snprintf(betas, 20, "State(%d)", i + 1);
                        inputs.parNames.push_back(betas);
                }
        } else if (nu - nOut > 0){
                char betas[20];
                for (int i = 0; i < nu - nOut; i++){
                        snprintf(betas, 20, "Beta(%d)", i + 1);
                        inputs.parNames.push_back(betas);
                }
        }
        // Outliers
        if (nOut > 0){
                char betas[20], typeO[5];
                for (int i = 0; i < nOut; i++){
                        if (inputs.typeOutliers(i, 0) == 0){
                                snprintf(typeO, 5, "AO");
                        } else if (inputs.typeOutliers(i, 0) == 1){
                                snprintf(typeO, 5, "LS");
                        } else if (inputs.typeOutliers(i, 0) == 2){
                                snprintf(typeO, 5, "SC");
                        }
                        snprintf(betas, 20, "%s%0.0f", typeO, inputs.typeOutliers(i, 1) + 1);
                        inputs.parNames.push_back(betas);
                }
        }
        if (inputs.pureARMA){
                inputs.parNames.push_back("Const");
        }
}
// validate: build the printable parameter / diagnostics table on a fitted
// model and stash it in SSmodel::inputs.table.  ~169 lines.  Outline:
//
//   1. Run SSmodel::validate() (computes residual statistics) and parCov()
//      (parameter covariance from the Hessian).
//   2. Compute p / stdP, joining the BSM params with the betas if there
//      are external regressors.
//   3. Insert the table header (model spec, Box-Cox lambda, periods,
//      footnote markers for concentrated/constrained parameters).
//   4. Compute t-stats, p-values, gradient column; mark constrained rows
//      with stars / suppress invalid stats.
//   5. Emit one formatted row per parameter into SSmodel::inputs.table.
//   6. Store the final coefficient vector in SSmodel::inputs.coef.
//   7. Optionally print the whole table.
//
// Section 5 (the format loop) is the largest, ~70 lines, and is the most
// obvious candidate for extraction into writeParamRows(...).
void BSMclass::validate(bool showTable){
        // SSpace validate
        SSmodel::validate(false, sum(inputs.nPar));
        vec scores, innovations = SSmodel::inputs.v;
        mat iHess = parCov(scores);
        SSmodel::inputs.covp = iHess;
        // Parameter names
        parLabels();
        // Parameter values
        int nu = SSmodel::inputs.u.n_rows;
        vec p, stdP, stdPBSM;
        stdPBSM = sqrt(abs(iHess.diag()));
        if (nu == 0 ){
                p = scores;
                stdP = stdPBSM;
        } else {
                if (SSmodel::inputs.system.Z.n_rows == 1){
                        uword ind1 = SSmodel::inputs.betaAug.n_elem - nu;
                        p = join_vert(scores, SSmodel::inputs.betaAug.rows(ind1, ind1 + nu - 1));
                        stdP = join_vert(stdPBSM, sqrt(SSmodel::inputs.betaAugVar.rows(ind1, ind1 + nu - 1)));
                } else {
                        uword ind1 = SSmodel::inputs.aEnd.n_rows - nu;
                        p = join_vert(scores, SSmodel::inputs.aEnd.rows(ind1, SSmodel::inputs.aEnd.n_rows - 1));
                        vec dPEnd = SSmodel::inputs.PEnd.diag();
                        stdP = join_vert(stdPBSM, sqrt(dPEnd.rows(ind1, SSmodel::inputs.aEnd.n_rows - 1)));
                }
        }
        vec parValues = p;
        // Calculating t stats and pValues
        char str[70];
        vec t = stdP;
        vec pValue = abs(p / stdP);
        // Creating table
        string fullModel;
        if (SSmodel::inputs.u.n_rows == 0){
                fullModel = inputs.model;
        } else {
                fullModel = inputs.model + " + inputs";
        }
        if (SSmodel::inputs.system.Z.n_rows > 1){
                fullModel += " + TVP";
        }
        string MODEL = fullModel;
        if (inputs.PTSnames)
                MODEL = UC2PTS(fullModel, inputs.lambda);
        snprintf(str, 70, " Model: %s\n", MODEL.c_str());
        auto it = SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 1, str);
        snprintf(str, 70, " Box-Cox lambda: %3.2f\n", inputs.lambda);
        SSmodel::inputs.table.insert(it, str);
        //if (SSmodel::inputs.cLlik)
        //    SSmodel::inputs.table.insert(it, " Concentrated Maximum-Likelihood\n");
        //else
        //    SSmodel::inputs.table.insert(it, " Maximum-Likelihood\n");
        vec col1;
        int insert = 0;
        // Periods
        vec periods1 = inputs.periods(find(inputs.rhos > 0));
        if (inputs.seasonal[0] == 'l')
                periods1.reset();
        int lPer = periods1.n_elem;
        if (lPer > 0 && periods1(0) > 1 && inputs.nPar(2) > 0){
                string line;
                snprintf(str, 70, " Periods: %5.1f", periods1(0));
                line = str;
                for (int i = 1; i < lPer; i++){
                        snprintf(str, 70, " /%5.1f", periods1(i));
                        line += str;
                }
                SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, line + "\n");
                insert++;
        } else {
                SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 3, " Periods: \n");
                insert++;
        }
        if (any(inputs.constPar == 1)){
                snprintf(str, 70, " (*)  concentrated out parameters\n");
                SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
                insert++;
        }
        if (any(inputs.constPar > 1)){
                snprintf(str, 70, " (**) constrained parameters during estimation\n");
                SSmodel::inputs.table.insert(SSmodel::inputs.table.begin() + 4 + insert, str);
                insert++;
        }
        SSmodel::inputs.table.at(5 + insert) = "                     Param   asymp.s.e.        |T|     |Grad| \n";
        col1 = parValues;
        // Pretty numbers for constrained parameters
        if (inputs.nPar(0) == 3 && inputs.constPar(0) > 0){  // DT trend
                if (p(0) < -100)
                        col1(0) = 0;
                if (p(0) > 100)
                        col1(0) = 1;
        }
        uvec indConst = find(inputs.constPar == 2);
        if (indConst.n_elem > 0){
                col1(indConst).fill(0);
        }
        uvec ind = find(inputs.constPar > 1);
        t(ind).fill(datum::nan);
        pValue(ind).fill(datum::nan);
        SSmodel::inputs.grad(ind).fill(datum::nan);
        vec gradBetas(nu);
        gradBetas.fill(datum::nan);
        vec grad = join_vert(SSmodel::inputs.grad, gradBetas);
        // stars for constrained parameters
        vector<string> col2;
        string chari;
        vec constPar;
        if (nu > 0){
                if (SSmodel::inputs.system.Z.n_rows == 1)
                        constPar = join_vert(inputs.constPar, ones(nu, 1));
                else
                        constPar = join_vert(inputs.constPar, zeros(nu, 1));
        } else {
                constPar = inputs.constPar;
        }
        for (uword i = 0; i < constPar.n_elem; i++){
                if (constPar(i) == 0)
                        chari = "  ";
                else if (constPar(i) == 1)
                        chari = "* ";
                else
                        chari = "**";
                col2.push_back(chari);
        }
        // Adding spaces for betas
        for (int i = 0; i < nu; i++){
                col2.push_back("  ");
        }
        // for (unsigned i = 0; i < p.n_elem; i++){
        //     snprintf(str, 70, "%s:  \n", inputs.parNames.at(i).c_str());
        // }
        for (unsigned i = 0; i < sum(inputs.nPar) + SSmodel::inputs.u.n_rows; i++){
                if (abs(col1(i)) > 1e-3 || abs(col1(i)) == 0 || abs(col1(i)) == 1){
                        if (isnan(pValue(i))){
                                snprintf(str, 70, "%*s: %12.4f%2s \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str());
                        } else {
                                if (constPar(i) > 0 || isnan(grad(i))){
                                        snprintf(str, 70, "%*s: %12.4f%2s %10.4f %10.4f \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i));
                                } else {
                                        snprintf(str, 70, "%*s: %12.4f%2s %10.4f %10.4f %10.2e\n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
                                }
                        }
                } else {
                        if (isnan(pValue(i))){
                                snprintf(str, 70, "%*s: %12.2e%2s \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str());
                        } else {
                                if (constPar(i) > 0 || isnan(grad(i))){
                                        snprintf(str, 70, "%*s: %12.2e%2s %10.2e %10.4f \n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i));
                                } else {
                                        snprintf(str, 70, "%*s: %12.2e%2s %10.2e %10.4f %10.2e\n", 12, inputs.parNames.at(i).c_str(), col1(i), col2.at(i).c_str(), t(i), pValue(i), abs(grad(i)));
                                }
                        }
                }
                SSmodel::inputs.table.at(i + 7 + insert) = str;
        }
        // mat coef = join_horiz(join_horiz(col1, t), pValue);
        SSmodel::inputs.coef = col1;
        // if (showTable){
        //     for (auto i = SSmodel::inputs.table.begin(); i != SSmodel::inputs.table.end(); i++){
        //         cout << *i << " ";
        //     }
        // }
        if (showTable){
                for (unsigned int i = 0; i < SSmodel::inputs.table.size(); i++){
                        Rprintf("%s ", SSmodel::inputs.table[i].c_str());
                }
        }
        SSmodel::inputs.v = innovations;
}
// Disturbance smoother (to recover just trend and epsilons)
void BSMclass::disturb(){
        inputs.eps.zeros(SSmodel::inputs.y.n_elem);
        if (inputs.irregular[0] == 'a' && inputs.ar == 0 && inputs.ma == 0){
                // Modification adding the observation noise as a final state
                SSinputs  copiaSS  = SSmodel::getInputs();
                // Modifying system
                int nsAll = sum(inputs.ns) + 1;
                // int nu = SSmodel::inputs.u.n_rows;
                bool TVP = (copiaSS.system.Z.n_rows > 1);
                mat T(nsAll, nsAll), Z(1, nsAll); //, R(nsAll, nsAll), Q(nsAll, nsAll); //, D(1, nu);
                uword colR = copiaSS.system.R.n_cols + 1;
                mat R(nsAll, colR), Q(colR, colR);
                if (TVP)
                    Z.resize(copiaSS.u.n_cols, nsAll);
                T.fill(0);
                R.fill(0.0);
                Q.fill(0);
                Z.fill(0);
                nsAll -= 2;
                colR -= 2;
                T(arma::span(0, nsAll), arma::span(0, nsAll)) = copiaSS.system.T;
                R(arma::span(0, nsAll), arma::span(0, colR)) = copiaSS.system.R;
                R(nsAll + 1, colR + 1) = 1;
                Q(arma::span(0, colR), arma::span(0, colR)) = copiaSS.system.Q;
                Q(colR + 1, colR + 1) = copiaSS.system.H(0, 0);
                //Q(arma::span(0, nsAll), arma::span(0, nsAll)) = copiaSS.system.Q;
                if (TVP){
                        Z.cols(0, nsAll) = copiaSS.system.Z;
                        Z.col(nsAll + 1).fill(copiaSS.system.C(0, 0));
                } else {
                        Z(0, arma::span(0, nsAll)) = copiaSS.system.Z;
                        Z(0, nsAll + 1) = copiaSS.system.C(0, 0);
                }
                // copying into copiaSS
                copiaSS.system.T = T;
                copiaSS.system.R = R;
                copiaSS.system.Q = Q;
                copiaSS.system.Z = Z;
                copiaSS.system.H(0, 0) = 0;
                copiaSS.system.C(0, 0) = 0;
                // Creating new system
                SSmodel copia = SSmodel(copiaSS);
                copia.disturb();
                // Saving in system the disturbances (just trend and irregular)
                copiaSS = copia.getInputs();
                inputs.eps = copiaSS.eta.row(copiaSS.eta.n_rows - 1).t();
                SSmodel::inputs.eta = copiaSS.eta.rows(arma::span(0, inputs.ns(0) - 1));
        } else {
                // No need of system modification, just run disturb()
                SSmodel::disturb();
                if (inputs.nPar(2) > 1)
                        inputs.eps = SSmodel::inputs.eta.row(SSmodel::inputs.eta.n_rows - 1).t();
                SSmodel::inputs.eta = SSmodel::inputs.eta.rows(arma::span(0, inputs.ns(0) - 1));
        }
}
// quasiNewtonBSM: BFGS-style optimiser with concentrated-likelihood
// support and adaptive switching of the concentrated-out parameter.
// ~241 lines.  Outline:
//
//   1. Initial setup: detect cLlik, zero the concentrated position, do an
//      initial objFun / gradFun evaluation at xNew.
//   2. Main BFGS loop:
//        a. Compute search direction d = -iHess * grad (mask constrained
//           positions).
//        b. Line search with backtracking + Armijo.
//        c. Translate xNew between concentrated and "true-variance" space
//           when cLlik is active (the xUncon rescaling at line 2550-ish).
//        d. Detect zero variances; mark them in constPar.
//        e. Switch the concentrated-out parameter if a different variance
//           now dominates (the largestVar check).
//        f. Re-evaluate the gradient at the new x.
//        g. BFGS Hessian update.
//        h. Convergence / nan-handling / new-try restart.
//   3. Store flag, iter, pTransform on SSmodel::inputs.
//
// Section 2c (xUncon transformation) and 2e (concentrated-param swap) are
// the two pieces most worth extracting into helpers; both have clear
// mathematical contracts.
int BSMclass::quasiNewtonBSM(std::function <double (vec& x, void* inputsFake)> objFun,
                             std::function <vec (vec& x, void* inputsFake, double obj, int& nFuns)> gradFun,
                             vec& xNew, void* inputsFake, double& objNew, vec& gradNew, mat& iHess,
                             bool verbosef){
        // Code for inputs.constPar
        // 0: not constrained; 1: concentrated-out; 2: zero variance; 3: alpha constrained
        int nx = xNew.n_elem,
                flag = 0,
                nOverallFuns,
                nFuns = 0,
                nIter = 0,
                plateauStreak = 0;  // see plateau-exit block below
        double objOld, alpha_i;
        vec gradOld(nx),
        xOld = xNew,
        d(nx);
        // Convergence thresholds.  crit(1) is the |dobj| threshold for
        // flag=2 ("objective stopped changing").  1e-7 was too tight: on
        // raw-scale objectives BFGS hits a plateau where dobj is ~1e-5
        // per iter but |grad| is small and parameters barely move; the
        // criterion never fires and BFGS limps to crit(4) = 10000 fun
        // evals (~500 iter for the typical 20-fun line search).  1e-4 is
        // still 4-decimal precision in the log-lik objective -- well
        // beyond what AIC / BIC differences can distinguish -- and lets
        // BFGS terminate at the plateau in O(2-5) iter.  The fast-
        // convergence path (lambda ~ -0.29 on AirPassengers etc.) is
        // unaffected because it exits at iter 2 via flag=6 (obj
        // overshoot) before flag=2 ever evaluates.
        vec crit(5); crit(0) = 1e-6; crit(1) = 1e-4; crit(2) = 1e-5; crit(3) = 1000; crit(4) = 10000;
        iHess.eye(nx, nx);
        uvec isVar = find(inputs.typePar == 0); //, nonConstrained = find(inputs.constPar == 0);
        bool cLlik = (sum(inputs.constPar) > 0);
        int newTry = 0;
        // Initial concentrated variance
        uvec concentratedOutPar = find(inputs.constPar == 1);
        uvec initialConcentratedPar = concentratedOutPar;
        if (cLlik){
                xNew(concentratedOutPar).fill(0);
        }
        // Calculating objective function and gradient
        objNew = objFun(xNew, inputsFake);
        vec xUncon = xNew;
        // xUncon is xNew de-converted to original variances (xNew is ratio of variances)
        if (cLlik){
                xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;
        }
        if (inputs.pureARMA){
                gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
        } else {
                gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
        }
        nOverallFuns = nFuns + 1;
        if (cLlik){
                gradNew(concentratedOutPar).fill(0);
        }
        // Head of table
        if (verbosef){
                Rprintf(" Iter FunEval  Objective       Step\n");
                Rprintf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, 1.0);
        }
        // Main loop
        uvec zeroVar, largestVar, allVar = find(inputs.typePar == 0); //, nonConst;
        vec newVar, variances, maxVar, d_old, critPar(1); //, critGrad(1);
        double innVar, objBest = 1e6;
        bool diagHess = false;
        int counter = 0;    // Regulates diagonal or full hessian
        vec xBest;
        // Main loop
        do{
                nIter++;
                // Search direction
                d = -iHess * gradNew;
                d(find(inputs.constPar > 0)).fill(0);
                // Descent-direction safeguard: a BFGS inverse-Hessian that has
                // lost positive-definiteness (ill-conditioned / non-convex
                // region) can yield d with d'grad >= 0, i.e. NOT a descent
                // direction -- the line search then backtracks to its floor
                // without decreasing and the optimiser stalls at a
                // non-stationary point.  Reset to steepest descent so progress
                // is guaranteed whenever the curvature estimate goes bad.
                if (as_scalar(dot(d, gradNew)) >= 0){
                        iHess.eye(nx, nx);
                        d = -gradNew;
                        d(find(inputs.constPar > 0)).fill(0);
                }
                if (counter < 6 && as_scalar(abs(d.t() * gradNew)) > 0.01)
                        diagHess = true;
                // Line Search
                xOld = xNew; gradOld = gradNew; objOld = objNew;
                alpha_i = 0.5;
                // storing in case objNew becomes nan
                innVar = SSmodel::inputs.innVariance;
                d_old = d;
                lineSearch(objFun, alpha_i, xNew, objNew, gradNew, d, nIter, nFuns, inputsFake);
                // Correcting when function becomes nan
                if (isnan(objNew)){   // Linesearch failed
                        xNew = xOld;
                        objNew = objOld;
                        objOld = datum::nan;
                        gradNew = gradOld;
                        SSmodel::inputs.innVariance = innVar;
                        alpha_i = 1;
                        d = d_old;
                }
                nOverallFuns = nOverallFuns + nFuns;
                xUncon = xNew;
                if (cLlik){
                        xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;
                }
                // Checking for zero variances.  Pin a variance to zero (mark it
                // deterministic, constPar = 2) when EITHER:
                //   (a) it is negligibly small (xUncon < -15, var < exp(-30) ~
                //       9e-14) AND its gradient is essentially flat (|grad| <
                //       1e-6) -- a genuine boundary optimum; OR
                //   (b) it has underflowed to an absurd value (xUncon < -25,
                //       var < exp(-50) ~ 2e-22) REGARDLESS of the gradient.
                // Without (b), a collapsing variance whose gradient never quite
                // reaches the 1e-6 flat threshold keeps drifting to ~1e-117; an
                // active component at that extreme corrupts the concentrated
                // log-likelihood (the sum-log-F / diffuse terms blow up, giving
                // a bogus +logLik while sigma stays sane).  Pinning it makes the
                // component properly deterministic and keeps the likelihood
                // well-conditioned.  Healthy variances (xUncon ~ -1..1) are far
                // from both thresholds and unaffected.
                zeroVar = find((((xUncon % (inputs.constPar == 0) % (inputs.typePar == 0)) < -15) +
                        ((abs(gradNew) % (inputs.constPar == 0)) % (inputs.typePar == 0) < 0.000001)) == 2);
                if (zeroVar.n_elem > 0){
                        xNew(zeroVar).fill(-300);
                        inputs.constPar(zeroVar).fill(2);
                }
                // Checking for boundaries in trend damping
                if (inputs.nPar(0) > 2 && inputs.constPar(0) == 0){    // DT trend
                        if (xNew(0) > 20){
                                xNew(0) = 300;
                                inputs.constPar(0) = 3;
                        }
                        if (xNew(0) < -4){
                                xNew(0) = -300;
                                inputs.constPar(0) = 3;
                        }
                }
                // Changing concentrated-out parameter (code 1)
                if (cLlik){
                        variances = exp(2 * xUncon) % (inputs.typePar == 0); //  % (inputs.constPar == 0);
                        if (inputs.nPar(0) == 3){      // DT trend
                                variances(0) = -300;
                        }
                        largestVar = (variances).index_max();
                        maxVar = variances(largestVar);
                        if (concentratedOutPar(0) != largestVar(0)){
                                inputs.constPar(concentratedOutPar).fill(0);
                                concentratedOutPar = largestVar;
                                inputs.constPar(concentratedOutPar).fill(1);
                                newVar = exp(2 * xNew(concentratedOutPar));
                                xNew(allVar) = log(exp(2 * xNew(allVar)) / newVar(0)) / 2;
                                xNew(find(inputs.constPar == 2)).fill(-300);
                                diagHess = true;
                        }
                }
                if (inputs.pureARMA){
                        gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
                } else {
                        gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
                }
                // Correcting gradient for constrained parameters
                if (cLlik){
                        gradNew(find(inputs.constPar)).fill(0);
                }
                nOverallFuns += nFuns;
                // Verbose
                if (verbosef){
                        Rprintf("%5.0i %5.0i %12.5f %12.5f\n", nIter, nOverallFuns, objNew, alpha_i);
                }
                // Stop Criteria
                flag = stopCriteria(crit, max(abs(gradNew)), objOld - objNew, 1e5, nIter, nOverallFuns);
                // Plateau exit: on ill-conditioned BC objectives
                // (e.g. fixed lambda=2 on raw-scale y ~ 1e6), Armijo's
                // linear-extrapolation target |dir| = beta*|grad.d| is
                // huge (~1e6) while the achievable per-step decrease
                // is small (~1e-4): every line search backtracks to its
                // 1e-5 alpha floor and BFGS limps for 500+ iters making
                // dobj ~ 3e-4 progress, ending only when the 10000 fun
                // eval cap fires.  Detect this and exit cleanly: when
                // the relative dobj is tiny AND alpha is at the floor
                // for K=5 consecutive iters, we are on a plateau the
                // line search cannot escape -- the IC ranking between
                // candidates is unaffected by the residual 0.1-unit
                // grind, and the fast paths (iter <= 2 via flag=6 /
                // flag=2) never accumulate enough streak to trigger.
                if (!flag){
                        double dobj_rel = std::abs(objOld - objNew) /
                                          std::max(std::abs(objNew), 1.0);
                        bool alphaAtFloor = alpha_i <= 2e-5;
                        if (dobj_rel < 1e-3 && alphaAtFloor){
                                if (++plateauStreak >= 5) flag = 2;
                        } else {
                                plateauStreak = 0;
                        }
                }
                // Inverse Hessian BFGS update
                if (!flag){
                        bfgs(iHess, gradNew - gradOld, xNew - xOld, nx, nIter);
                        if (diagHess){
                                diagHess = false;
                                iHess = diagmat(iHess);
                        }
                }
                // Try other initial conditions because non decreasing or nan function
                // Provisions when  problems with optimisation.
                // Guard: only revert when objOld is itself finite.  The
                // in-loop revert (around line 3005) sets objNew = objOld
                // and *then* writes objOld = NaN as a marker that the
                // revert happened (so dobj = objOld - objNew triggers
                // flag = 7 on the next stopCriteria call).  Without the
                // finite check here we'd blindly re-revert objNew to the
                // marker NaN, destroying the genuinely good objNew left
                // by the in-loop revert and propagating NaN out to the
                // caller as objFunValue.
                if (flag > 5 && std::isfinite(objOld)){
                        objNew = objOld;
                        gradNew = gradOld;
                        xNew = xOld;
                }
                if (flag > 105 && newTry < 4){
                        newTry++;
                        flag = 0;
                        if (SSmodel::inputs.verbose){
                                // cout << "    Trying new point..." << endl;
                                Rprintf("    Trying new point...\n");
                        }
                        xNew(allVar) = round(xNew(allVar)); //  % inputs.typePar;
                        if (inputs.nPar(0) > 2){
                                xNew(0) = 2;
                        }
                        if (objNew < objBest){
                                objBest = objNew;
                                xBest = xNew;
                        } else {
                                objNew = objBest;
                                xNew = xBest;
                        }
                        xNew(allVar).fill(-newTry);
                        objNew = objFun(xNew, inputsFake);
                        xUncon = xNew;
                        // xUncon is xNew de-converted to original variances (xNew is ratio of variances)
                        if (cLlik)
                                xUncon(isVar) = log(exp(2 * xNew(isVar)) * SSmodel::inputs.innVariance) / 2;


                        if (inputs.pureARMA){
                                gradNew = gradFun(xNew, inputsFake, objNew, nFuns);
                        } else {
                                gradNew = gradFun(xUncon, inputsFake, objNew, nFuns);
                        }
                        nOverallFuns += nFuns;
                        if (cLlik){
                                gradNew(concentratedOutPar).fill(0);
                        }
                        iHess.eye(nx, nx);
                }
                counter++;
        } while (!flag);
        SSmodel::inputs.flag = flag;
        SSmodel::inputs.Iter = nIter;
        SSmodel::inputs.pTransform = xUncon;
        return flag;
}
// Get states names
string stateNames(BSMmodel sys){
        string namesStates = "Level";
        uword j;
        if (sys.ns(0) > 1)
                namesStates += "/Slope";
        if (sys.ns(1) > 0){
                j = 1;
                for (uword i = 0; i < sys.ns(1); i = i + 2){
                        namesStates += "/Cycle" + to_string(j) +
                                "/Cycle" + to_string(j) + "*";
                        j++;
                }
        }
        if (sys.ns(2) > 0){
                if (sys.seasonal[0] != 'l'){
                        uword nharm = ceil(sys.ns(2) / 2);
                        vec periods = sys.periods.tail_rows(nharm);
                        j = 1;
                        for (uword i = 0; i < sys.ns(2); i = i + 2){
                                namesStates += "/Seasonal" + to_string(j);
                                if (sys.periods(j - 1) != 2)
                                        namesStates += "/Seasonal" + to_string(j) + "*";
                                j++;
                        }
                } else {
                        for (uword i = 0; i < sys.ns(2); i++){
                                namesStates += "/Seasonal" + to_string(i + 1);
                        }
                }
        }
        if (sys.ns(3) > 0){
                for (uword i = 0; i < sys.ns(3); i++){
                        if (i == 0)
                                namesStates += "/Irregular" + to_string(i + 1);
                        else
                                namesStates += "/Irr*";
                }
        }
        if (sys.ns(6) > 0){
                for (uword i = 0; i < sys.TVP.n_elem; i++){
                        namesStates += "/TVP(" + to_string(i + 1) + ")";
                }
        }
        return namesStates;
}
// Count states and parameters of BSM model
void BSMclass::countStates(vec periods, string trend, string cycle, string seasonal, string irregular){
        // string trend, string cycle, string seasonal, string irregular,
        // int nu, vec P, vec rhos, vec& ns, vec& nPar, int& arOrder,
        // int& maOrder, bool& exact
        inputs.ns = zeros(7);
        inputs.nPar = inputs.ns;
        SSmodel::inputs.exact = true;
        if (SSmodel::inputs.augmented){
                SSmodel::inputs.exact = false;
                SSmodel::inputs.cLlik = true;
        }
        // Trend
        if (trend[0] == 'l'){         // LLT trend
                inputs.ns(0) = 2;
                inputs.nPar(0) = 2;
        } else if (trend[0] == 'd' || trend[0] == 's'){  // Damped or Hyndman damped trend
                inputs.ns(0) = 2;
                inputs.nPar(0) = 3;
                SSmodel::inputs.exact = false;
                SSmodel::inputs.cLlik = true;
                SSmodel::inputs.augmented = false;
        } else if(trend[0] == 'r'){   // RW trend
                inputs.ns(0) = 1;
                inputs.nPar(0) = 1;
        } else if(trend[0] == 'i' || trend[0] == 't'){   // IRW trend or trend with drift
                inputs.ns(0) = 2;
                inputs.nPar(0) = 1;
        } else {                      // No trend
                inputs.ns(0) = 1;
                inputs.nPar(0) = 0;
        }
        if (trend[0] == 't')
                inputs.Drift = true;
        else
                inputs.Drift = false;
        // Cycle
        int nCycles = 0;
        if (cycle[0] != 'n'){
                string cycle0 = cycle;
                strReplace("+", "", cycle0);
                strReplace("-", "", cycle0);
                nCycles = cycle.length() - cycle0.length();
                inputs.ns(1) = nCycles * 2;
                inputs.nPar(1) = inputs.ns(1) + sum(periods < 0);
                SSmodel::inputs.exact = false;
                SSmodel::inputs.augmented = false;
        }
        // Seasonal
        int minus;
        int nHarm = periods.n_elem - nCycles;
        if (any(periods == 2))
                minus = 1;
        else
                minus = 0;
        if (seasonal[0] == 'e'){          // All equal
                inputs.ns(2) = nHarm * 2 - minus;
                inputs.nPar(2) = 1;
        } else if (seasonal[0] == 'd'){  // All different
                inputs.ns(2) = nHarm * 2 - minus;
                inputs.nPar(2) = nHarm;
        } else if (seasonal[0] == 'l'){   // Linear
                inputs.ns(2) = inputs.seas - 1;
                inputs.nPar(2) = 1;
        } else {                        // No seasonal
                inputs.ns(2) = 0;
                inputs.nPar(2) = 0;
        }
        // Irregular
        inputs.ar = 0;
        inputs.ma = 0;
        inputs.arDeg = 0;
        inputs.maDeg = 0;
        inputs.arOrders = ivec();
        inputs.maOrders = ivec();
        inputs.armaLags = ivec();
        if (irregular[0] == 'a'){      // ARMA / SARMA
                // Parse "arma(p,q)" or "arma(p,q,P,Q,s)" — count commas to
                // decide which grammar.  stoi-on-substring is used to stay
                // dependency-free.
                int ind1 = irregular.find("(");
                int ind3 = irregular.find(")");
                string body = irregular.substr(ind1 + 1,
                                                ind3 - ind1 - 1);
                // Split on commas.
                vector<int> parts;
                size_t start = 0, pos;
                while ((pos = body.find(',', start)) != string::npos){
                        parts.push_back(stoi(body.substr(start, pos - start)));
                        start = pos + 1;
                }
                parts.push_back(stoi(body.substr(start)));
                int p_local, q_local, P_local = 0, Q_local = 0, s_local = 1;
                if (parts.size() == 2){
                        p_local = parts[0]; q_local = parts[1];
                } else if (parts.size() == 5){
                        p_local = parts[0]; q_local = parts[1];
                        P_local = parts[2]; Q_local = parts[3];
                        s_local = parts[4];
                } else {
                        // Bad encoding — fall back to non-seasonal arma(0,0)
                        p_local = 0; q_local = 0;
                }
                inputs.arOrders = ivec(P_local > 0 || Q_local > 0 ? 2 : 1);
                inputs.maOrders = ivec(inputs.arOrders.n_elem);
                inputs.armaLags = ivec(inputs.arOrders.n_elem);
                inputs.arOrders(0) = p_local; inputs.maOrders(0) = q_local;
                inputs.armaLags(0) = 1;
                if (inputs.arOrders.n_elem == 2){
                        inputs.arOrders(1) = P_local;
                        inputs.maOrders(1) = Q_local;
                        inputs.armaLags(1) = s_local;
                }
                // ar / ma — free coef counts (BFGS-visible).
                inputs.ar = p_local + P_local;
                inputs.ma = q_local + Q_local;
                // arDeg / maDeg — expanded polynomial degrees (state-block).
                inputs.arDeg = p_local + P_local * s_local;
                inputs.maDeg = q_local + Q_local * s_local;
                if (inputs.ar == 0 && inputs.ma == 0){   // Just noise
                        inputs.ns(3) = 0;
                        inputs.nPar(3) = 1;
                } else {                      // ARMA
                        inputs.ns(3) = max(inputs.arDeg, inputs.maDeg + 1);
                        inputs.nPar(3) = inputs.ar + inputs.ma + 1;
                        SSmodel::inputs.exact = false;
                        SSmodel::inputs.augmented = false;
                }
        } else if (irregular[0] == 'n'){
                inputs.ns(3) = 0;
                inputs.nPar(3) = 0;
        }
        // inputs
        int nu = SSmodel::inputs.u.n_rows;
        if (nu > 0){
                SSmodel::inputs.exact = false;
                SSmodel::inputs.cLlik = true;
                if (sum(inputs.TVP) > 0){
                        SSmodel::inputs.augmented = false;
                        inputs.ns(6) = nu;
                        ///////**********
                        vec uniqueTVP = unique(inputs.TVP);
                        inputs.nPar(6) = sum(uniqueTVP > 0);
                        // inputs.nPar(6) = sum(inputs.TVP > 0);
                } else {
                        SSmodel::inputs.augmented = true;
                }
        }
        // Checking pureARMA model without inputs or with just constant
        inputs.pureARMA = false;
        if (trend[0] == 'n' && cycle[0] == 'n' && seasonal[0] == 'n' && irregular[0] == 'a' && SSmodel::inputs.u.n_rows == 0
                    && (inputs.ar > 0 || inputs.ma > 0)){
                inputs.pureARMA = true;
                SSmodel::inputs.augmented = false;
                inputs.nPar(5) = 1;
        }
}
// Fix matrices in standard BSM models (all except variances)
void BSMclass::initMatricesBsm(vec periods, vec rhos, string trend, string cycle, string seasonal, string irregular){
        uword nsCol, nsColTVP;
        countStates(periods, trend, cycle, seasonal, irregular);
        // Initializing system matrices
        int nsAll = sum(inputs.ns);
        SSmodel::inputs.system.T.eye(nsAll, nsAll);
        nsCol = sum(inputs.ns(arma::span(0, 2))) + 1;
        //////*****
        uword nTVP = sum(inputs.TVP > 0);
        vec uniqueTVP;
        if (nTVP > 0) {
            uniqueTVP = unique(inputs.TVP);
            nsColTVP = nsCol + sum(inputs.TVP == 0) + sum(uniqueTVP > 0) - 1;
        } else {
            nsColTVP = nsCol + sum(inputs.TVP == 0) + sum(uniqueTVP > 0);
        }
        // nsColTVP = nsCol + inputs.ns(6);
        //SSmodel::inputs.system.R.eye(nsAll, nsCol);
        SSmodel::inputs.system.R.resize(nsAll, nsColTVP);
        SSmodel::inputs.system.R.submat(0, 0, nsAll - 1, nsCol - 1) = eye(nsAll, nsCol);
        /////////****************
        if (nTVP > 0){
            uword nu = SSmodel::inputs.u.n_rows;
            mat R(nu, nsColTVP - nsCol + 1, fill::zeros);
            uword r = 0;
            uvec ind;
            for (uword i = 0; i < R.n_cols; i++) {
                if (inputs.TVP(i) == 0.0) {
                    R(r, i) = 1.0;
                    r++;
                } else {
                    ind = find(inputs.TVP == inputs.TVP(r));
                    R.submat(ind, uvec({i})).fill(1.0);
                    r += ind.n_elem;
                }
            }
            // R.print("R 2809 bsmmodelTVPconstrain.h");
            // SSmodel::inputs.system.R.print("R final 2821");
            SSmodel::inputs.system.R.submat(nsAll - nu, nsCol - 1, nsAll - 1, nsColTVP - 1) = R;
            // SSmodel::inputs.system.R.print("R final 2823");
            // SSmodel::inputs.system.R.submat(nsAll - nu, nsColTVP - nu, nsAll - 1, nsColTVP - 1) = eye(nu, uniqueTVP.n_elem);
        }
        SSmodel::inputs.system.Q.zeros(nsColTVP, nsColTVP);
        SSmodel::inputs.system.Gam = SSmodel::inputs.system.D = SSmodel::inputs.system.S = 0.0;
        SSmodel::inputs.system.Z.zeros(1, nsAll);
        SSmodel::inputs.system.C.ones(1, 1);
        SSmodel::inputs.system.H.zeros(1, 1);
        // Trends
        if (inputs.ns(0) > 0){
                trend2ss(inputs.ns(0), &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
        }
        // Cycles
        uvec aux;
        if (inputs.ns(1) > 0){
                aux = find(rhos < 0);
                bsm2ss(inputs.ns(0), inputs.ns(1), abs(periods(aux)), abs(rhos(aux)),
                       &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
        }
        // Seasonal
        if (inputs.ns(2) > 0){
                if (seasonal[0] != 'l'){
                        aux = find(rhos > 0);
                        bsm2ss(inputs.ns(0) + inputs.ns(1), inputs.ns(2), abs(periods(aux)),
                               abs(rhos(aux)), &SSmodel::inputs.system.T, &SSmodel::inputs.system.Z);
                } else {
                        // Linear (Harvey dummy) seasonal: ns(2) = seas - 1 states
                        // with constraint gamma_t + gamma_{t-1} + ... + gamma_{t-(s-2)} = 0.
                        //   state[i1]    = gamma_t       (current)
                        //   state[i1+k]  = gamma_{t-k}   (k periods ago)
                        // T's first row is the constraint (-1, -1, ..., -1), the
                        // remaining rows shift past values down (sub-diagonal id).
                        // Z must therefore pick state[i1] (current) -- not the last
                        // state, which would index gamma_{t-(s-2)} and produce a
                        // (s-2)-period seasonal phase shift in the observation.
                        uword i1 = inputs.ns(0) + inputs.ns(1), i2 = i1 + inputs.ns(2) - 1;
                        SSmodel::inputs.system.Z(i1) = 1.0;
                        SSmodel::inputs.system.T(arma::span(i1 + 1, i2), arma::span(i1, i2)) = eye(inputs.seas - 2, inputs.seas - 1);
                        SSmodel::inputs.system.T(arma::span(i1, i1), arma::span(i1, i2)).fill(-1.0);
                }
        }
        // Irregular as ARMA
        if (inputs.ar > 0 || inputs.ma > 0){   // ARMA
                SSmodel::inputs.system.C.zeros(1, 1);
                SSmodel::inputs.system.Z.col(nsCol - 1) = 1.0;
        }
        // Inputs in case of pure regression
        if (SSmodel::inputs.u.n_elem > 0 && sum(inputs.nPar.rows(0, 3)) == 1
            && inputs.nPar(0) == 0 && !inputs.pureARMA){
                SSmodel::inputs.system.T(0, 0) = 0;
        }
        // Inputs as TVP
        if (sum(inputs.TVP) > 0){
                SSmodel::inputs.system.Z = repmat(SSmodel::inputs.system.Z, SSmodel::inputs.u.n_cols, 1);
                SSmodel::inputs.system.Z.cols(nsAll - SSmodel::inputs.u.n_rows, nsAll -1) = SSmodel::inputs.u.t();
        }
        // Pure ARMA
        if (inputs.pureARMA){
                // SSmodel::inputs.system.D = SSmodel::inputs.u;
                SSmodel::inputs.system.T(0, 0) = 0;
        }
}
// Initializing parameters of BSM model
// initParBsm: build the initial parameter vector (SSmodel::inputs.p0) and
// the parameter-type tags (inputs.typePar) that the optimiser needs.
//
// The function is long (~785 lines) because it covers every model family
// in one place; the section markers below group code by what kind of
// parameter is being initialised:
//
//   1. Bookkeeping        - detect whether the user supplied p0, size the
//                           vector, reset typePar.
//   2. Trend params       - RW / LLT / DT / IRW / drift initial values.
//   3. Cycle params       - rhos, periods, variances.
//   4. ARMA params        - initial conditions + invertibility / stationarity
//                           guards if the user supplied them.
//   5. Inputs / TVP       - external regressor and time-varying-parameter
//                           starting values.
//   6. Pure-ARMA const    - constant term for pureARMA models.
//   7. Concentrated var   - pick which variance is concentrated out.
//   8. User-p0 conversion - if the user supplied natural-scale values,
//                           transform them into the optimiser's space.
//
// Extracting these sections into separate methods is a desirable future
// refactor but is non-trivial because they thread shared local state
// (p0, aux, aux1, periods, ...) and a covariance-style integration test
// would be needed first.
void BSMclass::initParBsm(){
        // ---- Section 1: bookkeeping ----------------------------------
        int nTrue = sum(inputs.nPar);
        bool userP0 = true;
        uvec indNaN = find_nonfinite(SSmodel::inputs.p0);
        if ((SSmodel::inputs.p0(0) == -9999.9) || (indNaN.n_elem == SSmodel::inputs.p0.n_elem)){
                userP0 = false;
        }
        SSmodel::inputs.p0.resize(nTrue);
        uvec aux, aux1;
        vec p0 = SSmodel::inputs.p0;
        inputs.typePar = zeros(nTrue);
        // ---- Section 2: trend params ---------------------------------
        SSmodel::inputs.p0.fill(-1.15);
        if (inputs.nPar(0) == 3){           // DT trend
                SSmodel::inputs.p0(0) = 2;                 // alpha
                SSmodel::inputs.p0(2) = -1.5;              // slope
                inputs.typePar(0) = -1;
        } else if (inputs.nPar(0) == 2){    // LLT trend
                SSmodel::inputs.p0(1) = -1.5;              // slope
        } else if (inputs.nPar(0) == 1 && inputs.ns(0) > 1){   // IRW or trend with drift
                if (inputs.Drift)                          // level of trend with drift
                        SSmodel::inputs.p0(0) = -1.15;
                else
                        SSmodel::inputs.p0(0) = -1.5;              // slope
        }
        // ---- Section 3: cycle params (rhos, periods, variances) -----
        if (inputs.nPar(1) > 0){
                // Cycle inputs.rhos
                int nRhos = sum(inputs.rhos < 0), pos;
                pos = inputs.nPar(0) + nRhos;
                aux = regspace<uvec>(inputs.nPar(0), 1, pos - 1);
                SSmodel::inputs.p0(aux).fill(2);
                inputs.typePar(aux).fill(1);
                // Cycle inputs.periods
                aux1 = find(inputs.periods < 0);
                int nPer = aux1.n_elem;
                aux = regspace<uvec>(pos, 1, pos + nPer - 1);
                SSmodel::inputs.p0(aux) = -inputs.periods(aux1);
                vec aaa = SSmodel::inputs.p0(aux);
                unconstrain(aaa, inputs.cycleLimits.rows(aux1));
                SSmodel::inputs.p0(aux) = aaa;
                inputs.typePar(aux).fill(2);   // Marking inputs.periods in overall parameter vector
                // Cycle variances
                pos += nRhos;
                aux = regspace<uvec>(pos, 1, inputs.nPar(0) + inputs.nPar(1) - 1);
        }
        // ---- Section 4: ARMA params (incl. invertibility checks) ----
        vec periods = inputs.periods(find(inputs.rhos > 0));
        if (periods.n_rows == 0){
                aux = 0.0;
        } else {
                aux = max(periods);
        }
        vec stdBeta, e, orders(2), betaHR;
        //double BIC, AIC, AICc;
        // Estimating initial conditions for ARMA from innovations
        if (inputs.nPar(3) > 1){
                uword ini = sum(inputs.nPar(arma::span(0, 2))) + 1;
                aux = regspace<uvec>(ini, 1, sum(inputs.nPar.rows(0, 3)) - 1);
                inputs.typePar(aux).fill(3);
                orders(0) = inputs.ar;
                orders(1) = inputs.ma; //nPar(3) - inputs.ar - 1;
                // ACF/PACF initial conditions for ARMA, laid out block-by-
                // block matching how armaMatrices() reads its BFGS slice.
                // Per-lag layout (arOrders / maOrders / armaLags already set
                // by setModel()):
                //   block 0 (lag 1)   : AR — PACF[1..p],     MA — ACF[1..q]
                //   block i (lag L_i) : AR — PACF[L_i, 2L_i,...]
                //                       MA — ACF [L_i, 2L_i,...]
                // MA is sign-flipped — in muse's (1 + θ·B) convention the
                // empirical ACF (positive for positive autocorrelation) maps
                // to a negative θ via -ACF, breaking the (φ_i, θ_i)
                // cancellation symmetry the optimiser would otherwise drift
                // into.  A final tie-breaker forces AR_i ≠ -MA_i so the
                // BFGS isn't dropped on the alternative cancellation
                // manifold either.  Fallbacks 0.1 / -0.1 (adam's defaults
                // at smooth/R/adam.R:1541) when the empirical estimate is
                // non-finite, clamped to [-0.85, 0.85] to stay clear of
                // the polyStationary boundary at ±0.98.
                inputs.beta0ARMA.zeros(inputs.ar + inputs.ma);
                {
                    arma::ivec arOrders = inputs.arOrders;
                    arma::ivec maOrders = inputs.maOrders;
                    arma::ivec armaLags = inputs.armaLags;
                    if (arOrders.n_elem == 0){
                        arOrders = arma::ivec(1); arOrders(0) = inputs.ar;
                        armaLags = arma::ivec(1); armaLags(0) = 1;
                    }
                    if (maOrders.n_elem == 0){
                        maOrders = arma::ivec(1); maOrders(0) = inputs.ma;
                        if (armaLags.n_elem == 0){
                            armaLags = arma::ivec(1); armaLags(0) = 1;
                        }
                    }
                    int maxLag = 0;
                    for (uword b = 0; b < arOrders.n_elem; ++b)
                        maxLag = std::max(maxLag,
                                          static_cast<int>(arOrders(b) * armaLags(b)));
                    for (uword b = 0; b < maOrders.n_elem; ++b)
                        maxLag = std::max(maxLag,
                                          static_cast<int>(maOrders(b) * armaLags(b)));
                    arma::vec pacf = (maxLag > 0)
                        ? sampleYWpacf(SSmodel::inputs.y, maxLag)
                        : arma::vec();
                    arma::vec acf  = (maxLag > 0)
                        ? sampleACF (SSmodel::inputs.y, maxLag)
                        : arma::vec();
                    auto clamp = [](double v, double fallback){
                        if (!std::isfinite(v)) return fallback;
                        if (v >  0.85) return  0.85;
                        if (v < -0.85) return -0.85;
                        return v;
                    };
                    // AR blocks.
                    uword pos = 0;
                    for (uword b = 0; b < arOrders.n_elem; ++b){
                        int pi = arOrders(b);
                        int Li = armaLags(b);
                        for (int j = 0; j < pi; ++j){
                            int lagIdx = (j + 1) * Li;
                            double seed = (lagIdx <= (int)pacf.n_elem)
                                ? pacf(lagIdx - 1) : 0.1;
                            inputs.beta0ARMA(pos + j) = clamp(seed, 0.1);
                        }
                        pos += pi;
                    }
                    uword maStart = pos;
                    // MA blocks — empirical ACF directly (matches muse's
                    // (1 + θ·B) convention: a negative-ACF series wants
                    // negative θ).
                    for (uword b = 0; b < maOrders.n_elem; ++b){
                        int qi = maOrders(b);
                        int Li = armaLags(b);
                        for (int j = 0; j < qi; ++j){
                            int lagIdx = (j + 1) * Li;
                            double seed = (lagIdx <= (int)acf.n_elem)
                                ? acf(lagIdx - 1) : -0.1;
                            inputs.beta0ARMA(pos + j) = clamp(seed, -0.1);
                        }
                        pos += qi;
                    }
                    // Tie-breaker: PACF and ACF on the same series often
                    // agree numerically — when AR_i ≈ MA_i the BFGS sits
                    // on the (φ_i − θ_i) cancellation manifold and drifts
                    // there.  When the leading pair of each block collides
                    // by < 0.1, push MA down by 0.2 (or up if MA is near
                    // the upper clamp) to land in an asymmetric starting
                    // basin.
                    uword apos = 0, mpos = maStart;
                    for (uword b = 0; b < arOrders.n_elem
                                       && b < maOrders.n_elem; ++b){
                        if (arOrders(b) > 0 && maOrders(b) > 0){
                            double a = inputs.beta0ARMA(apos);
                            double m = inputs.beta0ARMA(mpos);
                            if (std::abs(a - m) < 0.1){
                                double offset = (m > 0.0) ? -0.2 : 0.2;
                                inputs.beta0ARMA(mpos) =
                                    clamp(m + offset, -0.1);
                            }
                        }
                        apos += arOrders(b);
                        mpos += maOrders(b);
                    }
                }
                vec beta0aux, beta0aux1;
                uvec ind;
                vec armas = p0(ind);
                // Testing for unit roots in ARMA parameters chosen by user
                if (userP0 && !armas.has_nan()){
                        ind = find(inputs.typePar == 3);
                        inputs.beta0ARMA = p0(ind);
                        vec absRoots, uno = {1};
                        if (inputs.ar > 0){
                                // AR model
                                // Checking for non-stationary polynomial
                                beta0aux = inputs.beta0ARMA(arma::span(0, inputs.ar - 1));
                                beta0aux1 = -beta0aux;
                                absRoots = abs(roots(join_vert(uno, -beta0aux1)));
                                if (any(absRoots >= 1)){
                                        myError("\n\nERROR: Non-stationary model for AR initial conditions!!!\n");
                                }
                        }
                        if (inputs.ma > 0){
                                // MA model
                                // Checking for non-invertible polynomial
                                beta0aux = inputs.beta0ARMA(arma::span(inputs.ar, inputs.ar + inputs.ma - 1));
                                // Bringing MA polynomial to invertibility
                                absRoots = abs(roots(join_vert(uno, beta0aux)));
                                if (any(absRoots >= 1)){
                                        myError("\n\nERROR: Non-invertible model for MA initial conditions!!!\n");
                                }
                        }
                }
                // Converting to estimation space
                // AR pars
                if (inputs.ar > 0){
                        beta0aux = inputs.beta0ARMA(arma::span(0, inputs.ar - 1));
                        beta0aux1 = -beta0aux;
                        // Correction for non-stationary polynomial
                        arToPacf(beta0aux1);
                        ind = find(abs(beta0aux1) >= 1);
                        if (ind.n_elem > 0){
                                beta0aux1(ind) = sign(beta0aux1(ind)) * 0.96;
                                pacfToAr(beta0aux1);
                                beta0aux = -beta0aux1;
                        }
                        invPolyStationary(beta0aux);
                        beta0aux.elem(find_nonfinite(beta0aux)).zeros();
                        aux = regspace<uvec>(ini, 1, ini + inputs.ar - 1);
                        SSmodel::inputs.p0(aux) = beta0aux;
                }
                // MA pars
                if (inputs.ma > 0){
                        beta0aux = inputs.beta0ARMA(arma::span(inputs.ar, inputs.ar + inputs.ma - 1));
                        // Bringing MA polynomial to invertibility
                        maInvert(beta0aux);
                        inputs.beta0ARMA(arma::span(inputs.ar, inputs.ar + inputs.ma - 1)) = beta0aux;
                        // Parameterising polynomial to be invertible
                        invPolyStationary(beta0aux);
                        aux = regspace<uvec>(ini + inputs.ar, 1, ini + inputs.ar + inputs.ma - 1);
                        SSmodel::inputs.p0(aux) = beta0aux;
                }
        }
        // ---- Section 5: inputs / TVP starting values -----------------
        int nu = SSmodel::inputs.u.n_rows;
        if (sum(inputs.TVP) > 0){
            //////////**************
            vec uniqueTVP = unique(inputs.TVP);
            uword ini = sum(inputs.nPar.rows(0, 5));
            aux = regspace<uvec>(ini, ini + sum(uniqueTVP > 0.0) - 1);
            SSmodel::inputs.p0(aux).fill(-1.5);
        } else {
            int ns = SSmodel::inputs.betaAug.n_rows;
            if (nu > 0 && ns > 1){
                SSmodel::inputs.system.D = SSmodel::inputs.betaAug.rows(ns - nu, ns - 1);
            }
        }
        // ---- Section 6: pure-ARMA constant term ----------------------
        if (inputs.pureARMA){
            // setting value for constant
            SSmodel::inputs.system.D = nanMean(SSmodel::inputs.y);
            SSmodel::inputs.p0(nTrue - 1) = SSmodel::inputs.system.D(0, 0);
            inputs.typePar(nTrue - 1) = 5;
        }
        // ---- Section 7: choose concentrated-out variance ------------
        inputs.constPar = zeros(nTrue);
        if (SSmodel::inputs.cLlik){
                if (inputs.nPar(3) > 0){         // arma(0,0) or arma(p,q)
                        inputs.constPar(inputs.nPar(0) + inputs.nPar(1) + inputs.nPar(2)) = 1;
                } else if (inputs.nPar(3) == 0){   // no irregular component
                        uvec minIndex = find(inputs.typePar == 0);
                        inputs.constPar(minIndex(0)) = 1;
                }
                SSmodel::inputs.p0(find(inputs.constPar)).fill(0);
        }
        // ---- Section 8: convert user-supplied natural-scale p0 ------
        if (userP0){
                // type of parameter (0: variance;
                //        -1: damped of trend;
                //         1: cycle rhos;
                //         2: cycle periods;
                //         3: ARMA;
                // Converting user initial parameters to UComp initial
                inputs.p0Return = p0;
                // variances
                // concentrated out variance
                vec conc = p0(find(inputs.constPar));
                if (conc.has_nan()){
                        conc = 1;
                }
                if (conc(0) < 1e-6){
                        myError("\n\nERROR: Cannot select such small value for concentrated out variance!!!\n");
                }
                uvec ind2 = find(inputs.typePar == 0);
                vec variances = p0(ind2) / conc(0);
                if (any(variances < 0)){
                        myError("\n\nERROR: Initial conditions for variances must be non-negative!!!\n");
                }
                variances(find(variances == 0)).fill(1e-70);
                variances = log(variances) / 2;
                uvec ind = find_nonfinite(variances);
                if (ind.n_elem > 0){
                        // Replacing nan value selected by computer
                        variances(ind) = exp(2 * SSmodel::inputs.p0(ind)) * conc;
                }
                SSmodel::inputs.p0(ind2) = variances;
                // Rhos
                ind = join_vert(find(inputs.typePar == -1), find(inputs.typePar == 1));
                vec pp;
                if (ind.n_elem > 0){
                        pp = p0(ind);
                        if (any(pp > 1) || any(pp < 0)){
                                myError("\n\nERROR: Initial conditions for damping parameters must be between 0 and 1!!!\n");
                        }
                        vec lim1(pp.n_elem); lim1.fill(0);
                        vec lim2(pp.n_elem); lim2.fill(1);
                        mat limit = join_rows(lim1, lim2);
                        unconstrain(pp, limit);
                        ind2 = find_nonfinite(pp);
                        if (ind2.n_elem > 0){
                                pp(ind2) = SSmodel::inputs.p0(ind(ind2));
                        }
                        SSmodel::inputs.p0(ind) = pp;
                }
        } else {
                // type of parameter (0: variance;
                //        -1: damped of trend;
                //         1: cycle rhos;
                //         2: cycle periods;
                //         3: ARMA;
                // Converting initial parameters to user understandable
                inputs.p0Return = SSmodel::inputs.p0;
                // concentrated out variance
                inputs.p0Return.rows(find(inputs.constPar)).fill(0);
                // variances
                uvec ind = find(inputs.typePar == 0);
                inputs.p0Return(ind) = exp(2 * SSmodel::inputs.p0(ind));
                // Rhos
                ind = join_vert(find(inputs.typePar == -1), find(inputs.typePar == 1));
                vec pp;
                if (ind.n_elem > 0){
                        pp = SSmodel::inputs.p0(ind);
                        constrain(pp, regspace<vec>(0, 1));
                        inputs.p0Return(ind) = pp;
                }
                // cycle periods
                uvec aux = find(inputs.periods < 0);
                ind = find(inputs.typePar == 2);
                if (ind.n_elem > 0){
                        pp = SSmodel::inputs.p0(ind);
                        constrain(pp, inputs.cycleLimits.rows(aux));
                        inputs.p0Return(ind) = pp;
                }
                // ARMA
                ind = find(inputs.typePar == 3);
                if (ind.n_elem > 0){
                        inputs.p0Return(ind) = inputs.beta0ARMA;
                }
        }
        // Section 9: Lambda parameter (joint estimation).
        // When SSmodel::inputs.estimateLambda is true, lambda is appended as
        // the last element of p (typePar=6 keeps it out of the concentrated-
        // variance machinery; constPar=0 means it is a free parameter).
        // inputs.lambda holds the warm-start value set by estimUCs or musecore.h.
        if (SSmodel::inputs.estimateLambda){
                SSmodel::inputs.p0 = join_vert(SSmodel::inputs.p0, vec({inputs.lambda}));
                inputs.typePar = join_vert(inputs.typePar, vec({6.0}));
                inputs.constPar = join_vert(inputs.constPar, vec({0.0}));
        }
}
/*************************************************************
 //  * Implementation of auxiliar functions
 //  ************************************************************/
 // Variance matrices in standard BSM on top of fixed structure
 void bsmMatrices(vec p, SSmatrix* model, void* userInputs){
         BSMmodel* inp = (BSMmodel*)userInputs;
         // Floor the variance log-parameters so exp(2*p) cannot underflow to
         // EXACTLY zero.  A zero disturbance variance collapses the Kalman
         // innovation variance F_t = Z P Z' + H to 0, so the gain K = P Z'/F_t
         // becomes 0/0 = NaN and the filtered states -- and any forecast built
         // from the terminal state -- blow up (observed as 1e12-scale forecasts
         // when a flexible model's variances are driven to ~0).  typePar == 0
         // marks the variance log-params; clamp to var >= exp(-23) ~ 1e-10,
         // numerically "zero" for the model but keeping the filter well-defined.
         // Structural zeros (companion states) are not touched.
         if (inp->typePar.n_elem > 0){
                 uword nv = std::min(inp->typePar.n_elem, p.n_elem);
                 uvec vpos = find(inp->typePar.head(nv) == 0);
                 if (vpos.n_elem > 0) p(vpos) = arma::clamp(p(vpos), -11.5, arma::datum::inf);
         }
         // Lambda is no longer in p; no trailing element to shed.
         vec nsCum = cumsum(inp->ns);
         vec nparCum = cumsum(inp->nPar);
         // Trend
         // if (inp->nPar(0) != 0)
         //         model->T(0, 0) = 1.0;
         if (inp->nPar(0) == 2 && inp->ns(0) == 2){        // LLT
                 model->Q(0, 0) = exp(2 * p(0));
                 model->Q(1, 1) = exp(2 * p(1));
         } else if (inp->nPar(0) == 1 && inp->ns(0) == 1){  // RW trend
                 model->Q(0, 0) = exp(2 * p(0));
         } else if (inp->nPar(0) == 3){                     // Damped trend
                 constrain(p(0), regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
                 model->Q(0, 0) = exp(2 * p(1));
                 model->Q(1, 1) = exp(2 * p(2));
                 model->T(1, 1) = p(0);
                 if (inp->trend[0] == 's' && inp->MSOE){
                         model->T(0, 1) = p(0);
                         model->T(nsCum(nsCum.n_elem - 1), 1) = p(0);
                 }
         } else if (inp->nPar(0) == 1 && inp->ns(0) == 2){    // IRW or trend with drift
                 if (inp->Drift)
                         model->Q(0, 0) = exp(2 * p(0));   // Trend with drift
                 else
                         model->Q(1, 1) = exp(2 * p(0));   // IRW
         } else if (inp->nPar(0) == 0 && inp->ns(0) == 1){  // No trend
                 model->Q(0, 0) = 0;
         }
         // Cycle
         uvec ind, ind1;
         vec aux, aux1, periods, rhos, variances;
         uword pos;
         if (inp->nPar(1) > 0){
                 // Rhos
                 int nRhos = sum(inp->typePar == 1);
                 pos = inp->nPar(0) + nRhos;
                 ind = regspace<uvec>(inp->nPar(0), 1, pos - 1);
                 vec pInd = p(ind);
                 constrain(pInd, regspace<vec>(0, 1));
                 p(ind) = pInd;
                 rhos = pInd;
                 // Cycle periods
                 int nPerEstim = sum(inp->typePar == 2);
                 periods = inp->periods(arma::span(0, nRhos - 1));
                 if (nPerEstim > 0){
                         ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
                         pos += nPerEstim;
                         ind1 = find(inp->periods < 0);
                         pInd = p(ind);
                         constrain(pInd, inp->cycleLimits.rows(ind1));
                         p(ind) = pInd;
                         periods(ind1) = pInd;
                 }
                 ind1 = regspace<uvec>(pos, 1, nparCum(1) - 1);
                 variances = exp(2 * p(ind1));
                 // Matrices for cycles
                 bsm2ss(inp->ns(0), inp->ns(1), abs(periods), rhos, &model->T, &model->Z);
                 ind = regspace<uvec>(nsCum(0), 1, nsCum(1) - 1);
                 aux = vectorise(repmat(variances.t(), 2, 1));
                 aux1 = aux(arma::span(0, inp->ns(1) - 1));
                 model->Q(ind, ind) = diagmat(aux1);
         }
         // Seasonal
         if (inp->nPar(2) > 0){                               // With seasonal
                 // Index of states for seasonal
                 ind = regspace<uvec>(nsCum(1), 1, nsCum(2) - 1);
                 if (inp->seasonal[0] == 'l')
                         model->Q(ind(0), ind(0)) = exp(2 * p(nparCum(1)));
                 else if (inp->nPar(2) == 1){                         // Equal variances
                         model->Q(ind, ind) = eye(inp->ns(2), inp->ns(2)) *
                                 exp(2 * p(nparCum(1)));
                 } else {                                        // Different variances
                         ind1 = regspace<uvec>(nparCum(1), 1, nparCum(2) - 1);
                         aux = vectorise(repmat(exp(2 * p(ind1)).t(), 2, 1));
                         aux1 = aux(arma::span(0, inp->ns(2) - 1));
                         model->Q(ind, ind) = diagmat(aux1);
                 }
         }
         // Irregular
         if (inp->nPar(3) == 1){              // Irregular no ARMA model
                 model->H = exp(2 * p(nparCum(3) - 1));
         } else if (inp->ar > 0 || inp->ma > 0) {  // ARMA model
                 SSmatrix mARMA;
                 ARMAinputs iARMA;
                 iARMA.ar       = inp->ar;
                 iARMA.ma       = inp->ma;
                 iARMA.arDeg    = inp->arDeg;
                 iARMA.maDeg    = inp->maDeg;
                 iARMA.arOrders = inp->arOrders;
                 iARMA.maOrders = inp->maOrders;
                 iARMA.armaLags = inp->armaLags;
                 int aux4;
                 uvec aux2 = regspace<uvec>(nparCum(2), 1, nparCum(3) - 1);
                 initMatricesArma(inp->arDeg, inp->maDeg, aux4, mARMA);
                 armaMatrices(p(aux2), &mARMA, &iARMA);
                 uvec ind2 = regspace<uvec>(nsCum(2), 1, nsCum(3) - 1);
                 model->T(ind2, ind2) = mARMA.T;
                 uvec nsCol(1); nsCol(0) = nsCum(2); //sum(inp->ns(arma::span(0, 1)));
                 model->R(ind2, nsCol) = mARMA.R;
                 model->Q(nsCol, nsCol) = mARMA.Q;
         }
         if (inp->pureARMA){
                 model->D(0, 0) = p(nparCum(5) - 1);
         }
         // inputs
         //showSS(*model);
         if (sum(inp->TVP) > 0){
            /////////////***********
            vec uniqueTVP = unique(inp->TVP);
            uvec aux1 = nsCum(5) + find(uniqueTVP > 0) - 1;
            uvec aux2 = regspace<uvec>(nparCum(5), 1, nparCum(6) - 1);
            model->Q(aux1, aux1) = diagmat(exp(2 * p(aux2)));
         }
         //showSS(*model);
 }
// Variance matrices in standard BSM on top of fixed structure for true parameters
void bsmMatricesTrue(vec p, SSmatrix* model, void* userInputs){
        BSMmodel* inp = (BSMmodel*)userInputs;
        // Floor the variance parameters at a small POSITIVE value.  NOTE: unlike
        // bsmMatrices (where p holds log-params and Q = exp(2*p)), here p holds
        // the absolute variances directly (Q(i,i) = p(i)), so the floor must be
        // a variance, not a log-param.  This both (a) keeps a collapsed variance
        // from making F_t = Z P Z' + H exactly zero, and (b) guards against a
        // negative variance leaking in (it can, when the concentrated scale
        // innVariance is corrupted by a non-PSD filter state upstream).
        if (inp->typePar.n_elem > 0){
                uword nv = std::min(inp->typePar.n_elem, p.n_elem);
                uvec vpos = find(inp->typePar.head(nv) == 0);
                if (vpos.n_elem > 0) p(vpos) = arma::clamp(p(vpos), std::exp(2.0 * -11.5), arma::datum::inf);
        }
        vec nsCum = cumsum(inp->ns);
        vec nparCum = cumsum(inp->nPar);
        // Trend
        // if (inp->nPar(0) != 0)
        //         model->T(0, 0) = 1.0;
        if (inp->nPar(0) == 2 && inp->ns(0) == 2){        // LLT
                model->Q(0, 0) = p(0);
                model->Q(1, 1) = p(1);
        } else if (inp->nPar(0) == 1 && inp->ns(0) == 1){  // RW trend
                model->Q(0, 0) = p(0);
        } else if (inp->nPar(0) == 3){                     // Damped trend
                //constrain(p(0), regspace<vec>(0, 1)); //exp(p(0)) / (1+ exp(p(0)));
                model->Q(0, 0) = p(1);
                model->Q(1, 1) = p(2);
                model->T(1, 1) = p(0);
                if (inp->trend[0] == 's' && inp->MSOE){
                        model->T(0, 1) = p(0);
                        model->T(nsCum(nsCum.n_elem - 1), 1) = p(0);
                }
        } else if (inp->nPar(0) == 1 && inp->ns(0) == 2){    // IRW or trend with drift
                if (inp->Drift)
                        model->Q(0, 0) = p(0);    // Trend with drift
                else
                        model->Q(1, 1) = p(0);     // IRW
        } else if (inp->nPar(0) == 0 && inp->ns(0) == 1){  // No trend
                model->Q(0, 0) = 0;
        }
        // Cycle
        uvec ind, ind1;
        vec aux, aux1, periods, rhos, variances;
        uword pos;
        if (inp->nPar(1) > 0){
                // Rhos
                int nRhos = sum(inp->typePar == 1);
                pos = inp->nPar(0) + nRhos;
                ind = regspace<uvec>(inp->nPar(0), 1, pos - 1);
                vec pInd = p(ind);
                //constrain(pInd, regspace<vec>(0, 1));
                //p(ind) = pInd;
                rhos = pInd;
                // Cycle periods
                int nPerEstim = sum(inp->typePar == 2);
                periods = inp->periods(arma::span(0, nRhos - 1));
                if (nPerEstim > 0){
                        ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
                        pos += nPerEstim;
                        ind1 = find(inp->periods < 0);
                        pInd = p(ind);
                        //constrain(pInd, inp->cycleLimits.rows(ind1));
                        //p(ind) = pInd;
                        periods(ind1) = pInd;
                }
                ind1 = regspace<uvec>(pos, 1, nparCum(1) - 1);
                variances = p(ind1);
                // Matrices for cycles
                bsm2ss(inp->ns(0), inp->ns(1), abs(periods), rhos, &model->T, &model->Z);
                ind = regspace<uvec>(nsCum(0), 1, nsCum(1) - 1);
                aux = vectorise(repmat(variances.t(), 2, 1));
                aux1 = aux(arma::span(0, inp->ns(1) - 1));
                model->Q(ind, ind) = diagmat(aux1);
        }
        // Seasonal
        if (inp->nPar(2) > 0){                               // With seasonal
                // Index of states for seasonal
                ind = regspace<uvec>(nsCum(1), 1, nsCum(2) - 1);
                if (inp->seasonal[0] == 'l')
                        model->Q(ind(0), ind(0)) = p(nparCum(1));
                else if (inp->nPar(2) == 1){                         // Equal variances
                        model->Q(ind, ind) = eye(inp->ns(2), inp->ns(2)) *
                                p(nparCum(1));
                } else {                                        // Different variances
                        ind1 = regspace<uvec>(nparCum(1), 1, nparCum(2) - 1);
                        aux = vectorise(repmat(p(ind1).t(), 2, 1));
                        aux1 = aux(arma::span(0, inp->ns(2) - 1));
                        model->Q(ind, ind) = diagmat(aux1);
                }
        }
        // Irregular
        if (inp->nPar(3) == 1){              // Irregular no ARMA model
                model->H = p(nparCum(3) - 1);
        } else if (inp->ar > 0 || inp->ma > 0) {  // ARMA model
                SSmatrix mARMA;
                ARMAinputs iARMA;
                iARMA.ar       = inp->ar;
                iARMA.ma       = inp->ma;
                iARMA.arDeg    = inp->arDeg;
                iARMA.maDeg    = inp->maDeg;
                iARMA.arOrders = inp->arOrders;
                iARMA.maOrders = inp->maOrders;
                iARMA.armaLags = inp->armaLags;
                int aux4;
                uvec aux2 = regspace<uvec>(nparCum(2), 1, nparCum(3) - 1);
                initMatricesArma(inp->arDeg, inp->maDeg, aux4, mARMA);
                armaMatricesTrue(p(aux2), &mARMA, &iARMA);
                uvec ind2 = regspace<uvec>(nsCum(2), 1, nsCum(3) - 1);
                model->T(ind2, ind2) = mARMA.T;
                uvec nsCol(1); nsCol(0) = nsCum(2); //sum(inp->ns(arma::span(0, 1)));
                model->R(ind2, nsCol) = mARMA.R;
                model->Q(nsCol, nsCol) = mARMA.Q;
        }
        if (inp->pureARMA){
                model->D(0, 0) = p(nparCum(5) - 1);
        }
        // inputs
        if (sum(inp->TVP) > 0){
            //////////////*************
            vec uniqueTVP = unique(inp->TVP);
            uvec aux1 = nsCum(5) + find(uniqueTVP > 0.0) - 1;
            uvec aux2 = regspace<uvec>(nparCum(5), 1, nparCum(6) - 1);
            model->Q(aux1, aux1) = diagmat(p(aux2));
        }
}
// Remove elements of vector in n adjacent points
uvec selectOutliers(vec& val, int nTogether, float limit){
        int n = val.n_elem - 1,
                indMax;
        uvec indAround;
        bool next = true;
        uvec times;
        do{
                indMax = index_max(val);
                if (val(indMax) > limit){
                        times.resize(times.n_elem + 1);
                        times(times.n_elem - 1) = indMax;
                        val.rows(std::max(0, (int)indMax - nTogether), std::min(n, (int)indMax + nTogether)).fill(0);
                } else {
                        next = false;
                }
        } while (next);
        return times;
}
// Create dummy variable for outliers 0: AO, 1: LS, 2: SC
void dummy(uword indMax, uword typeO, rowvec& u){
        int n = u.n_elem;
        u.fill(0);
        if (typeO == 0){
                u(indMax) = 1.0;
        } else if (typeO == 1){
                u.cols(indMax, n - 1).fill(1.0);
        } else if (typeO == 2){
                u.cols(indMax, n - 1) = regspace(1, n - indMax).t();
        }
}
// Extract trend seasonal and irregular of model in a string
void splitModel(string model, string& trend, string& cycle, string& seasonal, string& irregular){
        int ind1, ind2, ind3;
        string aux1, aux2;

        lower(model);
        deblank(model);
        ind1 = model.find("/");
        aux1 = model.substr(ind1 + 1);
        ind2 = aux1.find("/");
        aux2 = aux1.substr(ind2 + 1);
        ind3 = aux2.find("/");
        trend = model.substr(0, ind1);
        cycle = aux1.substr(0, ind2);
        seasonal = aux2.substr(0, ind3);
        irregular = aux2.substr(ind3 + 1);
}
// SS form of trend models
void trend2ss(int ns, mat* T, mat* Z){
        if (ns > 1){
                (*T)(0, 1) = 1;
        }
        (*Z)(0) = 1;
}
// SS form of seasonal models
void bsm2ss(int ns0, int nsSeas, vec P, vec rhos, mat* T, mat* Z){
        bool minus = 1 - any(P == 2);
        uvec aux1 = regspace<uvec>(0, 2, nsSeas - 1) + ns0;
        (*Z).cols(aux1) = ones(Z->n_rows, aux1.n_elem);
        vec sinf = sin(2 * (datum::pi) / P) % rhos;
        vec cosf = cos(2 * (datum::pi) / P) % rhos;
        vec oneZero(2); oneZero(0) = 1; oneZero(1) = 0;
        vec oneOne(2); oneOne.fill(1);
        vec sines = kron(sinf, oneZero);
        vec cosines = kron(cosf, oneOne);
        int nDiag = nsSeas - 1 - minus;
        uvec aux3 = regspace<uvec>(0, 1, nDiag);
        mat aux = diagmat(cosines) + diagmat(sines(aux3), 1) + diagmat(-sines(aux3), -1);
        uvec aux2 = regspace<uvec>(0, 1, nDiag + minus);
        (*T)(aux2 + ns0, aux2 + ns0) = aux(aux2, aux2);
}
// Combining models for components
// Parse the AR order out of an "arma(...)" candidate string.  Sums the
// non-seasonal `p` and seasonal `P` fields (the 1st and 3rd commaseparated
// entries in arma(p,q[,P,Q,s])); returns 0 for "none" / non-arma tokens.
inline int armaARorder(const string& irr){
    if (irr.size() < 5 || irr.compare(0, 5, "arma(") != 0) return 0;
    size_t open  = irr.find('(');
    size_t close = irr.find(')');
    if (open == string::npos || close == string::npos) return 0;
    string body = irr.substr(open + 1, close - open - 1);
    vector<string> parts;
    chopString(body, ",", parts);
    int p_field = (parts.size() >= 1) ? stoi(parts[0]) : 0;
    int P_field = (parts.size() >= 3) ? stoi(parts[2]) : 0;
    return p_field + P_field;
}

void findUCmodels(string trend, string cycle, string seasonal, string irregular, vector<string>& allModels){
        int nTrendModels, nCycleModels, nSeasonalModels, nIrregularModels;
        vector <string> trendModels, cycleModels, seasonalModels, irrModels;
        // Possible trends
        chopString(trend, "/", trendModels);
        nTrendModels = trendModels.size();
        // Possible cycles
        chopString(cycle, "/", cycleModels);
        nCycleModels = cycleModels.size();
        // Possible seasonals
        chopString(seasonal, "/", seasonalModels);
        nSeasonalModels = seasonalModels.size();
        // Possible irregulars
        chopString(irregular, "/", irrModels);
        nIrregularModels = irrModels.size();
        // All possible models
        // int count = 0;
        string cModel;
        for (int i = 0; i < nTrendModels; i++){
                // Damped-style trends (srw, dt) share their persistence
                // parameter with an AR component — fitting them jointly
                // leads to an unidentified joint optimum.  Skip the
                // specific (damped trend, AR > 0) cells here; MA-only
                // ARMA candidates with the same trend remain allowed.
                bool dampedTrend = (trendModels[i] == "srw" ||
                                    trendModels[i] == "dt");
                for (int l = 0; l < nCycleModels; l++){
                        for (int j = 0; j < nSeasonalModels; j++){
                                for (int k = 0; k < nIrregularModels; k++){
                                        if (trendModels[i] == "none" && cycleModels[l] == "none" && seasonalModels[j] == "none" && irrModels[k] == "none"){
                                                continue;
                                        }
                                        if (dampedTrend && armaARorder(irrModels[k]) > 0)
                                                continue;
                                        cModel = trendModels[i];
                                        cModel.append("/").append(cycleModels[l]).append("/").append(seasonalModels[j]).append("/").append(irrModels[k]);
                                        allModels.push_back(cModel);
                                }
                        }
                }
        }
}
// Find first observation of n non-nan contiguous values
int findFirst(vec y, int n){
        uvec indFinite, poly(n, fill::ones), aux3;
        indFinite = find_finite(y);
        aux3 = conv(diff(indFinite), poly);
        int iniObs = indFinite(min(find(aux3.rows(n - 1, aux3.n_elem - 1) == n)));
        return iniObs;
}
// Translates names from UC to PTS
string UC2PTS(string modelUC, double lambda){
    vector<string> aux;
    chopString(modelUC, "/", aux);
    // noise
    char modelt[2] = "N", models[2] = "N";
    // trend
    if (aux[0] == "dt")
        strcpy(modelt, "F");
    else if (aux[0] == "llt")
        strcpy(modelt, "L");
    else if (aux[0] == "td")
        strcpy(modelt, "G");
    else if (aux[0] == "srw")
        strcpy(modelt, "D");
    // seasonal
    if (aux[2] == "equal")
        strcpy(models, "T");
    else if (aux[2] == "linear")
        strcpy(models, "D");
    else if (aux[2] == "different")
        strcpy(models, "F");
    char model[30];
    snprintf(model, 30, "(%3.1f, %1s, %1s)", lambda, modelt, models);
    return model;
}
// // Translates names from UC to PTS
// string UC2PTS(string modelUC){
//         vector<string> aux;
//         chopString(modelUC, "/", aux);
//         // noise
//         string model = "(A,";
//         if (aux[3] == "none")
//                 model = "(N,";
//         // trend
//         if (aux[0] == "rw" || aux[0] == "none")
//                 model += "N,";
//         else if (aux[0] == "srw")
//                 model += "Ad,";
//         else if (aux[0] == "llt")
//                 model += "A,";
//         else if (aux[0] == "td")
//                 model += "L,";
//         // seasonal
//         if (aux[2] == "none")
//                 model += "N)";
//         else if (aux[2] == "equal")
//                 model += "E)";
//         else if (aux[2] == "different")
//                 model += "D)";
//         else if (aux[2] == "linear")
//                 model += "L)";
//         return model;
// }
// Show SSmodel
void showSS(SSmatrix m){
        Rprintf("*********** SS system start *********\n");
        m.T.print("Matrix T:");
        m.R.print("Matrix R:");
        m.Q.print("Matrix Q:");
        mat RQR = m.R * m.Q * m.R.t();
        RQR.print("Matrix RQR:");
        if (m.Z.n_rows > 10)
                m.Z.rows(0, 9).print("First 10 rows of matrix Z:");
        else
                m.Z.print("Matrix Z:");
        Rprintf("*********** SS system end *********\n");
}
// Show BSMmodel
void showBSM(BSMmodel m){
        Rprintf("*********** BSM model start *********\n");
        Rprintf("model: %s\n", m.model.c_str());
        Rprintf("criterion: %s\n", m.criterion.c_str());
        Rprintf("stepwise: %10i\n", m.stepwise);
        Rprintf("tTest: %10i\n", m.tTest);
        Rprintf("arma: %10i\n", m.arma);
        m.periods.t().print("periods:");
        Rprintf("trend: %s\n", m.trend.c_str());
        Rprintf("seasonal: %s\n", m.seasonal.c_str());
        Rprintf("irregular: %s\n", m.irregular.c_str());
        Rprintf("compNames: %s\n", m.compNames.c_str());
        Rprintf("ar: %10i\n", m.ar);
        Rprintf("ma: %10i\n", m.ma);
        m.rhos.t().print("rhos:");
        m.ns.t().print("ns:");
        m.nPar.t().print("nPar:");
        m.p0Return.t().print("p0Return:");
        m.typePar.t().print("typePar:");
        m.eps.t().print("eps:");
        m.beta0ARMA.t().print("beta0ARMA:");
        m.constPar.t().print("constPar:");
        m.harmonics.t().print("harmonics:");
        m.comp.print("comp:");
        m.typeOutliers.t().print("typeOutliers:");
        m.cycleLimits.print("cycleLimits:");
        Rprintf("pureARMA: %10i\n", m.pureARMA);
        for (uword i = 0; i < m.parNames.size(); i++){
                Rprintf("%s / ", m.parNames[i].c_str());
        }
        Rprintf("\n*********** BSM model end *********");
}
