#include <iostream>
#include <math.h>
#include <string.h>
#include <armadillo>
using namespace arma;
using namespace std;
#include "DJPTtools.h"
#include "optim.h"
#include "stats.h"
// #include "boxcox.h"
#include "SSpaceTV.h"
// #include "ARMAmodel.h"
// #include "ETSmodel.h"

/**************************
 * Model CLASS Continuous Time level model
 ***************************/
struct CTLEVELmodel{
    string obsEq = "stock"; // "stock" or "flow"
    mat delta;
};
class CTLEVELclass : public SSmodel{
public:
    bool errorExit = false;
    string obsEq = "stock"; // "stock" or "flow"
    // SSmodel mSS;
    CTLEVELmodel userInputs;
    // Constructors
    CTLEVELclass(){};
    CTLEVELclass(SSinputs, vec, mat, vec, string, bool, vec);
    // Rest of methods
    // void CTLEVELmatrices(vec, SSmatrix*, void*);
    // void estim(){mSS.estim();};
    // void forecast(){mSS.forecast();};
    // void filter(){mSS.filter();};
    // void smooth(){mSS.smooth(false);};
    // void validate(){mSS.validate(true, 2);};
};
/**************************
 * Functions declarations
 ***************************/
void CTLEVEL(vec, mat, vec, string, bool, vec);
// CTLEVELclass preProcess(vec, mat, vec, string, bool, vec);
void CTLEVELmatrices(vec, SSmatrix*, void*);
/**************************
* Functions implementations
***************************/
// Constructor
CTLEVELclass::CTLEVELclass(SSinputs data, vec y, mat u, vec t, string obsEq,
                           bool verbose, vec p0) : SSmodel(data){
    this->obsEq = obsEq;
    bool errorExit = false;
    // NaNs in u
    if (u.n_rows > 0 && u.has_nonfinite()){
        printf("%s", "ERROR: missing values not allowed in input variables!!!\n");
        this->errorExit = true;
    }
    if (u.n_rows > u.n_cols){
        u = u.t();
    }
    // Checking length of t with respect to u
    if (u.n_cols > 0 && u.n_rows != t.n_rows){
        printf("%s", "ERROR: Time index length and inputs length should be equal!!!\n");
        this->errorExit = true;
    }
    // Checking length of t with respect to y
    if (y.n_rows > t.n_rows){
        printf("%s", "ERROR: Time index length should be equal or greater to output length!!!\n");
        this->errorExit = true;
    }
    // obsEq options
    lower(obsEq);
    if (obsEq[0] != 's' && obsEq[0] != 'f'){
        printf("%s", "ERROR: obsEq input should be either \"stock\" or \"flow\"!!!\n");
        this->errorExit = true;
    }
    if (any(p0 < 0.0)){
        printf("%s", "ERROR: Initial parameter values should be positive!!!\n");
        this->errorExit = true;
    }
    // Initial estimate
    if (p0.n_rows >= 2)
        p0 = log(p0.rows(0, 1)) / 2;
    else if (p0.n_rows == 1){
        vec p1(1, fill::value(-1.15));
        p0 = join_vert(log(p0.row(0)) / 2, p1);
    } else
        p0.resize(2).fill(-1.15);
    int h = t.n_rows - y.n_rows;
    // Setting up system matrices for CTL model
    // SSinputs input = this->mSS.getInputs();
    // SSmatrix sys = input.system;
    this->SSmodel::inputs.exact = false;
    if (u.n_rows > 0){
        this->SSmodel::inputs.augmented = true;
        this->SSmodel::inputs.llikFUN = llikAug;
    } else {
        this->SSmodel::inputs.augmented = false;
        this->SSmodel::inputs.llikFUN = llik;
    }
    // CTLEVELclass mClass(obsEq);
    // mClass.errorExit = errorExit;
    mat delta(t.n_elem, 1);
    delta(0, 0) = 1.0;
    delta.rows(1, t.n_elem - 1) = diff(t);
    SSmatrix sys;
    if (!errorExit){
        if (obsEq[0] == 's'){
            sys.T.set_size(1, 1); sys.T(0, 0) = 1.0;
            sys.R = sys.T;
            sys.Q = delta;
            sys.Z = sys.T;
            sys.C = sys.T;
            sys.H = sys.T;
            sys.D = sys.Gam = sys.S = 0.0;
            // sys.delta = delta;
        } else{
            printf("%s", "Not done yet!!!!!");
            // sys.T.set_size(1, 1); sys.T(0, 0) = 1.0;
            // sys.R = sys.T;
            // sys.Q.set_size(t.n_elem, 1); sys.Q.fill(1.0);
            // sys.Z = sys.T;
            // sys.C = sys.T;
            // sys.H = sys.T;
        }
        errorExit = checkSystem(sys);
        // Creating model object
        this->SSmodel::inputs.h = h;
        this->SSmodel::inputs.y = y;
        this->SSmodel::inputs.u = u;
        this->SSmodel::inputs.verbose = verbose;
        this->SSmodel::inputs.system = sys;
        this->SSmodel::inputs.p0 = p0;
        this->SSmodel::inputs.cLlik = false;   // Concentrated likelihood on/off
        // CTLEVELmodel userInputs;
        this->userInputs.delta.resize(delta.size());
        this->userInputs.delta = delta;
        this->userInputs.obsEq = obsEq;
        this->SSmodel::inputs.userInputs = &this->userInputs;
        this->SSmodel::inputs.userModel = CTLEVELmatrices;
    }
    // Creating class
    // SSmodel mSS = SSmodel(input);
    // mSS.setInputs(input);
}
// Main function
void CTLEVEL(vec y, mat u, vec t, string obsEq, bool verbose, vec p0){
    // Building standard SS model
    SSinputs mSS;
    CTLEVELclass mClass(mSS, y, u, t, obsEq, verbose, p0);
    // mClass = preProcess(y, u, t, obsEq, verbose, p0);
    mClass.estim();
    mClass.filter();
    mClass.validate(true, 2);
}
// matrices del sistema
void CTLEVELmatrices(vec p, SSmatrix* model, void* userInputs){
    // CTLEVELmodel* funInputs = (CTLEVELmodel*)userInputs;
    CTLEVELmodel* funInputs = static_cast<CTLEVELmodel*>(userInputs);
    if (funInputs->obsEq[0] == 's'){
        // model->Q = exp(2 * p(0)) * model->delta;
        model->Q = exp(2 * p(0)) * funInputs->delta;
        model->H = exp(2 * p(1));
    } else{
        printf("%s", "flow variables not done yet!!!");
    }
}
