#include "CTLEVELmodel.h"

/**************************
 * Model CLASS Intermittent model as a Continuous Time level model
 ***************************/
class INTLEVELclass{
public:
    bool errorExit = false, logTransform = true;
    string obsEq = "stock"; // "stock" or "flow"
    mat comp, compV;
    uword nObs;
    string compNames = "Error/Fit/Level";
    uvec tNonZero, thCum;
    // Other classes
    SSinputs mSS;
    CTLEVELmodel userInputs;
    CTLEVELclass m;
    // Constructor
    INTLEVELclass(vec, mat, int, string, bool, vec, bool);
    // Rest of methods
    void estim(){m.estim();};
    void forecast(){m.forecast();};
    // void filter(){m.filter();};
    void smooth();   // Smooth and forecast
    void validate();
};
/**************************
 * Functions declarations
 ***************************/
void INTLEVEL(vec, mat, int, string, bool, vec);
// INTLEVELclass preProcess(vec, mat, int, string, bool, vec);
/**************************
 * Functions implementations
 ***************************/
// Constructor
INTLEVELclass::INTLEVELclass(vec y, mat u, int h, string obsEq, bool verbose,
                             vec p0, bool logTransform){
    bool errorExit = false;
    // Correcting h in case there are inputs
    if (u.n_cols > 0){
        h = u.n_cols - y.n_elem;
        if (h < 0){
            printf("%s", "ERROR: Inputs should be at least as long as the output!!!\n");
            this->errorExit = true;
        }
    }
    uvec t = find(y != 0.0);
    if (logTransform)
        y(t) = log(y(t));
    uvec aux = t.tail_rows(1); uword lastT = aux(0);
    // vec th = join_vert(conv_to<vec>::from(t), regspace<vec>(lastT + 1, lastT + h));
    uvec th = join_vert(t, regspace<uvec>(lastT + 1, lastT + h));
    uvec thCum = join_vert(t, lastT + cumsum(regspace<uvec>(1, h)));
    if (!errorExit){
        CTLEVELclass m(mSS, y(t), u, conv_to<vec>::from(thCum), obsEq, verbose, p0);
        this->m = m;
        this->userInputs = m.userInputs;
        mSS = this->m.getInputs();
        mSS.userInputs = &this->userInputs;
        this->m.setInputs(mSS);
        this->errorExit = m.errorExit;
        this->tNonZero = th;
        // this->th = th;
        this->thCum = thCum;
        this->nObs = y.n_rows + h;
        this->logTransform = logTransform;
    }
    this->obsEq = obsEq;
}
void INTLEVELclass::smooth(){
    m.smooth(false);
    SSinputs mSS = m.SSmodel::getInputs();
    // uword nObs = mSS.y.n_rows;
    // Components
    uword nBefore = mSS.y.n_rows - mSS.v.n_rows;
    mat v(nObs, 1, fill::value(datum::nan)), F(nObs, 1), Ffit = v,
        yFit = v, a = v, P = v;
    v.rows(tNonZero.rows(nBefore, mSS.v.n_rows)) = mSS.v;
    F.rows(tNonZero.rows(nBefore, mSS.v.n_rows)) = mSS.F;
    Ffit = F;
    // Ffit.rows(tNonZero.rows(mSS.y.n_rows, tNonZero.n_rows - 1)) = mSS.FFor;
    yFit.rows(tNonZero) = mSS.yFit;
    a.rows(tNonZero) = mSS.a.row(0).t();
    P.rows(tNonZero) = mSS.P.row(0).t();
    comp = join_horiz(v, yFit, a);
    compV = join_horiz(F, F, P);
}
void INTLEVELclass::validate(){
    char str[45];
    mSS = m.getInputs();
    mSS.table.clear();
    mSS.table.push_back("-------------------------------------\n");
    string MODEL = "Intermittent-continuous time model";
    if (mSS.u.n_rows > 0){
        MODEL += " + inputs";
    }
    if (mSS.system.Z.n_rows > 1){
        MODEL += " + TVP";
    }
    snprintf(str, 45, " %s\n", MODEL.c_str());
    mSS.table.push_back(str);
    if (logTransform)
        snprintf(str, 45, " Logarthm transformation: Yes\n");
    else
        snprintf(str, 45, " Logarthm transformation: No\n");
    mSS.table.push_back(str);
    snprintf(str, 45, " %s", mSS.estimOk.c_str());
    mSS.table.push_back(str);
    mSS.table.push_back("------------------------------------\n");
    mSS.table.push_back("                 Param     |Grad| \n");
    mSS.table.push_back("------------------------------------\n");
    snprintf(str, 45, " Var(eta):  %10.4f %10.6f\n", exp(2 * mSS.p(0)), abs(mSS.grad(0)));
    mSS.table.push_back(str);
    snprintf(str, 45, " Var(eps):  %10.4f %10.6f\n", exp(2 * mSS.p(1)), abs(mSS.grad(1)));
    mSS.table.push_back(str);
    mSS.table.push_back("------------------------------------\n");
    snprintf(str, 45, "  AIC: %8.4f     BIC: %8.4f\n", mSS.criteria(1), mSS.criteria(2));
    mSS.table.push_back(str);
    snprintf(str, 45, " AICc: %8.4f Log-Lik: %8.4f\n", mSS.criteria(3), mSS.criteria(0));
    mSS.table.push_back(str);
    mSS.table.push_back("------------------------------------\n");
    m.setInputs(mSS);
}
// Main function
void INTLEVEL(vec y, mat u, int h, string obsEq, bool verbose, vec p0, bool logTransform){
    // Building standard SS model
    INTLEVELclass mClass(y, u, h, obsEq, verbose, p0, logTransform);
    // mClass = preProcess(y, u, h, obsEq, verbose, p0);
    mClass.estim();
    mClass.forecast();
    // mClass.filter();
    mClass.smooth();
    mClass.validate();
    SSinputs m = mClass.m.SSmodel::getInputs();
    m.yFor.t().print("yfor");
    m.FFor.t().print("FFor");
    SSinputs mSS = mClass.m.getInputs();
    for (unsigned int i = 0; i < mSS.table.size(); i++){
        printf("%s ", mSS.table[i].c_str());
    }
}

