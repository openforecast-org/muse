#include "CTLEVELmodel.h"

/**************************
 * Model CLASS Intermittent model as a Continuous Time level model
 ***************************/
class INTLEVELclass{
public:
    vec y;
    bool errorExit = false, logTransform = true;
    string obsEq = "stock"; // "stock" or "flow"
    mat u, comp, compV;
    uword nObs, h;
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
    void forecast();
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
    // Detecting logs and negative numbers
    if (logTransform && any(y < 0)){
        printf("%s", "ERROR: Data should be positive with log transformation!!!\n");
        this->errorExit = true;
    }
    uvec t = find(y != 0.0);
    this->u = u;
    if (u.n_rows > 0)
        u = u.cols(t);
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
        this->h = h;
        // y for smooth/forecast
        // this->y.resize(nObs);
        // this->y.fill(datum::nan);
        // this->y(t) = y(t);
    }
    this->obsEq = obsEq;
}
// Forecast function
void INTLEVELclass::forecast(){
    // Cumulative predictions   **********************
    // m.forecast();
    // mSS.aEnd and mSS.PEnd already are the one step ahead forecasts
    mSS = m.getInputs();
    uword delta = tNonZero(mSS.y.n_rows - 1) - tNonZero(mSS.y.n_rows - 2);
    vec beta, l = regspace<vec>(delta, h + delta - 1);
    double varEta = exp(2 * mSS.p(0)), varEps = exp(2 * mSS.p(1));
    mat PT;
    mat u;
    if (mSS.u.n_rows > 0){
        u = mSS.u.tail_cols(h);
        beta = mSS.p.rows(2, 1 + u.n_rows);
    }
    if (obsEq[0] == 's'){
        mSS.yFor.resize(h);
        mSS.yFor.fill(mSS.aEnd(0));
        if (u.n_rows > 0)
            mSS.yFor += u.t() * beta;
        PT = (mSS.PEnd(0, 0) - varEps) / delta - varEta;
        mSS.FFor = l * (PT + varEta) + varEps;   // 9.2.17 Harvey
    } else{
        mSS.yFor = mSS.aEnd(1)+ (l - delta) * mSS.aEnd(1) / delta;  // divided by delta_T
        if (u.n_rows > 0)
            mSS.yFor += u.t() * beta;
        PT = (mSS.PEnd(1, 1) - delta * varEps - 1.0 / 3.0  * delta * delta * delta * varEta) / (delta * delta);
        mSS.FFor = pow(l, 2) * PT + 1.0 / 3.0 * pow(l, 3) * varEta + l * varEps;  // 9.3.26 Harvey
    }
    m.setInputs(mSS);
}
// Smoothing function
void INTLEVELclass::smooth(){
    mat v(nObs, 1, fill::value(datum::nan)), F(nObs, 1), Ffit = v,
        yFit = v, a = v, P = v;
    if (true){    // interpolating and forecasting with smoothing algorithm
        SSinputs mSS = m.SSmodel::getInputs();
        vec y(nObs, fill::value(datum::nan));
        y(tNonZero.rows(0, mSS.y.n_rows - 1)) = mSS.y;
        CTLEVELclass mSmooth(mSS, y, u, regspace<vec>(1, y.n_rows), obsEq, this->mSS.verbose,
                             exp(2 * this->mSS.p));
        mSmooth.smooth(false);
        mSS = mSmooth.SSmodel::getInputs();
        v.tail_rows(mSS.v.n_rows) = mSS.v;
        if (obsEq[0] == 's')
            F.tail_rows(mSS.v.n_rows) = mSS.F;
        else
            F.tail_rows(mSS.v.n_rows) = mSS.P(span(1, 1), span(1, mSS.v.n_rows)).t();
        Ffit = F;
        // Ffit.rows(tNonZero.rows(mSS.y.n_rows, tNonZero.n_rows - 1)) = mSS.FFor;
        yFit = mSS.yFit;
        a = mSS.a.row(0).t();
        P = mSS.P.row(0).t();
    } else{
        // Just components without interpolation
        m.smooth(false);
        SSinputs mSS = m.SSmodel::getInputs();
        // uword nObs = mSS.y.n_rows;
        // Components
        // uword nBefore = mSS.y.n_rows - mSS.v.n_rows;
        v.rows(tNonZero.rows(1, mSS.v.n_rows)) = mSS.v;
        if (obsEq[0] == 's')
            F.rows(tNonZero.rows(1, mSS.v.n_rows)) = mSS.F;
        else
            F.rows(tNonZero.rows(1, mSS.v.n_rows)) = mSS.P(span(1, 1), span(1, mSS.v.n_rows)).t();
        Ffit = F;
        // Ffit.rows(tNonZero.rows(mSS.y.n_rows, tNonZero.n_rows - 1)) = mSS.FFor;
        yFit.rows(tNonZero) = mSS.yFit;
        a.rows(tNonZero) = mSS.a.row(0).t();
        P.rows(tNonZero) = mSS.P.row(0).t();
    }
    comp = join_horiz(v, yFit, a);
    compV = join_horiz(F, F, P);
}
// Validation function
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
    if (obsEq[1] == 's')
        snprintf(str, 45, " Stock output\n");
    else
        snprintf(str, 45, " Flow output\n");
    mSS.table.push_back(str);
    if (logTransform)
        snprintf(str, 45, " Log-transformed data\n");
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

