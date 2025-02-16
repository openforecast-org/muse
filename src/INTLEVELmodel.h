#include "CTLEVELmodel.h"

/**************************
 * Model CLASS Intermittent model as a Continuous Time level model
 ***************************/
class INTLEVELclass{
public:
    vec y;
    bool errorExit = false, logTransform = true, cllik = true, aggStock = true;
    string obsEq = "stock"; // "stock" or "flow"
    mat u, comp, compV; //, simul, noiseETA;
    uword nObs, h;
    string compNames = "Error/Fit/Level";
    uvec tNonZero, thCum;
    vec yForAgg, yForVAgg;
    vec coef;
    // Other classes
    SSinputs mSS;
    CTLEVELmodel userInputs;
    CTLEVELclass m;
    // Constructor
    INTLEVELclass(vec, mat, int, string, bool, vec, bool, bool); //, mat, mat);
    // Rest of methods
    // void estim(){m.estim();};
    void forecast();
    // void filter(){m.filter();};
    void smooth();   // Smooth and forecast
    void validate();
};
/**************************
 * Functions declarations
 ***************************/
void INTLEVEL(vec, mat, int, string, bool, vec, bool);  //, mat, mat);
// INTLEVELclass preProcess(vec, mat, int, string, bool, vec);
/**************************
 * Functions implementations
 ***************************/
// Constructor
INTLEVELclass::INTLEVELclass(vec y, mat u, int h, string obsEq, bool verbose,
                             vec p0, bool logTransform, bool cllik) {
                             // mat noiseETA, mat simul) {
    // simul is epsilon noise in reallity
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
    // Ignore the missing values. This is how we encode the gaps
    // uvec t =  find_finite(y);
    uvec t = find(y != 0.0);
    this->u = u;
    if (u.n_rows > 0){
        if (u.n_cols > t.n_rows)
            u = join_horiz(u.cols(t), u.cols(t(t.n_rows), u.n_cols - 1));
        else
            u = u.cols(t);
    }
    // this->simul = simul;  // noise coming from R
    // this->noiseETA = noiseETA;
    if (logTransform)
        y(t) = log(y(t));
    uvec aux = t.tail_rows(1); uword lastT = aux(0);
    // vec th = join_vert(conv_to<vec>::from(t), regspace<vec>(lastT + 1, lastT + h));
    uvec th = join_vert(t, regspace<uvec>(lastT + 1, lastT + h));
    uvec thCum = join_vert(t, lastT + cumsum(regspace<uvec>(1, h)));
    if (!errorExit){
        CTLEVELclass m(mSS, y(t), u, conv_to<vec>::from(thCum), obsEq, verbose, p0, cllik);
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
        this->cllik = cllik;
        this->aggStock = false;
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
    m.estim();
    mSS = m.getInputs();
    uword delta = tNonZero(mSS.y.n_rows - 1) - tNonZero(mSS.y.n_rows - 2);
    vec beta, l;
    double varEta = exp(2 * mSS.p(0));
    double varEps;
    if (mSS.p.n_rows > 1 + mSS.u.n_rows)
        varEps = exp(2 * mSS.p(1));
    else
        varEps = mSS.innVariance;
    double PT;
    mat u;
    if (mSS.u.n_rows > 0){
        u = mSS.u.tail_cols(h);
        beta = mSS.p.rows(2, 1 + u.n_rows);
    }
    if (mSS.cLlik) {
        mSS.PEnd *= mSS.innVariance;
        varEta *= mSS.innVariance;
        varEps = mSS.innVariance;
    }
    coef.resize(2);
    coef(0) = varEta; coef(1) = varEps;
    PT = mSS.PEnd(0, 0);
    if (obsEq[0] == 's'){
        l = regspace<vec>(delta, delta + h - 1);
        mSS.yFor.resize(h);
        mSS.FFor = mSS.yFor;
            // SS for aggregated output
            l = regspace<vec>(1, h);
            mSS.yFor = l * mSS.aEnd;
            mat F(2, 2, fill::ones); F(0, 1) = 0;
            mat Pt(2, 2, fill::value(varEta));
            Pt(0, 0) = mSS.PEnd(0, 0);
            Pt(1, 1) = mSS.PEnd(0, 0) + varEps;
            mat Q(2, 2, fill::zeros);
            Q(0, 0) = varEta;
            Q(1, 1) = varEps;
            for (uword i = 0; i < h; i++){
                mSS.FFor(i) = Pt(1, 1);
                Pt = F * Pt * F.t() + F * Q * F.t();
            }

            yForAgg = mSS.yFor;
            yForVAgg = mSS.FFor;

        // } else {
            mSS.yFor.fill(mSS.aEnd(0));
            mSS.FFor = PT + (l - delta) * varEta + varEps;   // 9.2.17 Harvey (490) or 3.5.8b (148)
        // }
        // if (simul.n_rows > 0) {  // Simulations //////////////////////////////
        //     simul *= sqrt(varEps);     // Re-scaling noise of epsilon
        //     noiseETA *= sqrt(varEta);    // Re-scaling noise
        //     noiseETA += repmat(mSS.yFor, 1, noiseETA.n_cols);
        //     simul += noiseETA;
        //     if (logTransform)
        //         simul = cumsum(exp(simul));
        //     else
        //         simul = cumsum(simul);
        // }
    } else{
        // aEnd and PEnd is one step ahead forecast for end + 1
        // Danger when the last observation is preceded by zeros
        l = regspace<vec>(1, h + delta);
        vec aux(h + delta);
        aux = (l - 1) * mSS.aEnd(0) / delta;
        mSS.yFor = aux.tail_rows(h);
        PT -= delta * varEta;
        aux = pow(l, 2) * PT + 1.0 / 3.0 * pow(l, 3) * varEta + l * varEps;  // 9.3.26 Harvey (498)
        mSS.FFor = aux.tail_rows(h);
    }
    if (u.n_rows > 0)
        mSS.yFor += u.rows(span(mSS.y.n_rows, u.n_cols)).t() * beta;
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
                             exp(2 * this->mSS.p), this->cllik);
        // Set up system with correct parameters
        mSS = mSmooth.getInputs();
        CTLEVELmodel aux;
        aux.delta.resize(mSS.y.n_rows);
        aux.delta.fill(1.0);
        aux.obsEq = obsEq;
        CTLEVELmatrices(mSS.p, &mSS.system, &aux);
        mSmooth.setInputs(mSS);
        // Smoothing
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
    if (obsEq[0] == 's')
        snprintf(str, 45, " Stock output\n");
    else
        snprintf(str, 45, " Flow output\n");
    mSS.table.push_back(str);
    if (logTransform){
        snprintf(str, 45, " Log-transformed data\n");
        mSS.table.push_back(str);
    }
    snprintf(str, 45, " %s", mSS.estimOk.c_str());
    mSS.table.push_back(str);
    mSS.table.push_back("------------------------------------\n");
    mSS.table.push_back("                 Param     |Grad| \n");
    mSS.table.push_back("------------------------------------\n");
    double varEta = exp(2 * mSS.p(0)), varEps, ggrad = 0.0;
    if (varEta < 1e-4)
        snprintf(str, 45, " Var(eta):  %10.2e %10.6f\n", varEta, abs(mSS.grad(0)));
    else
        snprintf(str, 45, " Var(eta):  %10.4f %10.6f\n", varEta, abs(mSS.grad(0)));
    mSS.table.push_back(str);
    if (mSS.p.n_rows + mSS.u.n_rows > 1){
        varEps = exp(2 * mSS.p(1));
        ggrad = abs(mSS.grad(1));
    } else{
        varEps= mSS.innVariance;
    }
    if (varEps < 1e-4)
        snprintf(str, 45, " Var(eps):  %10.2e %10.6f\n", varEps, ggrad);
    else
        snprintf(str, 45, " Var(eps):  %10.4f %10.6f\n", varEps, ggrad);
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
void INTLEVEL(vec y, mat u, int h, string obsEq, bool verbose, vec p0, bool logTransform,
              bool cllik) {  //, mat noiseETA, mat noiseEPS){
    // Building standard SS model
    INTLEVELclass mClass(y, u, h, obsEq, verbose, p0, logTransform, cllik);  //, noiseETA, noiseEPS);
    // mClass = preProcess(y, u, h, obsEq, verbose, p0);
    // mClass.estim();
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
    mat comp = exp(mClass.comp);
    comp.col(1).t().print("components");
}

