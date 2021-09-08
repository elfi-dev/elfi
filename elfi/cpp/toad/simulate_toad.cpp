// Rcpp code for the Fowler's Toads example
// The example is from the following paper
// Philippe Marchand, Morgan Boenke, David M. Green (2017). A stochastic movement model
// reproduces patterns of site fidelity and long-distance dispersal in a population
// of Fowlers toads (Anaxyrus fowleri). Ecological Modelling.
// https://doi.org/10.1016/j.ecolmodel.2017.06.025.
//
// This simulation code is based on the original R code of Marchand et al (2017).


#include <iostream>
#include <cmath>       /* pow */
#include <math.h>       /* sin, cos, floor */
#include <vector>
// #include <boost/python/raw_function.hpp>

// namespace py = boost::python;

#include <Rcpp.h>
// [[Rcpp::plugins("cpp11")]]

using namespace std;
using namespace Rcpp;


// sample from the Uniform integer distribuition
int sample(int a) {
    int x = floor(R::runif(0,a+1));
    return (x);
}

// probability distribution function of an univariate normal distribution
double dnormal(double x, double mu, double sigma) {
    double y;
    y = (1 / (sigma * sqrt(2*M_PI))) * exp(-0.5 * pow((x-mu)/sigma, 2.0));
    return (y);
}


//' Generate a random sample from the zero-centered stable distribution
//'
//' @description Draw a sample from a symmetric, zero-centered stable distribution with
//' given scale and stability (alpha) parameters, using the CMS algorithm.
//' @param scale The scale parameter.
//' @param alpha The stability parameter.
//' @return  A random sample from the zero-centered stable distribution.
//' @keywords internal
// [[Rcpp::export]]
double rstable(double scale, double alpha) {
    double x;
    if (alpha == 1) {
        x = R::rcauchy(0.0,1.0);
        return (x);
    }
    if (alpha == 2) {
        x = R::rnorm(0.0,sqrt(2)*scale);
        return (x);
    }
    if (alpha > 0 && alpha < 2) {
        double u = R::runif(-0.5*M_PI,0.5*M_PI);
        double v = R::rexp(1.0);
        double t = sin(alpha * u) / pow(cos(u), (1 / alpha));
        double s = pow((cos((1 - alpha) * u) / v), ((1 - alpha) / alpha));
        x = scale * t * s;
        return (x);
    }
    cout << "rstable is not defined for alpha = " << alpha << endl;
    return (0);
}

double prod(NumericVector x) {
    double ret = 1;
    for (int i = 0; i < x.length(); i++) ret = ret * x[i];
    return (ret);
}

//' The simulation function for the toad example
//'
//' @description The simulation function for the toad example.
//' @param params A vector of proposed model parameters, \ifelse{html}{\out{<i>&#945</i>}}{\eqn{\alpha}},
//'   \ifelse{html}{\out{<i>&#947</i>}}{\eqn{gamma}} and \ifelse{html}{\out{p<sub>0</sub>}}{\eqn{p_0}}.
//' @param ntoad The number of toads to simulate in the observation.
//' @param nday The number of days lasted of the observation.
//' @param model Which model to be used. 1 for the random return model, 2 for the nearest return model,
//'   and 3 for the distance-based return probability model.
//' @param d0 Characteristic distance for model 3. Only used if model is 3.
//' @return A data matrix.
//' @examples sim_toad(c(1.7,36,0.6), 10, 8, 1)
//' @export
// [[Rcpp::export]]
NumericMatrix sim_toad(NumericVector params, int ntoad, int nday, int model = 1, double d0 = 100) {
    double alpha = params[0];
    double scale = params[1];
    double p0 = params[2];
    NumericMatrix xs(nday, ntoad);;
    double x, xn, p_noret;
    NumericVector ref(1), dist, pi_ret;
    int ret_pt;

    for (int j = 0; j < ntoad; j++) {
        NumericVector curr(1);
        if (model == 3) {
            NumericVector ref(1);
        }
        for (int i = 0; i < nday - 1; i++) {
            x = rstable(scale, alpha);
            xn = curr(i) + x;
            if (model == 3) {
                dist = abs(xn - ref); //distances to all previous unique refuge sites
                pi_ret = p0 * exp(-dist / d0); // return prob to the each unique site
                p_noret = prod(1 - pi_ret);

            } else {
                p_noret = 1 - p0;
            }

            Rcpp::NumericVector r = Rcpp::runif(1);
            if (r[0] < p_noret) {
                // If no return, take refuge here
                curr.push_back(xn);
                if (model == 3) {
                    ref.push_back(xn);
                }
            } else {
                if (model == 1) {
                    // model 1: random return
                    ret_pt = sample(i);
                    curr.push_back(curr(ret_pt));
                }
                if (model == 2) {
                    // model 2: nearest return
                    ret_pt = which_min(abs(xn - curr));
                    curr.push_back(curr(ret_pt));
                }
                if (model == 3) {
                    // model 3: distance-based return probability
                    ret_pt = as<int>(sample(pi_ret.length(), 1, true, pi_ret, true)) - 1;
                    curr.push_back(ref(ret_pt));
                }
            }
        }
        xs(_, j) = curr;
    }
    return (xs);
}

// Convert an observation matrix to a vector of n-day displacements
vector<double> obsMat2deltaxCpp(Rcpp::NumericMatrix X, unsigned int lag) {
    unsigned int ndays = X.nrow();
    unsigned int ntoads =  X.ncol();
    unsigned int i, j;
    vector<double> x;
    double x0, x1, temp;
    for (j=0; j<ntoads; j++) {
        for (i=0; i<ndays-lag; i++) {
            x0 = X(i,j);
            x1 = X(i+lag,j);
            if (NumericVector::is_na(x0) | NumericVector::is_na(x1)) continue;
            temp = x1 - x0;
            x.push_back(abs(temp));
        }
    }
    return (x);
}

//' Convert an observation matrix to a vector of n-day displacements
//'
//' @description Convert an observation matrix to a vector of n-day
//' displacements. This is a function for the toad example.
//' @param X The observation matrix to be converted.
//' @param lag Interger, the number of day lags to compute the displacement.
//' @return A vector of displacements.
//' @export
// [[Rcpp::export]]
NumericVector obsMat2deltax(Rcpp::NumericMatrix X, unsigned int lag) {
    unsigned int ndays = X.nrow();
    unsigned int ntoads =  X.ncol();
    unsigned int i, j;
    NumericVector x;
    double x0, x1, temp;
    for (j=0; j<ntoads; j++) {
        for (i=0; i<ndays-lag; i++) {
            x0 = X(i,j);
            x1 = X(i+lag,j);
            if (NumericVector::is_na(x0) | NumericVector::is_na(x1)) continue;
            temp = x1 - x0;
            x.push_back(abs(temp));
        }
    }
    return (x);
}

// //' This function computes the scores of Gaussian mixture models for the toad example
// //'
// //' @description The summary statistics for the toad example is taken to be the
// //' scores of Gaussian mixture model fitted to the displacement distribution. The
// //' displacements are computed with a given day lag.
// //' @param X The observation matrix.
// //' @param gmm A matrix containing the parameters for a Gaussian mixture model.
// //' The first row should be component proportions. The second row should be
// //' component means. The last row should be component Variances.
// //' @param lag The lag of days to compute the displacements.
// //' @return A list of the following vectors: return frequencies, scores of
// //' component proportion, scores of mean and scores of variance.
// Rcpp::List gmmScores(Rcpp::NumericMatrix X, NumericMatrix gmm, unsigned int lag) {
//     int n, i, j, k;
//     k = gmm.ncol();
//     double temp, freq_ret;
//     NumericVector sigma(k);
//     vector<double> deltax, x_ret, x_noret;
// 
//     deltax = obsMat2deltaxCpp(X,lag);
//     for (i=0; i<(int) deltax.size(); i++) {
//         temp = deltax[i];
//         if (temp < 10) {
//             x_ret.push_back(temp);
//         } else {
//             x_noret.push_back(log(temp - 10));
//         }
//     }
//     freq_ret = (double) x_ret.size() / (double) deltax.size();
//     n = (int) x_noret.size();
// 
// 
//     NumericMatrix::Row p = gmm(0,_);
//     NumericMatrix::Row mu = gmm(1,_);
//     NumericMatrix::Row Sigma = gmm(2,_);
//     transform(Sigma.begin(),Sigma.end(),sigma.begin(),(double(*)(double)) sqrt);
// 
//     NumericVector f(n), scorepi(k-1), scoremu(k), scoreSigma(k);
//     //  NumericVector scoresigma(k);
//     NumericMatrix phi(n,k), xmu(n,k), A(n,k), w(n,k);
// 
//     for (i=0; i<n; i++) {
//         for (j=0; j<k; j++) {
//             phi(i,j) = dnormal(x_noret[i],mu[j],sigma[j]);
//             xmu(i,j) = x_noret[i] - mu[j];
//             A(i,j) = pow(xmu(i,j)/sigma[j],2) - 1;
//         }
//         f[i] = sum(phi(i,_) * p);
//         for (j=0; j<k; j++) {
//             w(i,j) = phi(i,j) / f[i];
//         }
//     }
// 
//     for (j=0; j<k-1; j++) {
//         scorepi[j] = sum(w(_,j) - w(_,k-1));
//     }
//     for (j=0; j<k; j++) {
//         scoremu[j] = p[j] / Sigma[j] * sum(w(_,j) * xmu(_,j));
//         //  scoresigma[j] = p[j] / sigma[j] * sum(w(_,j) * A(_,j));
//         scoreSigma[j] = 0.5 * p[j] / Sigma[j] * sum(w(_,j) * A(_,j));
//     }
// 
//     return (Rcpp::List::create(Named("freq_ret")=freq_ret,Named("scorepi")=scorepi,Named("scoremu")=scoremu,Named("scoreSigma")=scoreSigma));
// }