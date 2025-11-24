#include <chrono>
#include "../core/GainFunc.hpp"
#include "Model.hpp"
#include "MCMC.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' @export
// [[Rcpp::export]]
arma::mat sample_car(
    const unsigned int &k,
    const arma::mat &V,
    const arma::vec &car_params
)
{
    BYM2 spatial(V);
    spatial.mu = car_params[0];
    spatial.tau_b = car_params[1];
    spatial.phi = car_params[2];
    return spatial.sample_spatial_effects(k);
}

//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_default_algo_settings(const std::string &method)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string method_name = tolower(method);

    Rcpp::List opts;
    switch (algo_list[method_name])
    {
        case AVAIL::MCMC:
            opts = MCMC::get_default_settings();
            break;
        default:
            throw std::invalid_argument("mdgtf_default_algo_settings - unknown method: " + method);
    }

    return opts;
} // end of mdgtf_default_algo_settings()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_simulate(
    const Rcpp::List &settings,
    const unsigned int &ntime
)
{
    Model model(settings);
    arma::mat Y, Lambda, wt, Psi;
    model.simulate(Y, Lambda, wt, Psi, ntime);

    return Rcpp::List::create(
        Rcpp::Named("Y") = Y,
        Rcpp::Named("Lambda") = Lambda,
        Rcpp::Named("wt") = wt,
        Rcpp::Named("Psi") = Psi,
        Rcpp::Named("Rt") = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain),
        Rcpp::Named("model") = settings
    );
} // end of mdgtf_simulate()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_infer(
    const Rcpp::List &model_settings,
    const arma::mat &Y_in,
    const arma::mat &wt_in,
    const std::string &method,
    const Rcpp::List &method_settings
)
{
    Model model(model_settings);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string algo_name = tolower(method);

    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output;
    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::MCMC:
    {
        MCMC mcmc(method_settings);
        mcmc.infer(model, Y_in, wt_in);
        output = mcmc.get_output();
        break;
    }
    default:
        break;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

    return output;
} // end of mdgtf_infer()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_posterior_predictive(
    const Rcpp::List &output, 
    const Rcpp::List &model_opts, 
    const arma::mat &Y, // nS x (nT + 1)
    const unsigned int &nrep = 1,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Rt = R_NilValue // nS x (nT + 1)
)
{
    Model model(model_opts);
    arma::cube Y_pred, Y_residual, hPsi;
    // Y_pred: nS x (nT + 1) x (nsample * nrep)
    // Y_residual: nS x (nT + 1) x nsample
    // hPsi: nS x (nT + 1) x nsample
    double chi_sqr = model.sample_posterior_predictive_y(Y_pred, Y_residual, hPsi, output, Y, nrep);

    arma::cube Y_pred_trunc = Y_pred.cols(1, Y_pred.n_cols - 1); // nS x nT x (nsample * nrep)
    arma::mat Y_trunc = Y.cols(1, Y.n_cols - 1); // nS x nT
    arma::vec crps = calculate_crps(Y_trunc, Y_pred_trunc);

    arma::vec prob = {0.025, 0.5, 0.975};
    arma::cube Y_pred_ci(Y_pred.n_rows, Y_pred.n_cols, prob.n_elem);
    arma::vec Y_coverage(Y_pred.n_rows, arma::fill::zeros);
    arma::vec Y_width(Y_pred.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < Y_pred.n_rows; i++)
    {
        for (unsigned int j = 0; j < Y_pred.n_cols; j++)
        {
            arma::vec samples = Y_pred.tube(i, j); // (nsample * nrep) x 1
            Y_pred_ci.tube(i, j) = arma::quantile(samples, prob); // 3 x 1
        }

        arma::vec lobnd = Y_pred_ci.slice(0).row(i).t();
        arma::vec upbnd = Y_pred_ci.slice(2).row(i).t();
        arma::vec true_val = Y.row(i).t();

        double covered = arma::accu((true_val >= lobnd) % (true_val <= upbnd));
        Y_coverage.at(i) = covered / static_cast<double>(true_val.n_elem);
        Y_width.at(i) = arma::mean(upbnd - lobnd);
    }

    Rcpp::List stats = Rcpp::List::create(
        Rcpp::Named("Y_pred") = Y_pred_ci, // nS x (nT + 1) x 3
        Rcpp::Named("Y_residual") = Y_residual,
        Rcpp::Named("chi_sqr_avg") = chi_sqr,
        Rcpp::Named("crps") = crps, // nS x 1
        Rcpp::Named("crps_avg") = arma::mean(crps),
        Rcpp::Named("Y_coverage") = Y_coverage, // nS x 1
        Rcpp::Named("Y_coverage_avg") = arma::mean(Y_coverage),
        Rcpp::Named("Y_width") = Y_width, // nS x 1
        Rcpp::Named("Y_width_avg") = arma::mean(Y_width)
    );

    if (Rt.isNotNull())
    {
        arma::mat Rt_true = Rcpp::as<arma::mat>(Rt);

        arma::cube hPsi_ci(hPsi.n_rows, hPsi.n_cols, prob.n_elem);
        arma::vec hPsi_coverage(hPsi.n_rows, arma::fill::zeros);
        arma::vec hPsi_width(hPsi.n_rows, arma::fill::zeros);
        for (unsigned int i = 0; i < hPsi.n_rows; i++)
        {
            for (unsigned int j = 0; j < hPsi.n_cols; j++)
            {
                arma::vec samples = hPsi.tube(i, j);                // nsample x 1
                hPsi_ci.tube(i, j) = arma::quantile(samples, prob); // 3 x 1
            }

            if (Rt.isNotNull())
            {
                arma::vec lobnd = hPsi_ci.slice(0).row(i).t();
                arma::vec upbnd = hPsi_ci.slice(2).row(i).t();
                arma::vec true_val = Rt_true.row(i).t();

                double covered = arma::accu((true_val >= lobnd) % (true_val <= upbnd));
                hPsi_coverage.at(i) = covered / static_cast<double>(true_val.n_elem);
                hPsi_width.at(i) = arma::mean(upbnd - lobnd);
            }
        }
        arma::cube Rt_res_all = arma::abs(hPsi.each_slice() - Rt_true);
        arma::mat Rt_res = arma::mean(Rt_res_all, 2); // nS x (nT + 1)
        arma::vec mae_Rt = arma::mean(Rt_res, 1);

        stats["Rt"] = hPsi_ci; // nS x (nT + 1) x 3
        stats["Rt_mae"] = mae_Rt; // nS x 1
        stats["Rt_mae_avg"] = arma::mean(mae_Rt);
        stats["Rt_coverage"] = hPsi_coverage; // nS x 1
        stats["Rt_coverage_avg"] = arma::mean(hPsi_coverage);
        stats["Rt_width"] = hPsi_width; // nS x 1
        stats["Rt_width_avg"] = arma::mean(hPsi_width);
    }

    return stats;
} // end of mdgtf_posterior_predictive()