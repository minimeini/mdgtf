#include <chrono>
#include "Model.hpp"
#include "LinearBayes.hpp"
#include "SequentialMonteCarlo.hpp"
#include "MCMC.hpp"
#include "VariationalBayes.hpp"

#include <progress.hpp>
#include <progress_bar.hpp>

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr, BH, RcppProgress)]]

//' @export
// [[Rcpp::export]]
Rcpp::NumericVector lognorm2serial(const double &mu, const double &sd2)
{
    return lognorm::lognorm2serial(mu, sd2);
}

//' @export
// [[Rcpp::export]]
Rcpp::NumericVector serial2lognorm(const double &m, const double &s2)
{
    return lognorm::serial2lognorm(m, s2);
}


//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dnbinom0(
        const unsigned int &nL,
        const double &kappa,
        const double &r)
{
    return Rcpp::wrap(nbinom::dnbinom(nL, kappa, r));
}


//' @export
// [[Rcpp::export]]
Rcpp::NumericVector dlognorm0(
        const unsigned int &nL,
        const double &mu,
        const double &sd2)
{
    return Rcpp::wrap(lognorm::dlognorm(nL, mu, sd2));
}




//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_default_algo_settings(const std::string &method)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string method_name = tolower(method);

    Rcpp::List opts = SMC::SequentialMonteCarlo::default_settings();
    switch (algo_list[method_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        opts = LBA::LinearBayes::default_settings();
        break;
    }
    case AVAIL::Algo::MCS:
    {
        opts = SMC::MCS::default_settings();
        break;
    }
    case AVAIL::Algo::FFBS:
    {
        opts = SMC::FFBS::default_settings();
        break;
    }
    case AVAIL::Algo::TFS:
    {
        opts = SMC::TFS::default_settings();
        break;
    }
    case AVAIL::Algo::ParticleLearning:
    {
        opts = SMC::PL::default_settings();
        break;
    }
    case AVAIL::Algo::MCMC:
    {
        opts = MCMC::Disturbance::default_settings();
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        opts = VB::Hybrid::default_settings();
        break;
    }
    default:
        throw std::invalid_argument("dgtf_default_algo_settings: Invalid algorithm.");
        break;
    }

    return opts;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_default_model()
{
    return Model::default_settings();
}


//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_simulate(
    const Rcpp::List &settings,
    const unsigned int &ntime,
    const double &y0 = 0.)
{
    Model model(settings);

    arma::vec psi, ft, lambda, y;
    arma::mat Theta;
    StateSpace::simulate(y, lambda, ft, Theta, psi, model, ntime, y0);

    Rcpp::List output;
    output["model"] = model.info();
    output["y"] = Rcpp::wrap(y.t());
    output["nlag"] = model.dlag.nL;
    output["psi"] = Rcpp::wrap(psi.t());
    output["Theta"] = Rcpp::wrap(Theta);
    output["ft"] = Rcpp::wrap(ft.t());
    output["lambda"] = Rcpp::wrap(lambda.t());
    return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_infer(
    const Rcpp::List &model_settings,
    const arma::vec &y_in,
    const std::string &method,
    const Rcpp::List &method_settings,
    const std::string &loss_func = "quadratic",
    const unsigned int &k = 1,
    const Rcpp::Nullable<unsigned int> &tstart_forecast = R_NilValue,
    const Rcpp::Nullable<unsigned int> &tend_forecast = R_NilValue,
    const bool &add_y0 = false)
{
    // Model model = dgtf_initialize(model_settings);
    Model model(model_settings);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string algo_name = tolower(method);
    std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

    arma::vec y;
    if (add_y0)
    {
        y.set_size(y_in.n_elem + 1);
        y.at(0) = 0.;
        y.tail(y_in.n_elem) = y_in;
    }
    else
    {
        y = y_in;
    }

    const unsigned int nT = y.n_elem - 1;
    model.seas.X = Season::setX(nT, model.seas.period, model.seas.P);

    unsigned int nforecast = 0;
    if (method_settings.containsElementNamed("num_step_ahead_forecast"))
    {
        nforecast = Rcpp::as<unsigned int>(method_settings["num_step_ahead_forecast"]);
    }

    Rcpp::List output, forecast, error;
    arma::mat psi(nT + 1, 3);
    arma::vec ci_prob = {0.025, 0.5, 0.975};

    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        if (LBA_FILL_ZERO)
        {
            y.clamp(0.01 / static_cast<double>(model.nP), y.max());
        }
        LBA::LinearBayes linear_bayes(method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        linear_bayes.filter(model, y);
        linear_bayes.smoother(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = linear_bayes.get_output(model);

        Rcpp::List tmp = linear_bayes.forecast_error(model, y, 1000, loss_func, k, tstart_forecast, tend_forecast);
        error["forecast"] = tmp;

        Rcpp::List tmp = linear_bayes.fitted_error(model, y, 1000, loss_func);
        error["fitted"] = tmp;

        break;
    } // case Linear Bayes
    case AVAIL::Algo::MCS:
    {
        SMC::MCS mcs(model, method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        mcs.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = mcs.get_output();

        Rcpp::List tmp = mcs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
        error["forecast"] = tmp;

        Rcpp::List tmp = mcs.fitted_error(model, y, loss_func);
        error["fitted"] = tmp;

        break;
    } // case MCS
    case AVAIL::Algo::FFBS:
    {
        SMC::FFBS ffbs(model, method_settings);
        auto start = std::chrono::high_resolution_clock::now();
        ffbs.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;
        output = ffbs.get_output();

        Rcpp::List tmp = ffbs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
        error["forecast"] = tmp;

        Rcpp::List tmp = ffbs.fitted_error(model, y, loss_func);
        error["fitted"] = tmp;

        break;
    } // case FFBS
    case AVAIL::Algo::TFS:
    {
        SMC::TFS tfs(model, method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        tfs.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = tfs.get_output();

        if (nforecast > 0)
        {
            forecast = tfs.forecast(model, y);
        }

        Rcpp::List tmp = tfs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
        error["forecast"] = tmp;
        Rcpp::List tmp = tfs.fitted_error(model, y, loss_func);
        error["fitted"] = tmp;

        break;
    } // case TFS
    case AVAIL::Algo::ParticleLearning:
    {
        SMC::PL pl(model, method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        pl.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = pl.get_output();

        if (nforecast > 0)
        {
            forecast = pl.forecast(model, y);
        }

        Rcpp::List tmp = pl.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
        error["forecast"] = tmp;
        Rcpp::List tmp = pl.fitted_error(model, y, loss_func);
        error["fitted"] = tmp;

        break;
    } // case particle learning
    case AVAIL::Algo::MCMC:
    {
        MCMC::Disturbance mcmc(model, method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        mcmc.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = mcmc.get_output();
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        VB::Hybrid hvb(model, method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        hvb.infer(model, y);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

        output = hvb.get_output();
        break;
    }
    default:
    {
        throw std::invalid_argument("Unknown algorithm " + algo_name);
        break;
    }
    } // switch by algorithm

    Rcpp::List out;
    out["fit"] = output;
    out["error"] = error;

    return out;
}




//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_posterior_predictive(
    const Rcpp::List &output, 
    const Rcpp::List &model_opts, 
    const arma::vec &y, 
    const unsigned int &nrep = 100)
{
    const unsigned int ntime = y.n_elem - 1;
    Model model(model_opts);
    model.seas.X = Season::setX(ntime, model.seas.period, model.seas.P);

    arma::mat psi_stored;
    if (output.containsElementNamed("psi_stored"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi_stored"]);
    }
    else if (output.containsElementNamed("psi"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi"]);
    }
    else if (output.containsElementNamed("psi_filter"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi_filter"]);
    }
    else
    {
        throw std::invalid_argument("No state samples.");
    }

    const unsigned int nsample = psi_stored.n_cols;

    arma::vec W_stored(nsample);
    W_stored.fill(model.derr.par1);
    if (output.containsElementNamed("W"))
    {
        W_stored = Rcpp::as<arma::vec>(output["W"]);
    }

    arma::vec rho_stored(nsample);
    rho_stored.fill(model.dobs.par2);
    if (output.containsElementNamed("rho"))
    {
        rho_stored = Rcpp::as<arma::vec>(output["rho"]);
    }


    arma::vec par1_stored(nsample);
    par1_stored.fill(model.dlag.par1);
    if (output.containsElementNamed("par1"))
    {
        par1_stored = Rcpp::as<arma::vec>(output["par1"]);
    }


    arma::vec par2_stored(nsample);
    par2_stored.fill(model.dlag.par2);
    if (output.containsElementNamed("par2"))
    {
        par2_stored = Rcpp::as<arma::vec>(output["par2"]);
    }

    arma::mat seas_stored(model.seas.period, nsample, arma::fill::zeros);
    if (output.containsElementNamed("seas"))
    {
        seas_stored = Rcpp::as<arma::mat>(output["seas"]);
    }
    else
    {
        for (unsigned int i = 0; i < nsample; i++)
        {
            seas_stored.col(i) = model.seas.val;
        }
    }

    arma::cube yhat = arma::zeros<arma::cube>(ntime + 1, nsample, nrep);
    arma::cube res = arma::zeros<arma::cube>(ntime + 1, nsample, nrep);
    Progress p(nsample*ntime, true);
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < nsample; i++)
    {
        arma::vec ft(ntime + 1, arma::fill::zeros);
        arma::vec psi = psi_stored.col(i);
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, model.fgain);

        Model mod = model;
        mod.dobs.par2 = rho_stored.at(i);
        mod.derr.par1 = W_stored.at(i);
        mod.dlag.par1 = par1_stored.at(i);
        mod.dlag.par2 = par2_stored.at(i);
        mod.seas.val = seas_stored.col(i);

        for (unsigned int t = 1; t <= ntime; t++)
        {
            ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, mod.dlag, mod.ftrans);
            double eta = ft.at(t);
            if (mod.seas.period > 0)
            {
                eta += arma::as_scalar(mod.seas.X.col(t).t() * mod.seas.val);
            }

            double lambda = LinkFunc::ft2mu(eta, model.flink);

            for (unsigned int j = 0; j < nrep; j++)
            {
                yhat.at(t, i, j) = ObsDist::sample(lambda, mod.dobs.par2, mod.dobs.name);
                res.at(t, i, j) = std::abs(yhat.at(t, i, j) - y.at(t));
            }

            p.increment(); 
        }
    }

    arma::cube ytmp = yhat.reshape(ntime + 1, nsample * nrep, 1);
    arma::mat yhat2 = ytmp.slice(0);

    Rcpp::List output2;
    // output2["yhat"] = Rcpp::wrap(yhat2);

    arma::vec prob = {0.025, 0.5, 0.975};
    arma::mat yqt = arma::quantile(yhat2, prob, 1);
    output2["yhat"] = Rcpp::wrap(yqt);

    arma::cube rtmp = res.reshape(ntime + 1, nsample * nrep, 1);
    arma::mat res2 = rtmp.slice(0); // (ntime + 1) x (nsample * nrep)
    arma::mat res_qt = arma::quantile(res2, prob, 1);

    double rmse = std::sqrt(arma::mean(arma::mean(arma::pow(res2, 2))));
    double mae = arma::mean(arma::mean(res2));
    output2["res"] = Rcpp::wrap(res_qt);
    output2["rmse"] = rmse;
    output2["mae"] = mae;

    return output2;
}



//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_forecast(
    const Rcpp::List &output,
    const Rcpp::List &model_settings,
    const arma::vec &y,   // (nT + 1) x 1
    const unsigned int nrep = 100,
    const unsigned int &k = 1, // k-step-ahead forecasting,
    const bool &verbose = false)
{
    const unsigned int ntime = y.n_elem - 1;
    Model model(model_settings);
    model.seas.X = Season::setX(ntime + k, model.seas.period, model.seas.P); 
    // X: period x (ntime + k + 1)

    arma::mat psi_stored;
    if (output.containsElementNamed("psi_stored"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi_stored"]);
    }
    else if (output.containsElementNamed("psi"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi"]);
    }
    else if (output.containsElementNamed("psi_filter"))
    {
        psi_stored = Rcpp::as<arma::mat>(output["psi_filter"]);
    }
    else
    {
        throw std::invalid_argument("No state samples.");
    }

    const unsigned int nsample = psi_stored.n_cols;

    arma::vec W_stored(nsample);
    W_stored.fill(model.derr.par1);
    if (output.containsElementNamed("W"))
    {
        W_stored = Rcpp::as<arma::vec>(output["W"]);
    }

    arma::vec rho_stored(nsample);
    rho_stored.fill(model.dobs.par2);
    if (output.containsElementNamed("rho"))
    {
        rho_stored = Rcpp::as<arma::vec>(output["rho"]);
    }


    arma::vec par1_stored(nsample);
    par1_stored.fill(model.dlag.par1);
    if (output.containsElementNamed("par1"))
    {
        par1_stored = Rcpp::as<arma::vec>(output["par1"]);
    }


    arma::vec par2_stored(nsample);
    par2_stored.fill(model.dlag.par2);
    if (output.containsElementNamed("par2"))
    {
        par2_stored = Rcpp::as<arma::vec>(output["par2"]);
    }

    arma::mat seas_stored(model.seas.period, nsample, arma::fill::zeros);
    if (output.containsElementNamed("seas"))
    {
        seas_stored = Rcpp::as<arma::mat>(output["seas"]);
    }
    else
    {
        for (unsigned int i = 0; i < nsample; i++)
        {
            seas_stored.col(i) = model.seas.val;
        }
    }

    arma::mat ycast = arma::zeros<arma::mat>(nsample, nrep);
    Progress p(nsample, true);
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    // #endif
    for (unsigned int i = 0; i < nsample; i++)
    {
        arma::mat ytmp(ntime + 1 + k, nrep, arma::fill::zeros);
        arma::mat ft = ytmp;
        arma::mat psi = ytmp;
        for (unsigned int s = 0; s < nrep; s++)
        {
            ytmp.submat(0, s, ntime, s) = y;
            psi.submat(0, s, ntime, s) = psi_stored.col(i);
            psi.submat(ntime + 1, s, ntime + k, s).fill(psi.at(ntime, s));
        }

        arma::mat hpsi = GainFunc::psi2hpsi<arma::mat>(psi, model.fgain);

        Model mod = model;
        mod.dobs.par2 = rho_stored.at(i);
        mod.derr.par1 = W_stored.at(i);
        mod.dlag.par1 = par1_stored.at(i);
        mod.dlag.par2 = par2_stored.at(i);
        mod.seas.val = seas_stored.col(i);

        for (unsigned int j = 1; j <= k; j++)
        {
            unsigned int tidx = ntime + j;
            for (unsigned int s = 0; s < nrep; s++)
            {
                ft.at(tidx, s) = TransFunc::func_ft(
                    tidx, ytmp.col(s), ft.col(s), hpsi.col(s), mod.dlag, mod.ftrans);

                double eta = ft.at(tidx, s);
                if (mod.seas.period > 0)
                {
                    eta += arma::as_scalar(mod.seas.X.col(tidx).t() * mod.seas.val);
                }

                double lambda = LinkFunc::ft2mu(eta, model.flink);
                ytmp.at(tidx, s) = ObsDist::sample(lambda, mod.dobs.par2, mod.dobs.name);
            }
        }

        ycast.row(i) = ytmp.row(ntime + k);
        p.increment(); 
    }

    arma::vec ycast2 = ycast.as_col();
    arma::vec prob = {0.025, 0.5, 0.975};
    arma::vec yqt = arma::quantile(ycast2, prob);

    Rcpp::List out;
    out["ycast"] = Rcpp::wrap(ycast2.t());
    out["yqt"] = Rcpp::wrap(yqt.t());

    return out;
}

//' @export
// [[Rcpp::export]]
arma::vec dgtf_evaluate(
    const arma::vec &yest, // nsample x 1
    const double &ytrue,
    const std::string &loss_func = "quadratic",
    const bool &eval_covarage_width = true,
    const bool &eval_covarage_pct = true)
{
    return evaluate(yest, ytrue, loss_func, eval_covarage_width, eval_covarage_pct);
}


//' @export
// [[Rcpp::export]]
arma::mat dgtf_tuning(
    const Rcpp::List &model_opts,
    const arma::vec &y_in,
    const std::string &algo_name,
    const Rcpp::List &algo_opts,
    const std::string &param_name,
    const double &from = 0.7,
    const double &to = 0.99,
    const double &delta = 0.01,
    const Rcpp::Nullable<Rcpp::NumericVector> &grid = R_NilValue,
    const std::string &loss = "quadratic",
    const bool &add_y0 = false,
    const bool &verbose = true)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
    std::map<std::string, AVAIL::Param> param_list = AVAIL::tuning_param_list;

    arma::vec y;
    if (add_y0)
    {
        y.set_size(y_in.n_elem + 1);
        y.at(0) = 0.;
        y.tail(y_in.n_elem) = y_in;
    }
    else
    {
        y = y_in;
    }

    Model model(model_opts);
    model.seas.X = Season::setX(y.n_elem - 1, model.seas.period, model.seas.P);

    arma::vec param_grid;
    if (!grid.isNull())
    {
        param_grid = Rcpp::as<arma::vec>(grid);
    }
    else
    {
        param_grid = arma::regspace<arma::vec>(from, delta, to);
    }

    arma::mat stats(param_grid.n_elem, 4, arma::fill::zeros);
    for (unsigned int i = 0; i < param_grid.n_elem; i++)
    {
        Rcpp::checkUserInterrupt();

        Rcpp::List algo = algo_opts;
        algo["do_smoothing"] = false;

        if (param_list[param_name] == AVAIL::Param::discount_factor)
        {
            algo["use_discount"] = true;
            algo["discount_factor"] = param_grid.at(i);
        }
        else if (param_list[param_name] == AVAIL::Param::W)
        {
            model.derr.par1 = param_grid.at(i);
            model.derr.var.at(0, 0) = param_grid.at(i);
            algo["use_discount"] = false;
        }
        else if (param_list[param_name] == AVAIL::Param::num_backward)
        {
            algo["num_backward"] = param_grid.at(i);
        }
        else if (param_list[param_name] == AVAIL::Param::step_size)
        {
            algo["learning_rate"] = param_grid.at(i);
        }
        else if (param_list[param_name] == AVAIL::Param::learning_rate)
        {
            algo["eps_step_size"] = param_grid.at(i);
        }
        else if (param_list[param_name] == AVAIL::Param::k)
        {
            algo["k"] = param_grid.at(i);
        }
        else
        {
            throw std::invalid_argument("Unknown tuning parameter " + param_name + ".");
        }


        stats.at(i, 0) = param_grid.at(i);
        double err_forecast = 0.;
        double cov_forecast = 0.;
        double width_forecast = 0.;

        bool success = false;
        try
        {
            switch (algo_list[algo_name])
            {
            case AVAIL::Algo::LinearBayes:
            {
                if (LBA_FILL_ZERO)
                {
                    y.clamp(0.01 / static_cast<double>(model.nP), y.max());
                }
                LBA::LinearBayes linear_bayes(algo);
                linear_bayes.filter(model, y);
                linear_bayes.forecast_error(model, y, err_forecast, cov_forecast, width_forecast, 1000, loss);
                break;
            }
            case AVAIL::Algo::MCS:
            {
                SMC::MCS mcs(model, algo);
                mcs.infer(model, y, false);
                arma::cube theta_tmp = mcs.Theta.tail_slices(y.n_elem);
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                break;
            }
            case AVAIL::Algo::FFBS:
            {
                SMC::FFBS ffbs(model, algo);
                ffbs.infer(model, y, false);
                arma::cube theta_tmp = ffbs.Theta.tail_slices(y.n_elem);
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                break;
            }
            case AVAIL::Algo::TFS:
            {
                SMC::TFS tfs(model, algo);
                tfs.infer(model, y, false);
                arma::cube theta_tmp = tfs.Theta.tail_slices(y.n_elem);
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                break;
            }
            case AVAIL::Algo::ParticleLearning:
            {
                if (param_list[param_name] == AVAIL::Param::W)
                {
                    Rcpp::List W_opts;
                    if (algo.containsElementNamed("W"))
                    {
                        Rcpp::List W_opts = Rcpp::as<Rcpp::List>(algo["W"]);
                    }
                    W_opts["infer"] = false;
                    algo["W"] = W_opts;
                }
                SMC::PL pl(model, algo);
                pl.infer(model, y, false);
                arma::cube theta_tmp = pl.Theta.tail_slices(y.n_elem);
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                break;
            }
            case AVAIL::Algo::HybridVariation:
            {
                Rcpp::List W_opts;
                if (algo.containsElementNamed("W"))
                {
                    Rcpp::List W_opts = Rcpp::as<Rcpp::List>(algo["W"]);
                }
                W_opts["infer"] = false;
                algo["W"] = W_opts;

                VB::Hybrid hvb(model, algo);
                hvb.infer(model, y, false);
                Model::forecast_error(err_forecast, cov_forecast, width_forecast, hvb.psi_stored, y, model, loss, false);
                break;
            }
            default:
            {
                throw std::invalid_argument("Unknown algorithm: " + algo_name + ".");
            }
            }

            success = true;
        }
        catch (const std::exception &e)
        {
            success = false;
            std::cerr << std::endl;
            std::cerr << e.what() << ' - failed at delta = ' << param_grid.at(i) << std::endl;
        }

        if (success)
        {
            stats.at(i, 1) = err_forecast;
            stats.at(i, 2) = cov_forecast;
            stats.at(i, 3) = width_forecast;
        }

        if (verbose)
        {
            Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << param_grid.n_elem;
        }
    }

    if (verbose)
    {
        Rcpp::Rcout << std::endl;
    }

    return stats;
}
