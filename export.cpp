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
                res.at(t, i, j) = std::abs(yhat.at(t, i, j) - y.at(i));
            }

            p.increment(); 
        }
    }

    arma::cube ytmp = yhat.reshape(ntime + 1, nsample * nrep, 1);
    arma::mat yhat2 = ytmp.slice(0);

    Rcpp::List output2;
    output2["yhat"] = Rcpp::wrap(yhat2);

    arma::vec prob = {0.025, 0.5, 0.975};
    arma::mat yqt = arma::quantile(yhat2, prob, 1);
    output2["yest"] = Rcpp::wrap(yqt);

    arma::cube rtmp = res.reshape(ntime + 1, nsample * nrep, 1);
    arma::mat res2 = rtmp.slice(0); // (ntime + 1) x (nsample * nrep)

    double rmse = std::sqrt(arma::mean(arma::mean(arma::pow(res2, 2))));
    double mae = arma::mean(arma::mean(res2));
    output2["res"] = Rcpp::wrap(res2);
    output2["rmse"] = rmse;
    output2["mae"] = mae;

    return output2;
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
    const bool &summarize = true,
    const bool &forecast_error = true,
    const bool &fitted_error = true,
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

        if (forecast_error)
        {
            Rcpp::List tmp = linear_bayes.forecast_error(model, y, 1000, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = linear_bayes.fitted_error(model, y, 1000, loss_func);
            error["fitted"] = tmp;
        }

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

        if (nforecast > 0)
        {
            forecast = mcs.forecast(model, y);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = mcs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = mcs.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }

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

        if (nforecast > 0)
        {
            forecast = ffbs.forecast(model, y);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = ffbs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }

        if (fitted_error)
        {
            Rcpp::List tmp = ffbs.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }

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

        if (forecast_error)
        {
            Rcpp::List tmp = tfs.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = tfs.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }

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

        if (forecast_error)
        {
            Rcpp::List tmp = pl.forecast_error(model, y, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = pl.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }

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

        if (nforecast > 0)
        {
            forecast = mcmc.forecast(model, y);
        }

        if (fitted_error)
        {
            Rcpp::List tmp = mcmc.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }
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

        if (nforecast > 0)
        {
            forecast = hvb.forecast(model, y);
        }

        if (fitted_error)
        {
            Rcpp::List tmp = hvb.fitted_error(model, y, loss_func);
            error["fitted"] = tmp;
        }
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
    if (nforecast > 0)
    {
        out["pred"] = forecast;
    }
    if (forecast_error || fitted_error)
    {
        out["error"] = error;
    }

    return out;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_forecast(
    const Rcpp::List &model_settings,
    const arma::vec &y,   // (nT + 1) x 1
    const arma::mat &psi, // (nT + 1) x nsample
    const arma::vec &W_stored,
    const double &mu0 = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector> &ycast_true = R_NilValue, // k x 1
    const std::string &loss_func = "quadratic",
    const unsigned int &k = 1, // k-step-ahead forecasting,
    const bool &verbose = false)
{
    Model model(model_settings);
    model.seas.X = Season::setX(y.n_elem - 1, model.seas.period, model.seas.P);

    arma::mat ycast = Model::forecast(
        y, psi, W_stored, model.dlag, model.seas,
        model.ftrans, model.flink, model.fgain, k); // k x nsample

    Rcpp::List out;
    out["ycast_all"] = Rcpp::wrap(ycast);

    if (ycast_true.isNotNull())
    {
        arma::vec y_loss(k, arma::fill::zeros);
        arma::vec y_covered = y_loss;
        arma::vec y_width = y_loss;

        arma::mat ycast_err = ycast;
        arma::vec ycast_true_arma = Rcpp::as<arma::vec>(ycast_true);
        ycast_err.each_col([&ycast_true_arma](arma::vec &col)
                           { col = arma::abs(col - ycast_true_arma); });

        for (unsigned int j = 0; j < k; j++)
        {
            arma::rowvec ysamples = ycast.row(j);
            double ymin = ysamples.min();
            double ymax = ysamples.max();
            double ytrue = ycast_true_arma.at(j);

            y_width.at(j) = std::abs(ymax - ymin);
            y_covered.at(j) = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;
        }

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            y_loss = arma::vectorise(arma::mean(ycast_err, 1)); // k x 1
            break;
        }
        case AVAIL::L2: // rmse
        {
            ycast_err = arma::square(ycast_err);
            y_loss = arma::vectorise(arma::mean(ycast_err, 1)); // k x 1
            y_loss = arma::sqrt(y_loss);
            break;
        }
        default:
        {
            break;
        }
        } // switch by loss

        out["y_loss"] = Rcpp::wrap(y_loss); // k x 1
        out["y_covered"] = Rcpp::wrap(y_covered);
        out["y_width"] = Rcpp::wrap(y_width);
    }

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



//' @export
// [[Rcpp::export]]
arma::mat dgtf_optimal_lag(
    const Rcpp::List &model_opts,
    const arma::vec &y,
    const std::string &algo_name,
    const Rcpp::List &algo_opts,
    const arma::vec &par1_grid,
    const arma::vec &par2_grid,
    const std::string &loss = "quadratic")
{
    unsigned int npar1 = par1_grid.n_elem;
    unsigned int npar2 = par2_grid.n_elem;
    unsigned int ntotal = npar1 * npar2;

    arma::mat stats(npar1 * npar2, 7, arma::fill::zeros);

    Model model(model_opts);
    model.seas.X = Season::setX(y.n_elem - 1, model.seas.period, model.seas.P);

    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;

    unsigned int idx = 0;

    for (unsigned int i = 0; i < npar1; i++)
    {
        Rcpp::checkUserInterrupt();

        double par1 = par1_grid.at(i);
        model.dlag.par1 = par1;

        for (unsigned int j = 0; j < npar2; j++)
        {
            Rcpp::checkUserInterrupt();

            stats.at(idx, 0) = par1;
            double par2 = par2_grid.at(j);
            stats.at(idx, 1) = par2;
            model.dlag.par2 = par2;
            model.dlag.Fphi = LagDist::get_Fphi(model.dlag);

            switch (lag_list[model.dlag.name])
            {
            case AVAIL::Dist::lognorm:
            {
                stats.at(idx, 2) = lognorm::mean(par1, par2);
                stats.at(idx, 3) = lognorm::var(par1, par2);
                stats.at(idx, 4) = lognorm::mode(par1, par2);
                break;
            }
            case AVAIL::Dist::nbinomp:
            {
                stats.at(idx, 2) = nbinom::mean(par1, par2);
                stats.at(idx, 3) = nbinom::var(par1, par2);
                stats.at(idx, 4) = nbinom::mode(par1, par2);
                break;
            }
            default:
                break;
            }

            double err_forecast = 0.;
            double err_fit = 0.;
            double cov_forecast = 0.;
            double width_forecast = 0.;

            switch (algo_list[algo_name])
            {
            case AVAIL::Algo::LinearBayes:
            {
                LBA::LinearBayes linear_bayes(algo_opts);

                try
                {
                    linear_bayes.filter(model, y);
                    linear_bayes.smoother(model, y);

                    linear_bayes.fitted_error(model, y, err_fit, 1000, loss);
                    linear_bayes.forecast_error(model, y, err_forecast, cov_forecast, width_forecast, 1000, loss);
                }
                catch (...)
                {
                    err_fit = NA_REAL;
                    err_forecast = NA_REAL;
                }

                break;
            } // case Linear Bayes
            case AVAIL::Algo::MCS:
            {
                SMC::MCS mcs(model, algo_opts);
                try
                {
                    mcs.infer(model, y);

                    mcs.fitted_error(err_fit, model, y, loss);
                    mcs.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);
                }
                catch (...)
                {
                    err_fit = NA_REAL;
                    err_forecast = NA_REAL;
                }
                break;
            } // case MCS
            case AVAIL::Algo::FFBS:
            {
                SMC::FFBS ffbs(model, algo_opts);
                ffbs.infer(model, y);

                ffbs.fitted_error(err_fit, model, y, loss);
                ffbs.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

                break;
            } // case FFBS
            case AVAIL::Algo::ParticleLearning:
            {
                SMC::PL pl(model, algo_opts);
                pl.infer(model, y);

                pl.fitted_error(err_fit, model, y, loss);
                pl.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

                break;
            } // case particle learning
            case AVAIL::Algo::MCMC:
            {
                MCMC::Disturbance mcmc(model, algo_opts);
                mcmc.infer(model, y);

                mcmc.fitted_error(err_fit, model, loss);
                // mcmc.forecast_error(err_forecast, model, loss);
                break;
            }
            case AVAIL::Algo::HybridVariation:
            {
                VB::Hybrid hvb(model, algo_opts);

                try
                {
                    hvb.infer(model, y);

                    hvb.fitted_error(err_fit, model, loss);
                    // hvb.forecast_error(err_forecast, model, loss);
                }
                catch (...)
                {
                    err_fit = NA_REAL;
                    err_forecast = NA_REAL;
                }
                break;
            }
            default:
            {
                throw std::invalid_argument("Unknown algorithm " + algo_name);
                break;
            }
            } // switch by algorithm

            stats.at(idx, 5) = err_forecast;
            stats.at(idx, 6) = width_forecast;

            Rcpp::Rcout << "\rProgress: " << idx + 1 << "/" << ntotal;
            idx++;

        } // loop over par2
    } // loop over par1

    Rcpp::Rcout << std::endl;

    return stats;
}

//' @export
// [[Rcpp::export]]
arma::mat dgtf_optimal_obs(
    const Rcpp::List &model_opts,
    const arma::vec &y,
    const std::string &algo_name,
    const Rcpp::List &algo_opts,
    const arma::vec &delta_grid,
    const std::string &loss = "quadratic")
{
    unsigned int npar = delta_grid.n_elem;

    arma::mat stats(npar, 3);
    Model model(model_opts);
    model.seas.X = Season::setX(y.n_elem - 1, model.seas.period, model.seas.P);

    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;

    for (unsigned int i = 0; i < npar; i++)
    {
        Rcpp::checkUserInterrupt();

        double delta = delta_grid.at(i);
        model.dobs.par2 = delta;
        stats.at(i, 0) = delta;

        double err_forecast = 0.;
        // double err_fit = 0.;
        double cov_forecast = 0.;
        double width_forecast = 0.;

        switch (algo_list[algo_name])
        {
        case AVAIL::Algo::LinearBayes:
        {
            LBA::LinearBayes linear_bayes(algo_opts);

            try
            {
                linear_bayes.filter(model, y);
                linear_bayes.smoother(model, y);

                // linear_bayes.fitted_error(err_fit, 1000, loss);
                linear_bayes.forecast_error(model, y, err_forecast, cov_forecast, width_forecast, 1000, loss);
            }
            catch (...)
            {
                // err_fit = NA_REAL;
                err_forecast = NA_REAL;
            }

            break;
        } // case Linear Bayes
        case AVAIL::Algo::MCS:
        {
            SMC::MCS mcs(model, algo_opts);
            mcs.infer(model, y);

            // mcs.fitted_error(err_fit, model, loss);
            mcs.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

            break;
        } // case MCS
        case AVAIL::Algo::FFBS:
        {
            SMC::FFBS ffbs(model, algo_opts);
            ffbs.infer(model, y);

            // ffbs.fitted_error(err_fit, model, loss);
            ffbs.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

            break;
        } // case FFBS
        case AVAIL::Algo::ParticleLearning:
        {
            SMC::PL pl(model, algo_opts);
            pl.infer(model, y);

            // pl.fitted_error(err_fit, model, loss);
            pl.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

            break;
        } // case particle learning
        case AVAIL::Algo::MCMC:
        {
            MCMC::Disturbance mcmc(model, algo_opts);
            mcmc.infer(model, y);

            // mcmc.fitted_error(err_fit, model, loss);
            // mcmc.forecast_error(err_forecast, model, loss);
            break;
        }
        case AVAIL::Algo::HybridVariation:
        {
            VB::Hybrid hvb(model, algo_opts);
            hvb.infer(model, y);

            // hvb.fitted_error(err_fit, model, loss);
            // hvb.forecast_error(err_forecast, model, loss);
            break;
        }
        default:
        {
            throw std::invalid_argument("Unknown algorithm " + algo_name);
            break;
        }
        } // switch by algorithm

        stats.at(i, 1) = err_forecast;
        // stats.at(i, 2) = err_fit;
        stats.at(i, 2) = width_forecast;

        Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << npar;

    } // loop over par1

    Rcpp::Rcout << std::endl;

    return stats;
}
