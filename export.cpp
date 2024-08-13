#include <chrono>
#include "Model.hpp"
#include "LinearBayes.hpp"
#include "SequentialMonteCarlo.hpp"
#include "MCMC.hpp"
#include "VariationalBayes.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr, BH)]]

//' @export
// [[Rcpp::export]]
arma::vec dlognorm(const unsigned int &nlag, const double &mu, const double &sd2)
{
    return lognorm::dlognorm(nlag, mu, sd2);
}


//' @export
// [[Rcpp::export]]
arma::vec dnbinom(const unsigned int &nlag, const double &kappa, const double &r)
{
    return nbinom::dnbinom(nlag, kappa, r);
}

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
Rcpp::List dgtf_default_algo_settings(const std::string &method)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string method_name = tolower(method);

    Rcpp::List opts = SMC::SequentialMonteCarlo::default_settings();
    switch (algo_list[method_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        opts =  LBA::LinearBayes::default_settings();
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
Rcpp::List dgtf_model(
    const std::string &obs_dist = "nbinom",
    const std::string &link_func = "identity",
    const std::string &trans_func = "sliding",
    const std::string &gain_func = "softplus",
    const std::string &lag_dist = "lognorm",
    const std::string &err_dist = "gaussian",
    const unsigned int &nlag = 10,
    const unsigned int &ntime = 200,
    const bool &truncated = true,
    const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),  // (mu0, delta_nb)
    const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(1.4, 0.3), // (kappa, r) or (mu, sd2)
    const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.)  // (W, w[0]))
)
{
    // Consolidate the model definitions into a Rcpp list.s
    Rcpp::List model;
    model["obs_dist"] = tolower(obs_dist);
    model["link_func"] = tolower(link_func);
    model["trans_func"] = tolower(trans_func);
    model["gain_func"] = tolower(gain_func);
    model["lag_dist"] = tolower(lag_dist);
    model["err_dist"] = tolower(err_dist);

    Rcpp::List dim;
    dim["nlag"] = nlag;
    dim["ntime"] = ntime;
    dim["truncated"] = truncated;

    Rcpp::List param;
    param["obs"] = obs_param;
    param["lag"] = lag_param;
    param["truncated"] = truncated;

    Rcpp::List settings;
    settings["model"] = model;
    settings["dim"] = dim;
    settings["param"] = param;

    return settings;
}





//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_simulate(
    const Rcpp::List &settings, 
    const std::string &sim_algo = "transfunc",
    const double &y0 = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector> theta0 = R_NilValue)
{
    std::string sim_method = tolower(sim_algo);
    enum method {
        TransFunc,
        StateSpace
    };

    std::map<std::string, method> map;
    map["transfunc"] = method::TransFunc;
    map["transfer"] = method::TransFunc;
    map["trans_func"] = method::TransFunc;
    map["tf"] = method::TransFunc;

    map["statespace"] = method::StateSpace;
    map["state_space"] = method::StateSpace;
    map["ss"] = method::StateSpace;

    Rcpp::List output = settings;
    Model model(settings);

    switch (map[sim_algo])
    {
    case method::TransFunc:
    {
        arma::vec psi, lambda, y;
        Model::simulate(y, lambda, psi, model, y0);

        output["y"] = Rcpp::wrap(y);
        output["psi"] = Rcpp::wrap(psi);
        // output["wt"] = Rcpp::wrap(wt);
        output["lambda"] = Rcpp::wrap(model.lambda);
        break;
    }
    case method::StateSpace:
    {
        output = StateSpace::simulate(model, y0, theta0);
        break;
    }
    default:
    {
        break;
    }
    }
    
    
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

    unsigned int nforecast = 0;
    if (method_settings.containsElementNamed("num_step_ahead_forecast"))
    {
        nforecast = Rcpp::as<unsigned int>(method_settings["num_step_ahead_forecast"]);
    }

    // arma::vec ypad(model.dim.nT + 1, arma::fill::zeros);
    // ypad.tail(y.n_elem) = y;

    Rcpp::List output, forecast, error;
    arma::mat psi(model.dim.nT + 1, 3);
    arma::vec ci_prob = {0.025, 0.5, 0.975};

    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        LBA::LinearBayes linear_bayes(model, y);
        linear_bayes.init(method_settings);

        auto start = std::chrono::high_resolution_clock::now();
        linear_bayes.filter();
        linear_bayes.smoother();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;
        
         output = linear_bayes.get_output();

        if (forecast_error)
        {
            Rcpp::List tmp = linear_bayes.forecast_error(1000, loss_func, k, tstart_forecast, tend_forecast);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = linear_bayes.fitted_error(1000, loss_func);
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
    }// switch by algorithm


    Rcpp::List out;
    if (nforecast > 0 || forecast_error || fitted_error)
    {
        out["fit"] = output;
        if (nforecast > 0) { out["pred"] = forecast; }
        if (forecast_error || fitted_error)
        {
            out["error"] = error;
        }
    }
    else
    {
        out = output;
    }


    return out;
}


//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_forecast(
    const Rcpp::List &model_settings,
    const arma::vec &y, // (nT + 1) x 1
    const arma::mat &psi, // (nT + 1) x nsample
    const arma::vec &W_stored,
    const double &mu0 = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector> &ycast_true = R_NilValue, // k x 1
    const std::string &loss_func = "quadratic",
    const unsigned int &k = 1, // k-step-ahead forecasting,
    const bool &verbose = false
)
{
    Model model(model_settings);
    arma::mat ycast = Model::forecast(
        y, psi, W_stored, model.dim, model.dlag,
        model.transfer, 
        model.flink, model.fgain, mu0, k); // k x nsample

    Rcpp::List out;
    out["ycast_all"] = Rcpp::wrap(ycast);

    
    if (ycast_true.isNotNull())
    {
        arma::vec y_loss(k, arma::fill::zeros);
        arma::vec y_covered = y_loss;
        arma::vec y_width = y_loss;

        arma::mat ycast_err = ycast;
        arma::vec ycast_true_arma = Rcpp::as<arma::vec>(ycast_true);
        ycast_err.each_col([&ycast_true_arma](arma::vec &col) {
            col = arma::abs(col - ycast_true_arma);
        });

        for (unsigned int j = 0; j < k; j ++)
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
            // psi_tmp = arma::mean(psi_loss_tmp, 1); // // (nT - i) x 1
            // psi_loss.submat(1, i, model.dim.nT - i - 1, i) = psi_tmp;
            // psi_loss_all.at(i) = arma::mean(psi_tmp);

            y_loss = arma::vectorise(arma::mean(ycast_err, 1)); // k x 1
            break;
        }
        case AVAIL::L2: // rmse
        {
            // psi_loss_tmp = arma::square(psi_loss_tmp); // (nT - i) x nsample

            // psi_tmp = arma::mean(psi_loss_tmp, 1); // (nT - i) x 1
            // psi_loss.submat(1, i, model.dim.nT - i - 1, i) = arma::sqrt(psi_tmp);

            // psi_loss_all.at(i) = arma::mean(psi_tmp);
            // psi_loss_all.at(i) = std::sqrt(psi_loss_all.at(i));
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

// //' @export
// // [[Rcpp::export]]
// Rcpp::List dgtf_post_predictive_check(
//     const Rcpp::List &model_settings,
//     const arma::vec &y,   // (nT + 1) x 1
//     const std::string &method, 
//     const Rcpp::List &method_settings,
//     const std::string &loss_func = "quadratic",
//     const unsigned int &k = 1)
// {
//     Model model(model_settings);
//     Rcpp::List forecast = Model::forecast_error(psi, y, model, loss_func, k);
//     Rcpp::List fitted = Model::fitted_error(psi, y, model, loss_func);

//     Rcpp::List out;
//     out["forecast"] = forecast;
//     out["fitted"] = fitted;

//     return out;
// }

//' @export
// [[Rcpp::export]]
arma::mat dgtf_tuning(
    const Rcpp::List &model_opts,
    const arma::vec &y,
    const std::string &algo_name,
    const Rcpp::List &algo_opts,
    const std::string &tuning_param,
    const double &from = 0.,
    const double &to = 0.,
    const double &delta = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector> &grid = R_NilValue,
    const std::string &loss = "quadratic"
)
{
    Model model(model_opts);

    arma::vec param_grid;
    if (!grid.isNull())
    {
        param_grid = Rcpp::as<arma::vec>(grid);
    }

    std::map<std::string, AVAIL::Param> tuning_param_list = AVAIL::tuning_param_list;
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;

    arma::mat stats;

    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        LBA::LinearBayes linear_bayes(model, y);
        linear_bayes.init(algo_opts);

        switch (tuning_param_list[tuning_param])
        {
        case AVAIL::Param::discount_factor:
        {
            stats = linear_bayes.optimal_discount_factor(from, to, delta, loss);
            break;
        }
        case AVAIL::Param::W:
        {
            stats = linear_bayes.optimal_W(param_grid, loss);
            break;
        }
        default:
        {
            throw std::invalid_argument(algo_name + " doesn't have tuning parameter " + tuning_param);
        }
        } // switch by tuning parameter
        

        break;
    } // case Linear Bayes
    case AVAIL::Algo::MCS:
    {
        SMC::MCS mcs(model, algo_opts);

        switch (tuning_param_list[tuning_param])
        {
        case AVAIL::Param::discount_factor:
        {
            stats = mcs.optimal_discount_factor(model, y, from, to, delta, loss);
            break;
        }
        case AVAIL::Param::W:
        {
            stats = mcs.optimal_W(model, param_grid, loss);
            break;
        }
        case AVAIL::Param::num_backward:
        {
            stats = mcs.optimal_num_backward(model, y, from, to, delta, loss);
            break;
        }
        default:
        {
            throw std::invalid_argument(algo_name + " doesn't have tuning parameter " + tuning_param);
        }
        } // switch by tuning parameter

        break;
    } // case MCS
    case AVAIL::Algo::FFBS:
    {
        SMC::FFBS ffbs(model, algo_opts);
        switch (tuning_param_list[tuning_param])
        {
        case AVAIL::Param::discount_factor:
        {
            stats = ffbs.optimal_discount_factor(model, y, from, to, delta, loss);
            break;
        }
        case AVAIL::Param::W:
        {
            stats = ffbs.optimal_W(model, y, param_grid, loss);
            break;
        }
        default:
        {
            throw std::invalid_argument(algo_name + " doesn't have tuning parameter " + tuning_param);
        }
        } // switch by tuning parameter

        break;
    } // case FFBS
    case AVAIL::Algo::ParticleLearning:
    {
        SMC::PL pl(model, algo_opts);
        break;
    } // case particle learning
    case AVAIL::Algo::MCMC:
    {
        MCMC::Disturbance mcmc(model, algo_opts);        
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        VB::Hybrid hva(model, algo_opts);
        hva.infer(model, y);

        switch (tuning_param_list[tuning_param])
        {
        case AVAIL::Param::learning_rate:
        {
            stats = hva.optimal_learning_rate(model, y, from, to, delta, loss);
            break;
        }
        case AVAIL::Param::step_size:
        {
            stats = hva.optimal_step_size(model, param_grid, loss);
            break;
        }
        case AVAIL::Param::num_backward:
        {
            stats = hva.optimal_num_backward(model, y, from, to, delta, loss);
            break;
        }
        default:
        {
            throw std::invalid_argument(algo_name + " doesn't have tuning parameter " + tuning_param);
        }
        } // switch by tuning parameter

        break;
    }
    default:
    {
        throw std::invalid_argument("Unknown algorithm " + algo_name);
        break;
    }
    } // switch by algorithm


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
    const std::string &loss = "quadratic"
)
{
    unsigned int npar1 = par1_grid.n_elem;
    unsigned int npar2 = par2_grid.n_elem;
    unsigned int ntotal = npar1 * npar2;

    arma::mat stats(npar1 * npar2, 7, arma::fill::zeros);

    Model model(model_opts);


    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;

    unsigned int idx = 0;

    for (unsigned int i = 0; i < npar1; i ++)
    {
        Rcpp::checkUserInterrupt();
        
        double par1 = par1_grid.at(i);
        model.dlag.update_par1(par1);
        

        for (unsigned int j = 0; j < npar2; j ++)
        {
            Rcpp::checkUserInterrupt();

            stats.at(idx, 0) = par1;
            double par2 = par2_grid.at(j);
            stats.at(idx, 1) = par2;
            model.dlag.update_par2(par2);
            model.dlag.get_Fphi(model.dim.nL);


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
                LBA::LinearBayes linear_bayes(model, y);
                linear_bayes.init(algo_opts);

                try
                {
                    linear_bayes.filter();
                    linear_bayes.smoother();

                    linear_bayes.fitted_error(err_fit, 1000, loss);
                    linear_bayes.forecast_error(err_forecast, cov_forecast, width_forecast, 1000, loss);
                }
                catch(...)
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
                catch(...)
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
                catch(...)
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

    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;

    for (unsigned int i = 0; i < npar; i++)
    {
        Rcpp::checkUserInterrupt();

        double delta = delta_grid.at(i);
        model.dobs.update_par2(delta);
        stats.at(i, 0) = delta;

        double err_forecast = 0.;
        // double err_fit = 0.;
        double cov_forecast = 0.;
        double width_forecast = 0.;

        switch (algo_list[algo_name])
        {
        case AVAIL::Algo::LinearBayes:
        {
            LBA::LinearBayes linear_bayes(model, y);
            linear_bayes.init(algo_opts);

            try
            {
                linear_bayes.filter();
                linear_bayes.smoother();

                // linear_bayes.fitted_error(err_fit, 1000, loss);
                linear_bayes.forecast_error(err_forecast, cov_forecast, width_forecast, 1000, loss);
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

    }     // loop over par1

    Rcpp::Rcout << std::endl;

    return stats;
}

//' @export
// [[Rcpp::export]]
arma::mat dgtf_optimal_nlag(
    const Rcpp::List &model_opts,
    const arma::vec &y,
    const std::string &algo_name,
    const Rcpp::List &algo_opts,
    const arma::uvec &nlag_grid,
    const std::string &loss = "quadratic")
{
    unsigned int npar = nlag_grid.n_elem;
    Rcpp::List opts = model_opts;
    Rcpp::List dim_opts = Rcpp::as<Rcpp::List>(opts["dim"]);

    arma::mat stats(npar, 3);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;

    for (unsigned int i = 0; i < npar; i++)
    {
        Rcpp::checkUserInterrupt();

        unsigned int nlag = nlag_grid.at(i);
        dim_opts["nlag"] = nlag;
        opts["dim"] = dim_opts;
        Model model(opts);

        stats.at(i, 0) = nlag;

        double err_forecast = 0.;
        double err_fit = 0.;
        double cov_forecast = 0.;
        double width_forecast = 0.;

        switch (algo_list[algo_name])
        {
        case AVAIL::Algo::LinearBayes:
        {
            LBA::LinearBayes linear_bayes(model, y);
            linear_bayes.init(algo_opts);

            try
            {
                linear_bayes.filter();
                linear_bayes.smoother();

                linear_bayes.fitted_error(err_fit, 1000, loss);
                linear_bayes.forecast_error(err_forecast, cov_forecast, width_forecast, 1000, loss);
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
            mcs.infer(model, y);

            mcs.fitted_error(err_fit, model, y, loss);
            mcs.forecast_error(err_forecast, cov_forecast, width_forecast, model, y, loss);

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
            hvb.infer(model, y);

            hvb.fitted_error(err_fit, model, loss);
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
        stats.at(i, 2) = width_forecast;

        Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << npar;

    } // loop over par1

    Rcpp::Rcout << std::endl;

    return stats;
}

// [[Rcpp::export]]
arma::vec rmvnorm_arma_solve(const arma::mat &precision, const arma::vec &location, const arma::vec &scaled_mu)
{

    arma::vec epsilon = arma::randn(precision.n_rows);
    arma::mat precision_chol = arma::chol(precision);
    // arma::vec scaled_mu = arma::solve(arma::trimatu(precision_chol.t()), location);
    arma::vec draw = arma::solve(arma::trimatu(precision_chol), epsilon + scaled_mu);

    return draw;
}

// [[Rcpp::export]]
arma::vec rmvnorm_arma_inv(const arma::mat &precision, const arma::vec &location)
{

    arma::vec epsilon = arma::randn(precision.n_rows);
    arma::mat precision_chol = arma::chol(precision);
    arma::mat precision_chol_inv = arma::inv(arma::trimatu(precision_chol));
    arma::vec draw = precision_chol_inv * (precision_chol_inv.t() * location + epsilon);

    return draw;
}


