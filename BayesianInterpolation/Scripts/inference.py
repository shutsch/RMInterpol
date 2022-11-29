import os
import sys
import importlib
import nifty8 as ift
import numpy as np
import scipy.stats as st

from Functions.data import get_mock_data, get_real_data
from Functions.plot import data_and_prior_plot, progress_plot, scatter
from Functions.helpers import load_field_model
from Operators.SkyProjector import SkyProjector
from Operators.PlaneProjector import PlaneProjector
from Operators.IVG import InverseGammaOperator


def main(run_name, use_mock_data, do_plot, full_sphere, seed, n_iterations,   # general setup
         domain_parameters, amplitude_params, sign_params, extragal_params,  # model parameter dictionaries

         ):

    ift.random.push_sseq_from_seed(seed)
    # Setting up directories if necessary
    plotting_path = './Runs/plots/' + run_name
    result_path = './Runs/results/' + run_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Construct the signal domains, i.e. the spaces on which the fields live
    if full_sphere:
        # Us healpix discretisation for full sphere
        signal_domain = ift.makeDomain(ift.HPSpace(domain_parameters['nside']))
    else:
        # rectanglar grid (i.e flat sky approximation!) for coutout
        signal_domain = ift.makeDomain(ift.RGSpace(shape=domain_parameters['shape'], distances=domain_parameters['distances']))


    # initiliaze some dictionaries to collect stuff for plotting
    noise_estimate_dict = {}
    data_dict = {}
    data_adjoint_dict = {}
    model_dict = {}
    power_dict = {}

    #################### BUILDING THE MODELS ##############

    # Setting up the sky models.

    # First, the amplitude sky
    log_amplitude, log_amplitude_power = load_field_model('log_amplitude', signal_domain, amplitude_params)
    amplitude_sky = (log_amplitude.clip(None, 30)).exp()

    # Second, the sign sky
    sign, b_par_power = load_field_model('b_par', signal_domain, sign_params)
    faraday_sky = amplitude_sky*sign

    model_dict.update({'faraday_sky': faraday_sky, 'b_par': sign, 'amplitude': amplitude_sky, 'log_dm': log_amplitude})
    power_dict.update({'b_par': b_par_power, 'log_amplitude': log_amplitude_power})

    # load data

    if use_mock_data:
        theta, phi, data, error = get_mock_data()
    else:
        theta, phi, data, error = get_real_data()

    # nifty conversion
    data_domain = ift.makeDomain(ift.UnstructuredDomain((len(data),)))
    data = ift.Field(data_domain, data)
    error = ift.Field(data_domain, error)

    # construct a projection operator connecting the faraday sky to data space, depending on if we are on the full sky or on a coutout
    if full_sphere:
        response = SkyProjector(theta=theta, phi=phi, target=data_domain, domain=signal_domain)
    else:
        response = PlaneProjector(theta=theta, phi=phi, target=data_domain, domain=signal_domain, center=domain_parameters['center'])

    # Connect the response to the sky
    psky = response @ faraday_sky

    # input for the likelihood function
    noise_obs = ift.makeOp(error ** 2)
    residual = ift.Adder(data) @ (-psky)

    # We have three options to deal with the extraglactic component, all of which imply a different likelihood

    if extragal_params['type'] == 'explicit':
        # Option A: Introduce additional parameters eta, which modify the observational noise to account for the
        # additional systematics.
        # eta is assumed to be inverse gamma distributed.
        eta = InverseGammaOperator(data_domain, alpha=extragal_params['alpha'], q=extragal_params['q']) @ \
            ift.FieldAdapter(data_domain, 'noise_excitations')
        noise_estimate = noise_obs @ eta
        # Some nifty ducktape
        new_dom = ift.MultiDomain.make({'icov': noise_estimate.target, 'rm_residual': residual.target})
        nres = ift.FieldAdapter(new_dom, 'rm_icov')(noise_estimate ** (-1)) + \
            ift.FieldAdapter(new_dom, 'rm_residual')(residual)

        noise_estimate_dict.update({'rm': noise_estimate, })
        # Construct likelihood, a Gaussian with a variable noise term
        likelihood = ift.VariableCovarianceGaussianEnergy(domain=data_domain,
                                                          residual_key='rm_residual',
                                                          inverse_covariance_key='rm_icov',
                                                          sampling_dtype=np.dtype(np.float64))(nres)
    elif extragal_params['type'] == 'implicit':
        # Option B: Marginalize over the parameter eta introduced above, which modifies the Gaussian to a StudentT
        # likelihood. I can provide the analytic calcuation that justifies this if needed
        theta = 2*extragal_params['alpha']
        factor = np.sqrt(2*extragal_params['q']/theta)
        inv_stddev = ift.makeOp((factor * error)).inverse
        likelihood = ift.StudentTEnergy(domain=data_domain, theta=theta)(inv_stddev @ residual)
    elif extragal_params['type'] is None:
        # Option C: Ignore the extragalactic component and fit the sky anyhow
        likelihood = ift.GaussianEnergy(domain=data_domain,
                                        inverse_covariance=noise_obs.inverse,
                                        sampling_dtype=np.float64)(residual)
    else:
        raise KeyError('Unkown extragalactic RM strategy (allowed are explicit, implicit or None)')


    #################  Inference  ######################
    # The inference must start at an initial position in the latent space, which we draw randomly.
    # The only exception is the noise model, which set such that eta is one in the beginning (i.e. we start with the observed errors).

    initial_position = {}
    for k, v in likelihood.domain.items():
        if k in ['noise_excitations']:
            initial_position.update({k: ift.full(v, st.norm.ppf(st.invgamma.cdf(1., extragal_params['alpha'], scale=extragal_params['q'])))})
        else:
            initial_position.update({k: 0.1*ift.from_random(v, 'normal')})

    position = ift.MultiField.from_dict(initial_position)

    def number_of_samples(i):
        nsamps = 6 * [2, ] + 3 * [4, ] + 3 * [8, ] + 2 * [12, ] + 2 * [16, ] + [20, ] + [30, ]
        return nsamps[i] if i < len(nsamps) else 30

    def interation_limit_sampling(i):
        il = 20 * [100, ] + 2 * [150, ] + 2 * [200, ] + [250, ] + [400, ]
        return il[i] if i < len(il) else 500

    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=1.e-6, iteration_limit=25)
    minimizer = ift.NewtonCG(ic_newton)

    if do_plot:
        data_adjoint_dict = {'data': response.adjoint(data), 'error': response.adjoint(error),
                             'angular_position': response.adjoint(ift.full(data, 1.))/response.counts()}
        data_and_prior_plot(plotting_path, data_adjoint_dict, data_dict, model_dict, 10)

    ic_newton_nonlin = ift.AbsDeltaEnergyController(name='Newton nonlinear', deltaE=1.e-6, iteration_limit=20)
    nonlinear_minimizer = ift.NewtonCG(ic_newton_nonlin)

    for i in range(0, n_iterations):
        print('Starting global iteration #' + str(i))
        ic_sampling = ift.AbsDeltaEnergyController(deltaE=1e-7,
                                                   iteration_limit=interation_limit_sampling(i))
        H = ift.StandardHamiltonian(likelihood, ic_sampling)
        kl = ift.MetricGaussianKL.make(position, H, number_of_samples(i), nonlinear_minimizer, mirror_samples=True)

        if i == 0 and do_plot:
            progress_plot(plotting_path, kl, model_dict, power_dict, noise_estimate_dict, error.val, 'initial')

        kl, convergence = minimizer(kl)
        position = kl.position
        if do_plot:
            progress_plot(plotting_path, kl, model_dict, power_dict, noise_estimate_dict, error.val, i)
        for name, sky in model_dict.items():
            sc = ift.StatCalculator()
            for sample in kl.samples:
                sc.add(sky.force(sample + kl.position))
            np.save(result_path + '/' + name + '_mean', sc.mean.val)
            np.save(result_path + '/' + name + '_std', sc.mean.val)
        ift.extra.minisanity(data, lambda x: noise_obs.inverse, faraday_sky, kl.position, kl.samples)


if __name__ == '__main__':
    paramfile_name = str(sys.argv[1])

    pm = importlib.import_module('.' + paramfile_name, 'Parameters')
    run_parameter_dict = getattr(pm, 'run_params')
    domain_dict = getattr(pm, 'domain_params')
    amp_dict = getattr(pm, 'amplitude_params')
    sign_dict = getattr(pm, 'sign_params')
    egal_dict = getattr(pm, 'extragal_params')
    main(**run_parameter_dict, domain_parameters=domain_dict, amplitude_params=amp_dict, sign_params=sign_dict,
         extragal_params=egal_dict, )
