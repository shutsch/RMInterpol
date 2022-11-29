import os
import healpy as hp
import pandas as pd
import nifty7 as ift
import numpy as np
import scipy.stats as st

from Functions.data import get_real_data, get_mock_data
from Functions.plot import data_and_prior_plot, progress_plot
from Functions.helpers import load_field_model
from Operators.SkyProjector import SkyProjector


def main(data_name, n_data, # general setup
         full_sphere, nside, nlat, nlon, dlat, dlon, center,# domain parameters,
         extragal_model, amplitude_params, sign_params, extragal_params, # model parameter dictionaries

         ):

    # Setting up directories if necessary
    data_path = './Runs/results/' + data_name
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Construct the signal domains, i.e. the spaces on which the fields live
    if full_sphere:
        # Us healpix discretisatoion for full sphere
        if nside is None:
            raise ValueError('Nside needs to be provided for full sphere')
        signal_domain = ift.makeDomain(ift.HPSpace(nside))
    else:
        # rectanglar grid (i.e flat sky approximation!) for coutout
        signal_domain = ift.makeDomain(ift.RGSpace(shape=(nlon, nlat), distances=(dlon, dlat)))


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

    # generate positions

    # nifty conversion
    data_domain = ift.makeDomain(ift.UnstructuredDomain((len(n_data),)))
    data = ift.Field(data_domain, data)
    error = ift.Field(data_domain, error)

    # construct a projection operator connecting the faraday sky to data space, depending on if we are on the full sky or on a coutout
    if full_sphere:
        response = SkyProjector(theta=theta, phi=phi, target=data_domain, domain=signal_domain)
    else:
        response = PlaneProjector(theta=theta, phi=phi, target=data_domain, domain=signal_domain, center=center)

    # Connect the response to the sky
    psky = response @ faraday_sky

    # input for the likelihood function
    noise_obs = ift.makeOp(error ** 2)
    residual = ift.Adder(data) @ (-psky)

    # We have three options to deal with the extraglactic component, all of which imply a different likelihood

    if extragal_model == 'Gaussian':
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
    elif extragal_model == 'StudentT':
        # Option B: Marginalize over the parameter eta introduced above, which modifies the Gaussian to a StudentT
        # likelihood. I can provide the analytic calcuation that justifies this if needed
        theta = 2*extragal_params['alpha']
        factor = np.sqrt(2*extragal_params['q']/theta)
        inv_stddev = ift.makeOp((factor * error)).inverse
        likelihood = ift.StudentTEnergy(domain=data_domain, theta=theta)(inv_stddev @ residual)
    elif extragal_model is None:
        # Option C: Ignore the extragalactic component and fit the sky anyhow
        likelihood = ift.GaussianEnergy(domain=data_domain,
                                        inverse_covariance=noise_obs.inverse,
                                        sampling_dtype=np.float64)(residual)
    else:
        raise KeyError('Unkown extragalactic RM strategy (allowed are explicit, implicit or None)')



if __name__ == '__main__':
    paramfile_name = str(sys.argv[1])

    pm = importlib.import_module('.' + paramfile_name, 'Parameters')
    run_parameter_dict = getattr(pm, 'run_parameter_dict')
    hyper_parameter_dict = getattr(pm, 'hyper_parameter_dict')

    main(run_parameter_dict, hyper_parameter_dict)
