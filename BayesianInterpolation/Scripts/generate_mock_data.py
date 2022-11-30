import os
import nifty8 as ift
import numpy as np
import payplot as pl
import sys
import importlib

from Functions.helpers import load_field_model
from Operators.SkyProjector import SkyProjector
from Operators.PlaneProjector import PlaneProjector
from Operators.IVG import InverseGammaOperator


def main(data_name, n_data, do_plot, noise_sigma, # general setup
         domain_parameters,  # domain parameters,
         extragal_model, amplitude_params, sign_params, extragal_params,  # model parameter dictionaries

         ):

    # Setting up directories if necessary
    data_path = './Data/' + data_name + '/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Construct the data and sky domains, i.e. the spaces on which the fields live and connet them via response operator

    data_domain = ift.makeDomain(ift.UnstructuredDomain((len(n_data),)))

    if domain_parameters['full_sphere']:
        # Us healpix discretisatoion for full sphere
        if domain_parameters['nside'] is None:
            raise ValueError('Nside needs to be provided for full sphere')
        signal_domain = ift.makeDomain(ift.HPSpace(domain_parameters['nside']))
        phi = 2*np.pi * np.random.uniform(0, 1, n_data)
        theta = np.arccos(np.random.uniform(-1, 1, n_data))  # arccos corrects volume
        response = SkyProjector(theta=theta, phi=phi, target=data_domain, domain=signal_domain)
    else:
        # rectanglar grid (i.e flat sky approximation!) for coutout
        nlon = domain_parameters['nlon']
        dlon = domain_parameters['dlon']
        nlat = domain_parameters['nlat']
        dlat = domain_parameters['dlat']
        center = domain_parameters['center']
        signal_domain = ift.makeDomain(ift.RGSpace(shape=(nlon, nlat), distances=(dlon, dlat)))
        lon = center[0] + nlon*dlon*np.random.uniform(0, 1, n_data) - nlon*dlon/2
        lat = center[1] + nlat*dlat*np.random.uniform(0, 1, n_data) - nlat*dlat/2
        response = PlaneProjector(theta=lat, phi=lon, target=data_domain, domain=signal_domain, center=center)


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

    # calculate the mocksk from a random sample

    mock_sky = faraday_sky(ift.from_random(faraday_sky.domain))

    # calculate the galactic contrbution to the data
    gal_rm = response(mock_sky)

    # We have three option implement the extraglactic component

    if extragal_model == 'UnivariateGaussian':
        # Option A:
        extragal_rm = extragal_params['extragal_sigma']*ift.from_random(data_domain)
        full_rm = gal_rm + extragal_rm

    elif extragal_model == 'MultivariateGaussian':
        # Option B:
        ivg = InverseGammaOperator(data_domain, alpha=extragal_params['alpha'], q=extragal_params['q'])
        eta = ivg(ift.from_random(data_domain))
        extragal_sigma = extragal_params['extragal_sigma']*eta
        extragal_rm = extragal_sigma*ift.from_random(data_domain)
        full_rm = gal_rm + extragal_rm

    elif extragal_model is None:
        extragal_sigma = None
        extragal_rm = None
        full_rm = gal_rm
    else:
        raise KeyError('Unkown extragalactic RM strategy (allowed are explicit, implicit or None)')

    noise = noise_sigma*ift.from_random(data_domain)

    data = full_rm + noise

    if do_plot:
        pl.figure()
        pl.scatter(gal_rm.val, data.val)
        pl.xlabel('truth')
        pl.ylabel('data')
        pl.savefig(data_path + 'truth_data_scatter.png')

    np.save(data_path + 'data', data.val)
    np.save(data_path + 'noise_sigma', noise_sigma)
    np.save(data_path + 'extragal_sigma', extragal_sigma)
    np.save(data_path + 'noise', noise)
    np.save(data_path + 'extragal_rm', extragal_rm)
    np.save(data_path + 'gal_rm', gal_rm)
    np.save(data_path + 'rm_sky', mock_sky)
    np.save(data_path + 'theta', theta)
    np.save(data_path + 'phi', phi)


if __name__ == '__main__':
    paramfile_name = str(sys.argv[1])

    pm = importlib.import_module('.' + paramfile_name, 'Parameters')
    run_parameter_dict = getattr(pm, 'run_parameter_dict')
    hyper_parameter_dict = getattr(pm, 'hyper_parameter_dict')

    main(run_parameter_dict, hyper_parameter_dict)
