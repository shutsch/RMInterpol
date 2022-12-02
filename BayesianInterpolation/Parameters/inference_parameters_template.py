run_params = {
    'seed': 534634,  # random seed fixed for reproducability, can be set to None
    'run_name': 'test',  # The results will be saved in ./Runs/plots/<run_name>/  and  ./Runs/results/<run_name>/, respectively.
    'use_mock_data': True,
    'data_name': 'test',  # data name, will be given to the data loading routine
    'do_plot': True,
    'n_iterations': 10  # number of global iterations (i.e resampling steps). The large and the more non-linear the proble, the more you need
}

domain_params = {
    'full_sphere': False,
    # only necessary for full sky
    'nside': 128,
    # only necessary for cutout
    # note that it is necessary to cover some padding area around the cutout, as the nifty models assume periodic boundary conditions
    # thats why the mock data script covers a smaller area than here
    'nx': 150,  # number of pixels in x
    'ny': 150,  # number of pixels in y
    'dx': .05,  # resolution in x
    'dy': .05,  # resolution in x
    'center': [0, 0],  # coordinate of the central pixels on the sky
}

# Model parameters

# Parameters for the log-amplitude sky map
amplitude_params = {# prior parameters on the shape of the power spectrum ([mean, std])
                    'asperity': [.001, 0.001], 'flexibility': [.001, 0.001], 'fluctuations': [2., .5], 'loglogavgslope': [-4., .5],
                    # Parameters on the mean of the log amplitude field
                    'offset_mean': 5., 'offset_std': [3., 2.]
                    }

# Parameters for the sign sky map
sign_params = {# Prior parameters on the shape of the power spectrum ([mean, std])
               'asperity': [.001, 0.001], 'flexibility': [1., .5], 'fluctuations': [1., 1.], 'loglogavgslope': [-4, .5],
                #  Parameters on the mean of the sign field
                'offset_mean': 0, 'offset_std': [1., .5],

              }

extragal_params = {'type': 'explicit', # Model for the generation of the extragalactic component, choices are explicit, implicit or <None>
                   'alpha': 2., 'q': 1. # Hyperparameters for the explicit and implicit cases
                   }
