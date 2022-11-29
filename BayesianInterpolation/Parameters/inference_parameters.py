run_params = {
    'seed': 534634,
    'run_name': 'test',
    'use_mock_data': True,
    'full_sphere': False,
    'do_plot': True,
    'n_iterations': 20
}

domain_params = {
    # necessary for full sky
    'nside': 128,
    # necessary for cutout
    'n_lon': 100,
    'n_lat': 100,
    'center': [0, 0],
}

amplitude_params = {
        'spectral': {
            'asperity': [.5, .3], 'flexibility': [.5, .2], 'fluctuations': [2., .5], 'loglogavgslope': [-4., .5],
                    },
        'offset': {
            'offset_mean': 5., 'offset_std': [3., 2.]
                  }, }


sign_params = {
        'spectral': {
            'asperity': [.001, .001], 'flexibility': [1., .5], 'fluctuations': [1., 1.], 'loglogavgslope': [-4, .5],
                },
        'offset': {
            'offset_mean': 0, 'offset_std': [1., .5],
                },
               }

extragal_params = {'type': 'explicit',  'alpha': 5., 'q': 4.}
