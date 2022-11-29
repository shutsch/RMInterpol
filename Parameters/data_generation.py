run_parameter_dict = {
    'seed': 24,
    'name': 'test_studentrmdm_32_9',
    'infer_RM': True,
    'infer_DM': True,
    'infer_extra_gal_RM': True,
    'infer_extra_gal_DM': True,
    'infer_extra_gal_DM_explicitly': False,
    'use_ymw_template': False,
    'sinb_model': True,
    'use_students_t_dm': False,
    'use_students_t_rm': False,
    'mock': False,
    'mock_ndata': 30000,
    'rm_data_file': './Data/agn_pop_full.csv',
    'dm_data_file': './Data/NEW_frb_pop_50k.csv',
    'only_load_models': False,
    'plot': True,
    'nside': 32,
}

hyper_parameter_dict = {
#hyper_parameters_logdm = {
#    'spectral': {
#        'asperity': [.5, .3], 'flexibility': [.5, .2], 'fluctuations': [2., .5], 'loglogavgslope': [-4., .5],
#                },
#    'offset': {
#        'offset_mean': 5., 'offset_std': [3., 2.]
#              }, }

    'log_dm': {
        'spectral': {
            'asperity': [.5, .3], 'flexibility': [1., 1.], 'fluctuations': [2., 4.], 'loglogavgslope': [-4., .5],
                    },
        'offset': {
            'offset_mean': 2., 'offset_std': [5., 3.]
                  }, },


    'b_par': {
        'spectral': {
            'asperity': [.001, .001], 'flexibility': [1., 1.], 'fluctuations': [1., 1.], 'loglogavgslope': [-4, .5],
                },
        'offset': {
            'offset_mean': 0, 'offset_std': [1., 3.],
                }, },

    'noise_est': {
                  'dm_noise_excitations': {'alpha': 2, 'q': 1},
                  'rm_noise_excitations': {'alpha': 5., 'q': 6.}
                            },

    'point_sources': {
                      'dm_points': {'alpha': 2, 'q': 1, 'offset': 500.},
}
}