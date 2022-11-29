import os
import nifty7 as ift
from Functions.helpers import load_field_model
from Functions.plot import prior_plot


def main():
    """
    Short script which exemplifies Nifty priors for a log-normal sky.
    The goal here is to visualize the impact of the 'fluctuations' and 'offset' parameters,
    as they are mostly responsible for setting the scale ('offset') and range ('fluctuations') of the sky maps.
    For most parameters, one has to specify a mean an std for the inference. For this script, the stds for
    the 'fluctuations' and 'offset' parameters are set very small, as we want to be able to set the values precisely.
    The other parameters are set looser, which implies that the sky maps will have rather strong
    differences in smoothness. One can of course also set these stds very small, if one wants more uniformity there.
    In an inference this should be somewhat loosened, except if one is very sure that these are correct.
    The code specifies a set of possibles values over which it iterates and plots 4 possible realisations of the prior.
    Of course, the script can be adapted to also visualize the other parameters.
    """

    plotting_path = './Runs/plots/testing_priors/'
    if not os.path.exists(plotting_path):
        os.makedirs(plotting_path)

    # The hyper-parameters, most follow the patter {name: [mean, std]}.
    # The exception is the offset parameter, where the std has its own mean and std (yes, that's intended;))
    hyper_parameters = {'spectral':
                            {'asperity': [.5, .2], 'flexibility': [1.5, 1.],
                             'fluctuations': [2, .001], 'loglogavgslope': [-3., .75],
                             },
                        'offset': {'offset_mean': 0, 'offset_std': [0.001, .0001]}
                        }

    fluctuation_priors = [0.1, 1, 3, 6]
    offset_mean_priors = [-1., 0, 2, 4]
    nside = 128

    hp = ift.makeDomain(ift.HPSpace(nside))

    for f in fluctuation_priors:
        for o in offset_mean_priors:
            hyper_parameters['spectral'].update({'fluctuations': [f, 0.001]})
            hyper_parameters['offset'].update({'offset_mean': o,})
            log_field, amp = load_field_model(a_dict=hyper_parameters, domain=hp, name='field')
            prior_plot(plotting_path, {'fluc-' + str(f) + '_offset-' + str(o): log_field.exp()}, 4)


if __name__ == '__main__':
    main()
