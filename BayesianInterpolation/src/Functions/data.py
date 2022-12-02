import numpy as np


def get_mock_data(data_name):
    # These
    data_path = './Data/mock/' + data_name + '/'
    data = np.load(data_path + 'data.npy')
    std = np.load(data_path + 'noise_sigma.npy')
    theta = np.load(data_path + 'theta.npy')
    phi = np.load(data_path + 'phi.npy')

    return theta, phi, data, std


def get_real_data():
    raise NotImplementedError()
