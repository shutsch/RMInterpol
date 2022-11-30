import numpy as np
import nifty8 as ift
import healpy as hp
import os
import pylab as pl
import Data.rmtable as rmt
from Operators.SkyProjector import SkyProjector
from Functions.helpers import gal2gal


def get_mock_data(name, full_sky):
    # draw

    data_domain = ift.makeDomain(ift.UnstructuredDomain((n_data,)))
    response = SkyProjector(theta=theta, phi=phi, target=data_domain, domain=full_model.target)
    if isinstance(std, float):
        std = ift.full(data_domain, std)
    else:
        std = ift.Field(data_domain, std)
    assert isinstance(std, ift.Field)
    N = ift.DiagonalOperator(std**2)
    pos = ift.from_random(full_model.domain, 'normal')
    path += '/Groundtruth/'
    if not os.path.exists(path):
        os.makedirs(path)
    for name, model in model_dict.items():
        plot = ift.Plot()
        plot.add(model.force(pos), title='groundtruth_' + name)
        plot.output(name=path + name + '.png')
    sres = (response @ full_model).force(pos)
    noise = N.draw_sample_with_dtype(np.dtype(float))
    data = sres + noise
    pl.figure()
    pl.scatter(sres.val, data.val)
    pl.xlabel('truth')
    pl.ylabel('data')
    pl.savefig(path + 'truth_data_scatter.png')
    return theta, phi, data.val, std.val


def get_real_data():
    raise NotImplementedError()
