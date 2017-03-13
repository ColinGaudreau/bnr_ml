import numpy as np
from scipy.special import kv, gamma
from scipy.stats import beta
from utils import format_data
from priors import *

import pdb

class Parameter(object):
    def __init__(self, value, prior):
        assert(isinstance(prior, BasePrior))
        self.value = value
        self.prior = prior

    def __repr__(self):
        return 'Parameter({}, {})'.format(self.value, self.prior)

class BaseKernel(object):
    def __init__(self, parameters=[]):
        self.parameters = parameters

    def compute_covariance(self, x, y):
        raise NotImplementedError

    def logprior(self):
        val = 0.
        for par in self.parameters:
            val += par.prior.logprob(par.value)
        return val

    def get_parameters(self):
        return np.asarray([par.value for par in self.parameters])

    def set_parameters(self, values):
        assert(values.size == self.parameters.__len__())
        for i, par in enumerate(self.parameters):
            par.value = values[i]

    def _format_vects(self, x, y, diag=False):
        x, y = format_data(x), format_data(y)

        if not diag:
            idx1, idx2 = np.meshgrid(np.arange(x.shape[0]), np.arange(y.shape[0]))
            x, y =  x[idx1], y[idx2]
        else:
            x, y = x.reshape(x.shape + (1,)).swapaxes(1,2), x.reshape(x.shape + (1,)).swapaxes(1,2)

        return x, y

    def __repr__(self):
        return 'Kernel({}, {})'.format(self.__class__, self.parameters.__str__())

class SquaredExpKernel(BaseKernel):
    def __init__(self, parameters=[]):
        if parameters.__len__() == 0:
            self.parameters = [Parameter(0.1, GammaPrior())]
        else:
            for par in parameters:
                assert(isinstance(par, Parameter))
            self.parameters = parameters

    def compute_covariance(self, x, y, diag=False):
        x, y = self._format_vects(x, y, diag=diag)
        cov = 0
        if self.parameters.__len__() < 2:
            cov = np.exp(-((x - y)**2).sum(axis=2) / self.parameters[0].value)
        else:
            cov = self.parameters[1].value * np.exp(-((x - y)**2).sum(axis=2) / self.parameters[0].value)

        if self.parameters.__len__() > 2:
            cov += (self.parameters[2].value * np.eye(cov.shape[0]))

        return cov

class LinearKernel(BaseKernel):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def compute_covariance(self, x, y, diag=False):
        x, y = self._format_vects(x, y, diag=diag)
        return (x * y).sum(axis=2)


class MaternKernel(BaseKernel):
    def __init__(self, parameters=[], type='5/2'):
        if parameters.__len__() == 0:
            self.parameters = [Parameter(0.1, GammaPrior())]
        else:
            for par in parameters:
                assert(isinstance(par, Parameter))
            self.parameters = parameters
        assert(type == '5/2' or type == '3/2')
        self.type = type

    def compute_covariance(self, x, y, diag=False):
        x, y = self._format_vects(x, y, diag=diag)
        d = np.sqrt(((x - y)**2).sum(axis=2))
        l = self.parameters[0].value
        if self.type == '3/2':
            val = (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)
        else:
            val = (1 + np.sqrt(5)*d/l + (5 * d**2) / (3 * l**2)) * np.exp(-np.sqrt(5)*d/l)

        return val

class InputWarpedKernel(BaseKernel):
    '''
    Input warped kernel
    '''
    def __init__(self, kernel, N, parameters=None):
        self.kernel = kernel
        self.N = N

        if parameters is not None:
            assert(parameters.__len__() == 2 * N)
            for par in parameters:
                assert(isinstance(par, Parameter))
        else:
            parameters = []
            for i in range(N):
                parameters.extend([Parameter(1., LogNormalPrior(0., 0.5)), Parameter(1., LogNormalPrior(0., 0.5))])

        # add warping parameters
        self.parameters = kernel.parameters
        self.parameters.extend(parameters)

    def compute_covariance(self, x, y, diag=False):
        x, y = format_data(x), format_data(y)
        x, y = np.copy(x), np.copy(y)
        assert(x.shape[1] == self.N and y.shape[1] == self.N)

        for i in range(x.shape[1]):
            a, b = self.parameters[-2*self.N + 2*i].value, self.parameters[-2*self.N + 2*i + 1].value
            x[:,i] = beta.cdf(x[:,i], a, b)
            y[:,i] = beta.cdf(y[:,i], a, b)

        val =  self.kernel.compute_covariance(x, y, diag=diag)

        return val







