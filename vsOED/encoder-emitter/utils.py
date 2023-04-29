import numpy as np
import torch
import torch.nn as nn
# import warnings
# warnings.filterwarnings("ignore")

# logpdf of independent normal distribution.
# x is of size (n_sample, n_param or n_obs).
# loc and scale are int or numpy.ndarray of size n_param or n_obs.
# output is of size n_sample.
def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))

# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=-1)

class Encoder(nn.Module):
    def __init__(self, dimns, activate, device, dtype):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            # if i < len(dimns) - 1:
                # layers.append(nn.BatchNorm1d(dimns[i + 1]))
            layers.append(activate())
        self.dimns = dimns
        self.net = nn.Sequential(*layers).to(device, dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, ds_hist, ys_hist, xps_hist=None, return_all_stages=False):
        n_sample = max(len(ds_hist), len(ys_hist))
        n_stage = ds_hist.shape[1]
        if xps_hist is None:
            xps_hist = torch.empty(n_sample, n_stage, 0).to(self.device, self.dtype)
        if isinstance(ds_hist, np.ndarray):
            ds_hist = torch.from_numpy(ds_hist)
        if isinstance(ys_hist, np.ndarray):
            ys_hist = torch.from_numpy(ys_hist)
        if isinstance(xps_hist, np.ndarray):
            xps_hist = torch.from_numpy(xps_hist)
        ds_hist = ds_hist.to(self.device, self.dtype)
        ys_hist = ys_hist.to(self.device, self.dtype)
        xps_hist = xps_hist.to(self.device, self.dtype)
        if return_all_stages:
            rets = torch.zeros(n_sample, n_stage + 1, self.dimns[-1]).to(self.device, self.dtype)
        current_sum = torch.zeros(n_sample, self.dimns[-1]).to(self.device, self.dtype)
        for i in range(n_stage):
            ds = ds_hist[:, i, :]
            ys = ys_hist[:, i, :]
            xps = xps_hist[:, i, :]
            x = torch.cat([ds, ys, xps], dim=1)
            output = self.net(x)
            current_sum += output
            if return_all_stages:
                rets[:, i + 1, :] = current_sum
        if return_all_stages:
            return rets 
        else:
            return current_sum

# Construct neural network
class Net(nn.Module):
    def __init__(self, dimns, activate, bounds, net_type, device, dtype):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            if i < len(dimns) - 2:
                # layers.append(nn.BatchNorm1d(dimns[i + 1]))
                layers.append(activate())
        self.net = nn.Sequential(*layers).to(device, dtype)
        self.bounds = torch.from_numpy(bounds).to(device, dtype)
        self.has_inf = torch.isinf(self.bounds).sum()
        self.net_type = net_type
        self.device = device
        self.dtype = dtype

    def forward(self, state, action=None):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.to(self.device, self.dtype)
        if action is not None:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            action = action.to(self.device, self.dtype)
        if self.net_type == 'actor':
            x = self.net(state)
        elif self.net_type == 'critic':
            x = self.net(torch.cat([state, action], dim=-1))
        if self.has_inf:
            x = torch.maximum(x, self.bounds[:, 0])
            x = torch.minimum(x, self.bounds[:, 1])
        else:
#             x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
#                                      self.bounds[:, 0]) + self.bounds[:, 0])
            N = 1
            x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
                                     self.bounds[:, 0]) * N + (self.bounds[:, 0] * (N + 1) - self.bounds[:, 1] * (N - 1)) / 2)
            x = torch.maximum(x, self.bounds[:, 0])
            x = torch.minimum(x, self.bounds[:, 1])
        return x

    

import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale