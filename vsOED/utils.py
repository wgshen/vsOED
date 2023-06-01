import math
from numbers import Number

import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import numpy as np

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

TRAIN_SEEDS = [48563549, 1705737, 44028299, 84087361, 59006523,
       38325122, 65802546, 55469522, 60059890, 23897215]

EVAL_SEEDS = [63777278, 70378855, 12832782, 82173358, 86956867,
       16458555, 75560991, 86169801, 31694485, 16138827]


# import warnings
# warnings.filterwarnings("ignore")

def set_random_seed(random_state=None):
    NoneType = type(None)
    assert isinstance(random_state, (int, NoneType)), (
           "random_state should be an integer or None.")
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

def log(content, dowel, update, verbose):
    if dowel is not None and update % verbose == 0:
        dowel.logger.log(content)

def tmp_fun(*args, **kws):
    if kws['ret'] == '0':
        return 0
    elif kws['ret'] == 'empty tensor':
        return torch.tensor([])
    else:
        return
        
# logpdf of independent normal distribution.
def norm_logpdf(x, loc=0, scale=1):
    if isinstance(scale, (int, float)):
        scale = torch.tensor(scale)
    logpdf = (-torch.log(math.sqrt(2 * math.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(dim=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return torch.exp(norm_logpdf(x, loc, scale))

# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return torch.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(dim=-1)

# encoder network
class Encoder(nn.Module):
    def __init__(self, dimns, activate):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            # if i < len(dimns) - 1:
                # layers.append(nn.BatchNorm1d(dimns[i + 1]))
            layers.append(activate())
        self.dimns = dimns
        self.net = nn.Sequential(*layers)

    def forward(self, ds_hist, ys_hist, xps_hist=None, return_all_stages=False):
        n_sample = max(len(ds_hist), len(ys_hist))
        n_stage = ds_hist.shape[1]
        if xps_hist is None:
            xps_hist = torch.empty(n_sample, n_stage, 0)
        if return_all_stages:
            rets = torch.zeros(n_sample, n_stage + 1, self.dimns[-1])
        current_sum = torch.zeros(n_sample, self.dimns[-1])
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

# initialize the encoder net
def initialize_encoder(encoder_dimns, activate=nn.ReLU):
    return Encoder(encoder_dimns, activate)

# pass the input data through the encoder network to obtain encoded states
def get_encoded_states(encoder_net, ds_hist, ys_hist, 
                       xps_hist, return_all_stages):
    assert ds_hist.shape[1] == ys_hist.shape[1]
    assert xps_hist.shape[1] == ds_hist.shape[1] + 1
    xbs_encoded = encoder_net(ds_hist, ys_hist, 
                              xps_hist[:, :-1], return_all_stages)
    if return_all_stages:
        xps = xps_hist
    else:
        xps = xps_hist[:, -1]
    states = torch.cat([xbs_encoded, xps], dim=-1)
    return states

# policy/critic/backend network
class Net(nn.Module):
    def __init__(self, dimns, activate, bounds, net_type, 
        backend_net=None, n_design=None):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            if i < len(dimns) - 2:
                # layers.append(nn.BatchNorm1d(dimns[i + 1]))
                layers.append(activate())
        self.net = nn.Sequential(*layers)
        if net_type != 'backend':
            self.bounds = bounds
            self.has_inf = torch.isinf(self.bounds).sum()
        self.backend_net = backend_net
        self.net_type = net_type
        self.n_design = n_design

    def forward(self, x):
        if self.net_type == 'backend':
            x = self.net(x)
            return x
        elif self.net_type == 'actor':
            if self.backend_net is not None:
                x = self.backend_net(x)
            x = self.net(x)
        elif self.net_type == 'critic':
            if self.backend_net is not None:
                ds = x[:, -self.n_design:]
                x = self.backend_net(x[:, :-self.n_design])
                x = torch.cat([x, ds], dim=1)
            x = self.net(x)
        if not self.has_inf:
            x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
                                     self.bounds[:, 0]) + self.bounds[:, 0])
        x = torch.maximum(x, self.bounds[:, 0])
        x = torch.minimum(x, self.bounds[:, 1])
        return x


###########################################################
# Copied from https://github.com/toshas/torch_truncnorm   #
###########################################################
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
        super(TruncatedStandardNormal, self).__init__(batch_shape, 
            validate_args=validate_args)
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
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b 
            - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b 
            - self._little_phi_a) / self._Z) ** 2
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
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, 
            self._dtype_max_lt_1)
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