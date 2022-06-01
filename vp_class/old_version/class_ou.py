#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.distributions import Normal
import torch.optim
from torch.autograd import Variable
import numpy as np


# In[ ]:


""" OU (Ornstein-Uhlenbeck) process

    dX = -A(X-alpha)dt + v dB
    """
class OU:
    def __init__(self, t0, t1, z0, alpha, beta, sigma, timegrid=False, dN=500):
        self.t0 = t0
        self.t1 = t1
        self.z0 = z0
        self.alpha = alpha
        """
            check whether every element in sigma is positive
        """
        assert all(beta > 0), "beta should be positive"    
        self.beta = beta
        assert all(sigma > 0), "variance should be positive"
        self.sigma = sigma
        assert alpha.shape[0] == beta.shape[0], "parameter dimension must be equal"
        D = alpha.shape[0]
        if timegrid == True:
            self.pts = torch.linspace(t0, t1, dN).repeat(D, 1)
        else:
            self.pts = torch.sort(torch.cat([(t1 - t0) * torch.rand(D, dN-2) + t0, torch.tensor([self.t0, self.t1]).repeat(D,1)], axis=1), axis=1)[0]
        self.trj, self.dt = self.path()
          
    def path(self):
        """ Simulates a sample path"""
        alpha = self.alpha
        beta = self.beta
        sigma = self.sigma
        x0 = self.z0
        t = self.pts
        
        def variance(t, beta, sigma):
            assert all(beta >= 0)
            assert all(sigma >= 0)
            return sigma * sigma * (1.0 - torch.exp( -2.0 * beta * t)) / (2 * beta)

        def std(t, beta, sigma):
            return torch.sqrt(variance(t, beta, sigma))

        def mean(x0, t, beta, alpha):
            assert all(beta >= 0)
            return x0 * torch.exp(-1 * beta * t) + (1.0 - torch.exp( -1 * beta * t)) * alpha

        assert t.shape[1] > 1
        normal = Normal(loc=0., scale=1.)
        x = normal.sample(t.size())
        if torch.is_tensor(x0):
            x[:, 0] = x0.flatten()
        else:
            x[:, 0] = x0
        dt = torch.diff(t)
        scale = std(dt, beta, sigma)
        x[:, 1:] = x[:, 1:] * scale
        for i in range(1, x.shape[1]):
            x[:, i] += mean(x[:, i - 1].reshape(-1, 1), dt[:, i - 1].reshape(-1, 1), beta, alpha).flatten()
        return x, dt


# In[ ]:


def ou_gradient(t, x, params):
    
    """Calculates log likelihood of a path"""
    
    def variance(t, beta, sigma):
        assert all(beta >= 0)
        assert all(sigma >= 0)
        return sigma * sigma * (1.0 - torch.exp( -2.0 * beta * t)) / (2 * beta)

    def std(t, beta, sigma):
        return torch.sqrt(variance(t, beta, sigma))

    def mean(x0, t, beta, alpha):
        assert all(beta >= 0)
        return x0 * torch.exp(-1 * beta * t) + (1.0 - torch.exp( -1 * beta * t)) * alpha

    
    params_ = Variable(params, requires_grad=True)
    alpha, beta, sigma = params_
    dt = torch.diff(t)
    mu = mean(x[:, :-1], dt, beta, alpha)
    var = std(dt, beta, sigma)
    
    
    LL = torch.sum(Normal(loc=mu, scale=var).log_prob(x[:, 1:]), axis=1)
    LL.backward(torch.tensor([1.]).repeat(x.shape[0]))


    return {'alpha':params_.grad[0].clone().detach(), 'beta':params_.grad[1].clone().detach(), 'LL':LL.clone().detach().data}


