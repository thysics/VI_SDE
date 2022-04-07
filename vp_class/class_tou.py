#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal


class tOU:
    def __init__(self, t0, t1, z0, alpha, beta, m0, r, sigma, timegrid=False, dN=500):
        self.t0 = t0
        self.t1 = t1
        self.z0 = z0
        self.alpha = alpha
        """
            check whether every element in sigma is positive
        """  
        self.beta = beta
        assert all(sigma > 0), "variance must be positive"
        self.sigma = sigma
        assert alpha.shape[0] == beta.shape[0], "parameter dimension must be equal"
        
        self.m0 = m0
        assert all(r > 0), "r must be positive"
        self.r = r
        
        D = alpha.shape[0]
        if timegrid == True:
            self.pts = torch.linspace(t0, t1, dN).repeat(D, 1)
        else:
            self.pts = torch.sort(torch.cat([(t1 - t0) * torch.rand(D, dN-2) + t0, torch.tensor([self.t0, self.t1]).repeat(D,1)], axis=1), axis=1)[0]
        self.trj, self.dt = self.path()
    
    def path(self):
        
        # Get parameters
        alpha = self.alpha
        beta = self.beta
        sigma = self.sigma
        r = self.r
        m0 = self.m0
        z0 = self.z0
        t = self.pts
    
        def mean(x, dt, t0, alpha, beta, m0, r):
            """
                t0: we always start our process from t = 0
            """

            b_t = alpha * ( (t0[:,0].reshape(-1, 1) + 1) ** beta  - 1) + m0                      - (alpha  * ((t0[:, 1].reshape(-1, 1) + 1) ** beta - 1) + m0 ) * torch.exp(r * dt.reshape(-1, 1))

            return (x.reshape(-1, 1) - b_t) * torch.exp(-r * dt.reshape(-1, 1))

        def std(t, r, sigma):
            return torch.sqrt(variance(t, r, sigma))

        def variance(t, r, sigma):
            dt = torch.diff(t)
            return sigma * sigma * (1 - torch.exp(-2 * r * dt)) / (2*r)


        assert t.shape[1] > 1

        normal = Normal(loc=0., scale=1.)
        x = normal.sample(t.size())

        if torch.is_tensor(z0):
            x[:, 0] = z0.flatten()
        else:
            x[:, 0] = z0

        t0 = t - t[:, 0].reshape(-1, 1)

        dt = torch.diff(t)

        scale = std(t, r.reshape(-1,1), sigma.reshape(-1, 1))

        x[:, 1:] = x[:, 1:] * scale

        for i in range(1, x.shape[1]):
            x[:, i] += mean(x[:, i-1], dt[:, i-1], t0[:, i-1:i+1], alpha, beta, m0, r).flatten()


        return x, dt

def tou_gradient(t, x, params):
    
    """Calculates log likelihood of a path"""
    
    def mean(x, t, alpha, beta, m0, r):
        """
            t0: we always start our process from t = 0
        """
        t0 = t - t[:, 0].reshape(-1, 1)
        dt = torch.diff(t)
        m_t = alpha * ( (t0[:, :-1] + 1) ** beta  - 1) + m0 - (alpha * ((t0[:, 1:] + 1) ** beta - 1 ) + m0) * torch.exp(r * dt)

        return (x - m_t) * torch.exp(-r * dt)
    
    def std(t, r, sigma):
        return torch.sqrt(variance(t, r, sigma))

    def variance(t, r, sigma):
        dt = torch.diff(t)
        return sigma * sigma * (1 - torch.exp(-2 * r * dt)) / (2*r)

    
    params_ = Variable(params, requires_grad=True)
    alpha, beta, m0, r, sigma = params_
    
    mu = mean(x[:, :-1], t, alpha, beta, m0, r)
    var = std(t, r, sigma) + 1e-7 # To prevent the underflow
    
    
    LL = torch.sum(Normal(loc=mu, scale=var).log_prob(x[:, 1:]), axis=1)
    LL.backward(torch.tensor([1.]).repeat(x.shape[0]))


    return {'alpha':params_.grad[0].clone().detach(), 'beta':params_.grad[1].clone().detach(), 
            'm0':params_.grad[2].clone().detach(), 'r':params_.grad[3].clone().detach(), 
            'LL':LL.clone().detach().data}

