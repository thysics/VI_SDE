#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal


class tOU:
    def __init__(self, t0, t1, parameters, sde_sigma, init_dist, timegrid=False, dN=200):
        """
            t0, t1 = torch.tensor(t1)
            alpha, beta, m0, r, sigma = torch.tensor.shape (D, 1)
        """
        assert all(sde_sigma > 0), "variance must be positive"
        
        self.t0 = t0
        self.t1 = t1
        
        # set parameters
        self.a = parameters['a']
        self.b = parameters['b']
        self.c = parameters['c']
        self.d = parameters['d']
        
        # set hyper-parameter
        self.sigma = sde_sigma
        
        D = self.a.shape[0]
        if timegrid == True:
            self.pts = torch.linspace(t0, t1, dN).repeat(D, 1)
        else:
            self.pts = torch.sort(torch.cat([(t1 - t0) * torch.rand(D, dN-2) + t0, torch.tensor([self.t0, self.t1]).repeat(D,1)], axis=1), axis=1)[0]
            
        self.trj, self.dt = self.path(init_dist)
        
            
    def path(self, init_dist):
        """
            init_dist: distribution to draw an initial state
            init_dist = Normal(loc=E[X_{t_{i-1}}|lambda_{i-1}], scale = std[X_{t_{i-1}}|lambda_{i-1}])
        """
        
        # Get parameters

        a = self.a
        b = self.b
        c = self.c
        d = self.d
        
        sigma = self.sigma
        
        t = self.pts
        
        
        def mean(x, t0, a, b, c, d):
            """
                t0: we always start our process from t = 0
            """
            s = t0[:, 0].reshape(-1, 1)
            t = t0[:, 1].reshape(-1, 1)
            
            b_t = -1 / (a*s + b) * (a*c*(t**3 - s**3)/3 + (a*d + b*c)*(t**2 - s**2)/2 + d*b*(t-s))

            return (x.reshape(-1, 1) - b_t) * (a*s + b) / (a * t + b)
        
        def variance(t0, a, b, sigma):
            s = t0[:, :-1]
            t = t0[:, 1:]
            
            term1 = sigma**2
            term2 = (a**2 * (t**3 - s**3)/3 + a*b*(t**2 - s**2) + b**2 * (t-s)) / ((a*t + b)**2)

            
            return term1 * term2
        
        def std(t, a, b, sigma):
            return torch.sqrt(variance(t, a, b, sigma))

        
        normal = Normal(loc=0., scale=1.)
        x = normal.sample(t.size())

        if type(init_dist) == torch.distributions.normal.Normal:
            # sample initial state from approximate posterior distribution
            m0 = init_dist.sample([x.shape[0], 1])
            x[:, 0] = m0.flatten()
        else:
            m0 = init_dist
            x[:, 0] = m0.flatten()
            
        t0 = t - t[:, 0].reshape(-1, 1)

        dt = torch.diff(t)

        scale = std(t, a, b, sigma)
        x[:, 1:] = x[:, 1:] * scale
        for i in range(1, x.shape[1]):
            x[:, i] += mean(x[:, i-1], t0[:, i-1:i+1], a, b, c, d).flatten()
        
        return x, dt

def tou_gradient(pts, x, params, sde_sigma, init_dist):
    assert torch.is_tensor(sde_sigma), "sde_sigma must be a (D*1) tensor"
    assert torch.is_tensor(init_dist) or type(init_dist) == torch.distributions.normal.Normal,    "init_dist must either be tensor or torch.distributions"
    
    """
        Calculates log likelihood of a path
        Note that there are three parameters, alpha, beta, r
    """
    
    def mean(x, time, a, b, c, d):
        """
            t0: we always start our process from t = 0
        """
        t0 = time - time[:, 0].reshape(-1, 1)
        
        s = t0[:, :-1]
        t = t0[:, 1:]
        b_t = -1 / (a*s + b) * (a*c*(t**3 - s**3)/3 + (a*d + b*c)*(t**2 - s**2)/2 + d*b*(t-s))
        return (x - b_t) * (a*s + b) / (a * t + b)

    def variance(time, a, b, sigma):
        t0 = time - time[:, 0].reshape(-1, 1)
        s = t0[:, :-1]
        t = t0[:, 1:]

        term1 = sigma**2
        term2 = (a**2 * (t**3 - s**3)/3 + a*b*(t**2 - s**2) + b**2 * (t-s) )/ ((a*t + b) ** 2)

        return term1 * term2

    def std(t, a, b, sigma):
        return torch.sqrt(variance(t, a, b, sigma))
    
    param_stack = torch.stack([params['a'], params['b'], params['c'], params['d']])
    
    params_ = Variable(param_stack, requires_grad=True)
    a, b, c, d = params_
    sigma = sde_sigma

    m0 = x[:, 0].reshape(-1, 1)
    mu = mean(x[:, :-1], pts, a, b, c, d)
    
    var = std(pts, a, b, sigma) + 1e-7 # To prevent the underflow (some of the value becomes 0 due to lack of precision
    LL = torch.nansum(Normal(loc=mu, scale=var).log_prob(x[:, 1:]), axis=1)
    # At initialization (in case of random initialization)
    if type(init_dist) == torch.distributions.normal.Normal:
        LL += torch.nansum(init_dist.log_prob(x[:,0]))
    
    LL.backward(torch.tensor([1.]).repeat(x.shape[0]))

    return {'a':params_.grad[0].clone().detach(), 'b':params_.grad[1].clone().detach(), 
            'c':params_.grad[2].clone().detach(), 'd':params_.grad[3].clone().detach(), \
            'LL':LL.clone().detach().data}

