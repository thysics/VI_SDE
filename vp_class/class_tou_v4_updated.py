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
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.r = parameters['r']
        
        # set hyper-parameter
        self.sigma = sde_sigma
        
        D = self.alpha.shape[0]
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

        alpha = self.alpha
        beta = self.beta
        r = self.r
                
        sigma = self.sigma
        
        t = self.pts
        
        
        def mean(x, dt, t0, m0, alpha, beta, r):
            """
                t0: we always start our process from t = 0
            """
            s = t0[:, 0].reshape(-1, 1)
            t = t0[:, 1].reshape(-1, 1)
            
            b_t = alpha * ( (s + 1) ** beta  - 1) + m0 - (alpha  * ((t + 1) ** beta - 1) + m0 ) * torch.exp(r * dt.reshape(-1, 1))
            
            return (x.reshape(-1, 1) - b_t) * torch.exp(-r * dt.reshape(-1, 1))
        
        def variance(t0, r, sigma):
            dt = torch.diff(t0)
            return sigma * sigma * (1 - torch.exp(-2 * r * dt)) / (2*r)
        
        def std(t0, r, sigma):
            return torch.sqrt(variance(t0, r, sigma))

        
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

        scale = std(t0, r, sigma)
        
        x[:, 1:] = x[:, 1:] * scale
        for i in range(1, x.shape[1]):
            x[:, i] += mean(x[:, i-1], dt[:, i-1], t0[:, i-1:i+1], m0, alpha, beta, r).flatten()
        
        return x, dt

def tou_gradient(pts, x, params, sde_sigma, init_dist):
    assert torch.is_tensor(sde_sigma), "sde_sigma must be a (D*1) tensor"
    assert torch.is_tensor(init_dist) or type(init_dist) == torch.distributions.normal.Normal,    "init_dist must either be tensor or torch.distributions"
    
    """
        Calculates log likelihood of a path
        Note that there are three parameters, alpha, beta, r
    """
    
    def mean(x, time, m0, alpha, beta, r):
        """
            t0: we always start our process from t = 0
        """
        t0 = time - time[:, 0].reshape(-1, 1)
        
        dt = torch.diff(t0)
        
        s = t0[:, :-1]
        t = t0[:, 1:]

        b_t = alpha * ( (s + 1) ** beta  - 1) + m0 - (alpha  * ((t + 1) ** beta - 1) + m0 ) * torch.exp(r * dt)

        return (x - b_t) * torch.exp(-r * dt)

    def variance(time, r, sigma):
        dt = torch.diff(time)
        return sigma * sigma * (1 - torch.exp(-2 * r * dt)) / (2*r)
    

    def std(time, r, sigma):
        return torch.sqrt(variance(time, r, sigma))
    
    param_stack = torch.stack([params['alpha'], params['beta'], params['r']])
    
    params_ = Variable(param_stack, requires_grad=True)
    alpha, beta, r = params_
    sigma = sde_sigma

    m0 = x[:, 0].reshape(-1, 1)
    mu = mean(x[:, :-1], pts, m0, alpha, beta, r)

    var = std(pts, r, sigma) + 1e-7 # To prevent the underflow (some of the value becomes 0 due to lack of precision
    LL = torch.nansum(Normal(loc=mu, scale=var).log_prob(x[:, 1:]), axis=1)
    
    # At initialization (in case of random initialization)
    if type(init_dist) == torch.distributions.normal.Normal:
        LL += torch.sum(init_dist.log_prob(x[:,0]))
    
    LL.backward(torch.tensor([1.]).repeat(x.shape[0]))

    return {'alpha':params_.grad[0].clone().detach(), 'beta':params_.grad[1].clone().detach(), 
            'r':params_.grad[2].clone().detach(), 'LL':LL.clone().detach().data}

