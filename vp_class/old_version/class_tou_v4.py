#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal


class tOU:
    def __init__(self, t0, t1, alpha, beta, r, sde_sigma, init_dist, timegrid=False, dN=500):
        """
            t0, t1 = torch.tensor(t1)
            alpha, beta, m0, r, sigma = torch.tensor.shape (D, 1)
        """
        assert alpha.shape == beta.shape == r.shape == sde_sigma.shape, "parameters must have the same dimension"
        assert alpha.shape[1] == 1, "parameter must have a shape D*1"
        assert all(sde_sigma > 0), "variance must be positive"
        assert all(r > 0), "r must be positive"
        
        self.t0 = t0
        self.t1 = t1
        self.alpha = alpha
        self.beta = beta
        self.sigma = sde_sigma
        self.r = r
        D = alpha.shape[0]
        
        if timegrid == True:
            self.pts = torch.linspace(t0, t1, dN).repeat(D, 1)
        else:
            self.pts = torch.sort(torch.cat([(t1 - t0) * torch.rand(D, dN-2) + t0, torch.tensor([self.t0, self.t1]).repeat(D,1)], axis=1), axis=1)[0]
          
        if type(init_dist) == torch.distributions.normal.Normal:
            init_state = init_dist.sample([D, 1])
        else:
            init_state = init_dist
            
        self.init_state = init_state
        self.trj, self.dt = self.path(init_state)
        
            
    def path(self, init_state):
        """
            init_dist: distribution to draw an initial state
            init_dist = Normal(loc=E[X_{t_{i-1}}|lambda_{i-1}], scale = std[X_{t_{i-1}}|lambda_{i-1}])
        """
        
        # Get parameters
        alpha = self.alpha
        beta = self.beta
        sigma = self.sigma
        r = self.r
        t = self.pts
    
        def mean(x, dt, t0, alpha, beta, m0, r):
            """
                t0: we always start our process from t = 0
            """

            b_t = alpha * ( (t0[:,0].reshape(-1, 1) + 1) ** beta  - 1) + m0 - (alpha  * ((t0[:, 1].reshape(-1, 1) + 1) ** beta - 1) + m0 ) * torch.exp(r * dt.reshape(-1, 1))
            return (x.reshape(-1, 1) - b_t) * torch.exp(-r * dt.reshape(-1, 1))

        def std(t, r, sigma):
            return torch.sqrt(variance(t, r, sigma))

        def variance(t, r, sigma):
            dt = torch.diff(t)
            return sigma * sigma * (1 - torch.exp(-2 * r * dt)) / (2*r)
        
        normal = Normal(loc=0., scale=1.)
        x = normal.sample(t.size())

        x[:, 0] = init_state.flatten()
            
        t0 = t - t[:, 0].reshape(-1, 1)

        dt = torch.diff(t)

        scale = std(t, r.reshape(-1,1), sigma.reshape(-1, 1))

        x[:, 1:] = x[:, 1:] * scale
        for i in range(1, x.shape[1]):
            x[:, i] += mean(x[:, i-1], dt[:, i-1], t0[:, i-1:i+1], alpha, beta, init_state, r).flatten()


        return x, dt

def tou_gradient(t, x, params, sde_sigma, init_dist):
    assert torch.is_tensor(sde_sigma), "sde_sigma must be a (D*1) tensor"
    assert torch.is_tensor(init_dist) or type(init_dist) == torch.distributions.normal.Normal,    "init_dist must either be tensor or torch.distributions"
    
    """
        Calculates log likelihood of a path
        Note that there are three parameters, alpha, beta, r
    """
    
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
    alpha, beta, r, = params_
    sigma = sde_sigma

    m0 = x[:, 0].reshape(-1, 1)
    mu = mean(x[:, :-1], t, alpha, beta, m0, r)
    var = std(t, r, sigma) + 1e-7 # To prevent the underflow (some of the value becomes 0 due to lack of precision
    LL = torch.sum(Normal(loc=mu, scale=var).log_prob(x[:, 1:]), axis=1)

    # At initialization (in case of random initialization)
    if type(init_dist) == torch.distributions.normal.Normal:
        LL += torch.sum(init_dist.log_prob(x[:,0]))
    
    LL.backward(torch.tensor([1.]).repeat(x.shape[0]))
    
    return {'alpha':params_.grad[0].clone().detach(), 'beta':params_.grad[1].clone().detach(), 
            'r':params_.grad[2].clone().detach(), 'LL':LL.clone().detach().data}


