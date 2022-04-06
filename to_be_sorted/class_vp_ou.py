#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.autograd import Variable
import numpy as np
import scipy.stats as ss


# In[ ]:


torch.set_default_dtype(torch.float64)



class OU:
    """
        Class object: Ornstein Ulhenbeck process
        Parameters
            t0: time we wish to start the process
            t1: time we wish to stop the process
            z0: location we wish to start the process
            \alpha
            \beta
            \simga
            dN: the number of time points one wish to sample at
            D: the number of paths
            timegrid
                True: evenly space the time interval
                False: choose randomly selected points in each time interval
    """
    def __init__(self, t0, t1, z0, alpha, beta, sigma, dN, D, timegrid='True'):
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
        self.D = D
        if timegrid == 'True':
            self.pts = torch.linspace(t0, t1, dN).repeat(D, 1)
        else:
            self.pts = torch.sort(torch.cat([(t1 - t0) * torch.rand(D, dN-2) + t0, torch.tensor([self.t0, self.t1]).repeat(D,1)], axis=1), axis=1)[0]
        self.trj, self.dt = self.simulate()
        
    def simulate(self):
        """
            Simulate an OU process on a set of discrete points
                Make sure to match the dimension of each object;
                    note that memoryview of torch/python object flattens
        """
        output = torch.empty(self.pts.shape)
        output[:, 0] = self.z0.flatten()
        interval = self.pts[:, 1:] - self.pts[:, 0].reshape(-1,1)
        for t in range(1, self.pts.shape[1]):
            dt = interval[:, t-1].reshape(-1, 1)
            mean = self.alpha + (output[:, 0].reshape(-1,1) - self.alpha) * np.exp(-1 * self.beta * dt)
            var = np.sqrt((self.sigma ** 2) * (1 - np.exp(-2 * self.beta * dt)) / (2 * self.beta))
            if self.D > 1:
                output[:, t] = torch.from_numpy(ss.multivariate_normal.rvs(mean = mean.flatten(), cov = torch.diag(var.flatten())))
            else:
                assert var > 0, "variance is negative, sd:%.3f interval: %.3f" % (var, interval[t-1] )
                output[:, t] = ss.norm.rvs(loc = mean, scale = np.sqrt(var))
        return output, self.pts[:, 1:] - self.pts[:, :-1]


# In[ ]:


class OU_Score:
    """
        Class OU_Score
           It is to compute the transition densities of simulated path from the OU class
           One can take an autograd for parameters including alpha, beta (gradient with respect to its transition densities      (likelihood)
           
    """
    
    def __init__(self, ou):
        self.ou = ou
        
    def compute_score(self, alpha, beta):
        """
            Compute the value of the score function at given parameters
            return a dictionary matching each parameter to its gradient
        """
        # Hyperparameter
        sigma = self.ou.sigma
        
        # Parameters
        assert all(beta > 0), "beta should be positive"
        alpha = Variable(alpha, requires_grad = True)
        beta = Variable(beta, requires_grad = True)
        
        dt = self.ou.pts[:, 1:] - self.ou.pts[:, 0].reshape(-1, 1)
        X = self.ou.trj
        
        def compute_transition(X, dt, alpha, beta, sigma):
            """
            Compute the likelihood based on the transition density of the (simulated) path
            """
#             print("sigma = ", sigma.shape, "beta = ", beta.shape, "dt = ", dt.shape)
#             print("log sigma = ", torch.log(sigma), "log beta = ", torch.log(beta), "rest = ", torch.log(1 - torch.exp(-2 * beta * dt)) )
            term1 = -0.5 * (2 * torch.log(sigma) - torch.log(beta) + torch.log(1 - torch.exp(-2 * beta * dt)))
            term2 = beta * (X[:, 1:] - alpha - (X[:, 0].reshape(-1,1) - alpha) * torch.exp(-1 * beta * dt)) ** 2 
            term3 = (sigma ** 2) * (1 - torch.exp(-2 * beta * dt))
#             print("term1 = ", term1, "term2 = ", term2, "term3 = ", term3)
            return torch.sum(term1 - term2/term3, axis=1)
        
        LL = compute_transition(X, dt, alpha, beta, sigma)
#         print(LL.data.numpy())
#         print("D = ", self.ou.D)
        LL.backward(torch.tensor([1.]).repeat(self.ou.D))
         
        return {"alpha": alpha.grad.detach().clone(), 'beta':beta.grad.detach().clone(), "LL": LL.data.numpy()}

