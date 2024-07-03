#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:35 2024

@author: arjunmnanoj
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from MFGP import GaussianProcess
from MFGP import MFGPR


#%% Single Fidelity GP example


# Function
def f(x):
    return np.sin(3*np.pi*(x**3)) - np.sin(8*np.pi*(x**3))

# Design space
x = np.linspace(0, 1, 1000)


def LHS(num_samples, num_dimensions):
    """
    Generate samples using Latin Hypercube Sampling.
    """
    sampler = qmc.LatinHypercube(d=num_dimensions, seed=1)
    sample = sampler.random(n=num_samples)
    return sample


# Samples
X = LHS(5, 1)

y = f(X)

X, y = np.atleast_2d(X), np.atleast_2d(y)


# Gaussian Process Regression
GP = GaussianProcess(epsilon=0.1)
GP.learn(X, y)
GP_mean = np.array([GP(i) for i in x]).ravel()
GP_variance = np.array([GP.variance(i) for i in x]).ravel()

# Plot
plt.figure(figsize=(10,4))
plt.plot(x, f(x), label='Function')
plt.scatter(X, y, label='Samples')
plt.plot(x, GP_mean, label='GP mean')
plt.fill_between(
    x.ravel(),
    GP_mean - 1.96 * np.sqrt(GP_variance),
    GP_mean + 1.96 * np.sqrt(GP_variance),
    alpha=0.5,
    label=r"95% confidence interval",)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#%% Multi-fidelity GP example

# Low and high fidelity functions
def LF_function(x):
    return 0.5*((x*6-2)**2)*np.sin((x*6-2)*2)+(x-0.5)*10. - 5

def HF_function(x):
    return ((x*6-2)**2)*np.sin((x*6-2)*2)


# Design space
x = np.linspace(0, 1, 101, endpoint = True).reshape(-1,1)



# Samples
ndim=1
Xt_c = np.linspace(0, 1, 11, endpoint=True).reshape(-1, ndim)

# To have nested samples
random.seed(26)
Xt_e = np.array(random.sample(list(Xt_c.T[0]),4)).reshape(-1, ndim)


# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)


# Multi-fidelity GPR
MFK = MFGPR(sigma_l=0, epsilon_l=0.1, sigma_h=0, epsilon_h=0.1, rho=1)
MFK.learn(Xt_c, yt_c, Xt_e, yt_e)

# High-fidelity GP mean and variance
GP_mean = np.array([MFK.GP_h(i) for i in x]).ravel()
GP_variance = np.array([MFK.variance_h(i) for i in x]).ravel()


# Plots
plt.rc('font', family='serif')
lw=3
# Low fidelity and high fidelity functions and samples
plt.figure(figsize=(20, 12))
plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)

plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('x', fontsize=25)
plt.ylabel('y', fontsize=25)

plt.title('Samples', fontsize=35)




# Multi-fidelity GPR
plt.figure(figsize=(20, 12))
plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
plt.plot(x, GP_mean, linestyle = '-.' , label='Multifidelity GP', color='orange', lw=lw)

plt.fill_between(
    x.ravel(),
    GP_mean - 1.96 * (GP_variance),
    GP_mean + 1.96 * (GP_variance),
    alpha=0.3,
    label=r"95% confidence interval",
    color='orange')

plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)

plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('x', fontsize=25)
plt.ylabel(r'y and $\hat{y}$', fontsize=25)

plt.title('Multi-fidelity Gaussian Process Regression', fontsize=35)




#%% SIngle fidelity GP for the high fidelity function

GP = GaussianProcess(epsilon=0.1)
GP.learn(Xt_e, yt_e)
GP_mean = np.array([GP(i) for i in x]).ravel()
GP_variance = np.array([GP.variance(i) for i in x]).ravel()



lw=3

plt.figure(figsize=(20, 12))
plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
plt.plot(x, GP_mean, linestyle = '-.' , label='Standard GP', color='orange', lw=lw)

plt.fill_between(
    x.ravel(),
    GP_mean - 1.96 * (GP_variance),
    GP_mean + 1.96 * (GP_variance),
    alpha=0.3,
    label=r"95% confidence interval",
    color='orange')

plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)

plt.legend(fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('x', fontsize=25)
plt.ylabel(r'y and $\hat{y}$', fontsize=25)

plt.title('Single-fidelity Gaussian Process Regression', fontsize=35)

