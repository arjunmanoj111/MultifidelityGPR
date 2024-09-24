#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:20:07 2024

@author: arjunmnanoj
"""

import numpy as np
from mfgp import multifidelityGPR 
from mfucb import multifidelityUCB
import mf2

#%%
def low_fidelity_toy(x):
    return 0.5*high_fidelity_toy(x) + 10*(x-0.5) -5

def high_fidelity_toy(x):
    return ((x*6-2)**2)*np.sin((x*6-2)*2)

def lf(x):
    return -low_fidelity_toy(x)

def hf(x):
    return -high_fidelity_toy(x)


#%%

mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True, noise=0.0)
mfUCB.set_initial_data(12, 3, np.array([0, 1]))
mfUCB.set_model()
#mfgp.plot_model()
mfUCB.set_acquisition(50, 0.001)
mfUCB.true_model()
mfUCB.init_bayes_loop()

#%%
x, f = mfUCB.run_bayes_loop(4)
#%%
mfUCB.plot_results()


#%%

def lf(x):
    return mf2.bohachevsky.low(x).reshape(-1,1)

def hf(x):
    return mf2.bohachevsky.high(x).reshape(-1,1)

mfUCB = multifidelityUCB(lf, hf, ndim=2)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]))
mfUCB.set_model()
mfUCB.true_model()
#mfgp.plot_model()
mfUCB.set_acquisition(5, 1/35)
mfUCB.init_bayes_loop()
mfUCB.run_bayes_loop(30)

#%%

mfUCB.plot_results()

#%%

mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True)
n = np.array([12, 3])
mfgp.set_initial_data(12, 3, np.array([0, 1]))
mfgp.set_model()
mfgp.true_model()

mfgp.plot_model(figsize=(10,6))


mfgp.set_acquisition(10, 0.01)

mfgp.init_bayes_loop()

mfgp.run_bayes_loop(15)

mfgp.plot_results()



#%%

def lf(x):
    return mf2.bohachevsky.low(x).reshape(-1,1)

def hf(x):
    return mf2.bohachevsky.high(x).reshape(-1,1)

mfgp = multifidelityGPR(lf, hf, ndim=2)
mfgp.set_initial_data(12, 2, np.array([[0, 1], [0, 1]]))
mfgp.set_model()
mfgp.true_model()
mfgp.set_acquisition(5, 10)
mfgp.init_bayes_loop()

#%%
x, f = mfgp.run_bayes_loop(2)

#%%
mfgp.plot_results()


#%%

true_model = mfgp.multifidelity.f[1](mfgp.x_plot)
