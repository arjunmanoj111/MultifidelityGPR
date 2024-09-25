#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:20:07 2024

@author: arjun"mnanoj
"""

import numpy as np
import matplotlib.pyplot as plt
from mfgp import multifidelityGPR
from mfucb import multifidelityUCB
import mf2

#%% 1-D Forrester function

def low_fidelity_toy(x):
    return 0.5*high_fidelity_toy(x) + 10 * (x - 0.5) - 5

def high_fidelity_toy(x):
    return ((x*6 - 2)**2)*np.sin((x*6 - 2)*2)

def lf(x):
    return -low_fidelity_toy(x)
def hf(x):
    return -high_fidelity_toy(x)

#%% Multi-fidelity UCB
mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True)
mfUCB.set_initial_data(12, 3, np.array([0, 1]), seed=6)
mfUCB.set_model()
mfUCB.true_model()
mfUCB.plot_model()

mfUCB.set_acquisition(5, 1/10)
mfUCB.init_bayes_loop()

x, f = mfUCB.run_bayes_loop(10, plot_opt=True, plot_acq=True)
mfUCB.plot_results()

#%% Fidelity-Weighted Optimization

mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True)
n = np.array([12, 3])
mfgp.set_initial_data(12, 3, np.array([0, 1]), seed=6)
mfgp.set_model()
mfgp.true_model()

mfgp.plot_model(figsize=(10,6))
mfgp.set_acquisition(10, 0.01)
mfgp.init_bayes_loop()

mfgp.run_bayes_loop(10)
mfgp.plot_results()

#%% Comparison 50 experiments
exp = 50
iters = 20
ucbIterates = np.zeros((exp, iters+1))
mfgpIterates = np.zeros((exp, iters+1))
ucbFrac = []
mfgpFrac = []
for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True)
    mfUCB.set_initial_data(12, 3, np.array([0, 1]), seed=i)
    mfUCB.set_model()
    mfUCB.true_model()
    
    mfUCB.set_acquisition(5, 1/5)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    
    l = len(mfUCB.iterates[:,1])
    ucbIterates[i][:l] = mfUCB.iterates[:,1]
    
    ucbFrac.append(mfUCB.n_hf_evals/mfUCB.n_lf_evals)
    
    mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True)
    mfgp.set_initial_data(12, 3, np.array([0, 1]), seed=i)
    mfgp.set_model()
    mfgp.true_model()

    mfgp.plot_model(figsize=(10,6))
    mfgp.set_acquisition(10, 0.01)
    mfgp.init_bayes_loop()

    mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    l = len(mfgp.iterates[:,1])
    mfgpIterates[i][:l] = mfgp.iterates[:,1]
    mfgpFrac.append(mfgp.n_hf_evals/mfgp.n_lf_evals)
    
#%% Comparison plot 50 experiments

mfgpMean = -np.mean(mfgpIterates, axis=0)
ucbMean = -np.mean(ucbIterates, axis=0)

mfgpStd = np.sqrt(np.var(mfgpIterates, axis=0))
ucbStd = np.sqrt(np.var(ucbIterates, axis=0))

plt.figure()
plt.plot(np.arange(1, iters+2, 1), mfgpMean, label='Fidelity-Weighted Optimization; {:g}'.format(np.mean(mfgpFrac)), color='orange')
plt.plot(np.arange(1, iters+2, 1), ucbMean, label='Multifidelity UCB Optimization; {:g}'.format(np.mean(ucbFrac)), color='g')

plt.fill_between(np.arange(1, iters+2, 1), mfgpMean - 1.96*mfgpStd/np.sqrt(exp),
                 mfgpMean + 1.96*mfgpStd/np.sqrt(exp), 
                alpha=0.2, color='orange')

plt.fill_between(np.arange(1, iters+2, 1), ucbMean - 1.96*ucbStd/np.sqrt(exp),
                 ucbMean + 1.96*ucbStd/np.sqrt(exp), 
                alpha=0.2, color='g')
plt.legend()



#%% 2-D Bohachevsky function

def lf(x):
    return mf2.bohachevsky.low(x).reshape(-1,1)

def hf(x):
    return mf2.bohachevsky.high(x).reshape(-1,1)


#%% Multifidelity UCB

mfUCB = multifidelityUCB(lf, hf, ndim=2)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]))
mfUCB.set_model()
mfUCB.true_model()
mfUCB.plot_model(figsize=(10,5))
mfUCB.set_acquisition(5, 1/35)
mfUCB.init_bayes_loop()

x, f = mfUCB.run_bayes_loop(32)

mfUCB.plot_results()


#%% Fidelity-Weighted Optimization

mfgp = multifidelityGPR(lf, hf, ndim=2)
mfgp.set_initial_data(12, 2, np.array([[0, 1], [0, 1]]))
mfgp.set_model()
mfgp.true_model()
mfgp.plot_model()
mfgp.set_acquisition(5, 10)
mfgp.init_bayes_loop()

x, f = mfgp.run_bayes_loop(15)
mfgp.plot_results()

#%% Comparison 50 experiments
exp = 50
iters = 35
ucbIterates2 = np.zeros((exp, iters+1))
mfgpIterates2 = np.zeros((exp, iters+1))
ucbFrac2 = []
mfgpFrac2 = []
for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=2)
    mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=i)
    mfUCB.set_model()
    #mfUCB.true_model()
    
    mfUCB.set_acquisition(5, 1/35)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    
    l = len(mfUCB.iterates[:,1])
    ucbIterates2[i][:l] = mfUCB.iterates[:,1]
    
    ucbFrac2.append(mfUCB.n_hf_evals/mfUCB.n_lf_evals)
    
    mfgp = multifidelityGPR(lf, hf, ndim=2)
    mfgp.set_initial_data(12, 2, np.array([[0, 1], [0, 1]]), seed=i)
    mfgp.set_model()
    #mfgp.true_model()

    mfgp.set_acquisition(5, 10)
    mfgp.init_bayes_loop()

    mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    l = len(mfgp.iterates[:,1])
    mfgpIterates2[i][:l] = mfgp.iterates[:,1]
    mfgpFrac2.append(mfgp.n_hf_evals/mfgp.n_lf_evals)
    
#%% Comparison plot 50 experiments

mfgpMean2 = -np.mean(mfgpIterates2, axis=0)
ucbMean2 = -np.mean(ucbIterates2, axis=0)

mfgpStd2 = np.sqrt(np.var(mfgpIterates2, axis=0))
ucbStd2 = np.sqrt(np.var(ucbIterates2, axis=0))

plt.figure()
plt.plot(np.arange(1, iters+2, 1), mfgpMean2, label='Fidelity-Weighted Optimization; {:g}'.format(np.mean(mfgpFrac2)), color='orange')
plt.plot(np.arange(1, iters+2, 1), ucbMean2, label='Multifidelity UCB Optimization; {:g}'.format(np.mean(ucbFrac2)), color='g')

plt.fill_between(np.arange(1, iters+2, 1), mfgpMean2 - 1.96*mfgpStd2/np.sqrt(exp),
                 mfgpMean2 + 1.96*mfgpStd2/np.sqrt(exp), 
                alpha=0.2, color='orange')

plt.fill_between(np.arange(1, iters+2, 1), ucbMean2 - 1.96*ucbStd2/np.sqrt(exp),
                 ucbMean2 + 1.96*ucbStd2/np.sqrt(exp), 
                alpha=0.2, color='g')
plt.legend()



#%% Toy Enzyme model
from scipy.integrate import solve_ivp

# Enzyme Toy Models
def toyModel(t,x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1,k2,k3 = k
    dS0  = -k1*x[0]*x[3] + k2*x[1]
    dES0 =  k1*x[0]*x[3] - k2*x[1] - k3*x[1]
    dS1  =  k3*x[1] 
    dE   = -k1*x[3]*x[0] + k2*x[1] + k3*x[1] 
    return [dS0, dES0, dS1, dE]


def reducedToyModel(t,x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1,k2,k3, E0 = k
    ke = (k1*k3)/(k2+ k3)
    dS0  = -ke*x[0]*E0
    dS1  = ke*x[0]*E0
    return [dS0, dS1]


def high_fidelity_toy(E):
    S0 = 5.0
    C0 = np.array([S0, 0.0, 0.0, E]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    kvals = np.array([10, 50, 2])
    
    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals), 
                         [0, 100], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    return abs(10 - sol1.t[np.argmin(abs(sol1.y[2,:] - 0.67*S0))])


def low_fidelity_toy(E):
    S0 = 5.0
    C0 = np.array([S0, 0.0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    kvals = np.array([10, 50, 2, E])
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 100], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    return abs(10 - sol2.t[np.argmin(abs(sol2.y[1,:] - 0.67*S0))])

def lf(x):
    if (type(x)) == np.ndarray:
        lf_vals = np.array([low_fidelity_toy(i) for i in x.ravel()]).reshape(-1,1)
    else:
        lf_vals = low_fidelity_toy(x)
    return -lf_vals
def hf(x):
    if (type(x)) == np.ndarray:
        hf_vals = np.array([high_fidelity_toy(i) for i in x.ravel()]).reshape(-1,1)
    else:
        hf_vals = high_fidelity_toy(x)
    return -hf_vals

#%% Multifidelity UCB

mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True)
mfUCB.set_initial_data(12, 3, np.array([0, 1]))
mfUCB.set_model(ker='mat')
mfUCB.true_model()
mfgp.plot_model()
#%%
mfUCB.set_acquisition(5, 1/10)
mfUCB.init_bayes_loop()
x,f = mfUCB.run_bayes_loop(12)
mfUCB.plot_results()
#%%
x,f = mfUCB.run_bayes_loop(4)
mfUCB.plot_results()


#%% Fidelity-Weighted Optimization

mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True)
n = np.array([12, 3])
mfgp.set_initial_data(12, 3, np.array([0, 1]))
mfgp.set_model()
mfgp.true_model()

mfgp.plot_model(figsize=(10,6))

#%%
mfgp.set_acquisition(15, 10)
mfgp.init_bayes_loop()

mfgp.run_bayes_loop(15)

mfgp.plot_results()

#%% 8-D Borehole function

lb = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
ub = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])

def lf(x):
    x1 = x*(ub-lb) + lb
    return mf2.borehole.low(x1).reshape(-1,1)

def hf(x):
    x1 = x*(ub-lb) + lb
    return mf2.borehole.high(x1).reshape(-1,1)




#%% multifidelity UCB
mfUCB = multifidelityUCB(lf, hf, ndim=8, noise=0.0)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]))
mfUCB.set_model(ker='mat')
#mfUCB.true_model()

mfUCB.set_acquisition(5, 10)
mfUCB.init_bayes_loop()
x,f = mfUCB.run_bayes_loop(45, min_hf_evals=0)
mfUCB.plot_results()


#%% fidelity-weighted optimization
mfgp = multifidelityGPR(lf, hf, ndim=8, noise=0.0)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]))
mfUCB.set_model()
#mfUCB.true_model()

mfUCB.set_acquisition(5, 10)
mfUCB.init_bayes_loop()
x,f = mfUCB.run_bayes_loop(45, min_hf_evals=0)
mfUCB.plot_results()


#%%Experiment comparison
