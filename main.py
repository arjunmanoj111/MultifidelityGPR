#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:20:07 2024

@author: arjun"mnanoj
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True, noise=0.0, normalize=1)
mfUCB.set_initial_data(4,2, np.array([0, 1]), seed=3)

mfUCB.set_model()

kern = mfUCB.model.gpy_model.kern.kernels[1]
#%%
mfUCB.true_model()
mfUCB.plot_model()
#%%
forrester_min = mfUCB.max

mfUCB.set_acquisition(5, 0.001)
mfUCB.init_bayes_loop()

acq_low, acq_high, error = mfUCB.acquisition.acquisition_low_high(mfUCB.x_plot_low, mfUCB.x_plot_high)
#%%
x, f = mfUCB.run_bayes_loop(1, plot_opt=True, plot_acq=True)

mfUCB.plot_results()

#%% Fidelity-Weighted Optimization
mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True, normalize=0)
mfgp.set_initial_data(4, 1, np.array([0, 1]), seed=2)
mfgp.set_model()
mfgp.true_model()

mfgp.plot_model(figsize=(10,6))
mfgp.set_acquisition(5, 0.8)
# mfgp.set_acquisition(5, 1.1)
mfgp.init_bayes_loop()

mfgp.run_bayes_loop(15, plot_acq=True, plot_opt=True)
mfgp.plot_results()

#%% Comparison 50 experiments

exp = 50
iters = 13
ucbIterates = np.zeros((exp, iters+1))
ucbFidelities = np.zeros((exp, iters+1))

# mfgpIterates = np.zeros((exp, iters+1))
# mfgpFidelities = np.zeros((exp, iters+1))


for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True, noise=0.0, normalize=False)
    mfUCB.set_initial_data(4, 2, np.array([0, 1]), seed=i)
    mfUCB.set_model()
    mfUCB.true_model()

    mfUCB.set_acquisition(5, 0.001)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)

    l = len(mfUCB.iterates[:,1])
    ucbIterates[i][:l] = mfUCB.iterates[:,1]
    ucbFidelities[i][:l] = np.array(mfUCB.fidelity_list)

    # mfgp = multifidelityGPR(lf, hf, ndim=1, negate=True, noise=0.0, normalize=False)
    # mfgp.set_initial_data(4, 1, np.array([0, 1]), seed=i)
    # mfgp.set_model()
    # mfgp.true_model()

    # #mfgp.plot_model(figsize=(10,6))
    # mfgp.set_acquisition(5, 0.8)
    # mfgp.init_bayes_loop()

    # mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    # l = len(mfgp.iterates[:,1])
    # mfgpIterates[i][:l] = mfgp.iterates[:,1]
    # mfgpFidelities[i][:l] = np.array(mfgp.fidelity_list)

# np.save('datanew/mfgpIterates_forrester.npy', mfgpIterates)
# np.save('datanew/ucbIterates_forrester.npy', ucbIterates)

# np.save('datanew/mfgpFidelities_forrester.npy', mfgpFidelities)
# np.save('datanew/ucbFidelities_forrester.npy', ucbFidelities)


#%%
def multifidelity_plots(optimization_data, costs, true_opt=None, negate=False, bo=False):
    if bo:
        mfgpIterates, ucbIterates, mfgpFidelities, ucbFidelities, boIterates = optimization_data
    else:
        mfgpIterates, ucbIterates, mfgpFidelities, ucbFidelities = optimization_data[:4]
    iters = len(ucbIterates[0])
    exp = len(ucbIterates)
    
    mfgpMean = np.mean(mfgpIterates, axis=0)
    ucbMean = np.mean(ucbIterates, axis=0)
    
    if bo:
        boMean = np.mean(boIterates, axis=0)
        boStd = np.sqrt(np.var(boIterates, axis=0))
        iters2 = len(boIterates[0])
        
    if negate:
        mfgpMean = -mfgpMean
        ucbMean = -ucbMean
        if bo:
            boMean = -boMean


    mfgpStd = np.sqrt(np.var(mfgpIterates, axis=0))
    ucbStd = np.sqrt(np.var(ucbIterates, axis=0))
    
    
    mfgpFrac = np.mean(np.sum(mfgpFidelities, axis=1)/iters)
    ucbFrac = np.mean(np.sum(ucbFidelities, axis=1)/iters)
    
    # Convergence
    plt.figure(figsize=(10, 7))
    # plt.plot(np.arange(1, iters+1, 1), mfgpMean, label='Fidelity-Weighted Optimization ({:.2f})'.format(np.mean(mfgpFrac)), color='orange')
    plt.plot(np.arange(1, iters+1, 1), ucbMean, label='MF-GPR-UCB Optimization ({:.2f})'.format(np.mean(ucbFrac)), color='g')

    # plt.fill_between(np.arange(1, iters+1, 1), mfgpMean - 1.96*mfgpStd/np.sqrt(exp),
    #                  mfgpMean + 1.96*mfgpStd/np.sqrt(exp), 
    #                 alpha=0.2, color='orange')

    plt.fill_between(np.arange(1, iters+1, 1), ucbMean - 1.96*ucbStd/np.sqrt(exp),
                     ucbMean + 1.96*ucbStd/np.sqrt(exp), 
                    alpha=0.2, color='g')
    
    if bo:
        plt.plot(np.arange(1, iters2+1, 1), boMean, label='Standard Bayesian Optimization', color='r')
        
        plt.fill_between(np.arange(1, iters2+1, 1), boMean - 1.96*boStd/np.sqrt(exp),
                          boMean + 1.96*boStd/np.sqrt(exp), 
                        alpha=0.2, color='r')
    
    if true_opt:
        plt.hlines(true_opt, 1, iters, color='k', linestyle='-.', label='True optimum')
    
    plt.xlim(1, iters)
    plt.legend(fontsize=15)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Best high-fidelity solution',fontsize=15)
    
    # Cost plot
    l1, l2 = costs
    
    ucbCosts = ucbFidelities.copy()
    mfgpCosts = mfgpFidelities.copy()
    for i in range(len(ucbFidelities)):
        ucbCosts[i][ucbFidelities[i] == 0] += l1
        ucbCosts[i][ucbFidelities[i] == 1] += (l2-1)
        ucbCosts[i] = np.cumsum(ucbCosts[i])
        
    mfgpCosts = mfgpFidelities.copy()
    for i in range(len(ucbFidelities)):
        mfgpCosts[i][mfgpFidelities[i] == 0] += l1
        mfgpCosts[i][mfgpFidelities[i] == 1] += (l2-1)
        mfgpCosts[i] = np.cumsum(mfgpCosts[i])
        
    mfgpCostMean = np.mean(mfgpCosts, axis=0)
    ucbCostMean = np.mean(ucbCosts, axis=0)

    mfgpCostStd = np.sqrt(np.var(mfgpCosts, axis=0))
    ucbCostStd = np.sqrt(np.var(ucbCosts, axis=0))
    
    
    plt.figure(figsize=(10, 7))
    # plt.plot(np.arange(1, iters+1, 1), mfgpCostMean, label='Fidelity-Weighted Optimization', color='orange')
    plt.plot(np.arange(1, iters+1, 1), ucbCostMean, label='MF-GPR-UCB Optimization', color='g')

    # plt.fill_between(np.arange(1, iters+1, 1), mfgpCostMean - 1.96*mfgpCostStd/np.sqrt(exp),
    #                  mfgpCostMean + 1.96*mfgpCostStd/np.sqrt(exp), 
    #                 alpha=0.2, color='orange')

    plt.fill_between(np.arange(1, iters+1, 1), ucbCostMean - 1.96*ucbCostStd/np.sqrt(exp),
                     ucbCostMean + 1.96*ucbCostStd/np.sqrt(exp), 
                    alpha=0.2, color='g')
    
    if bo:
        plt.hlines(l2*iters2, 1, iters, color='k', linestyle='-.', label='Standard BO Cost')
    plt.xlim(1, iters)
    plt.legend(fontsize=15, loc = 'upper left')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Cost',fontsize=15)


mfgpIterates = np.load('datanew/mfgpIterates_forrester.npy')
# ucbIterates = np.load('datanew/ucbIterates_forrester.npy')

mfgpFidelities = np.load('datanew/mfgpFidelities_forrester.npy')
# ucbFidelities = np.load('datanew/ucbFidelities_forrester.npy')

# boIterates = np.load('datanew/boIterates_forrester.npy')

optim_data = [mfgpIterates, ucbIterates, mfgpFidelities, ucbFidelities]#, boIterates]
costs = [1,10]

multifidelity_plots(optim_data, costs, negate=True, true_opt=forrester_min, bo=0)

#%%

ucbIterates_amm = np.load('datanew/amm/ucbIterates_amm.npy')
ucbFidelities_amm = np.load('datanew/amm/ucbFidelities_amm.npy')
#%%
mfgpIterates_amm = np.load('datanew/amm/mfgpIterates_amm3.npy')
mfgpFidelities_amm = np.load('datanew/amm/mfgpFidelities_amm3.npy')

boIterates_amm = np.load('datanew/boIterates_amm.npy')

#%%
plt.plot(boIterates_amm[:, -1])
plt.plot(mfgpIterates_amm[:, -1])

#%%

def video(i, optimization_data, costs, true_opt=None, negate=False, ylim=None):
    
    mfgpIterates, ucbIterates, mfgpFidelities, ucbFidelities = optimization_data
    iters = len(mfgpIterates[0])
    exp = len(mfgpIterates)
    
    mfgpMean = np.mean(mfgpIterates, axis=0)
    ucbMean = np.mean(ucbIterates, axis=0)
    
    if negate:
        mfgpMean = -mfgpMean
        ucbMean = -ucbMean
        
    mfgpStd = np.sqrt(np.var(mfgpIterates, axis=0))
    ucbStd = np.sqrt(np.var(ucbIterates, axis=0))
    
    mfgpFrac = np.mean(np.sum(mfgpFidelities, axis=1)/iters)
    ucbFrac = np.mean(np.sum(ucbFidelities, axis=1)/iters)
    
    # Convergence
    ax1.plot(np.arange(1, iters+1, 1), mfgpMean, label='Fidelity-Weighted Optimization ({:.2f})'.format(np.mean(mfgpFrac)), color='orange')
    ax1.plot(np.arange(1, iters+1, 1), ucbMean, label='MF-GPR-UCB Optimization ({:.2f})'.format(np.mean(ucbFrac)), color='g')

    ax1.fill_between(np.arange(1, iters+1, 1), mfgpMean - 1.96*mfgpStd/np.sqrt(exp),
                     mfgpMean + 1.96*mfgpStd/np.sqrt(exp), 
                    alpha=0.2, color='orange')

    ax1.fill_between(np.arange(1, iters+1, 1), ucbMean - 1.96*ucbStd/np.sqrt(exp),
                     ucbMean + 1.96*ucbStd/np.sqrt(exp), 
                    alpha=0.2, color='g')
    if true_opt:
        ax1.hlines(true_opt, 1, iters, color='k', linestyle='-.', label='True optimum')
    
    ax1.set_xlim(1, iters)
    ax1.legend(fontsize=18, loc = 'upper left')
    ax1.tick_params(axis='x' , labelsize=15)  # Increase pad between ticks and axis
    ax1.tick_params(axis='y',labelsize=15)
    ax1.set_xlabel('Iterations', fontsize=18)
    ax1.set_ylabel('Best high-fidelity solution ',fontsize=18)
    
    hf = np.mean(np.sum(ucbFidelities, axis=1))
    lf = np.mean(iters - np.sum(ucbFidelities, axis=1))
    
    hf2 = np.mean(np.sum(mfgpFidelities, axis=1))
    lf2 = np.mean(iters - np.sum(mfgpFidelities, axis=1))
    
    
    # Data
    types_of_eval = ['Low-Fidelity', 'High-fidelity']
    fidelities = ['Fidelity-Weighted Optimization', 'MF-GPR-UCB Optimization']
    values = [[lf2, lf], [hf2, hf]]  # values corresponding to (eval 1, fidelity 1), (eval 2, fidelity 1), etc.

    # Number of groups (evaluations) and bars (fidelities)
    n_groups = len(types_of_eval)
    n_bars = len(fidelities)

    # Bar width and positions
    bar_width = 0.25
    index = np.arange(n_groups)  # positions for the evaluations

    # Create the figure and axis

    # Plot bars for each fidelity
    bar1 = ax2.bar(index, [v[0] for v in values], bar_width, label=fidelities[0], color='orange', alpha=0.5)  # Fidelity 1
    bar2 = ax2.bar(index + bar_width, [v[1] for v in values], bar_width, label=fidelities[1], color='green', alpha=0.5)  # Fidelity 2

    ax2.tick_params(axis='x' , labelsize=15)  # Increase pad between ticks and axis
    ax2.tick_params(axis='y',labelsize=15)
    if ylim:
       ax2.set_ylim(ylim[0], ylim[1]) 
    else:
        ax2.set_ylim(0, 8)

    # Set labels and title
    ax2.set_xlabel('Fidelity Type', fontsize=18)
    ax2.set_ylabel('Average number of evaluations', fontsize=18)

    # Add ticks for evaluations
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(types_of_eval)
    
    # Add a legend
    ax2.legend(fontsize=15)


    # Customize layout
    fig.tight_layout()
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [3, 2]})

#%%
    
    
# Animation Function
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    ax1.clear()
    ax2.clear()    
    optim_data = [mfgpIterates[:, :i+2], ucbIterates[:, :i+2], mfgpFidelities[:, :i+2], ucbFidelities[:, :i+2]]
    video(i, optim_data, costs, true_opt=mfUCB.max, negate=True)
    
ani = animation.FuncAnimation(fig, animate, frames=len(mfgpIterates[0])+3) #animation function. frames will iterate i in animate
ani.save('datanew/forrester.gif', writer='pillow', fps=2) #save animation


#%% 2-D Bohachevsky function

def lf(x):
    return mf2.bohachevsky.low(x).reshape(-1,1)

def hf(x):
    return mf2.bohachevsky.high(x).reshape(-1,1)


#%% Multifidelity UCB

mfUCB = multifidelityUCB(lf, hf, ndim=2, noise=0)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=1)
mfUCB.set_model()
mfUCB.true_model()
mfUCB.plot_model(figsize=(10,5))
bohachevsky_max = mfUCB.max
#%%
mfUCB.set_acquisition(5, 0.001)
mfUCB.init_bayes_loop()

x, f = mfUCB.run_bayes_loop(30, plot_opt=True)

mfUCB.plot_results()

#%% Fidelity-Weighted Optimization

mfgp = multifidelityGPR(lf, hf, ndim=2, noise=0)
mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=1)
mfgp.set_model()
mfgp.true_model()
mfgp.plot_model()
# mfgp.set_acquisition(5, 0.5)
mfgp.set_acquisition(4, 0.5)
mfgp.init_bayes_loop()

x, f = mfgp.run_bayes_loop(30, plot_opt=True)
mfgp.plot_results()

#%% Comparison 50 experiments
exp = 50
iters = 30
# ucbIterates2 = np.zeros((exp, iters+1))
# ucbxIterates2 = np.zeros((exp, iters+1))
# ucbFidelities2 = np.zeros((exp, iters+1))

mfgpIterates2 = np.zeros((exp, iters+1))
mfgpFidelities2 = np.zeros((exp, iters+1))
mfgpxIterates2 = np.zeros((exp, iters+1))

for i in range(exp):
    print(i)
    # mfUCB = multifidelityUCB(lf, hf, ndim=2, noise=0)
    # mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=i)
    # mfUCB.set_model()
    # mfUCB.true_model()
    
    # mfUCB.set_acquisition(5, 0.001)
    # mfUCB.init_bayes_loop()
    
    
    # x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    
    # l = len(mfUCB.iterates[:,-1])
    # ucbIterates2[i][:l] = mfUCB.iterates[:,-1]
    # ucbFidelities2[i][:l] = np.array(mfUCB.fidelity_list)
    # ucbxIterates2[i][:l] = np.linalg.norm(mfUCB.iterates[:,:-1] - mfUCB.loc, axis=1)
    
    mfgp = multifidelityGPR(lf, hf, ndim=2, noise=0)
    mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=i)
    mfgp.set_model()
    #mfgp.true_model()

    mfgp.set_acquisition(2, 0.2)
    mfgp.init_bayes_loop()

    mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    l = len(mfgp.iterates[:,-1])
    mfgpIterates2[i][:l] = mfgp.iterates[:,-1]
    mfgpFidelities2[i][:l] = np.array(mfgp.fidelity_list)
    mfgpxIterates2[i][:l] = np.linalg.norm(mfgp.iterates[:,:-1] - mfUCB.loc, axis=1)

# np.save('datanew/mfgpIterates_bohachevsky.npy', mfgpIterates2)
# np.save('datanew/ucbIterates_bohachevsky.npy', ucbIterates2)

# np.save('datanew/mfgpFidelities_bohachevsky.npy', mfgpFidelities2)
# np.save('datanew/ucbFidelities_bohachevsky.npy', ucbFidelities2)

#%% Plots

# mfgpIterates2 = np.load('datanew/mfgpIterates_bohachevsky.npy')
ucbIterates2 = np.load('datanew/ucbIterates_bohachevsky.npy')
# boIterates2 = np.load('datanew/boIterates_bohachevsky.npy')

# mfgpFidelities2 = np.load('datanew/mfgpFidelities_bohachevsky.npy')
ucbFidelities2 = np.load('datanew/ucbFidelities_bohachevsky.npy')

#%%

optim_data2 = [mfgpIterates2, ucbIterates2, mfgpFidelities2, ucbFidelities2]
costs = [1,10]

multifidelity_plots(optim_data2, costs, true_opt=bohachevsky_max, bo=0)
#%%
# Animation Function
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    ax1.clear()
    ax2.clear()    
    optim_data = [mfgpIterates2[:, :i+2], ucbIterates2[:, :i+2], mfgpFidelities2[:, :i+2], ucbFidelities2[:, :i+2]]
    video(i, optim_data, costs, true_opt=bohachevsky_max, ylim=(0, 30))
    
ani = animation.FuncAnimation(fig, animate, frames=len(mfgpIterates2[0])+4) #animation function. frames will iterate i in animate
ani.save('datanew/bohachevsky.gif', writer='pillow', fps=4) #save animation

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
                         [0, 10], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    # return abs(10 - sol1.t[np.argmin(abs(sol1.y[2,:] - 0.67*S0))])
    return 0.2*abs(sol1.y[2,-1] - 0.67*S0)

 # exact_t.append(0.2*abs(sol1.y[2,-1] - 0.67*S0))
 # qssa_t.append(0.2*abs(sol2.y[1,-1] - 0.67*S0))


def low_fidelity_toy(E):
    S0 = 5.0
    C0 = np.array([S0, 0.0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    kvals = np.array([10, 50, 2, E])
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 10], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    # return abs(10 - sol2.t[np.argmin(abs(sol2.y[1,:] - 0.67*S0))])
    return 0.2*abs(sol2.y[1,-1] - 0.67*S0)

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

mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True, noise=0, normalize=1)
mfUCB.set_initial_data(4, 2, np.array([0, 1]), seed=3)
mfUCB.set_model(ker='mat')
# mfUCB.true_model()

mfUCB.set_acquisition(5, 0.01)
mfUCB.init_bayes_loop()
x,f = mfUCB.run_bayes_loop(15, plot_opt=True, plot_acq=True)
mfUCB.plot_results()
#%%
x,f = mfUCB.run_bayes_loop(4)


#%% Fidelity-Weighted Optimization

mfgp = multifidelityGPR(lf, hf, ndim=1, noise=0, negate=True, normalize=1)
mfgp.set_initial_data(4, 2, np.array([0, 1]), seed=1)
mfgp.set_model(ker='mat')
# mfgp.true_model()

mfgp.plot_model(figsize=(10,6))

mfgp.set_acquisition(5, 0.9)
mfgp.init_bayes_loop()

mfgp.run_bayes_loop(15, plot_opt=True)

mfgp.plot_results()

#%% Comparison 50 experiments
exp = 50
iters = 15
ucbIterates3 = np.zeros((exp, iters+1))
ucbFidelities3 = np.zeros((exp, iters+1))

# mfgpIterates3 = np.zeros((exp, iters+1))
# mfgpFidelities3 = np.zeros((exp, iters+1))

ucbEvals3 = []
mfgpEvals3 = []
for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=1, negate=True, noise=0, normalize=1)
    mfUCB.set_initial_data(4, 2, np.array([0, 1]), seed=i)
    mfUCB.set_model(ker='mat')
    #mfUCB.true_model()
    mfUCB.set_acquisition(5, 0.5)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    
    l = len(mfUCB.iterates[:,-1])
    ucbIterates3[i][:l] = mfUCB.iterates[:,-1]
    ucbFidelities3[i][:l] = np.array(mfUCB.fidelity_list)
    
    # mfgp = multifidelityGPR(lf, hf, ndim=1, noise=0, negate=True, normalize=1)
    # mfgp.set_initial_data(4, 2, np.array([0, 1]), seed=i)
    # mfgp.set_model(ker='mat')
    # #mfgp.true_model()

    # mfgp.set_acquisition(5, 0.9)
    # mfgp.init_bayes_loop()

    # mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    # l = len(mfgp.iterates[:,-1])
    # mfgpIterates3[i][:l] = mfgp.iterates[:,-1]
    # mfgpFidelities3[i][:l] = np.array(mfgp.fidelity_list)
    # mfgpEvals3.append([mfgp.n_hf_evals+1, mfgp.n_lf_evals])


#%% Save
np.save('datanew/mfgpIterates_enzyme.npy', mfgpIterates3)
np.save('datanew/ucbIterates_enzyme.npy', ucbIterates3)

np.save('datanew/mfgpFidelities_enzyme.npy', mfgpFidelities3)
np.save('datanew/ucbFidelities_enzyme.npy', ucbFidelities3)

#%%

mfgpIterates3 = np.load('datanew/mfgpIterates_enzyme.npy')
ucbIterates3 = np.load('datanew/ucbIterates_enzyme.npy')
boIterates3 = np.load('datanew/boIterates_enzyme.npy')

mfgpFidelities3 = np.load('datanew/mfgpFidelities_enzyme.npy')
ucbFidelities3 = np.load('datanew/ucbFidelities_enzyme.npy')
#%%

optim_data3 = [mfgpIterates3, ucbIterates3, mfgpFidelities3, ucbFidelities3]#, boIterates3]
costs = [1, 10]

multifidelity_plots(optim_data3, costs, negate=True, true_opt=0.0000523335, bo=0)

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [3, 2]})
# Animation Function
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    ax1.clear()
    ax2.clear()
    optim_data = [mfgpIterates3[:, :i+2], ucbIterates3[:, :i+2], mfgpFidelities3[:, :i+2], ucbFidelities3[:, :i+2]]
    video(i, optim_data, costs, true_opt=0.0001, negate=True, ylim=(0, 15))


ani = animation.FuncAnimation(fig, animate, frames=len(mfgpIterates3[0])+3) #animation function. frames will iterate i in animate
ani.save('datanew/enzyme.gif', writer='pillow', fps=3) #save animation


#%% Himmelblau 2D

def lf(x):
    lb = mf2.himmelblau.l_bound
    ub = mf2.himmelblau.u_bound
    x1 = x*(ub-lb) + lb
    return -mf2.himmelblau.low(x1).reshape(-1,1)

def hf(x):
    lb = mf2.himmelblau.l_bound
    ub = mf2.himmelblau.u_bound
    x1 = x*(ub-lb) + lb
    return -mf2.himmelblau.high(x1).reshape(-1,1)


mfUCB = multifidelityUCB(lf, hf, ndim=2, noise=0, negate=True)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=2)
mfUCB.set_model()
mfUCB.true_model()
mfUCB.plot_model(figsize=(10,5))
#%%
mfUCB.set_acquisition(5, 0.001)
mfUCB.init_bayes_loop()

x, f = mfUCB.run_bayes_loop(40, plot_opt=True, plot_acq=True)

mfUCB.plot_results()

#%%
mfgp = multifidelityGPR(lf, hf, ndim=2, noise=0, negate=True)
mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=6)
mfgp.set_model()
mfgp.true_model()
mfgp.plot_model()

mfgp.set_acquisition(5, -0.6)
mfgp.init_bayes_loop()

x, f = mfgp.run_bayes_loop(40, plot_opt=True)
mfgp.plot_results()




#%% Comparison 50 experiments
exp = 50
iters = 40
ucbIterates4 = np.zeros((exp, iters+1))
mfgpIterates4 = np.zeros((exp, iters+1))
ucbEvals4 = []
mfgpEvals4 = []
for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=2, noise=0, negate=True)
    mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=i)
    mfUCB.set_model()
    #mfUCB.true_model()
    
    mfUCB.set_acquisition(5, 0.001)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    
    l = len(mfUCB.iterates[:,-1])
    ucbIterates4[i][:l] = mfUCB.iterates[:,-1]
    
    ucbEvals4.append([mfUCB.n_hf_evals+1,mfUCB.n_lf_evals])
    
    mfgp = multifidelityGPR(lf, hf, ndim=2, noise=0, negate=True)
    mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1]]), seed=i)
    mfgp.set_model()
    #mfgp.true_model()

    mfgp.set_acquisition(5, -0.6)
    mfgp.init_bayes_loop()

    mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False)
    l = len(mfgp.iterates[:,-1])
    mfgpIterates4[i][:l] = mfgp.iterates[:,-1]
    mfgpEvals4.append([mfgp.n_hf_evals+1, mfgp.n_lf_evals])

np.save('datanew/mfgpIterates_himmelblau.npy', mfgpIterates4)
np.save('datanew/ucbIterates_himmelblau.npy', ucbIterates4)

np.save('datanew/mfgpEvals_himmelblau.npy', mfgpEvals4)
np.save('datanew/ucbEvals_himmelblau.npy', ucbEvals4)    



#%%

mfgpIterates4 = np.load('datanew/mfgpIterates_himmelblau.npy')
ucbIterates4 = np.load('datanew/ucbIterates_himmelblau.npy')
boIterates4 = np.load('datanew/boIterates_himmelblau.npy')

mfgpFidelities4 = np.load('datanew/mfgpFidelities_himmelblau.npy')
ucbFidelities4 = np.load('datanew/ucbFidelities_himmelblau.npy')
#%%

optim_data4 = [mfgpIterates4, ucbIterates4, mfgpFidelities4, ucbFidelities4, boIterates4]
costs = [1,10]

multifidelity_plots(optim_data4, costs, negate=True, true_opt=0.00001, bo=1)

#%%
# Animation Function
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    ax1.clear()
    ax2.clear()    
    optim_data = [mfgpIterates4[:, :i+2], ucbIterates4[:, :i+2], mfgpFidelities4[:, :i+2], ucbFidelities4[:, :i+2]]
    video(i, optim_data, costs, true_opt=0.0001, negate=True, ylim=(0, 25))
    
ani = animation.FuncAnimation(fig, animate, frames=len(mfgpIterates4[0])+4) #animation function. frames will iterate i in animate
ani.save('datanew/himmelblau.gif', writer='pillow', fps=4) #save animation


#%% 8-D Borehole function
def lf(x):
    lb = mf2.borehole.l_bound
    ub = mf2.borehole.u_bound
    x1 = x*(ub-lb) + lb
    return mf2.borehole.low(x1).reshape(-1,1)

def hf(x):
    lb = mf2.borehole.l_bound
    ub = mf2.borehole.u_bound
    x1 = x*(ub-lb) + lb
    return mf2.borehole.high(x1).reshape(-1,1)

#%% multifidelity UCB
mfUCB = multifidelityUCB(lf, hf, ndim=8, noise=0.0)
mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), seed=36)
mfUCB.set_model()
mfUCB.true_model()

#%%
mfUCB.set_acquisition(2, 2)
mfUCB.init_bayes_loop()
x,f = mfUCB.run_bayes_loop(100, min_hf_evals=0, is_GP=False)
mfUCB.plot_results(is_GP=False)


#%% fidelity-weighted optimization
mfgp = multifidelityGPR(lf, hf, ndim=8, noise=0.0)
mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]))
mfgp.set_model()
#mfUCB.true_model()

mfgp.set_acquisition(2, -0.1)
mfgp.init_bayes_loop()
x,f = mfgp.run_bayes_loop(45, min_hf_evals=0, is_GP=False)
mfgp.plot_results(is_GP=False)


#%% Comparison 50 experiments
exp = 50
iters = 100
ucbIterates5 = np.zeros((exp, iters+1))
mfgpIterates5 = np.zeros((exp, iters+1))
ucbEvals5 = []
mfgpEvals5 = []
for i in range(exp):
    print(i)
    mfUCB = multifidelityUCB(lf, hf, ndim=8, noise=0.0)
    mfUCB.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), seed=i)
    mfUCB.set_model()
    #mfUCB.true_model()
    
    mfUCB.set_acquisition(2, 2)
    mfUCB.init_bayes_loop()

    x, f = mfUCB.run_bayes_loop(iters, plot_opt=False, plot_acq=False, is_GP=False)
    
    l = len(mfUCB.iterates[:,-1])
    ucbIterates5[i][:l] = mfUCB.iterates[:,-1]
    
    ucbEvals5.append([mfUCB.n_hf_evals,mfUCB.n_lf_evals])
    
    mfgp = multifidelityGPR(lf, hf, ndim=8, noise=0.0)
    mfgp.set_initial_data(12, 3, np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]), seed=i)
    mfgp.set_model()
    #mfgp.true_model()

    mfgp.set_acquisition(2, -0.1)
    mfgp.init_bayes_loop()

    mfgp.run_bayes_loop(iters, plot_opt=False, plot_acq=False, is_GP=False)
    l = len(mfgp.iterates[:,-1])
    mfgpIterates5[i][:l] = mfgp.iterates[:,-1]
    mfgpEvals5.append([mfgp.n_hf_evals, mfgp.n_lf_evals])
    
np.save('datanew/mfgpIterates_borehole.npy', mfgpIterates5)
np.save('datanew/ucbIterates_borehole.npy', ucbIterates5)

np.save('datanew/mfgpEvals_borehole.npy', mfgpEvals5)
np.save('datanew/ucbEvals_borehole.npy', ucbEvals5)


#%% 

mfgpIterates5 = np.load('datanew/mfgpIterates_borehole.npy')
ucbIterates5 = np.load('datanew/ucbIterates_borehole.npy')

mfgpFidelities5 = np.load('datanew/mfgpFidelities_borehole.npy')
ucbFidelities5 = np.load('datanew/ucbFidelities_borehole.npy')

mfgpIterates5 = mfgpIterates5[:,:-1]
ucbIterates5 =  ucbIterates5[:,:-1]

mfgpFidelities5 = mfgpFidelities5[:,:-1]
ucbFidelities5 = ucbFidelities5[:,:-1]

optim_data5 = [mfgpIterates5, ucbIterates5, mfgpFidelities5, ucbFidelities5]
costs = [1,10]

multifidelity_plots(optim_data5, costs, true_opt=mfUCB.max)

# Animation Function
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    ax1.clear()
    ax2.clear()    
    optim_data = [mfgpIterates5[:, :i+2], ucbIterates5[:, :i+2], mfgpFidelities5[:, :i+2], ucbFidelities5[:, :i+2]]
    video(i, optim_data, costs, true_opt=mfUCB.max, ylim=(0, 95))
    
ani = animation.FuncAnimation(fig, animate, frames=len(mfgpIterates5[0])+4) #animation function. frames will iterate i in animate
ani.save('datanew/borehole.gif', writer='pillow', fps=6) #save animation

#%% High fidelity KDE plot

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

# # Generate random data (for demonstration)
# data = np.array(ucbEvals)[:,0]
# data2 = np.array(mfgpEvals)[:,0]

# # Create a gaussian KDE (kernel density estimate)
# kde = gaussian_kde(data, bw_method=0.4)  # 'bw_method' controls the smoothness of the plot

# kde2 = gaussian_kde(data2, bw_method=0.4)  # 'bw_method' controls the smoothness of the plot

# # Generate a range of values over which to evaluate the KDE
# x_values = np.linspace(0, 14, 1000)
# x_values2 = np.linspace(0, 14, 1000)

# # Evaluate the KDE at each point in the range
# kde_values = kde(x_values)
# kde_values2 = kde2(x_values2)



# # Create the plot
# plt.figure(figsize=(8,5))
# plt.plot(x_values, kde_values, label='Combined UCB method', color='g')
# plt.plot(x_values2, kde_values2, label='Fidelity-Weighted method', color='orange')
# plt.fill_between(x_values, kde_values, alpha=0.2, color='g')  # Add some shading under the curve for aesthetics
# plt.fill_between(x_values2, kde_values2, alpha=0.2, color='orange')  # Add some shading under the curve for aesthetics
# plt.title('Probability Density; High-fidelity evaluations', fontsize=14)
# plt.xlabel('Number of high-fidelity evaluations', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.xticks(fontsize=11)
# plt.yticks(fontsize=11)
# plt.xlim(1,14)
# plt.legend(fontsize=12)
# plt.show()