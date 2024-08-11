#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:35 2024

@author: arjunmnanoj
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
MFK = MFGPR(sigma_l=0, epsilon_l=0.1, sigma_h=0, epsilon_h=0.1, rho=0.5)
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
GP.learn(Xt_c, yt_c)
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



#%% Fidelity weighted acquisition function

# Low and high fidelity functions
def LF_function(x):
    return 0.5*((x*6-2)**2)*np.sin((x*6-2)*2)+(x-0.5)*10. - 5

def HF_function(x):
    return ((x*6-2)**2)*np.sin((x*6-2)*2)


# Initial samples

ndim=1
Xt_c = np.linspace(0, 1, 11, endpoint=True).reshape(-1, ndim)

# To have nested samples
random.seed(26)
Xt_e = np.array(random.sample(list(Xt_c.T[0]),4)).reshape(-1, ndim)


# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)

b = 1
n1, n2 = 0,0
n_iter = 0
b1,b2 = 1,1
for i in range(10):
    n_iter += 1
    # Multi-fidelity GPR
    MFK = MFGPR(sigma_l=0, epsilon_l=0.2, sigma_h=0, epsilon_h=0.2, rho=1)
    MFK.learn(Xt_c, yt_c, Xt_e, yt_e)
    
    # High-fidelity GP mean and variance
    GP_mean_h = np.array([MFK.GP_h(i) for i in x]).ravel()
    GP_variance_h = np.array([MFK.variance_h(i) for i in x]).ravel()
    
    # Low fidelity GP
    GP = GaussianProcess(epsilon=0.1)
    GP.learn(Xt_c, yt_c)
    GP_mean_l = np.array([GP(i) for i in x]).ravel()
    GP_variance_l = np.array([GP.variance(i) for i in x]).ravel()
    
    phi_h = GP_mean_h - b * GP_variance_h
    phi_l = GP_mean_l - b * GP_variance_l
    
    # Fidelity weighted cost
    cost_l = ((b1/b2)*(n1 + 1) + n2)
    cost_h = ((b2/b1)*(n1) +  (n2 + 1))
    
    # print(phi_l, cost_l)
    # print(phi_h, cost_h)
    print(np.min(phi_h), np.min(phi_l))
    
    phi_l += cost_l
    phi_h += cost_h
    
    print(np.min(phi_h), np.min(phi_l))
    
    if np.min(phi_h) < np.min(phi_l):
        xnew = x[np.argmin(phi_h)]
        ynew = HF_function(xnew)
        Xt_e = np.vstack([Xt_e, xnew])
        yt_e = np.vstack([yt_e, ynew])
        n2 += 1
        
    else:
        xnew = x[np.argmin(phi_l)]
        ynew = LF_function(xnew)
        Xt_c = np.vstack([Xt_c, xnew])
        yt_c = np.vstack([yt_c, ynew])
        n1 += 1
        
#%%        



#%% MF GP UCB


# Low and high fidelity functions
def LF_function(x):
    return 0.5*((x*6-2)**2)*np.sin((x*6-2)*2)+(x-0.5)*10. - 5

def HF_function(x):
    return ((x*6-2)**2)*np.sin((x*6-2)*2)


# Initial samples

ndim=1
Xt_c = np.linspace(0, 1, 4, endpoint=True).reshape(-1, ndim)

# To have nested samples
random.seed(26)
Xt_e = np.array(random.sample(list(Xt_c.T[0]),1)).reshape(-1, ndim)


# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)



e = np.array([9])
l = np.array([0.0001, 1])
v = np.array([e[i]*np.sqrt(l[i]/l[i+1]) for i in range(e.size)])

b = 100
b2 = 500
n1, n2 = 0,0


phi_1_list =[]
phi_2_list=[]
phi_list=[]
Xt_e_list=[]
Xt_c_list=[]
xnew_list=[]
yt_e_list=[]
yt_c_list=[]
ynew_list=[]


for i in range(22):
    n_iter += 1
    # Low fidelity GP
    
    GP1 = GaussianProcess(epsilon=0.1)
    GP1.learn(Xt_c, -yt_c)
    GP_mean_1 = np.array([GP1(i) for i in x]).ravel()
    GP_variance_1 = np.array([GP1.variance(i) for i in x]).ravel()
    GP_sig_1 = np.sqrt(GP_variance_1)
    
    phi_1 = GP_mean_1 + np.sqrt(b) * GP_sig_1 + e
    
    GP2 = GaussianProcess(epsilon=0.1)
    GP2.learn(Xt_e, -yt_e)
    GP_mean_2 = np.array([GP2(i) for i in x]).ravel()
    GP_variance_2 = np.array([GP2.variance(i) for i in x]).ravel()
    GP_sig_2 = np.sqrt(GP_variance_2)
    
    phi_2 = GP_mean_2 + np.sqrt(b2) * GP_sig_2
    
    phi = np.array([min(phi_1[i], phi_2[i]) for i in range(len(phi_1))])
    
    xnew = x[np.argmax(phi)]
    
    if np.sqrt(b) *  GP_sig_1[np.argmax(phi)] > v[0]:
        ynew = LF_function(xnew)
        Xt_c = np.vstack([Xt_c, xnew])
        yt_c = np.vstack([yt_c, ynew])
        n1 += 1
        
    else:
        ynew = HF_function(xnew)
        Xt_e = np.vstack([Xt_e, xnew])
        yt_e = np.vstack([yt_e, ynew])
        n2 += 1
        
    
    phi_1_list.append(phi_1)
    phi_2_list.append(phi_2)
    phi_list.append(phi)
    Xt_e_list.append(Xt_e)
    Xt_c_list.append(Xt_c)
    xnew_list.append(xnew)
    yt_e_list.append(yt_e)
    yt_c_list.append(yt_c)
    ynew_list.append(ynew)
    
    plt.figure(figsize=(20, 12))
    plt.plot(x, -HF_function(x), label ='High Fidelity function', lw=lw)
    plt.plot(x, -LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    plt.plot(x, phi_1, linestyle = '-.' , label=r'$\phi^{(1)}$', color='blue', lw=lw)
    plt.plot(x, phi_2, linestyle = '-.' , label=r'$\phi^{(2)}$', color='brown', lw=lw)
    plt.plot(x, phi, linestyle = '-.' , label=r'$\phi$', color='green', lw=lw)

    plt.scatter(Xt_e, -yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c, -yt_c, marker = '*' , color ='g', label ='LF samples', s=120)
    plt.scatter(xnew, -ynew, marker = '*' , color ='r', label ='Next evaluation', s=520, zorder=2)
    plt.vlines(xnew, -25, -ynew, linestyle='-.', color='r', linewidth=6)

    plt.legend(fontsize=25,loc='lower left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(-25, 25)

    plt.title('Multi-Fidelity UCB', fontsize=35)

#%%
fig = plt.figure(figsize=(20, 12))


# Animation Function 
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    fig.clear()

    plt.plot(x, -HF_function(x), label ='High Fidelity function', lw=lw)
    plt.plot(x, -LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    plt.plot(x, phi_1_list[i], linestyle = '-.' , label=r'$\phi^{(1)}$', color='blue', lw=lw)
    plt.plot(x, phi_2_list[i], linestyle = '-.' , label=r'$\phi^{(2)}$', color='brown', lw=lw)
    plt.plot(x, phi_list[i], linestyle = '-.' , label=r'$\phi$', color='green', lw=lw)

    plt.scatter(Xt_e_list[i], -yt_e_list[i], marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c_list[i], -yt_c_list[i], marker = '*' , color ='g', label ='LF samples', s=120)
    plt.scatter(xnew_list[i], -ynew_list[i], marker = '*' , color ='r', label ='Next evaluation', s=520, zorder=2)
    plt.vlines(xnew_list[i], -30, -ynew_list[i], linestyle='-.', color='r', linewidth=6)

    plt.legend(fontsize=25,loc='lower left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(-30, 25) 
    plt.title('Multi-Fidelity UCB', fontsize=35)
    
    
    

ani = animation.FuncAnimation(fig, animate, frames=(len(xnew_list))) #animation function. frames will iterate i in animate
ani.save('MF UCB.gif', writer='pillow', fps=1) #save animation
    

    
    
    
