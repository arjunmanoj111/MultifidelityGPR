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
from scipy.integrate import solve_ivp




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

GP = GaussianProcess(epsilon=0.05)
GP.learn(Xt_c, yt_c)
GP_mean = np.array([GP(i) for i in x]).ravel()
GP_variance = np.array([GP.variance(i) for i in x]).ravel()

lw=3

plt.figure(figsize=(20, 12))
plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
plt.plot(x, GP_mean, linestyle = '-.' , label='Standard GP', color='red', lw=lw)

plt.fill_between(
    x.ravel(),
    GP_mean - 1.96 * (GP_variance),
    GP_mean + 1.96 * (GP_variance),
    alpha=0.3,
    label=r"95% confidence interval",
    color='red')

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


GP_mean_h_list =[]
GP_mean_l_list=[]
GP_variance_h_list =[]
GP_variance_l_list=[]
Xt_e_list=[]
Xt_c_list=[]
xnew_list=[]
yt_e_list=[]
yt_c_list=[]
ynew_list=[]



beta1, beta2 = 500, 10
n1, n2 = 10,4
n_iter = 0
b1,b2 = 1, 1
for i in range(30):
    n_iter += 1
    # Multi-fidelity GPR
    MFK = MFGPR(sigma_l=0, epsilon_l=0.05, sigma_h=0, epsilon_h=0.05, rho=0.5)
    MFK.learn(Xt_c, yt_c, Xt_e, yt_e)
    
    # High-fidelity GP mean and variance
    GP_mean_h = np.array([MFK.GP_h(i) for i in x]).ravel()
    GP_variance_h = np.array([MFK.variance_h(i) for i in x]).ravel()
    
    # Low fidelity GP
    GP = GaussianProcess(epsilon=0.05, sigma=0)
    GP.learn(Xt_c, yt_c)
    GP_mean_l = np.array([GP(i) for i in x]).ravel()
    GP_variance_l = np.array([GP.variance(i) for i in x]).ravel()
    
    phi_l = GP_mean_l - beta1 * GP_variance_l
    phi_h = GP_mean_h - beta2 * GP_variance_h
    
    
    plt.figure(figsize=(20, 12))
    plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
    plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    plt.plot(x, GP_mean_h, linestyle = '-.' , label='Multifidelity GP', color='orange', lw=lw)
    plt.plot(x, GP_mean_l, linestyle = '-.' , label='Low-fidelity GP', color='red', lw=lw)

    plt.fill_between(
        x.ravel(),
        GP_mean_l - 1.96 * (GP_variance_l),
        GP_mean_l + 1.96 * (GP_variance_l),
        alpha=0.3,
        label=r"95% confidence interval",
        color='red')
    

    plt.fill_between(
        x.ravel(),
        GP_mean_h - 1.96 * (GP_variance_h),
        GP_mean_h + 1.96 * (GP_variance_h),
        alpha=0.3,
        label=r"95% confidence interval",
        color='orange')

    plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)

    
    # Fidelity weighted cost
    cost_l = ((b1/b2)*(n1 + 1) + n2)/n_iter
    cost_h = ((b2/b1)*(n1) +  (n2 + 1))/n_iter
    
    # print(phi_l, cost_l)
    # print(phi_h, cost_h)
    # print(np.min(phi_h), np.min(phi_l))
    
    phi_l += cost_l
    phi_h += cost_h
    
    # print(np.min(phi_h), np.min(phi_l))
    
    if np.min(phi_h) < np.min(phi_l):
        print('high fidelity')
        xnew = x[np.argmin(phi_h)]
        rec = False
        if xnew in Xt_e:
            rec = True
            phii = phi_h.copy()
        
        while rec == True:
            phii = np.delete(phii, np.argmin(phii)) 
            xnew = x[np.argmin(phii)]
            if xnew not in Xt_e:
                rec = False
            
        ynew = HF_function(xnew)
        Xt_e = np.vstack([Xt_e, xnew])
        yt_e = np.vstack([yt_e, ynew])
        n2 += 1
        
    else:
        print('low-fidelity')
        xnew = x[np.argmin(phi_l)]
        rec = False
        if xnew in Xt_c:
            rec = True
            phii = phi_l.copy()
        
        while rec == True:
            phii = np.delete(phii, np.argmin(phii)) 
            xnew = x[np.argmin(phii)]
            if xnew not in Xt_c:
                rec = False
        
        ynew = LF_function(xnew)
        Xt_c = np.vstack([Xt_c, xnew])
        yt_c = np.vstack([yt_c, ynew])
        n1 += 1
    
    plt.scatter(xnew, ynew, marker = '*' , color ='r', label ='next location', s=500, zorder=2)
    plt.vlines(xnew, -10, ynew, linestyle='-.', linewidth=5, color='r')
    plt.ylim(-10, 17)
    
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel(r'y and $\hat{y}$', fontsize=25)

    plt.title('Multi-fidelity Gaussian Process Regression', fontsize=35)
    
    
    GP_mean_h_list.append(GP_mean_h)
    GP_mean_l_list.append(GP_mean_l)
    GP_variance_h_list.append(GP_variance_h)
    GP_variance_l_list.append(GP_variance_l)
    Xt_e_list.append(Xt_e)
    Xt_c_list.append(Xt_c)
    xnew_list.append(xnew)
    yt_e_list.append(yt_e)
    yt_c_list.append(yt_c)
    ynew_list.append(ynew)
        
    
        
#%%        

fig = plt.figure(figsize=(20, 12))
fig.clear()

# Animation Function 
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    fig.clear()
    
    plt.plot(x, HF_function(x), label ='High Fidelity function', lw=lw)
    plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    plt.plot(x, GP_mean_h_list[i], linestyle = '-.' , label='Multifidelity GP', color='orange', lw=lw)
    plt.plot(x, GP_mean_l_list[i], linestyle = '-.' , label='Low-fidelity GP', color='red', lw=lw)
    
    plt.scatter(Xt_e_list[i], yt_e_list[i], marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c_list[i], yt_c_list[i], marker = '*' , color ='g', label ='LF samples', s=120)

    plt.fill_between(
        x.ravel(),
        GP_mean_l_list[i] - 1.96 * (GP_variance_l_list[i]),
        GP_mean_l_list[i] + 1.96 * (GP_variance_l_list[i]),
        alpha=0.3,
        label=r"95% confidence interval",
        color='red')
    

    plt.fill_between(
        x.ravel(),
        GP_mean_h_list[i] - 1.96 * (GP_variance_h_list[i]),
        GP_mean_h_list[i] + 1.96 * (GP_variance_h_list[i]),
        alpha=0.3,
        label=r"95% confidence interval",
        color='orange')

    plt.scatter(xnew_list[i], ynew_list[i], marker = '*' , color ='r', label ='next location', s=500, zorder=2)
    plt.vlines(xnew_list[i], -10, ynew_list[i], linestyle='-.', linewidth=5, color='r')
    plt.ylim(-10, 17)
    
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel(r'y and $\hat{y}$', fontsize=25)

    plt.title('Multi-fidelity Gaussian Process Regression', fontsize=35)
    
    
ani = animation.FuncAnimation(fig, animate, frames=(len(xnew_list))) #animation function. frames will iterate i in animate
ani.save('MFGP.gif', writer='pillow', fps=1) #save animation
    


#%% MF GP UCB

# Low and high fidelity functions
def LF_function(x):
    return 0.5*((x*6-2)**2)*np.sin((x*6-2)*2)+(x-0.5)*10. - 5

def HF_function(x):
    return ((x*6-2)**2)*np.sin((x*6-2)*2)



Xt_c, Xt_e, yt_c, yt_e = (np.array([0. , 0.2, 0.4, 0.6, 0.8, 1. ]),
 np.array([0.8, 0. , 0.6]),
 np.array([[-8.48639501],
        [-8.31986355],
        [-5.94261151],
        [-4.0747189 ],
        [-4.47456522],
        [ 7.91486597]]),
 np.array([[-4.94913044],
        [ 3.02720998],
        [-0.14943781]]))




Xt_c, Xt_e = (np.array([0.08, 0.12, 0.19, 0.22, 0.27, 0.43, 0.45, 0.53, 0.72, 0.88, 0.93, 0.96]),
 np.array([0.22, 0.45 , 0.93]))



# Initial samples

ndim=1

# Xt_c = np.linspace(0, 1, 4, endpoint=True).reshape(-1, ndim)

# # To have nested samples
# random.seed(26)
# Xt_e = np.array(random.sample(list(Xt_c.T[0]),1)).reshape(-1, ndim)


Xt_c = Xt_c.reshape(-1, ndim)
Xt_e = Xt_e.reshape(-1, ndim)

# Evaluate the HF and LF functions
yt_e = HF_function(Xt_e)
yt_c = LF_function(Xt_c)



e = np.array([9])
l = np.array([0.0005, 1])
v = np.array([e[i]*np.sqrt(l[i]/l[i+1]) for i in range(e.size)])

b = 10
b2 = 600
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
fidelitylist = []


best_min = []
best_loc = []
best_mean = []
best_mean_loc = []

n_iter = 0
for i in range(7):
    n_iter += 1
    # Low fidelity GP
    GP1 = GaussianProcess(epsilon=0.1)
    GP1.learn(Xt_c, -yt_c)
    GP_mean_1 = np.array([GP1(i) for i in x]).ravel()
    GP_variance_1 = np.array([GP1.variance(i) for i in x]).ravel()
    GP_sig_1 = np.sqrt(GP_variance_1)
    
    phi_1 = GP_mean_1 + np.sqrt(b) * GP_sig_1 + e
    
    GP2 = GaussianProcess(epsilon=0.2)
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
        fidelitynew = 0
        
    else:
        ynew = HF_function(xnew)
        Xt_e = np.vstack([Xt_e, xnew])
        yt_e = np.vstack([yt_e, ynew])
        n2 += 1
        fidelitynew=1
        
    
    phi_1_list.append(phi_1)
    phi_2_list.append(phi_2)
    phi_list.append(phi)
    Xt_e_list.append(Xt_e)
    Xt_c_list.append(Xt_c)
    xnew_list.append(xnew)
    yt_e_list.append(yt_e)
    yt_c_list.append(yt_c)
    ynew_list.append(ynew)
    fidelitylist.append(fidelitynew)
    
    best_min.append(np.min(yt_e))
    best_loc.append(Xt_e[np.argmin(yt_e)])
    best_mean.append(np.min(-GP_mean_2))
    best_mean_loc.append(x[np.argmin(-GP_mean_2)])
    
    
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, HF_function(x), c='k', linestyle='--', label ='High Fidelity function')
    #plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    #plt.plot(x, -phi_1, linestyle = '-.' , label=r'$\phi^{(1)}$', color='red', lw=lw)
    #plt.plot(x, -phi_2, linestyle = '-.' , label=r'$\phi^{(2)}$', color='orange', lw=lw)
    #plt.plot(x, -phi, linestyle = '-.' , label=r'$\phi$, Best bound', color='green', lw=1.5*lw)
    #plt.plot(x, -GP_mean_1 - e - v[0] ,linestyle='--')
    
    plt.plot(x, -GP_mean_1 ,c='b', label='Low-fidelity GP')
    plt.plot(x, -GP_mean_2 ,c='r', label='High-fidelity GP')
    
    # plt.fill_between(
    #     x.ravel(),
    #     -phi_1,
    #     -GP_mean_1,
    #     alpha=0.3,
    #     #label=r"95% confidence interval",
    #     color='blue')
    
    plt.fill_between(
        x.ravel(),
        -GP_mean_1 - 1.96 * (GP_sig_1),
        -GP_mean_1 + 1.96 * (GP_sig_1),
        alpha=0.3,
        #label=r"95% confidence interval",
        color='blue')
    
    plt.fill_between(
        x.ravel(),
        -GP_mean_2 - 1.96 * (GP_sig_2),
        -GP_mean_2 + 1.96 * (GP_sig_2),
        alpha=0.3,
        #label=r"95% confidence interval",
        color='red')

    
    # plt.fill_between(
    #     x.ravel(),
    #     -phi_2,
    #     -GP_mean_2,
    #     alpha=0.3,
    #     #label=r"95% confidence interval",
    #     color='red')

    plt.scatter(Xt_e, yt_e, color ='red', s=60, zorder=3)
    plt.scatter(Xt_c, yt_c, color ='blue', s=60, zorder=2)
    
    if fidelitynew == 0:
        plt.scatter(xnew, ynew, marker = '*' , color ='b', s=520, zorder=2)
        plt.vlines(xnew, -25, ynew, linestyle='-.', color='b', linewidth=6)
    if fidelitynew == 1:
        plt.scatter(xnew, ynew, marker = '*' , color ='r', s=520, zorder=2)
        plt.vlines(xnew, -25, ynew, linestyle='-.', color='r', linewidth=6)
    

    plt.legend(fontsize=10,loc='upper left')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(-25, 25)
    plt.title('Multi-Fidelity LCB; Iteration {}'.format(i+1), fontsize=15)
   
    
    
    plt.figure(figsize=(8, 4.5))
    
    plt.plot(x, phi_1, label='Low-fidelity', color='blue')
    plt.plot(x, phi_2, label='High-fidelity', color='red')
    plt.plot(x, phi, linestyle = '-.' , label='Combined', color='green')
    plt.plot(x, GP_mean_1 + e + v[0] ,linestyle='--', color = 'k', label='Gamma threshold')
   
    
    if fidelitynew == 0:
        plt.scatter(x[[np.argmax(phi)]], np.max(phi), color ='b', zorder=2)

    if fidelitynew == 1:
        plt.scatter(x[[np.argmax(phi)]], np.max(phi), color ='r', zorder=2)

    

    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(-25, 25)
    
    plt.title('Acquisition function; Iteration {}'.format(i+1), fontsize=15)


colours = ['b', 'r']

minval, loc = np.min(HF_function(x)), x[np.argmin(HF_function(x))]
#%%

fig, axes = plt.subplots(1, 2, figsize=(12, 4))


axes[0].plot(np.arange(1, n_iter+1, 1), best_min, label='Best solution')
axes[0].plot(np.arange(1, n_iter+1, 1), best_mean, label='Best GP mean')
axes[0].set_xlim(1, n_iter)
axes[0].hlines(minval, 0, n_iter, 'k', linestyle='--', label='True optimum')

for i in range(len(best_min)):
    axes[0].scatter(i+1, best_min[i], color = colours[fidelitylist[i]], zorder=1, s=50)
    axes[0].scatter(i+1, best_mean[i], color = colours[fidelitylist[i]], zorder=2, s=50)
axes[0].legend()
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel(r'$\hat{f}$')


axes[1].plot(np.arange(1, n_iter+1, 1), best_loc, label='Best location')
axes[1].plot(np.arange(1, n_iter+1, 1), best_mean_loc, label='Best GP location')
axes[1].set_xlim(1, n_iter)
axes[1].hlines(loc, 0, n_iter, 'k', linestyle='--', label='True location')
for i in range(len(best_min)):
    axes[1].scatter(i+1, best_loc[i], color = colours[fidelitylist[i]], zorder=1, s=50)
    axes[1].scatter(i+1, best_mean_loc[i], color = colours[fidelitylist[i]], zorder=2, s=50)
axes[1].legend()
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel(r'$\hat{x}$')

    
    
    
#%% Animation
fig = plt.figure(figsize=(24, 14))


# Animation Function 
def animate(i):  
    """This function will be called by the animation function iteratively to plot"""
    print(i)
    fig.clear()

    plt.plot(x, HF_function(x), c = 'b', label ='High Fidelity function', lw=lw)
    plt.plot(x, LF_function(x) , c ='k', label ='Low Fidelity function', lw=lw)
    plt.plot(x, -phi_1_list[i], linestyle = '-.' , label=r'$\phi^{(1)}$', color='r', lw=0.3*lw)
    plt.plot(x, -phi_2_list[i], linestyle = '-.' , label=r'$\phi^{(2)}$', color='orange', lw=0.3*lw)
    plt.plot(x, -phi_list[i], linestyle = '-.' , label=r'$\phi$', color='green', lw=1.4*lw)

    plt.scatter(Xt_e_list[i], yt_e_list[i], marker = 'o' , color ='b', label ='HF samples', s=220, zorder=2)
    plt.scatter(Xt_c_list[i], yt_c_list[i], marker = '*' , color ='k', label ='LF samples', s=220, zorder=3)
    
    plt.fill_between(
        x.ravel(),
        -phi_1_list[i],
        HF_function(x).ravel(),
        alpha=0.3,
        #label=r"95% confidence interval",
        color='red')
    
    plt.fill_between(
        x.ravel(),
        -phi_2_list[i],
        HF_function(x).ravel(),
        alpha=0.3,
        #label=r"95% confidence interval",
        color='orange')
    
    if fidelitylist[i] == 0:
        plt.scatter(xnew_list[i], ynew_list[i], marker = '*' , color ='k', label ='Next evaluation', s=520, zorder=2)
    if fidelitylist[i] == 1:
        plt.scatter(xnew_list[i], ynew_list[i], marker = '*' , color ='b', label ='Next evaluation', s=520, zorder=2)     
        
    plt.legend(fontsize=25,loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(-25, 25) 
    plt.title('Multi-Fidelity LCB, Iteration {}'.format(i+1), fontsize=35)
    
    
    

ani = animation.FuncAnimation(fig, animate, frames=(len(xnew_list))) #animation function. frames will iterate i in animate
ani.save('MF UCB.gif', writer='pillow', fps=1) #save animation
    


#%% Incomplete MFGPUCB


class MFGPUCB:
    def __init__(self, N, functions, errors, costs):
        self.N = N
        self.high_fidelity = functions[-1]
        self.low_fidelity = functions[:-1]
        self.errors= errors
        self.costs = costs
    
    def gammas(self):
        self.gammas = np.array([self.errors[i]*\
                                np.sqrt(self.costs[i]/self.costsl[i+1])\
                                    for i in range(self.errors.size)])
    
    def data(self, X):
        assert len(X) == self.N
        self.y_hf = self.high_fidelity(X[-1])
        self.y_lf = []
        for i in range(self.N-1):
            self.y_lf.append(self.low_fidelity[i](X[i]))


    
#%% Toy Model


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

exact_t = []
qssa_t = []
Elist = np.linspace(0,1.2,100)[1:]
S0 = 5.0

# Integrate the ODE for various (randomly perturbed) parameter values:
for i in range(len(Elist)):
    print(i)
    E0 = Elist[i] # initial concentration of enzyme
    C0 = np.array([S0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    kvals = np.array([10, 50, 2])
    
    C0 = np.array([5, 0.0, 0.0, E0]) # Initial concentrations of S0, ES0, ES1, S1, S2, E
    
    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals), 
                         [0, 100], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    kvals = np.array([10, 50, 2, E0])
    
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 100], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    exact_t.append(sol1.t[np.argmin(abs(sol1.y[2,:] - 0.67*S0))])
    qssa_t.append(sol2.t[np.argmin(abs(sol2.y[1,:] - 0.67*S0))])



#%%



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

    



# Initial samples

ndim=1
Xt_c = np.linspace(0,1.2,5, endpoint=True)[1:].reshape(-1, ndim)

# To have 2 samples
random.seed(2)
Xt_e = np.array(random.sample(list(Xt_c.T[0]),4)).reshape(-1, ndim)

    
# Evaluate the HF and LF functions
yt_e = np.array([high_fidelity_toy(i[0]) for i in Xt_e]).reshape(-1, ndim)
yt_c = np.array([low_fidelity_toy(i[0]) for i in Xt_c]).reshape(-1, ndim)

    

plt.figure(figsize=(10,7))
plt.plot(Elist, abs(10 - np.array(exact_t)), label='High-fidelity model')
plt.plot(Elist, abs(10 - np.array(qssa_t)), label='Low-fidelity model')
plt.legend(fontsize=15)
plt.xlabel('Enzyme concentration', fontsize=16)
plt.ylabel('Time (s)', fontsize=16)
plt.scatter(Xt_e, yt_e)
plt.scatter(Xt_c, yt_c)
# plt.title('Minimum enzyme concentration for 67% completion in 10s', fontsize=20)


#%% Fidelity weighted acquisition function






GP_mean_h_list =[]
GP_mean_l_list=[]
GP_variance_h_list =[]
GP_variance_l_list=[]
Xt_e_list=[]
Xt_c_list=[]
xnew_list=[]
yt_e_list=[]
yt_c_list=[]
ynew_list=[]

x = Elist

beta1, beta2 = 500, 10
n1, n2 = 10,4
n_iter = 0
b1,b2 = 1, 1
for i in range(30):
    n_iter += 1
    # Multi-fidelity GPR
    MFK = MFGPR(sigma_l=0, epsilon_l=0.05, sigma_h=0, epsilon_h=0.05, rho=0.5)
    MFK.learn(Xt_c, yt_c, Xt_e, yt_e)
    
    # High-fidelity GP mean and variance
    GP_mean_h = np.array([MFK.GP_h(i) for i in x]).ravel()
    GP_variance_h = np.array([MFK.variance_h(i) for i in x]).ravel()
    
    # Low fidelity GP
    GP = GaussianProcess(epsilon=0.05, sigma=0)
    GP.learn(Xt_c, yt_c)
    GP_mean_l = np.array([GP(i) for i in x]).ravel()
    GP_variance_l = np.array([GP.variance(i) for i in x]).ravel()
    
    phi_l = GP_mean_l - beta1 * GP_variance_l
    phi_h = GP_mean_h - beta2 * GP_variance_h
    
    
    plt.figure(figsize=(20, 12))
    plt.plot(Elist, abs(10 - np.array(exact_t)), label='High-fidelity model')
    plt.plot(Elist, abs(10 - np.array(qssa_t)), label='Low-fidelity model')
    plt.plot(x, GP_mean_h, linestyle = '-.' , label='Multifidelity GP', color='orange', lw=lw)
    plt.plot(x, GP_mean_l, linestyle = '-.' , label='Low-fidelity GP', color='red', lw=lw)

    plt.fill_between(
        x.ravel(),
        GP_mean_l - 1.96 * (GP_variance_l),
        GP_mean_l + 1.96 * (GP_variance_l),
        alpha=0.3,
        label=r"95% confidence interval",
        color='red')
    

    plt.fill_between(
        x.ravel(),
        GP_mean_h - 1.96 * (GP_variance_h),
        GP_mean_h + 1.96 * (GP_variance_h),
        alpha=0.3,
        label=r"95% confidence interval",
        color='orange')

    plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)

    
    # Fidelity weighted cost
    cost_l = ((b1/b2)*(n1 + 1) + n2)/n_iter
    cost_h = ((b2/b1)*(n1) +  (n2 + 1))/n_iter
    
    # print(phi_l, cost_l)
    # print(phi_h, cost_h)
    # print(np.min(phi_h), np.min(phi_l))
    
    phi_l += cost_l
    phi_h += cost_h
    
    # print(np.min(phi_h), np.min(phi_l))
    
    if np.min(phi_h) < np.min(phi_l):
        print('high fidelity')
        xnew = x[np.argmin(phi_h)]
        rec = False
        if xnew in Xt_e:
            rec = True
            phii = phi_h.copy()
        
        while rec == True:
            phii = np.delete(phii, np.argmin(phii)) 
            xnew = x[np.argmin(phii)]
            if xnew not in Xt_e:
                rec = False
            
        ynew = high_fidelity_toy(xnew)
        Xt_e = np.vstack([Xt_e, xnew])
        yt_e = np.vstack([yt_e, ynew])
        n2 += 1
        
    else:
        print('low-fidelity')
        xnew = x[np.argmin(phi_l)]
        rec = False
        if xnew in Xt_c:
            rec = True
            phii = phi_l.copy()
        
        while rec == True:
            phii = np.delete(phii, np.argmin(phii)) 
            xnew = x[np.argmin(phii)]
            if xnew not in Xt_c:
                rec = False
        
        ynew = low_fidelity_toy(xnew)
        Xt_c = np.vstack([Xt_c, xnew])
        yt_c = np.vstack([yt_c, ynew])
        n1 += 1
    
    plt.scatter(xnew, ynew, marker = '*' , color ='r', label ='next location', s=500, zorder=2)
    plt.vlines(xnew, -10, ynew, linestyle='-.', linewidth=5, color='r')
    plt.ylim(-10, 90)
    plt.xlim(0, 1.2)
    
    plt.legend(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel(r'y and $\hat{y}$', fontsize=25)

    plt.title('Multi-fidelity Gaussian Process Regression', fontsize=35)
    
    
    GP_mean_h_list.append(GP_mean_h)
    GP_mean_l_list.append(GP_mean_l)
    GP_variance_h_list.append(GP_variance_h)
    GP_variance_l_list.append(GP_variance_l)
    Xt_e_list.append(Xt_e)
    Xt_c_list.append(Xt_c)
    xnew_list.append(xnew)
    yt_e_list.append(yt_e)
    yt_c_list.append(yt_c)
    ynew_list.append(ynew)
        
    
    
#%% MF GP UCB

# Initial samples

ndim=1
Xt_c = np.linspace(0,1.2,5, endpoint=True)[1:].reshape(-1, ndim)

# To have 2 samples
random.seed(2)
Xt_e = np.array(random.sample(list(Xt_c.T[0]),4)).reshape(-1, ndim)

    
# Evaluate the HF and LF functions
yt_e = np.array([high_fidelity_toy(i[0]) for i in Xt_e]).reshape(-1, ndim)
yt_c = np.array([low_fidelity_toy(i[0]) for i in Xt_c]).reshape(-1, ndim)



e = np.array([10])
l = np.array([0.00001, 1])
v = np.array([e[i]*np.sqrt(l[i]/l[i+1]) for i in range(e.size)])

b = 10
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


for i in range(23):
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
        ynew = low_fidelity_toy(xnew)
        Xt_c = np.vstack([Xt_c, xnew])
        yt_c = np.vstack([yt_c, ynew])
        n1 += 1
        
    else:
        ynew = high_fidelity_toy(xnew)
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
    
    plt.figure(figsize=(24, 14))
    plt.plot(Elist, abs(10 - np.array(exact_t)), label='High-fidelity model')
    plt.plot(Elist, abs(10 - np.array(qssa_t)), label='Low-fidelity model')
    plt.plot(x, -phi_1, linestyle = '-.' , label=r'$\phi^{(1)}$', color='blue', lw=lw)
    plt.plot(x, -phi_2, linestyle = '-.' , label=r'$\phi^{(2)}$', color='brown', lw=lw)
    plt.plot(x, -phi, linestyle = '-.' , label=r'$\phi$', color='green', lw=lw)

    plt.scatter(Xt_e, yt_e, marker = 'o' , color ='k', label ='HF samples', s=120)
    plt.scatter(Xt_c, yt_c, marker = '*' , color ='g', label ='LF samples', s=120)
    plt.scatter(xnew, ynew, marker = '*' , color ='r', label ='Next evaluation', s=520, zorder=2)
    plt.vlines(xnew, -25, ynew, linestyle='-.', color='r', linewidth=6)

    plt.legend(fontsize=25,loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)
    plt.xlim(0, 1)
    plt.ylim(-25, 90)

    plt.title('Multi-Fidelity UCB', fontsize=35)



