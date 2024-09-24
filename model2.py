#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:53:17 2024

@author: arjunmnanoj
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rc('font', family='serif')


#%% Scheme 1

def exact(t, k, c0):
    ca = c0 * np.exp(-k[0]*t)
    cb = c0 * (k[0]/(k[1]-k[0])) * (np.exp(-k[0]*t) - np.exp(-k[1]*t))
    cc = c0 * (1/(k[1]-k[0])) * (k[1]*(1 - np.exp(-k[0]*t)) - k[0]*(1 - np.exp(-k[1]*t)))
    return np.array([ca,cb,cc])
    

def qssa(t, k, c0):
    ca = c0 * np.exp(-k[0]*t)
    cb = c0 * (k[0]/k[1]) * np.exp(-k[0]*t)
    cc = c0 * (1 - np.exp(-k[0]*t))
    return np.array([ca,cb,cc])


t = np.linspace(0, 5, 100)
klist = [[5,1], [2,1], [1,2], [1, 5], [1,10], [1, 50]]


fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()



for i in range(len(klist)):
    exact_data = exact(t, klist[i], 1)
    qssa_data = qssa(t, klist[i], 1)

    axs[i].plot(t, exact_data[2], label='exact model')
    axs[i].plot(t, qssa_data[2], linestyle='-.', label = 'QSSA model')
    if i>2:
        axs[i].set_xlabel('Time(s)')
    if i in [0,3]:
        axs[i].set_ylabel('Concentration of C')
    axs[i].set_title(r'$k_2/k_1$ = {}'.format(klist[i][1]/klist[i][0]))
    axs[i].legend()


#%% Scheme 1 Temperature

def exact(t, k, c0):
    ca = c0 * np.exp(-k[0]*t)
    cb = c0 * (k[0]/(k[1]-k[0])) * (np.exp(-k[0]*t) - np.exp(-k[1]*t))
    cc = c0 * (1/(k[1]-k[0])) * (k[1]*(1 - np.exp(-k[0]*t)) - k[0]*(1 - np.exp(-k[1]*t)))
    return np.array([ca,cb,cc])
    

def qssa(t, k, c0):
    ca = c0 * np.exp(-k[0]*t)
    cb = c0 * (k[0]/k[1]) * np.exp(-k[0]*t)
    cc = c0 * (1 - np.exp(-k[0]*t))
    return np.array([ca,cb,cc])


t = np.linspace(0, 5, 100)


fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()


def get_constants(T):
    E1, E2 = np.array([25, 2])*5e2
    A1, A2 = 200, 5
    R = 8.314
    k1 = A1*np.exp(-E1/(R*T))
    k2 = A2*np.exp(-E2/(R*T))
    return np.array([k1, k2])

T = np.linspace(300, 500, 6)

for i in range(len(T)):
    exact_data = exact(t, get_constants(T[i]), 1)
    qssa_data = qssa(t, get_constants(T[i]), 1)

    axs[i].plot(t, exact_data[2], label='exact model')
    axs[i].plot(t, qssa_data[2], linestyle='-.', label = 'QSSA model')
    if i>2:
        axs[i].set_xlabel('Time(s)')
    if i in [0,3]:
        axs[i].set_ylabel('Concentration of C')
    axs[i].set_title(r'T = {} K'.format(T[i]))
    axs[i].legend()
    
#%% Scheme 2

def qssa(t, k, c0):
    ca = c0 * np.exp(-k[0]*k[2]*t/(k[1] + k[2]))
    cb = c0 * (k[0]/(k[1]+k[2])) * np.exp(-k[0]*k[2]*t/(k[1] + k[2]))
    cc = c0 * (1 - np.exp(-k[0]*k[2]*t/(k[1] + k[2])))
    return np.array([ca,cb,cc])

    
def rxrate(t, x, p):
    c1 = x[0]
    c2 = x[1]
    c3 = x[2]

    r1 = p[0] * c1 - p[1] * c2
    r2 = p[2] * c2

    return [-r1, r1 - r2, r2]



t = np.linspace(0, 5, 100)
klist = [[5,1,1], [2,1,1], [1, 2, 1], [1, 1, 10], [1, 10, 1], [1, 10, 10]]

fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()

for i in range(len(klist)):
    qssa_data = qssa(t, klist[i], 1)
    
    sol = solve_ivp(lambda t, x: rxrate(t, x, klist[i]), 
                         [0, t[-1]], 
                         [1, 0, 0],
                         method='LSODA',
                         t_eval=t, 
                         atol=np.sqrt(np.finfo(float).eps), 
                        rtol=np.sqrt(np.finfo(float).eps),
                        )

    axs[i].plot(sol.t, sol.y[2], label='exact model')
    axs[i].plot(t, qssa_data[2], linestyle='-.', label = 'QSSA model')
    if i>2:
        axs[i].set_xlabel('Time(s)')
    if i in [0,3]:
        axs[i].set_ylabel('Concentration of C')
    axs[i].set_title(r'$k_2/k_1$ = {}, $k_-1/k_1$ = {}'.format(klist[i][2]/klist[i][0], klist[i][1]/klist[i][0]))
    axs[i].legend()

    