#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:22:26 2024

@author: arjunmnanoj
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def oregonator(t, x, p):
    e1, e, q = p
    dx = (q*x[1] - x[0]*x[1] + x[0] - x[0]**2)/e1
    dy = (-q*x[1] -x[0]*x[1] + x[2])/e
    dz = x[0] - x[2]
    return np.array([dx, dy, dz])


def reduced_oregonator(t,x,p):
    e1, e, q = p
    x1 = (1 - x[0])/2 + np.sqrt(q * x[0] + ((1 - x[0])**2)/4)
    dy = (-q*x[0] -x1*x[0] + x[1])/e
    dz = x1 - x[1]
    return np.array([dy, dz])


# def reduced_oregonator(t,x,p):
#     e1, e, q = p
#     y1 = x[1]/(q + x[0])
#     dx = (q*y1 - x[0]*y1 + x[0] - x[0]**2)/e1
#     dz = x[0] - x[1]
#     return np.array([dx, dz])


def get_params(T):
    T0 = 298
    R = 8.31446
    E = np.array([60, 25, 60, 75, 70])*1e3
    E = np.array([54, 25, 60, 64, 70])*1e3
    e1 = 0.1 * np.exp((E[2] - E[4])*(1/T - 1/T0)/R)
    e = 4e-4 * np.exp((E[1] + E[2] - E[3] - E[4])*(1/T - 1/T0)/R)
    q = 8e-4 * np.exp((E[1] + E[2] - E[0] - E[3])*(1/T - 1/T0)/R)
    return np.array([e1, e, q])

#%%


parray = np.array([8e-4, 4e-3, 5e-2, 1/12, 1/4, 0.5])


fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()

for i in range(len(parray)):
    p = [8e-4, parray[i], 1e-4]
    init = np.array([0.1, 0.3, 0.2])
    
    sol1 = solve_ivp(lambda t, x: oregonator(t, x, p), 
                         [0, 100], 
                         init,
                         method='BDF',
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                        rtol=np.sqrt(np.finfo(float).eps),
                        )
    
    sol2 = solve_ivp(lambda t, x: reduced_oregonator(t, x, p), 
                         [0, 100], 
                         init[1:],
                         # init[[0,2]],
                         method='BDF',
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps),
                         )
    
    axs[i].loglog(sol1.y[0], sol1.y[2], label='exact')
    axs[i].loglog(sol2.y[0], sol2.y[1], '--', label='QSSA')
    if i>2:
        axs[i].set_xlabel('Y')
    axs[i].set_ylabel('Z')
    axs[i].set_title('epsilon = {:g}'.format(parray[i]))
    axs[i].legend()
    




#%% Temperature

parray = np.linspace(300, 500, 6)


fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()

for i in range(len(parray)):
    p = get_params(parray[i])
    init = np.array([0.1, 0.3, 0.2])
    
    sol1 = solve_ivp(lambda t, x: oregonator(t, x, p), 
                         [0, 40], 
                         init,
                         method='BDF',
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                        rtol=np.sqrt(np.finfo(float).eps),
                        )
    
    sol2 = solve_ivp(lambda t, x: reduced_oregonator(t, x, p), 
                         [0, 40], 
                           init[1:],
                          # init[[0,2]],
                         method='BDF',
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps),
                         )
    
    axs[i].loglog(sol1.y[1], sol1.y[2], label='exact')
    axs[i].loglog(sol2.y[0], sol2.y[1], '--', label='QSSA')
    if i>2:
        axs[i].set_xlabel('Y')
    if i in [0,3]:
        axs[i].set_ylabel('Z')
    axs[i].set_title('T = {:g} K'.format(parray[i]))
    axs[i].legend()
    
