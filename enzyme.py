#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:22:05 2024

@author: arjunmnanoj
"""

import numpy as np
import tqdm as tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

        
        
#%%

def MSP(t, x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1, k2, k3, k4, k5, k6 = k
    dS0  = -k1*x[0]*x[5] + k2*x[1]
    dES0 =  k1*x[0]*x[5] - k2*x[1] - k3*x[1]
    dES1 =  k3*x[1] - k4*x[2] + k5*x[5]*x[3] - k6*x[2]
    dS1  =  k4*x[2] - k5*x[5]*x[3]
    dS2  =  k6*x[2]
    dE   = -k1*x[5]*x[0] + k2*x[1] + k4*x[2] - k5*x[5]*x[3] + k6*x[2]
    
    return [dS0, dES0, dES1, dS1, dS2, dE]



E0 = 1 # initial concentration of enzyme
C0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E


kfarray = np.array([10, 100, 500, 1000])


# Create a figure with 6 subplots (2 rows, 3 columns)
fig1, axs1 = plt.subplots(2, 2, figsize=(15, 8))  # 2 rows, 3 columns
axs1 = axs1.ravel()
plt.tight_layout(rect=[0.05, 0, 1, 1])

# Create a figure with 6 subplots (2 rows, 3 columns)
fig2, axs2 = plt.subplots(2, 2, figsize=(15, 8))  # 2 rows, 3 columns
axs2 = axs2.ravel()
plt.tight_layout(rect=[0.05, 0, 1, 1])

for i in range(len(kfarray)):
    print(i)
    # Integrate the ODE for various (randomly perturbed) parameter values:
    kvals = np.array([0.71, 19, 6700, 9200, 0.97, 5200])

    
    sol1 = solve_ivp(lambda t, x: MSP(t, x, kvals), 
                         [0, 20], 
                         C0,
                         # t_eval=time_points, 
                          atol=np.sqrt(np.finfo(float).eps), 
                          rtol=np.sqrt(np.finfo(float).eps)
                         )
    
    # Reduced MSP model

    def reducedMSP(t, x, k):
        """
        Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
        """
        kf1, kr1, kcat1, kf2, kr2, kcat2 = k
    
        k1 = (E0*kf1*kcat1)/(kr1 + kcat1)
        k2 = (E0*kf2*kcat2)/(kr2 + kcat2)
        p = kcat2/(kr2 + kcat2)
        
        dS0  = -k1*x[0]
        dS1  =  k1*(1 - p)*x[0] - k2*x[1]
        dS2  =  k1*p*x[0] + k2*x[1]
        return [dS0, dS1, dS2]
    
    sol2 = solve_ivp(lambda t, x: reducedMSP(t, x, kvals), 
                         [0, 20], 
                         C0[:3],
                         #method='BDF',
                         # t_eval=time_points,
                          atol=np.sqrt(np.finfo(float).eps), 
                          rtol=np.sqrt(np.finfo(float).eps),
                         )


    # plt.figure()
    axs1[i].plot(sol1.t, sol1.y[4,:], label ='exact S2')
    axs1[i].plot(sol2.t, sol2.y[2,:], color = 'r', zorder=2, label='QSSA S2')
    if i>1:
        axs1[i].set_xlabel('time')
    axs1[i].set_title('$kf_2$ = {}'.format(kfarray[i]))
    axs1[i].legend()
    
    # plt.figure()
    axs2[i].plot(sol1.t, sol1.y[3,:], label ='exact S1')
    axs2[i].plot(sol2.t, sol2.y[1,:], color = 'r', zorder=2, label ='QSSA S1')
    if i>1:
        axs2[i].set_xlabel('time')
    axs2[i].set_title('$kf_2$ = {}'.format(kfarray[i]))
    axs2[i].legend()


fig1.text(0.04, 0.5, 'S2 concentration', va='center', rotation='vertical', fontsize=12)
fig2.text(0.04, 0.5, 'S1 concentration', va='center', rotation='vertical', fontsize=12)

#%%

def MSP(t, x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1, k2, k3, k4, k5, k6 = k
    dS0  = -k1*x[0]*x[5] + k2*x[1]
    dES0 =  k1*x[0]*x[5] - k2*x[1] - k3*x[1]
    dES1 =  k3*x[1] - k4*x[2] + k5*x[5]*x[3] - k6*x[2]
    dS1  =  k4*x[2] - k5*x[5]*x[3]
    dS2  =  k6*x[2]
    dE   = -k1*x[5]*x[0] + k2*x[1] + k4*x[2] - k5*x[5]*x[3] + k6*x[2]
    
    return [dS0, dES0, dES1, dS1, dS2, dE]


use_S2_only = True
generator = np.random.RandomState(12345)


E0 = 1 # initial concentration of enzyme
C0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
time_points = np.linspace(0, 20, num_steps + 1)
delta_k = 0.1 # relative half-width of rate constant perturbations

exact_t = []
qssa_t = []
Elist = np.linspace(0,1.2,50)[1:]
S0 = 5.0


for i in range(len(Elist)):
    print(i)
    # Integrate the ODE for various (randomly perturbed) parameter values:
    kvals = np.array([0.71, 19, 6700, 9200, 0.97, 5200])
    E0 = Elist[i] # initial concentration of enzyme
    C0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    sol1 = solve_ivp(lambda t, x: MSP(t, x, kvals), 
                         [0, 150], 
                         C0,
                         # t_eval=time_points, 
                          atol=np.sqrt(np.finfo(float).eps), 
                          rtol=np.sqrt(np.finfo(float).eps)
                         )
    
    # Reduced MSP model

    def reducedMSP(t, x, k):
        """
        Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
        """
        kf1, kr1, kcat1, kf2, kr2, kcat2 = k
    
        k1 = (E0*kf1*kcat1)/(kr1 + kcat1)
        k2 = (E0*kf2*kcat2)/(kr2 + kcat2)
        p = kcat2/(kr2 + kcat2)
        
        dS0  = -k1*x[0]
        dS1  =  k1*(1 - p)*x[0] - k2*x[1]
        dS2  =  k1*p*x[0] + k2*x[1]
        return [dS0, dS1, dS2]
    
    sol2 = solve_ivp(lambda t, x: reducedMSP(t, x, kvals), 
                         [0, 150], 
                         C0[:3],
                         #method='BDF',
                         # t_eval=time_points,
                          atol=np.sqrt(np.finfo(float).eps), 
                          rtol=np.sqrt(np.finfo(float).eps),
                         )

    
    exact_t.append(sol1.t[np.argmin(abs(sol1.y[4,:] - 0.67*S0))])
    qssa_t.append(sol2.t[np.argmin(abs(sol2.y[2,:] - 0.67*S0))])

#%%



plt.figure(figsize=(10,7))
plt.plot(Elist, abs(10 - np.array(exact_t)), label='High-fidelity model', lw=4)
plt.plot(Elist, abs(10 - np.array(qssa_t)), label='Low-fidelity model', linestyle='-.', lw=4)
plt.legend(fontsize=15)
plt.xlabel('Enzyme concentration', fontsize=16)
plt.ylabel('Time (s)', fontsize=16)
# plt.title('Minimum enzyme concentration for 67% completion in 10s', fontsize=20)



#%% Toy problem


kfarray = np.array([0.1, 0.3, 0.5, 2, 5, 5000])

# Create a figure with 6 subplots (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
axs = axs.ravel()

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
 


E0 = 0.66 # initial concentration of enzyme
C0 = np.array([5.0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E


# Integrate the ODE for various (randomly perturbed) parameter values:
for i in range(len(kfarray)):
    kvals = np.array([kfarray[i], 19, 9200])
    
    # Reduced toy problem
    def reducedToyModel(t,x, k):
        """
        Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
        """
        k1,k2,k3 = k
        ke = E0*(k1*k3)/(k2+ k3)
        dS0  = -ke*x[0]*E0
        dS1  = ke*x[0]*E0
        return [dS0, dS1]
    
    C0 = np.array([5, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    
    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals), 
                         [0, 20], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 20], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    
    
    axs[i].plot(sol1.t, sol1.y[2,:], label ='exact model')
    axs[i].plot(sol2.t, sol2.y[1,:], color = 'r', zorder=2, label='QSSA model')
    axs[i].set_xlabel('time')
    axs[i].set_title('kf={:g}'.format(kfarray[i]))
    axs[i].legend()
    
plt.tight_layout()

#%%


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



# Define the ODEs for the reaction
def reaction(y, t, k_f, k_r, k_cat):
    E, S1, ES1, S2 = y
    
    # Rate equations
    dE_dt = -k_f * E * S1 + k_r * ES1 + k_cat * ES1
    dS1_dt = -k_f * E * S1 + k_r * ES1
    dES1_dt = k_f * E * S1 - k_r * ES1 - k_cat * ES1
    dS2_dt = k_cat * ES1
    
    return [dE_dt, dS1_dt, dES1_dt, dS2_dt]




exact_t = []
qssa_t = []
Elist = np.linspace(0,1.2,50)[1:]
S0 = 5.0

# Integrate the ODE for various (randomly perturbed) parameter values:
for i in range(len(Elist)):
    print(i)
    E0 = Elist[i] # initial concentration of enzyme
    C0 = np.array([S0, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    kvals = np.array([10, 50, 2])
    
    # Reduced toy problem
    
    def reducedToyModel(t,x, k):
        """
        Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
        """
        k1,k2,k3 = k
        ke = (k1*k3)/(k2+ k3)
        dS0  = -ke*x[0]*E0
        dS1  = ke*x[0]*E0
        
        return [dS0, dS1]
    
    C0 = np.array([5, 0.0, 0.0, E0]) # Initial concentrations of S0, ES0, ES1, S1, S2, E
    
    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals), 
                         [0, 100], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 100], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    exact_t.append(sol1.t[np.argmin(abs(sol1.y[2,:] - 0.67*S0))])
    qssa_t.append(sol2.t[np.argmin(abs(sol2.y[1,:] - 0.67*S0))])
    

#%%

plt.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": "Computer Modern",
})

plt.figure(figsize=(10,7))
plt.plot(Elist, abs(10 - np.array(exact_t)), label='High-fidelity model', lw=4)
plt.plot(Elist, abs(10 - np.array(qssa_t)), label='Low-fidelity model', linestyle='-.', lw=4)
plt.legend(fontsize=15)
plt.xlabel('Enzyme concentration', fontsize=16)
plt.ylabel('Time (s)', fontsize=16)
# plt.title('Minimum enzyme concentration for 67% completion in 10s', fontsize=20)


#%% Bounds

# kf1, kr1, kcat1, kf2, kr2, kcat2
kvals = np.array([0.71, 19, 6700, 9200, 0.97, 5200])
ub = [25, 1e4, 1e4, 2, 1e4, 1e4]




#%% Niko code


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the rate constants for the QSSA regime
k_f = 1.0    # Forward rate constant for E + S_1 -> ES_1
k_r = 500.0     # Reverse rate constant for ES_1 -> E + S_1
k_cat = 200.0   # Catalytic rate constant for ES_1 -> E + S_2

# Initial concentrations
E0 = 1.0      # Initial concentration of enzyme E
S1_0 = 10.0   # Initial concentration of substrate S_1
ES1_0 = 0.0   # Initial concentration of complex ES_1
S2_0 = 0.0    # Initial concentration of product S_2

# Time points for integration
t = np.linspace(0, 10, 1000)

# Define the ODEs for the reaction
def reaction(y, t, k_f, k_r, k_cat):
    E, S1, ES1, S2 = y
    
    # Rate equations
    dE_dt = -k_f * E * S1 + k_r * ES1 + k_cat * ES1
    dS1_dt = -k_f * E * S1 + k_r * ES1
    dES1_dt = k_f * E * S1 - k_r * ES1 - k_cat * ES1
    dS2_dt = k_cat * ES1
    
    return [dE_dt, dS1_dt, dES1_dt, dS2_dt]

# Initial conditions
y0 = [E0, S1_0, ES1_0, S2_0]

# Integrate the system of ODEs for the QSSA regime
solution_qssa = odeint(reaction, y0, t, args=(k_f, k_r, k_cat))

# Extract the results
E_qssa, S1_qssa, ES1_qssa, S2_qssa = solution_qssa.T

# Plot the results for the QSSA regime
plt.figure(figsize=(3, 3))

plt.plot(t, S2_qssa, label='[S_2]')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Reaction Dynamics in QSSA Regime: E + S_1 <-> ES_1 -> E + S_2')
plt.legend()
# plt.grid(True)
plt.show()

# Now let's change the parameters to simulate out of QSSA regime
k_f_fast = k_f*100  # Increase the forward rate constant significantly
k_r_slow = k_r/100     # Reverse rate constant for ES_1 -> E + S_1
k_cat_slow = k_cat/100   # Catalytic rate constant for ES_1 -> E + S_2

# Integrate again with the new parameters for the non-QSSA regime
solution_non_qssa = odeint(reaction, y0, t, args=(k_f_fast, k_r_slow, k_cat_slow))

# Extract the results for the non-QSSA regime
E_non_qssa, S1_non_qssa, ES1_non_qssa, S2_non_qssa = solution_non_qssa.T

# Plot the results for the non-QSSA regime
plt.figure(figsize=(3, 3))
plt.plot(t, E_non_qssa, label='[E] (non-QSSA regime)')
plt.plot(t, S1_non_qssa, label='[S_1] (non-QSSA regime)')
plt.plot(t, ES1_non_qssa, label='[ES_1] (non-QSSA regime)')
plt.plot(t, S2_non_qssa, label='[S_2] (non-QSSA regime)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Reaction Dynamics in Non-QSSA Regime: E + S_1 <-> ES_1 -> E + S_2')
plt.legend()
plt.grid(True)
plt.show()


#%%




kfarray = np.array([[1, 500, 200], [10, 50, 2]])

# Create a figure with 6 subplots (2 rows, 3 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 2 rows, 3 columns
axs = axs.ravel()

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
 


E0 = 0.66 # initial concentration of enzyme
C0 = np.array([10.0, 0.0, 0.0, 1.0]) # initial concentrations of S0, ES0, ES1, S1, S2, E


# Integrate the ODE for various (randomly perturbed) parameter values:
for i in range(len(kfarray)):
    kvals = np.array(kfarray[i])
    
    # Reduced toy problem
    def reducedToyModel(t,x, k):
        """
        Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
        """
        k1,k2,k3 = k
        ke = (k1*k3)/(k2+ k3)
        dS0  = -ke*x[0]*E0
        dS1  = ke*x[0]*E0
        return [dS0, dS1]
    
    C0 = np.array([10, 0.0, 0.0, E0]) # initial concentrations of S0, ES0, ES1, S1, S2, E
    
    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals), 
                         [0, 20], 
                         C0,
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals), 
                         [0, 20], 
                         C0[:2],
                         # t_eval=time_points, 
                         atol=np.sqrt(np.finfo(float).eps), 
                         rtol=np.sqrt(np.finfo(float).eps))
    
    
    
    axs[i].plot(sol1.t, sol1.y[2,:], label ='exact model')
    axs[i].plot(sol2.t, sol2.y[1,:], color = 'r', zorder=2, label='QSSA model')
    axs[i].set_xlabel('time')
    axs[i].set_title('kf = {}, kr = {}, kcat = {}'.format(kfarray[i][0], kfarray[i][1], kfarray[i][2]))
    axs[i].legend()
    
plt.tight_layout()
