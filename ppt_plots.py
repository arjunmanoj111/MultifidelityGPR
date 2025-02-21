#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:53:26 2024

@author: arjunmnanoj
"""

# General imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

np.random.seed(25)

plt.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": "Computer Modern",
})

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.core.loop.user_function import UserFunctionWrapper

## Generate samples from the Forrester function

high_fidelity = emukit.test_functions.forrester.forrester
low_fidelity = emukit.test_functions.forrester.forrester_low



x_plot = np.linspace(0, 1, 200)[:, None]
y_plot_l = low_fidelity(x_plot)-10
y_plot_h = high_fidelity(x_plot)

x_train_l = np.atleast_2d(np.random.rand(12)).T
x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])
y_train_l = low_fidelity(x_train_l)
y_train_h = high_fidelity(x_train_h)



from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])



plt.figure(figsize=(12, 8))
plt.plot(x_plot, y_plot_h, label='High-fidelity', lw=3)
plt.plot(x_plot, y_plot_l, '-.', label='Low-fidelity', lw=3)
# plt.scatter(x_train_l, y_train_l, color='b', s=40)
# plt.scatter(x_train_h, y_train_h, color='r', s=40)
plt.ylabel('f (x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.legend(fontsize=15)
plt.xlim(0,1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.ylim(-15, 20)

# plt.title('High and low fidelity Forrester functions');

#%%

## Construct a linear multi-fidelity model

kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)


## Wrap the model using the given 'GPyMultiOutputWrapper'

lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)

## Fit the model
  
lin_mf_model.optimize()



## Convert x_plot to its ndarray representation

X_plot = convert_x_list_to_array([x_plot, x_plot])
X_plot_l = X_plot[:len(x_plot)]
X_plot_h = X_plot[len(x_plot):]

## Compute mean predictions and associated variance

lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(X_plot_l)
lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)
hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(X_plot_h)
hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)


## Plot the posterior mean and variance

plt.figure(figsize=(12, 8))
plt.fill_between(x_plot.flatten(), (lf_mean_lin_mf_model - 1.96*lf_std_lin_mf_model).flatten(), 
                 (lf_mean_lin_mf_model + 1.96*lf_std_lin_mf_model).flatten(), facecolor='b', alpha=0.3)
plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(), 
                 (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), facecolor='r', alpha=0.3)

# plt.plot(x_plot, y_plot_l, 'k--.', label ='Low-fidelity function')
plt.plot(x_plot, y_plot_h, 'k--', label = 'High-fidelity function')
plt.plot(x_plot, lf_mean_lin_mf_model, color='b', label='Low-fidelity GP')
plt.plot(x_plot, hf_mean_lin_mf_model, color='r', label='Multi-fidelity GP')
plt.scatter(x_train_l, y_train_l, color='b', s=50)
plt.scatter(x_train_h, y_train_h, color='r', s=50)
plt.ylabel('f (x)', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.legend(fontsize=15)
plt.xlim(0,1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Multi-fidelity Gaussian Process Regression', fontsize=18)
plt.ylim(-15, 20)


## Create standard GP model using only high-fidelity data

kernel = GPy.kern.RBF(1)
high_gp_model = GPy.models.GPRegression(x_train_h, y_train_h, kernel)
high_gp_model.Gaussian_noise.fix(0)

## Fit the GP model

high_gp_model.optimize_restarts(5)

## Compute mean predictions and associated variance

hf_mean_high_gp_model, hf_var_high_gp_model  = high_gp_model.predict(x_plot)
hf_std_hf_gp_model = np.sqrt(hf_var_high_gp_model)



## Plot the posterior mean and variance for the high-fidelity GP model

plt.figure(figsize=(12, 8))


plt.fill_between(x_plot.flatten(), (hf_mean_high_gp_model - 1.96*hf_std_hf_gp_model).flatten(), 
                 (hf_mean_high_gp_model + 1.96*hf_std_hf_gp_model).flatten(), facecolor='g', alpha=0.2)

plt.fill_between(x_plot.flatten(), (hf_mean_lin_mf_model - 1.96*hf_std_lin_mf_model).flatten(), 
                  (hf_mean_lin_mf_model + 1.96*hf_std_lin_mf_model).flatten(), facecolor='r', alpha=0.33)

plt.plot(x_plot, y_plot_h, '--',color='k', label='True function')
plt.plot(x_plot, hf_mean_high_gp_model, 'g', label='Standard GP')
plt.plot(x_plot, hf_mean_lin_mf_model, '-', color='r', label='Multi-fidelity GP')
plt.scatter(x_train_h, y_train_h, color='k')
plt.xlabel('x', fontsize=15)
plt.ylabel('f (x)', fontsize=15)
plt.legend(fontsize=15, loc='upper left')
# plt.title('Multi-fidelity GPR vs Standard GPR', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,1)
plt.ylim(-15, 20)


