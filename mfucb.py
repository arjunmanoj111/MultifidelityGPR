#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:37:30 2024

@author: arjunmnanoj
"""


import warnings
warnings.simplefilter('always')

import warnings
warnings.filterwarnings("ignore")



import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

import GPy
from GPy.models.gp_regression import GPRegression
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer

from scipy.integrate import solve_ivp




plt.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": "Computer Modern",
})
FIG_SIZE = (12, 8)


class multifidelityUCB:
    def __init__(self, lf, hf, ndim=1, negate=False, noise=0.1):
        """
        multi-fidelity UCB optimization.
        
        :lf: Low-fidelity function
        :hf: High-fidelity function
        :ndim: Problem dimensionality
        :negate: True if minimzation problem
        :noise: Low-fidelity model noise
        """
        self.LF = lf
        self.HF = hf
        self.multifidelity = MultiSourceFunctionWrapper([lf, hf])
        self.iterates = []
        self.GP_iterates = []
        self.ndim = ndim
        self.negate = negate
        self.fidelity_list = []
        self.l_noise = noise
        self.is_max = False

    def set_initial_data(self, n1, n2, bounds, seed=12346):
        """
        Initialize dataset for the model
        
        :n1: Number of Low-fidelity points
        :n2: Number of High-fidelity points
        :bounds: Problem bounds
        :seed: random seed
        """
        np.random.seed(seed)
        x_low = np.random.rand(n1,self.ndim)
        x_high = x_low[:n2, :]
        y_low = self.multifidelity.f[0](x_low)
        y_high = self.multifidelity.f[1](x_high)
        self.x_array, self.y_array = convert_xy_lists_to_arrays([x_low, x_high], [y_low, y_high])
        self.n_init = n1+n2
        self.bounds = bounds
        
        if self.ndim==1:
            # Bounds
            lb, ub = bounds
            x_plot = np.linspace(lb, ub, 500)[:, None]
            
        if self.ndim==2:
            x1_range = np.linspace(0, 1, 200)
            x2_range = np.linspace(0, 1, 200)
            # Create mesh grid from x1 and x2 ranges
            X1, X2 = np.meshgrid(x1_range, x2_range)
            x_plot = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
            self.X1, self.X2 = X1, X2
        
        else:
           arrays = [np.linspace(0, 1, 10) for i in range(self.ndim)]

           # Create meshgrid from the arrays
           meshgrid = np.meshgrid(*arrays, indexing='ij')
           
           # Stack meshgrid and reshape to get an array of data points
           x_plot = np.stack(meshgrid, axis=-1).reshape(-1, self.ndim)
           
        self.x_plot = x_plot
        self.x_plot_low = np.concatenate([np.atleast_2d(x_plot), np.zeros((x_plot.shape[0], 1))], axis=1)
        self.x_plot_high = np.concatenate([np.atleast_2d(x_plot), np.ones((x_plot.shape[0], 1))], axis=1)

        

    def set_model(self, x_array=None, y_array=None, ker='rbf'):
        """
        Initialize model
        
        :x_array: X data if any
        :y_array: Y data if any
        :ker: Kernel (RBF by default. Will switch to Matern for any other input)
        """
        if x_array==None:
            x_array = self.x_array
        if y_array==None:
            y_array = self.y_array
        
        n_fidelities = 2
        if self.ndim==1:
            lb, ub = self.bounds
            self.parameter_space = ParameterSpace([ContinuousParameter('x', lb, ub), InformationSourceParameter(n_fidelities)])
        else:
            paramspace = []
            for i in range(self.ndim):
                lb, ub = self.bounds[i]
                paramspace.append(ContinuousParameter('x'+str(i+1), 0, 1))
            paramspace.append(InformationSourceParameter(n_fidelities))
            self.parameter_space = ParameterSpace(paramspace)

        if ker=='rbf':
            kern_low = GPy.kern.RBF(self.ndim)
            kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            
            kern_err = GPy.kern.RBF(self.ndim)
            kern_err.lengthscale.constrain_bounded(0.01, 0.5)

        else:
            kern_low = GPy.kern.Matern52(self.ndim)
            kern_low.lengthscale.constrain_bounded(0.005, 0.5)
            
            kern_err = GPy.kern.Matern52(self.ndim)
            kern_err.lengthscale.constrain_bounded(0.005, 0.5)
            
            
        multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
        gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, n_fidelities)
        
        gpy_model.likelihood.Gaussian_noise.fix(self.l_noise)
        gpy_model.likelihood.Gaussian_noise_1.fix(0)
        
        self.model = GPyMultiOutputWrapper(gpy_model, 2, 5, verbose_optimization=False)
        self.model.optimize()

    def set_acquisition(self, beta, cost_ratio):
        """
        Set the acquisition function for the optimization.
        
        :beta: Exploration term
        :cost_ratio: Cost coefficient
        """
        self.cost_ratio = cost_ratio
        self.beta = beta
        self.acquisition = CustomMultiFidelityUCB(self.model, 
                                                          self.parameter_space,                                                          
                                                          ei_beta = beta)
    def init_bayes_loop(self):
        """
        Initialize Optimization
        """
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(self.parameter_space), self.parameter_space)
        self.candidate_point_calculator = SequentialPointCalculator(self.acquisition, acquisition_optimizer)
        self.n_hf_evals, self.n_lf_evals = 0,0
        self.iterates = []
        self.GP_iterates = []
        self.n_iter = 0
        self.min_hf_evals = 0

    def run_bayes_loop(self, n_iter, min_hf_evals=1, plot_opt=False, plot_acq=False):
        """
        Run optimization loop
        
        :n_iter: Total number of iterations
        :min_hf_evals: Number of addditional High-fidelity evaluations
        :plot_opt: Plot optimization progress at each iteration
        :plot_acq: Plot acquisition function at each iteration
        """
        self.n_iter += n_iter
        self.min_hf_evals += min_hf_evals
        self.GP_iterates = list(self.GP_iterates)
        self.iterates = list(self.iterates)
    
        for i in range(n_iter):
            self.iter = i+1
            xnext = self.candidate_point_calculator.compute_next_points(self.model)[:, :-1]
            mean_l, variance_l = self.model.predict(np.array([np.hstack([xnext[0],0])]))
            mean_h, variance_h = self.model.predict(np.array([np.hstack([xnext[0],1])]))
            error = abs(mean_h - mean_l)
            
            self.gamma = error * np.sqrt(self.cost_ratio)
            
            if self.beta * np.sqrt(variance_l) > self.gamma:
                fidelity = 0
                self.n_lf_evals += 1
            else:
                fidelity = 1
                self.n_hf_evals += 1
        
            self.fidelity_list.append(fidelity)
            xnew = np.hstack([xnext[0], fidelity])
            ynew = self.multifidelity.f[fidelity](xnext)
            Y = np.vstack([self.model.Y, ynew])
            X = np.vstack([self.model.X, xnew])
           
            if plot_acq:
                self.plot_acquisition(xnew)
            
            self.model.set_data(X, Y)
            
            if plot_opt:
                self.plot_optimization()
            
            self.model.optimize()
                

            best_so_far = np.max(self.model.Y[self.model.X[:, -1] == 1])
            hf = self.model.X[self.model.X[:, -1] == 1][:, :-1]
            best_loc = hf[np.argmax(self.model.Y[self.model.X[:, -1] == 1])]
            mean, var = self.model.predict(self.x_plot_high)
            best_mean = np.max(mean)
            best_mean_loc = self.x_plot_high[np.argmax(mean), :-1]
            iterate = np.hstack([best_loc, best_so_far])
            GP_iterate = np.hstack([best_mean_loc, best_mean])
            self.iterates.append(iterate)
            self.GP_iterates.append(GP_iterate)

        y_last = self.multifidelity.f[1](best_mean_loc)

        if best_mean > best_so_far:
            if y_last[0] > best_so_far:
                best_so_far = y_last[0]
                best_loc = best_mean_loc
    
            iterate = np.hstack([best_loc, best_so_far])
            self.iterates.append(iterate)
            self.fidelity_list.append(1)
            self.GP_iterates.append(self.GP_iterates[-1])
            
            Y = np.vstack([self.model.Y, y_last[0]])
            
            np.array(np.hstack([best_mean_loc, 1]))
            
            X = np.vstack([self.model.X, np.array(np.hstack([best_mean_loc, 1]))])
            
            self.model.set_data(X, Y)
            
            if plot_opt:
                self.plot_optimization(label='Final evaluation')
            
        else:
            self.min_hf_evals -= min_hf_evals

        self.GP_iterates = np.array(self.GP_iterates)
        self.iterates = np.array(self.iterates)

        if self.negate:
            best_so_far = -best_so_far
        
        return best_loc, best_so_far

    def plot_with_error_bars(self, x, mean, var, color, label):
        """
        Plot the GP with variance
        
        :x: Search space
        :mean: Mean at every location
        :var: Variance at every location
        :color: Color of the plot
        :label: Label of the plot
        """
        plt.plot(x, mean, color=color, label=label)
        plt.fill_between(x.flatten(), mean.flatten() - 1.96*var.flatten(), mean.flatten() + 1.96*var.flatten(), 
                        alpha=0.2, color=color)
    def true_model(self):
        """
        Calculate the exact model, the optimum and the location
        """
        self.true_model = self.multifidelity.f[1](self.x_plot)
        self.max = np.max(self.true_model)
        self.loc = self.x_plot[np.argmax(self.true_model)]
        if self.negate:
            self.true_model = -self.true_model
            self.max = -self.max
        self.is_max =True
        
    
    def plot_model(self, figsize=(10,6)):
        """
        Plot the model
        
        :figsize: Figure size
        """
        if self.ndim==1:
            plt.figure(figsize=figsize)
            x_low = self.x_array[self.x_array[:, 1] == 0][:, 0]
            x_high = self.x_array[self.x_array[:, 1] == 1][:, 0]
            y_low = self.y_array[self.x_array[:, 1] == 0]
            y_high = self.y_array[self.x_array[:, 1] == 1]
            mean_low, var_low = self.model.predict(self.x_plot_low)
            mean_high, var_high = self.model.predict(self.x_plot_high)
            
            if self.negate:
                mean_high = -mean_high
                mean_low = -mean_low
                y_high = -y_high
                y_low = -y_low
                
            #plt.figure(figsize=FIG_SIZE)
            self.plot_with_error_bars(self.x_plot_high[:, 0], mean_low, var_low, 'b', label='Low-fidelity GP')
            self.plot_with_error_bars(self.x_plot_high[:, 0], mean_high, var_high, 'r', label='High-fidelity GP')
            
            if self.is_max:
                plt.scatter(self.loc, self.max, color='k',  marker='*', s=120, label='True optimum')
                plt.plot(self.x_plot, self.true_model, 'k--', label='True model')
                
            plt.scatter(x_low, y_low, color='b')
            plt.scatter(x_high, y_high, color='r')
            plt.legend(['Low fidelity model', 'High fidelity model', 'True high fidelity'])
            plt.title('Low and High Fidelity Models')
            lb, ub = self.bounds
            plt.xlim(lb, ub)
            plt.xlabel('x')
            plt.ylabel('y');
            plt.show()
            
        if self.ndim==2:
            Z1 = self.multifidelity.f[0](self.x_plot)
            Z2 = self.multifidelity.f[1](self.x_plot)

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            c1 = axes[0].contourf(self.X1, self.X2, Z1.reshape(self.X1.shape), levels=50, cmap='viridis')
            c2 = axes[1].contourf(self.X1, self.X2, Z2.reshape(self.X1.shape), levels=50, cmap='viridis')
            
            if self.is_max:
                plt.scatter(self.loc[0], self.loc[1], color='k',  marker='*', s=120, label='True optimum')

            cbar1 = fig.colorbar(c1, ax=axes[0], orientation='vertical', fraction=0.05, pad=0.1)

            cbar2 = fig.colorbar(c2, ax=axes[1], orientation='vertical', fraction=0.05, pad=0.1)
            plt.suptitle('Multifidelity Function')
            axes[0].set_title('Low-fidelity')
            axes[0].set_xlabel('x1')
            axes[0].set_ylabel('x2')
            axes[1].set_title('High-fidelity')
            axes[1].set_xlabel('x1')
            plt.tight_layout()
            plt.show()
            
    def plot_optimization(self, label=None):
        """
        Plot the optimization progress
        
        :label: Plot label
        """
        colours = ['b', 'r']
        is_high_fidelity = self.model.X[:, -1] == 1
        x_low = self.model.X[~is_high_fidelity, :-1]
        y_low = self.model.Y[~is_high_fidelity]
        x_high = self.model.X[is_high_fidelity, :-1]
        y_high = self.model.Y[is_high_fidelity]
        
        mean_low, var_low = self.model.predict(self.x_plot_low)
        mean_high, var_high = self.model.predict(self.x_plot_high)
        
        if self.negate:
            mean_low = -mean_low
            mean_high = -mean_high
            y_low = -y_low
            y_high = -y_high

        if self.ndim==1:
            plt.figure(figsize=(7,4))
     
            #plt.figure(figsize=FIG_SIZE)
            self.plot_with_error_bars(self.x_plot_high[:, :-1], mean_low, var_low, 'b', label='Low-fidelity GP')
            self.plot_with_error_bars(self.x_plot_high[:, :-1], mean_high, var_high, 'r', label='High-fidelity GP')
            
            if self.is_max:
                plt.scatter(self.loc, self.max, color='k',  marker='*', s=120, label='True optimum')
                plt.plot(self.x_plot, self.true_model, 'k--', label='True model')

            plt.scatter(x_low, y_low, color='b')
            plt.scatter(x_high, y_high, color='r')
        
            xnew = self.model.X[[-1], :]
            fidelity_idx = int(xnew[0, -1])
            ynew = self.multifidelity.f[fidelity_idx](xnew[0,0])
    
            if self.negate:
                ynew = -ynew
            plt.scatter(xnew[0, 0], 
                        ynew, 
                        color=colours[fidelity_idx], marker='*', s=420)
            
            plt.vlines(xnew[0,0], -15, ynew, linestyle='-.', linewidth=5, color=colours[fidelity_idx])
            
            plt.legend()
            if label:
                plt.title('Multi-fidelity UCB Optimization; ' + label)
            else:
                plt.title('Multi-fidelity UCB Optimization; Iteration {}'.format(self.iter))
            plt.xlim(0, 1)
            # plt.ylim(-15,25)
            plt.xlabel('x')
            plt.ylabel('y');
            plt.show()
        
        if self.ndim==2:
            plt.figure(figsize=(8,6))
            plt.contourf(self.X1, self.X2, mean_high.reshape(self.X1.shape), levels=50, cmap='viridis')
            plt.colorbar(label="High-fidelity GP Mean")
            plt.scatter(x_low[:, 0], x_low[:, 1], color='b', label='Low fidelity points')
            plt.scatter(x_high[:, 0], x_high[:, 1], color='r', label='High fidelity points')
        
            if self.is_max:
                plt.scatter(self.loc[0], self.loc[1], color='k',  marker='*', s=120, zorder=3, label='True optimum')
            
            xnew = self.model.X[[-1], :]
            fidelity_idx = int(xnew[0, -1])
            #ynew = self.multifidelity.f[fidelity_idx](xnew[0,:-1])
            plt.scatter(xnew[:, 0], 
                        xnew[:, 1], 
                        color=colours[fidelity_idx], marker='*', s=220, label='next location')
            plt.legend()
            if label:
                plt.title('Multi-fidelity UCB Optimization; ' + label)
            else:
                plt.title('Multi-fidelity UCB Optimization; Iteration {}'.format(self.iter))
            plt.xlabel('x1')
            plt.ylabel('x2');
            plt.show()

    def plot_acquisition(self, xnew):
        """
        Plot the acquisition function
        
        :xnew: New evaluation with fidelity
        """
        colours = ['b', 'r']
        
        if self.ndim==1:
            plt.figure(figsize=(6.5,4))
            acq_low, acq_high, error = self.acquisition.acquisition_low_high(self.x_plot_low, self.x_plot_high)
            mean_l = self.acquisition.acquisition_low_mean(self.x_plot_low)
            plt.plot(self.x_plot_low[:, 0], acq_low, 'b', label='Low-fidelity')
            plt.plot(self.x_plot_low[:, 0], acq_high, 'r', label='High-fidelity')
            plt.plot(self.x_plot_low[:, 0], self.acquisition.evaluate(self.x_plot_high), 'g', linestyle='--', label='Combined')
            xnew = np.array([xnew])
            fidelity_idx = int(xnew[0, -1])
            plt.scatter(xnew[0, :-1], 
                       self.acquisition.evaluate(xnew), 
                        color=colours[fidelity_idx])
            plt.plot(self.x_plot_low[:, 0], mean_l + self.gamma + error, color='k', linestyle='--', label='Gamma threshold')
            
            plt.legend()
            plt.title('Multifidelity UCB at Iteration {}'.format(self.iter))
            plt.xlabel('x')
            plt.xlim(0, 1)
            plt.ylabel('Acquisition Value')
            plt.tight_layout()
            plt.show()
            
        if self.ndim==2:
            
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(self.X1, self.X2, self.acquisition.evaluate(self.x_plot_high).reshape(self.X1.shape), levels=50, cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('Multi-fidelity UCB')
            xnew = np.array([xnew])
            fidelity_idx = int(xnew[0, -1])
            plt.scatter(xnew[:, 0], 
                        xnew[:, 1], 
                        color=colours[fidelity_idx], marker='*', s=220, label='next location')
            
            plt.legend()
            plt.title('Multi-fidelity UCB at Iteration {}'.format(self.iter))
            plt.xlabel('x1')
            plt.xlim(0,1)
            plt.ylabel('x2')
            plt.show()

    def plot_results(self):
        """
        Plot the convergence plots
        """
        colours = ['b', 'r']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        iterates = self.iterates[:,-1]
        GP_iterates = self.GP_iterates[:,-1]
        if self.negate:
            iterates = -iterates
            GP_iterates = -GP_iterates

        axes[0].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), iterates, label='Best solution')
        axes[0].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), GP_iterates, label='Best GP mean')
        axes[0].set_xlim(1, self.n_iter+self.min_hf_evals)
        if self.is_max:
            max, loc = self.max, self.loc
            axes[0].hlines(max, 0, self.n_iter+self.min_hf_evals, 'k', linestyle='--', label='True optimum')
        for i in range(len(iterates)):
            axes[0].scatter(i+1, iterates[i], color = colours[self.fidelity_list[i]], zorder=1, s=50)
            axes[0].scatter(i+1, GP_iterates[i], color = colours[self.fidelity_list[i]], zorder=2, s=50)
        axes[0].legend()
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel(r'$\hat{f}$')

        if self.ndim==1:
            axes[1].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), self.iterates[:,0], label='Best location')
            axes[1].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), self.GP_iterates[:,0], label='Best GP location')
            axes[1].set_xlim(1, self.n_iter+self.min_hf_evals)
            axes[1].hlines(loc, 0, self.n_iter+self.min_hf_evals, 'k', linestyle='--', label='True location')
            for i in range(len(iterates)):
                axes[1].scatter(i+1, self.iterates[:,0][i], color = colours[self.fidelity_list[i]], zorder=1, s=50)
                axes[1].scatter(i+1, self.GP_iterates[:,0][i], color = colours[self.fidelity_list[i]], zorder=2, s=50)
            axes[1].legend()
            axes[1].set_xlabel('Iterations')
            axes[1].set_ylabel(r'$\hat{x}$')
        else:
            if self.is_max:
                norm_iterates = [np.linalg.norm(i - loc) for i in self.iterates[:,:-1]]
                norm_GP_iterates = [np.linalg.norm(i - loc) for i in self.GP_iterates[:,:-1]]
                axes[1].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), norm_iterates, label='Best location')
                axes[1].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), norm_GP_iterates, label='Best GP location')
                for i in range(len(iterates)):
                    axes[1].scatter(i+1, norm_iterates[i], color = colours[self.fidelity_list[i]], zorder=1, s=50)
                    axes[1].scatter(i+1, norm_GP_iterates[i], color = colours[self.fidelity_list[i]], zorder=2, s=50)
                axes[1].set_xlim(1, self.n_iter+self.min_hf_evals)
                axes[1].legend()
                axes[1].set_xlabel('Iterations')
                axes[1].set_ylabel(r'$|\hat{x} - x_*|$')
            

# Define custom acquisition model

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
import numpy as np


class CustomMultiFidelityUCB(Acquisition):
    def __init__(self, model: IModel, space, ei_beta=15):
        """
        Custom acquisition function for multi-fidelity optimization.
        
        :param model: Multi-fidelity model (e.g., a multi-fidelity GP)
        :param space: Search space over which the optimization is performed
        :param cost_function: Optional cost function that assigns a cost to each fidelity level
        """
        super().__init__()
        self.model = model  # Multi-fidelity GP model
        self.space = space  # The parameter space (search space)
        self.ei_beta = ei_beta
    
    @property
    def has_gradients(self):
        # If gradients are implemented, return True. Otherwise, return False.
        return False

    def evaluate(self, x: np.ndarray):
        """
        Evaluate the custom acquisition function at the given points.

        :param x: Input points to evaluate (np.ndarray of shape (n_points, input_dim))
        :return: Acquisition values (np.ndarray of shape (n_points,))
        """
        # Multi-fidelity model's prediction (mean, variance) at point x
        xcopy_l, xcopy_h = x.copy(), x.copy()
        x_low = xcopy_l
        x_low[:, -1] = np.zeros(xcopy_l[:, -1].shape)
        x_high = xcopy_h
        x_high[:, -1] = np.ones(xcopy_h[:, -1].shape)
        
        mean_l, variance_l = self.model.predict(x_low)
        mean_h, variance_h = self.model.predict(x_high)

        acq_l = mean_l + self.ei_beta*(np.sqrt(variance_l)) + abs(mean_h - mean_l)
        acq_h = mean_h + self.ei_beta*(np.sqrt(variance_h))
        acquisition_value = np.array([np.min([acq_l[i], acq_h[i]]) for i in range(len(acq_l))]).reshape(-1,1)
        return acquisition_value

    def acquisition_low_high(self, x_low, x_high):
        """
        Evaluate the low-fidelity and high-fidelity acquisition function.

        :x_low: Low-fidelity evaluation points
        :x_high: Low-fidelity evaluation points
        
        Returns
        Low-fidelity acquisition function
        High-fidelity acquisition function
        """
        mean_h, variance_h = self.model.predict(x_high)
        mean_l, variance_l = self.model.predict(x_low)
        error = abs(mean_h - mean_l)
        return [mean_l + self.ei_beta* np.sqrt(variance_l) + error,
               mean_h + self.ei_beta * np.sqrt(variance_h), error]
    
    def acquisition_low_mean(self, x_low):
        """
        Evaluate the low-fidelity acquisition function.

        :x_low: Low-fidelity evaluation points
        
        Returns
        :mean_l: Low-fidelity GP mean
        """
        mean_l, variance_l = self.model.predict(x_low)
        return mean_l


    def evaluate_with_gradients(self, x: np.ndarray):
        """
        Evaluate the custom acquisition function and return gradients.

        :param x: Input points to evaluate (np.ndarray of shape (n_points, input_dim))
        :return: Tuple of (acquisition values, gradients)
        """
        raise NotImplementedError("Gradients are not implemented for this acquisition function.")