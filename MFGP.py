#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:22:12 2024

@author: arjunmnanoj
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
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

from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel

# Define custom acquisition model
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



plt.rcParams.update({
    "font.family": "serif",
    "font.sans-serif": "Computer Modern",
})
FIG_SIZE = (12, 8)


class multifidelityGPR:
    """
    Multifidelity Gaussian Process Regression
    lf -> Low-fidelity function
    hf -> High-fidelity function
    ndim -> Problem dimensionality
    negate -> True if minimization problem
    normalize ->
    """
    def __init__(self, lf, hf, ndim=1, negate=False, normalize=True, noise=0.0, linear=True):
        self.LF = lf
        self.HF = hf
        self.multifidelity = MultiSourceFunctionWrapper([lf, hf])
        self.iterates = []
        self.GP_iterates = []
        self.fidelity_list = []
        self.ndim = ndim
        self.negate = negate
        self.fidelity_list = []
        self.l_noise = noise
        self.is_max = False
        self.normalize = normalize
        self.scaler_low = MinMaxScaler()
        self.scaler_high = MinMaxScaler()
        self.linear=linear
        

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
        
        y_low_scaled = self.scaler_low.fit_transform(y_low)
        y_high_scaled = self.scaler_high.fit_transform(y_high)
        
        self.x_array, self.y_array = convert_xy_lists_to_arrays([x_low, x_high], [y_low, y_high])
        self.x_array, self.y_array_scaled = convert_xy_lists_to_arrays([x_low, x_high], [y_low_scaled, y_high_scaled])
        
        self.n_init = n1+n2
        self.bounds = bounds
        
        if self.ndim==1:
            # Bounds
            lb, ub = bounds
            x_plot = np.linspace(lb, ub, 500)[:, None]
            
        elif self.ndim==2:
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
        Create model
        """
        if x_array==None:
            x_array = self.x_array
        if y_array==None:
            if self.normalize:
                y_array = self.y_array_scaled
            else:
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


        if self.linear:
            if ker=='rbf':
                kern_low = GPy.kern.RBF(self.ndim)
                kern_low.lengthscale.constrain_bounded(0.01, 0.5, warning=False)
                
                kern_err = GPy.kern.RBF(self.ndim)
                kern_err.lengthscale.constrain_bounded(0.01, 0.5, warning=False)
    
            else:
                kern_low = GPy.kern.Matern52(self.ndim)
                kern_low.lengthscale.constrain_bounded(0.005, 0.5, warning=False)
                
                kern_err = GPy.kern.Matern52(self.ndim)
                kern_err.lengthscale.constrain_bounded(0.005, 0.5, warning=False)
                
                
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, n_fidelities)
            
            gpy_model.likelihood.Gaussian_noise.fix(self.l_noise)
            gpy_model.likelihood.Gaussian_noise_1.fix(0)
            self.model = GPyMultiOutputWrapper(gpy_model, 2, 5, verbose_optimization=False)
        else:
            base_kernel = GPy.kern.RBF
            kernels = make_non_linear_kernels(base_kernel, 2, x_array.shape[1] - 1)
            nonlin_mf_model = NonLinearMultiFidelityModel(x_array, y_array, n_fidelities=2, kernels=kernels, 
                                                          verbose=False, optimization_restarts=5)
            for m in nonlin_mf_model.models:
                m.Gaussian_noise.variance.fix(0)
                
            self.model = nonlin_mf_model
        self.model.optimize()

    def set_acquisition(self, beta, cost_param):
        self.cost_ratio = cost_param
        self.acquisition = FidelityWeightedAcquisition(self.model, 
                                                          self.parameter_space, 
                                                          cost_ratio=cost_param, 
                                                          ei_beta = beta, 
                                                          n_init=self.n_init)
        
    def init_bayes_loop(self):
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(self.parameter_space), self.parameter_space)
        self.candidate_point_calculator = SequentialPointCalculator(self.acquisition, acquisition_optimizer)
        self.n_hf_evals, self.n_lf_evals = 0,0
        self.iterates = []
        self.GP_iterates = []
        self.n_iter = 0
        self.min_hf_evals = 0

    
    def run_bayes_loop(self, n_iter, min_hf_evals=1, plot_opt=False, plot_acq=False, is_GP =True):
        self.n_iter += n_iter
        self.min_hf_evals += min_hf_evals
        self.GP_iterates = list(self.GP_iterates)
        self.iterates = list(self.iterates)
    
        for i in range(n_iter):
            self.iter = i+1
            # beta = np.sqrt(0.2 * self.ndim * np.log(2*(i+1)))
            # self.set_acquisition(beta, self.cost_ratio)
            # acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(self.parameter_space), self.parameter_space)
            # self.candidate_point_calculator = SequentialPointCalculator(self.acquisition, acquisition_optimizer)
            
            
            xnext = self.candidate_point_calculator.compute_next_points(self.model)
            
            if xnext[0][-1] == 0:
                fidelity = 0
                self.n_lf_evals += 1
                
                ynew = self.multifidelity.f[fidelity](xnext[0][:-1])
                if self.normalize:
                    ynew = self.scaler_low.transform(ynew.reshape(-1,1))
            else:
                fidelity = 1
                self.n_hf_evals += 1
                
                ynew = self.multifidelity.f[fidelity](xnext[0][:-1])
                if self.normalize:
                    ynew = self.scaler_high.transform(ynew.reshape(-1,1))
        
            self.fidelity_list.append(fidelity)
            # ynew = self.multifidelity.f[fidelity](xnext[0][:-1])
            Y = np.vstack([self.model.Y, ynew])
            X = np.vstack([self.model.X, xnext])
            
            if plot_acq:
                self.plot_acquisition(xnext[0])
        
            
            self.model.set_data(X, Y)
            
            if plot_opt:
                self.plot_optimization()
        
            self.model.optimize()
                
            
            hf = self.model.X[self.model.X[:, -1] == 1][:, :-1]
            best_loc = hf[np.argmax(self.model.Y[self.model.X[:, -1] == 1])]
            
            y_high = self.model.Y[self.model.X[:, -1] == 1]
            
            
            if self.normalize:
                y_high = self.scaler_high.inverse_transform(y_high)
                
            if is_GP:
                mean, var = self.model.predict(self.x_plot_high)
                if self.normalize:
                    mean = self.scaler_high.inverse_transform(mean)
                best_mean = np.max(mean)
                best_mean_loc = self.x_plot_high[np.argmax(mean), :-1]
                GP_iterate = np.hstack([best_mean_loc, best_mean])
                self.GP_iterates.append(GP_iterate)
                
                
            best_so_far = np.max(y_high)
            iterate = np.hstack([best_loc, best_so_far])
            self.iterates.append(iterate)
            
            if self.normalize:
                y_low = self.scaler_low.inverse_transform(self.model.Y[self.model.X[:,-1]==0])
                y_high = self.scaler_high.inverse_transform(self.model.Y[self.model.X[:,-1]==1])
                
                y_low_scaled = self.scaler_low.fit_transform(y_low)
                y_high_scaled = self.scaler_high.fit_transform(y_high)
                
                x_low = self.model.X[self.model.X[:,-1]==0]
                x_high = self.model.X[self.model.X[:,-1]==1]
                
                X = np.vstack([x_low, x_high])
                Y = np.vstack([y_low_scaled, y_high_scaled])
                
                self.model.set_data(X, Y)
                self.model.optimize()
            

        if is_GP:
        # if best_mean > best_so_far:
            if best_mean > -1e10:
                
                y_last = self.multifidelity.f[1](best_mean_loc)
                
                if y_last[0] > best_so_far:
                    best_so_far = y_last[0]
                    best_loc = best_mean_loc
        
                iterate = np.hstack([best_loc, best_so_far])
                self.iterates.append(iterate)
                self.fidelity_list.append(1)
                self.GP_iterates.append(self.GP_iterates[-1])
                if self.normalize:
                    y_last_norm = self.scaler_high.transform(y_last.reshape(-1,1))
                    Y = np.vstack([self.model.Y, y_last_norm[0]])
                else:
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


    def true_model(self):
        """Exact model"""
        self.true_model = self.multifidelity.f[1](self.x_plot)
        self.max = np.max(self.true_model)
        self.loc = self.x_plot[np.argmax(self.true_model)]
        if self.negate:
            self.true_model = -self.true_model
            self.max = -self.max
        self.is_max =True

    def plot_acquisition(self, xnew):
        if self.ndim==1:
            plt.figure(figsize=(6.5,4))
            colours = ['b', 'r']
            acq_low = self.acquisition.evaluate(self.x_plot_low)
            
            acq_high = self.acquisition.evaluate(self.x_plot_high)
            
            plt.plot(self.x_plot_low[:, 0], acq_low, 'b', label='Low-fidelity')
            plt.plot(self.x_plot_low[:, 0], acq_high, 'r', label='High-fidelity')
            xnew = np.array([xnew])
            fidelity_idx = int(xnew[0, -1])
            plt.scatter(xnew[0, :-1], 
                        self.acquisition.evaluate(xnew), 
                        color=colours[fidelity_idx])
            
            plt.legend()
            plt.title('Fidelity-Weighted Acquisition Function at Iteration {}'.format(self.iter))
            plt.xlabel('x')
            plt.xlim(0, 1)
            plt.ylabel('Acquisition Value')
            plt.tight_layout()
            plt.show()
            
            
        if self.ndim==2:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            colours = ['b', 'r']
            
            acq_low = self.acquisition.evaluate(self.x_plot_low)
            acq_high = self.acquisition.evaluate(self.x_plot_high)
            
            c1 = axes[0].contourf(self.X1, self.X2, acq_low.reshape(self.X1.shape), levels=50, cmap='viridis')
            c2 = axes[1].contourf(self.X1, self.X2, acq_high.reshape(self.X1.shape), levels=50, cmap='viridis')
            
            cbar1 = fig.colorbar(c1, ax=axes[0], orientation='vertical', fraction=0.05, pad=0.1)
            cbar1.set_label('Low-fidelity Acquisition function')

            cbar2 = fig.colorbar(c2, ax=axes[1], orientation='vertical', fraction=0.05, pad=0.1)
            cbar2.set_label('Multi-fidelity Acquisition function')
            
            xnew = np.array(xnew)
            fidelity_idx = int(xnew[-1])
            axes[fidelity_idx].scatter(xnew[0], 
                        xnew[1], 
                        color=colours[fidelity_idx], marker='*', s=220, label='next location')
            axes[fidelity_idx].legend()
            plt.suptitle('Acquisition Function at Iteration ' + str(self.iter))
            axes[0].set_title('Low-fidelity')
            axes[0].set_xlabel('x1')
            axes[0].set_xlim(0, 1)
            axes[0].set_ylabel('x2')
            axes[1].set_title('Multi-fidelity')
            axes[1].set_xlabel('x1')
            axes[1].set_xlim(0, 1)
            plt.tight_layout()
            plt.show()
        

    
    def plot_optimization(self, label=None):
        
        colours = ['b', 'r']
        is_high_fidelity = self.model.X[:, -1] == 1
        x_low = self.model.X[~is_high_fidelity, :-1]
        y_low = self.model.Y[~is_high_fidelity]
        x_high = self.model.X[is_high_fidelity, :-1]
        y_high = self.model.Y[is_high_fidelity]
        
        mean_low, var_low = self.model.predict(self.x_plot_low)
        mean_high, var_high = self.model.predict(self.x_plot_high)
        
        if self.ndim==1:
            plt.figure(figsize=(7,4))
            
            
            xnew = self.model.X[[-1], :]
            fidelity_idx = int(xnew[0, -1])
            ynew = self.multifidelity.f[fidelity_idx](xnew[0,0])
            
            if self.normalize:
                mean_high = self.scaler_high.inverse_transform(mean_high)
                mean_low = self.scaler_low.inverse_transform(mean_low)
                y_low = self.scaler_low.inverse_transform(y_low)
                y_high = self.scaler_high.inverse_transform(y_high)
    
            if self.negate:
                mean_low = -mean_low
                mean_high = -mean_high
                y_low = -y_low
                y_high = -y_high
                ynew = -ynew
            
            self.plot_with_error_bars(self.x_plot_high[:, :-1], mean_low, var_low, 'b', label='Low-fidelity GP')
            self.plot_with_error_bars(self.x_plot_high[:, :-1], mean_high, var_high, 'r', label='High-fidelity GP')
            
            if self.is_max:
                plt.scatter(self.loc, self.max, color='k',  marker='*', s=120)
                plt.plot(self.x_plot, self.true_model, 'k--', label='True model')
            
            plt.scatter(x_low, y_low, color='b')
            plt.scatter(x_high, y_high, color='r')
        
            plt.scatter(xnew[0, 0], 
                        ynew, 
                        color=colours[fidelity_idx], marker='*', s=420)
            
            plt.vlines(xnew[0,0], -15, ynew, linestyle='-.', linewidth=5, color=colours[fidelity_idx])
            
            plt.legend()
            if label:
                plt.title('Fidelity-Weighted Optimization; ' + label)
            else:
                plt.title('Fidelity-Weighted Optimization; Iteration {}'.format(self.iter))
            plt.xlim(0, 1)
            #plt.ylim(-15,25)
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
                plt.scatter(self.loc[0], self.loc[1], color='k',  marker='*', s=120, zorder=3)
        
            xnew = self.model.X[[-1], :]
            fidelity_idx = int(xnew[0, -1])
            #ynew = self.multifidelity.f[fidelity_idx](xnew[0,:-1])
            plt.scatter(xnew[:, 0], 
                        xnew[:, 1], 
                        color=colours[fidelity_idx], marker='*', s=220, label='next location')
            plt.legend()
            if label:
                plt.title('Fidelity-Weighted Optimization; ' + label)
            else:
                plt.title('Fidelity-Weighted Optimization; Iteration {}'.format(self.iter))
            plt.xlabel('x1')
            plt.ylabel('x2');
            plt.show()

    def plot_with_error_bars(self, x, mean, var, color, label):
            plt.plot(x, mean, color=color, label=label)
            plt.fill_between(x.flatten(), mean.flatten() - 1.96*var.flatten(), mean.flatten() + 1.96*var.flatten(), 
                            alpha=0.2, color=color)

    def plot_model(self, figsize=(10,5)):        
        if self.ndim==1:
            plt.figure(figsize=figsize)
            x_low = self.x_array[self.x_array[:, 1] == 0][:, 0]
            x_high = self.x_array[self.x_array[:, 1] == 1][:, 0]
            y_low = self.y_array[self.x_array[:, 1] == 0]
            y_high = self.y_array[self.x_array[:, 1] == 1]
            mean_low, var_low = self.model.predict(self.x_plot_low)
            mean_high, var_high = self.model.predict(self.x_plot_high)
            
            
            if self.normalize:
                mean_high = self.scaler_high.inverse_transform(mean_high)
                mean_low = self.scaler_low.inverse_transform(mean_low)
            
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
    
    def plot_results(self, is_GP=True):
        colours = ['b', 'r']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        iterates = self.iterates[:,-1]
        if is_GP:
            GP_iterates = self.GP_iterates[:,-1]
        if self.negate:
            iterates = -iterates
            if is_GP:
                GP_iterates = -GP_iterates

        axes[0].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), iterates, label='Best solution')
        if is_GP:
            axes[0].plot(np.arange(1, self.n_iter+self.min_hf_evals+1, 1), GP_iterates, label='Best GP mean')
        axes[0].set_xlim(1, self.n_iter+self.min_hf_evals)
        if self.is_max:
            max, loc = self.max, self.loc
            axes[0].hlines(max, 0, self.n_iter+self.min_hf_evals, 'k', linestyle='--', label='True optimum')
        for i in range(len(iterates)):
            axes[0].scatter(i+1, iterates[i], color = colours[self.fidelity_list[i]], zorder=1, s=50)
            if is_GP:
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
    
    
    
            

class FidelityWeightedAcquisition(Acquisition):
    def __init__(self, model: IModel, space, n_init, ei_beta=1, cost_ratio=1/5):
        """
        Custom acquisition function for multi-fidelity optimization.

        :model: Multi-fidelity model 
        :space: Search space over which the optimization is performed
        :n_init: Number of initial data points
        :ei_beta: Optional exploration parameter
        :cost_ratio: Relative cost ratio
        """
        super().__init__()
        self.model = model  # Multi-fidelity GP model
        self.space = space  # The parameter space (search space)
        self.cost_ratio = cost_ratio  # Optional cost ratio
        self.ei_beta = ei_beta
        self.n_init = n_init
    
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
        mean, variance = self.model.predict(x)

        # if x[0][-1] == 0:
        #     best_so_far = np.max(self.model.Y[self.model.X[:, -1] == 0])
            
        # if x[0][-1] == 1:
        #     best_so_far = np.max(self.model.Y[self.model.X[:, -1] == 1])

        # if self.cost_ratio==None:
        #     acquisition_value = self._expected_improvement(mean, variance, best_so_far)
        # else:
        #     acquisition_value = self._expected_improvement(mean, variance, best_so_far) - self._cost_function(x)
        
        
        if self.cost_ratio==None:
            acquisition_value = self.UCB(mean, variance)
        else:
            acquisition_value = self.UCB(mean, variance) - self._cost_function(x)
            
        return acquisition_value

    def _expected_improvement(self, mean, variance, best_so_far):
        """
        Expected Improvement calculation.

        :param mean: Predicted mean from the GP model
        :param variance: Predicted variance from the GP model
        :return: Expected Improvement (np.ndarray)
        """
        improvement = mean - best_so_far
        with np.errstate(divide='warn'):
            z = improvement / np.sqrt(variance)
            ei = improvement * norm.cdf(z) + self.ei_beta *(np.sqrt(variance) * norm.pdf(z))
            ei[ei == 0] = np.zeros(ei[ei == 0].shape)
        return ei
    
    def UCB(self, mean, variance):
        return mean + self.ei_beta*(np.sqrt(variance))

    def _cost_function(self, x):
        b = self.cost_ratio
        n1 = sum(self.model.X[:, -1] == 0) # LF points
        n2 = sum(self.model.X[:, -1] == 1) # HF points
        
        if x[0][-1] == 0:
            cost = b*(n1+ 1) + n2
        if x[0][-1] == 1:
            cost = b*n1 + (n2+1)
        return cost/(1 + len(self.model.X) - self.n_init)
    
    def evaluate_with_gradients(self, x: np.ndarray):
        raise NotImplementedError("Gradients are not implemented for this acquisition function.")
        
        
   