#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:52:22 2024

@author: arjunmnanoj
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace


import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound, ProbabilityOfImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer

from emukit.core.loop import SequentialPointCalculator

from sklearn.preprocessing import MinMaxScaler, StandardScaler


from scipy.stats import norm
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
import mf2


class bayesOptimization:
    def __init__(self, fun, ndim=1, negate=False, normalize=True, noise=0.0):
        """
        Bayesian Optimization.

        :fun: function
        :ndim: Problem dimensionality
        :negate: True if minimization problem
        :normalize: True if outputs are normalized
        :noise: Low-fidelity model noise
        """
        self.function = UserFunctionWrapper(fun)
        self.iterates = []
        self.GP_iterates = []
        self.ndim = ndim
        self.negate = negate
        self.noise = noise
        self.is_max = False
        self.normalize = normalize
        self.scaler = MinMaxScaler()

    def set_initial_data(self, n, bounds, seed=12346):
        """
        Initialize dataset for the model

        :n1: Number of Low-fidelity points
        :n2: Number of High-fidelity points
        :bounds: Problem bounds
        :seed: random seed
        """
        np.random.seed(seed)
        x = np.random.rand(n, self.ndim)
        y = self.function.f(x)
        self.y = y
        if self.normalize:
            y_scaled = self.scaler.fit_transform(y)
            y = y_scaled
            self.y_scaled = y_scaled

        self.n_init = n
        self.bounds = bounds

        if self.ndim == 1:
            # Bounds
            lb, ub = bounds
            x_plot = np.linspace(lb, ub, 500)[:, None]

        elif self.ndim == 2:
            x1_range = np.linspace(0, 1, 200)
            x2_range = np.linspace(0, 1, 200)
            # Create mesh grid from x1 and x2 ranges
            X1, X2 = np.meshgrid(x1_range, x2_range)
            x_plot = np.array([[x1, x2]
                              for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
            self.X1, self.X2 = X1, X2

        else:
            arrays = [np.linspace(0, 1, 5) for i in range(self.ndim)]

            # Create meshgrid from the arrays
            meshgrid = np.meshgrid(*arrays, indexing='ij')

            # Stack meshgrid and reshape to get an array of data points
            x_plot = np.stack(meshgrid, axis=-1).reshape(-1, self.ndim)

        self.x_plot = x_plot
        self.x_array, self.y_array = x, y.reshape(-1, 1)

    def set_model(self, x_array=None, y_array=None, ker='rbf'):
        """
        Initialize model

        :x_array: X data if any
        :y_array: Y data if any
        :ker: Kernel (RBF by default. Will switch to Matern for any other input)
        """
        if x_array == None:
            x_array = self.x_array
        if y_array == None:
            y_array = self.y_array

        if self.ndim == 1:
            lb, ub = self.bounds
            self.parameter_space = ParameterSpace(
                [ContinuousParameter('x', lb, ub)])
        else:
            paramspace = []
            for i in range(self.ndim):
                lb, ub = self.bounds[i]
                paramspace.append(ContinuousParameter('x'+str(i+1), 0, 1))
            self.parameter_space = ParameterSpace(paramspace)

        if ker == 'rbf':
            kernel = GPy.kern.RBF(self.ndim)
            kernel.lengthscale.constrain_bounded(0.05, 0.5, warning=False)

        else:
            kernel = GPy.kern.Matern52(self.ndim)
            kernel.lengthscale.constrain_bounded(0.005, 0.5, warning=False)

        gpy_model = GPy.models.GPRegression(
            x_array, y_array, kernel, noise_var=self.noise)

        self.model = GPyModelWrapper(gpy_model)
        self.model.optimize()

    def set_acquisition(self, beta, acq):
        """
        Set the acquisition function for the optimization.

        :beta: Exploration term
        :cost_ratio: Cost coefficient
        """
        self.beta = beta
        self.acquisition = CustomAcq(self.model,
                                     self.parameter_space,
                                     ei_beta=beta,
                                     acq=acq)

    def init_bayes_loop(self):
        """
        Initialize Optimization
        """
        acquisition_optimizer = GradientAcquisitionOptimizer(
            self.parameter_space)
        self.candidate_point_calculator = SequentialPointCalculator(
            self.acquisition, acquisition_optimizer)
        self.iterates = []
        self.GP_iterates = []
        self.n_iter = 0

    def run_bayes_loop(self, n_iter, plot_opt=False, plot_acq=False):
        """
        Run optimization loop

        :n_iter: Total number of iterations
        :min_hf_evals: Number of addditional High-fidelity evaluations
        :plot_opt: Plot optimization progress at each iteration
        :plot_acq: Plot acquisition function at each iteration
        """
        self.n_iter += n_iter
        self.GP_iterates = list(self.GP_iterates)
        self.iterates = list(self.iterates)

        y_high = self.model.Y

        if self.normalize:
            y_high = self.scaler.inverse_transform(y_high)

        best_so_far = np.max(y_high)

        best_loc = self.model.X[np.argmax(self.model.Y)]

        iterate = np.hstack([best_loc, best_so_far])
        self.iterates.append(iterate)

        for i in range(n_iter):
            self.iter = i+1
            xnext = self.candidate_point_calculator.compute_next_points(
                self.model)
            mean, variance = self.model.predict(np.array([xnext[0]]))

            ynew = self.function.f(xnext)
            if self.normalize:
                ynew = self.scaler.transform(ynew)

            xnew = xnext[0]

            Y = np.vstack([self.model.Y, ynew])
            X = np.vstack([self.model.X, xnew])

            if plot_acq:
                self.plot_acquisition(xnew)

            self.model.set_data(X, Y)

            if plot_opt:
                self.plot_optimization()

            self.model.optimize()

            y_high = self.model.Y

            if self.normalize:
                y_high = self.scaler.inverse_transform(y_high)

            best_so_far = np.max(y_high)

            best_loc = self.model.X[np.argmax(self.model.Y)]

            iterate = np.hstack([best_loc, best_so_far])
            self.iterates.append(iterate)

            if self.normalize:
                y_inverse = self.scaler.inverse_transform(self.model.Y)

                y_scaled = self.scaler.fit_transform(y_inverse)

                Y = y_scaled.reshape(-1, 1)

                self.model.set_data(X, Y)
                self.model.optimize()

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
        self.true_model = self.function.f(self.x_plot)
        self.max = np.max(self.true_model)
        self.loc = self.x_plot[np.argmax(self.true_model)]
        if self.negate:
            self.true_model = -self.true_model
            self.max = -self.max
        self.is_max = True

    def plot_model(self, figsize=(10, 6)):
        """
        Plot the model

        :figsize: Figure size
        """
        if self.ndim == 1:
            plt.figure(figsize=figsize)
            x = self.x_array
            y = self.y_array

            mean, var = self.model.predict(self.x_plot)

            if self.normalize:
                mean = self.scaler.inverse_transform(mean)
                y = self.scaler.inverse_transform(y)

            if self.negate:
                mean = -mean
                y = -y

            # plt.figure(figsize=FIG_SIZE)
            self.plot_with_error_bars(self.x_plot, mean, var, 'b', label='GP')

            if self.is_max:
                plt.scatter(self.loc, self.max, color='k',
                            marker='*', s=120, label='True optimum')
                plt.plot(self.x_plot, self.true_model,
                         'k--', label='True model')

            plt.scatter(x, y, color='b')
            plt.legend()
            plt.title('GP Model')
            lb, ub = self.bounds
            plt.xlim(lb, ub)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if self.ndim == 2:
            Z1 = self.function.f(self.x_plot)

            fig, axes = plt.subplots(1, 1, figsize=figsize)
            c1 = axes.contourf(self.X1, self.X2, Z1.reshape(
                self.X1.shape), levels=50, cmap='viridis')

            if self.is_max:
                plt.scatter(self.loc[0], self.loc[1], color='k',
                            marker='*', s=120, label='True optimum')

            cbar1 = fig.colorbar(
                c1, ax=axes, orientation='vertical', fraction=0.05, pad=0.1)

            plt.title('GP Model')
            axes.set_xlabel('x1')
            axes.set_ylabel('x2')
            plt.tight_layout()
            plt.show()

    def plot_optimization(self, label=None):
        """
        Plot the optimization progress

        :label: Plot label
        """
        colours = ['b', 'r']
        x = self.model.X
        y = self.model.Y

        mean, var = self.model.predict(self.x_plot)

        if self.normalize:
            mean = self.scaler.inverse_transform(mean)
            y = self.scaler.inverse_transform(y)

        if self.negate:
            mean = -mean
            y = -y

        if self.ndim == 1:
            plt.figure(figsize=(7, 4))

            # plt.figure(figsize=FIG_SIZE)
            self.plot_with_error_bars(self.x_plot, mean, var, 'b', label='LGP')

            if self.is_max:
                plt.scatter(self.loc, self.max, color='k',
                            marker='*', s=120, label='True optimum')
                plt.plot(self.x_plot, self.true_model,
                         'k--', label='True model')

            plt.scatter(x, y, color='b')

            xnew = self.model.X[-1]
            ynew = self.function.f(xnew[0])

            if self.negate:
                ynew = -ynew
            plt.scatter(xnew[0],
                        ynew,
                        color=colours[0], marker='*', s=420)

            plt.legend()
            if label:
                plt.title('Multi-fidelity UCB Optimization; ' + label)
            else:
                plt.title(
                    'Multi-fidelity UCB Optimization; Iteration {}'.format(self.iter))
            plt.xlim(0, 1)
            # plt.ylim(-15,25)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if self.ndim == 2:
            plt.figure(figsize=(8, 6))
            plt.contourf(self.X1, self.X2, mean.reshape(
                self.X1.shape), levels=50, cmap='viridis')
            plt.colorbar(label="High-fidelity GP Mean")
            plt.scatter(x[:, 0], x[:, 1], color='b', label='Data ppoints')

            if self.is_max:
                plt.scatter(self.loc[0], self.loc[1], color='k',
                            marker='*', s=120, zorder=3, label='True optimum')

            xnew = self.model.X[-1]
            plt.scatter(xnew[0],
                        xnew[1],
                        color=colours[0], marker='*', s=220, label='next location')
            plt.legend()
            if label:
                plt.title('Multi-fidelity UCB Optimization; ' + label)
            else:
                plt.title(
                    'Multi-fidelity UCB Optimization; Iteration {}'.format(self.iter))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.show()

    def plot_acquisition(self, xnew):
        if self.ndim == 1:
            plt.figure(figsize=(6.5, 4))
            colours = ['b', 'r']
            acq = self.acquisition.evaluate(self.x_plot)

            plt.plot(self.x_plot, acq, 'b', label='Low-fidelity')
            xnew = np.array([xnew])
            plt.scatter(xnew[0],
                        self.acquisition.evaluate(xnew),
                        color=colours[0])

            plt.legend()
            plt.title('Acquisition Function at Iteration {}'.format(self.iter))
            plt.xlabel('x')
            plt.xlim(0, 1)
            plt.ylabel('Acquisition Value')
            plt.tight_layout()
            plt.show()

        if self.ndim == 2:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            colours = ['b', 'r']

            acq = self.acquisition.evaluate(self.x_plot)

            c1 = axes.contourf(self.X1, self.X2, acq.reshape(
                self.X1.shape), levels=50, cmap='viridis')

            cbar1 = fig.colorbar(
                c1, ax=axes, orientation='vertical', fraction=0.05, pad=0.1)
            cbar1.set_label('Acquisition function')

            xnew = np.array(xnew)
            axes.scatter(xnew[0],
                         xnew[1],
                         color=colours[0], marker='*', s=220, label='next location')
            axes.legend()
            plt.title('Acquisition Function at Iteration ' + str(self.iter))
            axes.set_xlabel('x1')
            axes.set_xlim(0, 1)
            axes.set_ylabel('x2')
            plt.tight_layout()
            plt.show()

    def plot_results(self):
        """
        Plot the convergence plots
        """
        colours = ['b', 'r']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        iterates = self.iterates[:, -1]
        if self.negate:
            iterates = -iterates

        axes[0].plot(np.arange(1, self.n_iter+1, 1),
                     iterates, '-o', label='Best solution')
        axes[0].set_xlim(1, self.n_iter)

        if self.is_max:
            max, loc = self.max, self.loc
            axes[0].hlines(max, 0, self.n_iter, 'k',
                           linestyle='--', label='True optimum')

        axes[0].legend()
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel(r'$\hat{f}$')

        if self.ndim == 1:
            axes[1].plot(np.arange(1, self.n_iter+1, 1),
                         self.iterates[:, 0], '-o', label='Best location')
            axes[1].set_xlim(1, self.n_iter)
            axes[1].hlines(loc, 0, self.n_iter, 'k',
                           linestyle='--', label='True location')
            axes[1].legend()
            axes[1].set_xlabel('Iterations')
            axes[1].set_ylabel(r'$\hat{x}$')
        else:
            if self.is_max:
                norm_iterates = [np.linalg.norm(i - loc)
                                 for i in self.iterates[:, :-1]]
                axes[1].plot(np.arange(1, self.n_iter+1, 1),
                             norm_iterates, '-o', label='Best location')
                axes[1].set_xlim(1, self.n_iter)
                axes[1].legend()
                axes[1].set_xlabel('Iterations')
                axes[1].set_ylabel(r'$|\hat{x} - x_*|$')


class CustomAcq(Acquisition):
    def __init__(self, model: IModel, space, ei_beta=15, acq='ei'):
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
        self.acq = acq

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
        # Multi-fidelity model's prediction (mean, variance) at point x)

        mean, variance = self.model.predict(x)

        if self.acq == 'ei':
            best_so_far = np.max(self.model.Y)

            acquisition_value = self._expected_improvement(
                mean, variance, best_so_far)
        else:
            acquisition_value = mean + self.ei_beta*(np.sqrt(variance))

        return acquisition_value.reshape(-1, 1)

    def _expected_improvement(self, mean, variance, best_so_far):
        """
        Custom Expected Improvement calculation.

        :param mean: Predicted mean from the GP model
        :param variance: Predicted variance from the GP model
        :return: Expected Improvement (np.ndarray)
        """
        improvement = mean - best_so_far
        with np.errstate(divide='warn'):
            z = improvement / np.sqrt(variance)
            ei = improvement * \
                norm.cdf(z) + self.ei_beta * (np.sqrt(variance) * norm.pdf(z))
            ei[ei == 0] = np.zeros(ei[ei == 0].shape)
        return ei

    def evaluate_with_gradients(self, x: np.ndarray):
        """
        Evaluate the custom acquisition function and return gradients.

        :param x: Input points to evaluate (np.ndarray of shape (n_points, input_dim))
        :return: Tuple of (acquisition values, gradients)
        """
        raise NotImplementedError(
            "Gradients are not implemented for this acquisition function.")


# %%

def high_fidelity_toy(x):
    return ((x*6 - 2)**2)*np.sin((x*6 - 2)*2)


def hf(x):
    return -high_fidelity_toy(x)


bo = bayesOptimization(hf, ndim=1, negate=True, noise=0.0, normalize=1)
bo.set_initial_data(1, np.array([0, 1]), seed=2)
bo.set_model()
bo.true_model()
bo.plot_model()

forrester_min = bo.max

bo.set_acquisition(5, 'ucb')
bo.init_bayes_loop()

x, f = bo.run_bayes_loop(10, plot_opt=True, plot_acq=True)

bo.plot_results()


# %%

def high_fidelity_toy(x):
    return np.sin(3*np.pi*x**3) - np.sin(8*np.pi*x**3)


def hf(x):
    return -high_fidelity_toy(x)


bo = bayesOptimization(hf, ndim=1, negate=True, noise=0.0, normalize=1)
bo.set_initial_data(3, np.array([0, 1]), seed=2)
bo.set_model()
bo.true_model()
bo.plot_model()

forrester_min = bo.max

bo.set_acquisition(2, 'ucb')
bo.init_bayes_loop()

x, f = bo.run_bayes_loop(12, plot_opt=True, plot_acq=True)

bo.plot_results()


# %%
exp = 50
iters = 8
boIterates = np.zeros((exp, iters))


for i in range(exp):
    print(i)
    bo = bayesOptimization(hf, ndim=1, negate=True, noise=0.0, normalize=False)
    bo.set_initial_data(2, np.array([0, 1]), seed=i)
    bo.set_model()
    bo.true_model()

    bo.set_acquisition(5, 'ei')
    bo.init_bayes_loop()

    x, f = bo.run_bayes_loop(iters, plot_opt=False, plot_acq=False)

    l = len(bo.iterates[:, 1])
    boIterates[i][:l] = bo.iterates[:, 1]

# %%
boMean = np.mean(boIterates, axis=0)

negate = True
if negate:
    boMean = -boMean

boStd = np.sqrt(np.var(boIterates, axis=0))


# Convergence
plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, iters+1, 1), boMean,
         label='Bayesian Optimization', color='r')

plt.fill_between(np.arange(1, iters+1, 1), boMean - 1.96*boStd/np.sqrt(exp),
                 boMean + 1.96*boStd/np.sqrt(exp),
                 alpha=0.2, color='r')


plt.hlines(forrester_min, 1, iters, color='k',
           linestyle='-.', label='True optimum')

plt.xlim(1, iters)
plt.legend(fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Best solution', fontsize=15)


# %%
np.save('datanew/boIterates_forrester.npy', boIterates)

# %%


def hf(x):
    return mf2.bohachevsky.high(x).reshape(-1, 1)


bo = bayesOptimization(hf, ndim=2, negate=False, noise=0.0, normalize=1)
bo.set_initial_data(4, np.array([[0, 1], [0, 1]]), seed=2)

bo.set_model()
bo.true_model()
bo.plot_model()

bohachevsky_max = bo.max

bo.set_acquisition(5, 'ucb')
bo.init_bayes_loop()

x, f = bo.run_bayes_loop(10, plot_opt=True, plot_acq=True)

bo.plot_results()

# %%

exp = 50
iters = 18
boIterates2 = np.zeros((exp, iters))


for i in range(exp):
    print(i)
    bo = bayesOptimization(hf, ndim=2, negate=False, noise=0.0)
    bo.set_initial_data(5, np.array([[0, 1], [0, 1]]), seed=i)
    bo.set_model()
    bo.true_model()

    bo.set_acquisition(5, 'ei')
    bo.init_bayes_loop()

    x, f = bo.run_bayes_loop(iters, plot_opt=False, plot_acq=False)

    l = len(bo.iterates[:, 1])
    boIterates2[i][:l] = bo.iterates[:, -1]

# %%

boMean2 = np.mean(boIterates2, axis=0)

negate = False
if negate:
    boMean2 = -boMean2

boStd2 = np.sqrt(np.var(boIterates2, axis=0))


# Convergence
plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, iters+1, 1), boMean2,
         label='Bayesian Optimization', color='r')

plt.fill_between(np.arange(1, iters+1, 1), boMean2 - 1.96*boStd2/np.sqrt(exp),
                 boMean2 + 1.96*boStd2/np.sqrt(exp),
                 alpha=0.2, color='r')


plt.hlines(bohachevsky_max, 1, iters, color='k',
           linestyle='-.', label='True optimum')

plt.xlim(1, iters)
plt.legend(fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Best solution', fontsize=15)

# %%

np.save('datanew/boIterates_bohachevsky.npy', boIterates2)

# %%


def hf(x):
    lb = mf2.himmelblau.l_bound
    ub = mf2.himmelblau.u_bound
    x1 = x*(ub-lb) + lb
    return -mf2.himmelblau.high(x1).reshape(-1, 1)


bo = bayesOptimization(hf, ndim=2, negate=True, noise=0.0, normalize=1)
bo.set_initial_data(4, np.array([[0, 1], [0, 1]]), seed=2)

bo.set_model()
bo.true_model()
bo.plot_model()

bohachevsky_max = bo.max

bo.set_acquisition(5, 'ucb')
bo.init_bayes_loop()

x, f = bo.run_bayes_loop(10, plot_opt=True, plot_acq=True)

bo.plot_results()

# %%

exp = 50
iters = 28
boIterates3 = np.zeros((exp, iters))


for i in range(exp):
    print(i)
    bo = bayesOptimization(hf, ndim=2, negate=False, noise=0.0)
    bo.set_initial_data(4, np.array([[0, 1], [0, 1]]), seed=i)
    bo.set_model()
    bo.true_model()

    bo.set_acquisition(5, 'ei')
    bo.init_bayes_loop()

    x, f = bo.run_bayes_loop(iters, plot_opt=False, plot_acq=False)

    l = len(bo.iterates[:, 1])
    boIterates3[i][:l] = bo.iterates[:, -1]

# %%

boMean3 = np.mean(boIterates3, axis=0)

negate = True
if negate:
    boMean3 = -boMean3

boStd3 = np.sqrt(np.var(boIterates3, axis=0))


# Convergence
plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, iters+1, 1), boMean3,
         label='Bayesian Optimization', color='r')

plt.fill_between(np.arange(1, iters+1, 1), boMean3 - 1.96*boStd3/np.sqrt(exp),
                 boMean3 + 1.96*boStd3/np.sqrt(exp),
                 alpha=0.2, color='r')


plt.hlines(0, 1, iters, color='k', linestyle='-.', label='True optimum')

plt.xlim(1, iters)
plt.legend(fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Best solution', fontsize=15)

# %%

np.save('datanew/boIterates_himmelblau.npy', boIterates3)

# %% Toy Enzyme model

# Enzyme Toy Models


def toyModel(t, x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1, k2, k3 = k
    dS0 = -k1*x[0]*x[3] + k2*x[1]
    dES0 = k1*x[0]*x[3] - k2*x[1] - k3*x[1]
    dS1 = k3*x[1]
    dE = -k1*x[3]*x[0] + k2*x[1] + k3*x[1]
    return [dS0, dES0, dS1, dE]


def reducedToyModel(t, x, k):
    """
    Evaluate time-derivatives of the six concentrations in Yeung et al.'s kinetic model.
    """
    k1, k2, k3, E0 = k
    ke = (k1*k3)/(k2 + k3)
    dS0 = -ke*x[0]*E0
    dS1 = ke*x[0]*E0
    return [dS0, dS1]


def high_fidelity_toy(E):
    S0 = 5.0
    # initial concentrations of S0, ES0, ES1, S1, S2, E
    C0 = np.array([S0, 0.0, 0.0, E])
    kvals = np.array([10, 50, 2])

    sol1 = solve_ivp(lambda t, x: toyModel(t, x, kvals),
                     [0, 100],
                     C0,
                     # t_eval=time_points,
                     atol=np.sqrt(np.finfo(float).eps),
                     rtol=np.sqrt(np.finfo(float).eps))
    return abs(10 - sol1.t[np.argmin(abs(sol1.y[2, :] - 0.67*S0))])


def low_fidelity_toy(E):
    S0 = 5.0
    # initial concentrations of S0, ES0, ES1, S1, S2, E
    C0 = np.array([S0, 0.0])
    kvals = np.array([10, 50, 2, E])
    sol2 = solve_ivp(lambda t, x: reducedToyModel(t, x, kvals),
                     [0, 100],
                     C0[:2],
                     # t_eval=time_points,
                     atol=np.sqrt(np.finfo(float).eps),
                     rtol=np.sqrt(np.finfo(float).eps))
    return abs(10 - sol2.t[np.argmin(abs(sol2.y[1, :] - 0.67*S0))])


def lf(x):
    if (type(x)) == np.ndarray:
        lf_vals = np.array([low_fidelity_toy(i)
                           for i in x.ravel()]).reshape(-1, 1)
    else:
        lf_vals = low_fidelity_toy(x)
    return -lf_vals


def hf(x):
    if (type(x)) == np.ndarray:
        hf_vals = np.array([high_fidelity_toy(i)
                           for i in x.ravel()]).reshape(-1, 1)
    else:
        hf_vals = high_fidelity_toy(x)
    return -hf_vals


bo = bayesOptimization(hf, ndim=1, negate=True, noise=0.0, normalize=1)
bo.set_initial_data(2, np.array([0, 1]), seed=2)
bo.set_model()
# bo.true_model()
bo.plot_model()


bo.set_acquisition(5, 'ucb')
bo.init_bayes_loop()

x, f = bo.run_bayes_loop(10, plot_opt=True, plot_acq=True)

bo.plot_results()


# %%


exp = 50
iters = 23
boIterates2 = np.zeros((exp, iters))


for i in range(exp):
    print(i)
    bo = bayesOptimization(hf, ndim=1, negate=True, noise=0.0)
    bo.set_initial_data(2, np.array([0, 1]), seed=i)
    bo.set_model()
    # bo.true_model()

    bo.set_acquisition(2, 'ei')
    bo.init_bayes_loop()

    x, f = bo.run_bayes_loop(iters, plot_opt=False, plot_acq=False)

    l = len(bo.iterates[:, 1])
    boIterates2[i][:l] = bo.iterates[:, -1]

# %%
boMean2 = np.mean(boIterates2, axis=0)

negate = True
if negate:
    boMean2 = -boMean2

boStd2 = np.sqrt(np.var(boIterates2, axis=0))

# Convergence
plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, iters+1, 1), boMean2,
         label='Bayesian Optimization', color='r')

plt.fill_between(np.arange(1, iters+1, 1), boMean2 - 1.96*boStd2/np.sqrt(exp),
                 boMean2 + 1.96*boStd2/np.sqrt(exp),
                 alpha=0.2, color='r')


plt.hlines(0, 0.01523335, iters, color='k',
           linestyle='-.', label='True optimum')


plt.xlim(1, iters)
plt.legend(fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Best solution', fontsize=15)


np.save('datanew/boIterates_enzyme.npy', boIterates2)
