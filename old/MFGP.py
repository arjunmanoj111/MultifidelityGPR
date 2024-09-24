#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:39:41 2024

@author: arjunmnanoj
"""

#Multifidelity Gaussian Process

from typing import Optional
import numpy as np
import scipy
import scipy.spatial.distance as distance



DEFAULT_SIGMA: float = 1e-4


class GaussianProcess:
    """Gaussian process regressor."""
    sigma: float = DEFAULT_SIGMA         # Regularization term.
    epsilon: Optional[float] = None      # Spatial scale parameter.
    alphas: Optional[np.ndarray] = None  # Coefficients.

    def __init__(self, sigma: Optional[float] = None,
                 epsilon: Optional[float] = None) -> None:
        if sigma is not None:
            self.sigma = sigma
        if epsilon is not None:
            self.epsilon = epsilon

    def _learn(self, 
               points: np.ndarray,
               values: np.ndarray,
               epsilon: float, 
               kernel_matrix: np.ndarray,
               ) -> None:
        """Auxiliary method for fitting a Gaussian process."""
        self.points = points
        self.epsilon = epsilon
        

        sigma2_eye = self.sigma**2 * np.eye(kernel_matrix.shape[0])
        L, Lt = scipy.linalg.cho_factor(
            kernel_matrix + sigma2_eye, lower=True, check_finite=False, overwrite_a=True)
        self.cholesky_factor = L
        self.alphas = scipy.linalg.cho_solve((L, True), values,
                                                 check_finite=False)

    def learn(self, points: np.ndarray, values: np.ndarray) -> None:
        """Fit a Gaussian process

        Parameters
        ----------
        points: np.ndarray
            Data points arranged by rows.
        values: np.ndarray
            Values corresponding to the data points.

        """
        distances2 = distance.pdist(points, metric='sqeuclidean')

        if self.epsilon is None:
            threshold = np.finfo(points.dtype).eps * 1e2
            self.epsilon = np.sqrt(
                np.median(distances2[distances2 > threshold]))

        kernel_matrix = distance.squareform(
            np.exp(-distances2 / (2.0 * self.epsilon**2)))
        diagonal_indices = np.diag_indices_from(kernel_matrix)
        kernel_matrix[diagonal_indices] = 1.0
        self.kernel_matrix = kernel_matrix

        self._learn(points, values,
                    self.epsilon, self.kernel_matrix)

    def __call__(self, point: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian process at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned Gaussian process
            is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the GP at the given point.

        """
        kstar = np.exp(
            -np.sum((point - self.points)**2, axis=1)
            / (2.0 * self.epsilon**2))
        return kstar @ self.alphas
    
    def variance(self, point: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian process variance at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned Gaussian process
            is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the GP variance at the given point.

        """
        
        kstar = np.exp(
            -np.sum((point - self.points)**2, axis=1)
            / (2.0 * self.epsilon**2))
        
        k = np.exp(
            -np.sum((point - point)**2)
            / (2.0 * self.epsilon**2))
        
        L = self.cholesky_factor
        self.betas = scipy.linalg.cho_solve((L, True), kstar,
                                                 check_finite=False)
        
        return k - kstar @ self.betas
    
   


class MFGPR:
    """Multifidelity Gaussian process regressor."""
    sigma: float = DEFAULT_SIGMA         # Regularization term.
    epsilon: Optional[float] = None      # Spatial scale parameter.
    alphas: Optional[np.ndarray] = None  # Coefficients.

    def __init__(self,
                 sigma_l=0,
                 epsilon_l=None,
                 sigma_h=0,
                 epsilon_h=None,
                 rho=1,
                 mu=0) -> None:
        self.rho = rho
        self.mu = mu
        self.sigma_l = sigma_l
        self.epsilon_l = epsilon_l
        self.sigma_h = sigma_h
        self.epsilon_h = epsilon_h

    def _learn(self, 
               points_l: np.ndarray,
               values_l: np.ndarray,
               epsilon_l: float, 
               kernel_matrix_l: np.ndarray,
               points_h: np.ndarray,
               values_h: np.ndarray,
               epsilon_h: float, 
               kernel_matrix_h: np.ndarray
               ) -> None:
        """Auxiliary method for fitting both the Gaussian processes."""
        self.points_l = points_l
        self.epsilon_l = epsilon_l
        
        self.points_h = points_h
        self.epsilon_h = epsilon_h

        sigma2_eye_l = self.sigma_l**2 * np.eye(kernel_matrix_l.shape[0])
        L_l, _ = scipy.linalg.cho_factor(
            kernel_matrix_l + sigma2_eye_l, lower=True, check_finite=False, overwrite_a=True)
        self.cholesky_factor_l = L_l
        self.alphas_l = scipy.linalg.cho_solve((L_l, True), values_l,
                                                 check_finite=False)
        
        
        sigma2_eye_h = self.sigma_h**2 * np.eye(kernel_matrix_h.shape[0])
        L_h, _ = scipy.linalg.cho_factor(
            kernel_matrix_h + sigma2_eye_h, lower=True, check_finite=False, overwrite_a=True)
        self.cholesky_factor_h = L_h
        self.values_h = values_h

    def learn(self, 
              points_l: np.ndarray,
              values_l: np.ndarray,
              points_h: np.ndarray,
              values_h: np.ndarray) -> None:
        """Fit a Gaussian process 

        Parameters
        ----------
        points_l: np.ndarray
            Low fidelity data points arranged by rows.
        values_l: np.ndarray
            Low fidelity function values corresponding to the data points.
        points_h: np.ndarray
            High fidelity data points arranged by rows.
        values_h: np.ndarray
            High fidelity function values corresponding to the data points.
        """
        distances_l = distance.pdist(points_l, metric='sqeuclidean')
        distances_h = distance.pdist(points_h, metric='sqeuclidean')

        if self.epsilon_l is None:
            threshold = np.finfo(points_l.dtype).eps * 1e2
            self.epsilon_l = np.sqrt(
                np.median(distances_l[distances_l > threshold]))
            
        if self.epsilon_h is None:
            threshold = np.finfo(points_h.dtype).eps * 1e2
            self.epsilon_h = np.sqrt(
                np.median(distances_h[distances_h > threshold]))

        kernel_matrix_l = distance.squareform(
            np.exp(-distances_l / (2.0 * self.epsilon_l**2)))
        diagonal_indices = np.diag_indices_from(kernel_matrix_l)
        kernel_matrix_l[diagonal_indices] = 1.0
        self.kernel_matrix_l = kernel_matrix_l
        
        
        kernel_matrix_h = distance.squareform(
            np.exp(-distances_h / (2.0 * self.epsilon_h**2)))
        diagonal_indices = np.diag_indices_from(kernel_matrix_h)
        kernel_matrix_h[diagonal_indices] = 1.0
        self.kernel_matrix_h = kernel_matrix_h

        self._learn(points_l, values_l,
                    self.epsilon_l, self.kernel_matrix_l,
                    points_h, values_h,
                    self.epsilon_h, self.kernel_matrix_h,)

    def GP_l(self, point: np.ndarray) -> np.ndarray:
        """Evaluate low-fidelity Gaussian process at a new point.

        This function must be called after the Gaussian processes have been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned low fidelity 
            Gaussian process is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the low-fidelity GP at the given point.

        """
        kstar = np.exp(
            -np.sum((point - self.points_l)**2, axis=1)
            / (2.0 * self.epsilon_l**2))
        return kstar @ self.alphas_l
    
    
    
    def variance_l(self, point: np.ndarray) -> np.ndarray:
        """Evaluate low-fidelity Gaussian process variance at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned low-fidelity
            Gaussian process is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the low-fidelity GP variance at the given point.

        """
        
        kstar = np.exp(
            -np.sum((point - self.points_l)**2, axis=1)
            / (2.0 * self.epsilon_l**2))
        
        k = np.exp(
            -np.sum((point - point)**2)
            / (2.0 * self.epsilon_l**2))
        
        L = self.cholesky_factor_l
        self.betas_l = scipy.linalg.cho_solve((L, True), kstar,
                                                 check_finite=False)
        
        return k - kstar @ self.betas_l
    
    
    
    def GP_h(self, point: np.ndarray) -> np.ndarray:
        """Evaluate high-fidelity Gaussian process at a new point.

        This function must be called after the Gaussian processes have been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned high fidelity 
            Gaussian process is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the high-fidelity GP at the given point.

        """
        kstar = np.exp(
            -np.sum((point - self.points_h)**2, axis=1)
            / (2.0 * self.epsilon_h**2))
        
        GP_l_eval = np.array([self.GP_l(i) for i in self.points_h]).reshape(self.values_h.shape)
        
        L_h = self.cholesky_factor_h
        self.alphas_h = scipy.linalg.cho_solve((L_h, True), self.values_h - self.rho*GP_l_eval-self.mu,
                                                 check_finite=False)
        return self.rho*self.GP_l(point) + self.mu + kstar @ self.alphas_h
    
    
    
    def variance_h(self, point: np.ndarray) -> np.ndarray:
        """Evaluate high-fidelity Gaussian process variance at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: np.ndarray
            A single point on which the previously learned high-fidelity
            Gaussian process is to be evaluated.

        Returns
        -------
        value: np.ndarray
            Estimated value of the high-fidelity GP variance at the given point.

        """
        
        kstar = np.exp(
            -np.sum((point - self.points_h)**2, axis=1)
            / (2.0 * self.epsilon_h**2))
        
        k = np.exp(
            -np.sum((point - point)**2)
            / (2.0 * self.epsilon_h**2))
        
        L_h = self.cholesky_factor_h
        self.betas = scipy.linalg.cho_solve((L_h, True), kstar,
                                                 check_finite=False)
        return self.rho**2 * self.variance_l(point) + k - kstar @ self.betas