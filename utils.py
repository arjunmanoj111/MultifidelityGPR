#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:41:13 2024

@author: arjunmnanoj
"""

import numpy as np
from sklearn.metrics.pairwise import check_pairwise_arrays

def cross_distances(X, y=None):
    return X
    
    
def differences(X, Y):
    "Component wise difference between X and Y"
    X, Y = check_pairwise_arrays(X, Y)
    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    return D.reshape((-1, X.shape[1]))


# Nested LHS
# LHS of the hi fidelity model is a subset of the LHS on lower fidelity model.


            
          
            
            
            
            
            
            
            
            
            