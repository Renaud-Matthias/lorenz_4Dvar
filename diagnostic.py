#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:26:19 2021

@author: renamatt
"""

import numpy as np



def eval_rmse(x_ref,x) :
    '''
    Computes the rmse between x_ref and x
    '''
    square_xdiff = (x-x_ref)**2
    mean_square_diff = np.mean(square_xdiff) #(1/N)sum((x-xref)^2)
    rmse = np.sqrt(mean_square_diff)
    return rmse

def eval_score(x_ref,x) :
    '''
    Computes the score based error between x_ref and x
    '''
    std = np.std(x_ref)
    rmse = eval_rmse(x_ref,x)
    score = float(1. - (rmse/std))
    return score

def SCORE(X_ref,X) :
    n = len(X_ref)
    score_out = np.zeros(n)
    for i in range(n) :
        x_ref,x = X_ref[i],X[i]
        score = eval_score(x_ref,x)
        score_out[i] = score
    return score_out
    