#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:40:15 2021

@author: renamatt
"""

from lorenz import *
from obs import *
from ana import *

def ana_4Dvar(dt,param,n_iter,Xb,Pb,R,Obs) :
    '''
    Perform a 4Dvar assimilation and return the result of the assimilation
    PARAMETERS :
    - dt , time step
    - param , parameters of the lorenz model
    - n_iter , number of iteration in the assimilation
    - Xb , background state
    - Pb , background covariance error matrix
    - R , observation covariance error matrix
    - Obs , Observation object, contain the observations and the observation operators
    RETURN :
    - Xout , result of the assimilation, initial coordinates that best fits observations and background
    '''
    # model to analyse
    M = Model(dt,param,np.ones(3),n_iter)
    Var = Variational(Xb,Pb,R,M,Obs)
    res = minimize(Var.cost,np.zeros(3),jac=Var.grad)
    return res.x

