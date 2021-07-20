#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:46:57 2021

@author: Matthias
"""

from lorenz import Model
import ana
import windows as wdw
import numpy as np


#########################
# parameter of simulation
#########################

dt = 0.01 # temporal discretisation
parameters = [10.,28.,8/3] # true parameters of the model
n_simul = 30 # number of iteration of the simulation
scheme = 'RK4'
X0 = np.array([1.,3.,2.]) # initial condition

### ADJOINT & TANGENT TEST ###
Lor = Model(dt,parameters,X0,n_simul,scheme=scheme,test=True)


### GRADIENT TEST ###

n_sub = 5 # time step between two observations

Obs = wdw.create_Obs(Lor,n_simul,n_sub)

Xb = np.array([3.,10.,10.])
R = Obs.std*np.eye(3) # observation error covariance matrix
B = np.eye(3) # background error covariance matrix

Var = ana.Variational(Xb=Xb,B=B,R=R,M=Lor,Obs=Obs)

Var.grad_test(plot=True)