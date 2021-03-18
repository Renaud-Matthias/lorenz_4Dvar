#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:52:56 2021

@author: Matthias
"""

from lorenz import *
from obs import *
from ana import *
from scipy.optimize import minimize

#########################
# parameter of simulation
#########################

dt = 0.01 # temporal discretisation
parameters = [10.,11.,5.] # true parameters of the model
n_simul = 30 # number of iteration of the simulation


#########################
# assimilation parameter
#########################

# background state
Xb = np.array([0.,0.,0.])
# background covariance error matrix
sigma_b = 1.
Pb = sigma_b*np.eye(3)
# observation covariance error matrix
sigma_y = 0.01
R = sigma_y*np.eye(3)


#########################
# TRUE MODEL
#########################
condi_ini = np.array([10.,3.,2.]) # initial condition

# true model, reference
Lor_true = Model(dt,parameters,condi_ini,n_simul)
Lor_true.forward(n_simul)


#########################
# observation parameter
#########################
n_sub = 5 # number of iteration between two observation

T_obs = [i*n_sub*dt for i in range(n_simul//n_sub)]

Obs = Observation(T_obs,n_simul)

Obs.gen_obs(Lor_true)

#########################
# Data Assimilation
#########################

# new initial condition
coef = 1.
delta_x = coef * np.random.random(3)
new_condi_ini = condi_ini + delta_x

# analysed model
Lor_ana = Model(dt,parameters,new_condi_ini,n_simul)

Var = Variational(Xb, Pb, R, Lor_ana, Obs)

res = minimize(Var.cost,new_condi_ini)


#fig = plt.figure(figsize=(12,8))


