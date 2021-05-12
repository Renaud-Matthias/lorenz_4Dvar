#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:54:38 2021

@author: renamatt
"""

from lorenz import *
from obs import *
from ana import *
from windows import *
import config as cfg
import matplotlib.pyplot as plt

# CONFIGURATION
###############################################################################

#########################
# parameter of simulation
#########################


dt = 0.01 # temporal discretisation
parameters = [10.,28.,4/3] # true parameters of the model
X0 = np.array([10.,15.,6.])

# numerical scheme : euler,
sch = 'euler'
# sch = 'RK3'

# assimilation windows parameters
n_window = 30 # number of iteration contained in an assimilation window (need to be even)
n_step = n_window//2 # number of iteration between two assimilation, the window cross each other
n_assimil = 5 # number of assimilation windows
n_simul = (1+n_assimil)*n_step # number of iteration of the simulation


#########################
# observation parameter
#########################

# one observation every n_sub iteration
n_sub = 5 # number of iteration between two observations


#########################
# assimilation parameter
#########################

# new parametrs for assimilation
# delta for each parameters
d_param = [0., 0., 0.] # d sigma, d rho, d beta
par_assimil = []
for i in range(3) :
    new_param = parameters[i] + d_param[i]
    par_assimil.append(new_param)

# initial background state
Xb = np.array([8.,1.,5.])

# background covariance error matrix
sigma_b = 1.
B = sigma_b*np.eye(3)

# observation covariance error matrix
sigma_y = 0.1
R = sigma_y*np.eye(3)


###############################################################################

# RUN

#########################
# run assimilation
#########################

M_true,M_ana, Obs = assimil(n_window,n_step,n_assimil,n_simul,dt,parameters,par_assimil,n_sub,X0,Xb,B,R)

#########################
# plot results
#########################

time = [i*dt for i in range(n_simul)]
time_obs = [i*dt for i in Obs.iter_obs]

observations = np.array(list(Obs.obs.values()))

fig, ax = plt.subplots()

ax.plot(time,M_true.xvar_series[:,0])
ax.plot(time,M_ana.xvar_series[:,0])
ax.plot(time_obs,observations[:,0],'o')














