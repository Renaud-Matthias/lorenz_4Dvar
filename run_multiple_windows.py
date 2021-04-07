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
import matplotlib.pyplot as plt

# multiple assimilation windows

#########################
# parameter of simulation
#########################

dt = 0.01 # temporal discretisation
parameters = [10.,28.,4/3] # true parameters of the model
X0 = np.array([10.,15.,6.])

# numerical scheme : euler,
# sch = 'euler'
# sch = 'RK3'

# assimilation windows parameters
n_window = 30 # number of iteration contained in an assimilation window (need to be even)
n_step = n_window//2 # number of iteration between two assimilation, the window cross each other
n_assimil = 5 # number of assimilation windows
n_simul = (1+n_assimil)*n_step # number of iteration of the simulation

#########################
# assimilation parameter
#########################

# new parametrs for assimilation
# delta for each parameters
d_param = [1., 0., 0.5] # d sigma, d rho, d beta
par_assimil = []
for i in range(3) :
    new_param = parameters[i] + d_param[i]
    par_assimil.append(new_param)

# initial background state
Xb = np.array([8.,1.,5.])
# background covariance error matrix
sigma_b = 1.
Pb = sigma_b*np.eye(3)
# observation covariance error matrix
sigma_y = 0.1
R = sigma_y*np.eye(3)


#########################
# observation parameter
#########################

# one observation every n_sub iteration
n_sub = 5 # number of iteration between two observations

#########################
# run assimilation
#########################

M_true, M_ana = assimil(n_window,n_step,n_assimil,n_simul,dt,parameters,par_assimil,n_sub,X0,Xb,Pb,R)

#########################
# plot results
#########################

plt.figure()

coord = 2 # 0 for x, 1 for y and 2 for z

Time_plot = [dt*i for i in range(n_simul)]
Time_obs = [dt*i for i in range(n_sub,n_simul,n_sub)]
Obs_values = [M_true.xvar_series[i][coord] for i in range(n_sub,n_simul,n_sub)]

plt.plot(Time_plot,M_true.xvar_series[:,coord],label='true state')
plt.plot(Time_plot,M_ana.xvar_series[:,coord],label='analysed')
plt.plot(Time_obs,Obs_values,'o',color='red',label='observations')
plt.plot(0,X0[coord],'o',color='black')
for i in range(0,n_assimil) :
    plt.axvline(x=dt*n_step*i, color='green')
plt.legend()
plt.show()






    