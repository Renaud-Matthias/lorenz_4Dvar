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
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# multiple assimilation windows

#########################
# parameter of simulation
#########################

dt = 0.01 # temporal discretisation
parameters = [10.,11.,5.] # true parameters of the model
X0 = np.array([10.,2.,2.])

n_window = 30 # number of iteration contained in an assimilation window (need to be even)
n_step = n_window//2 # number of iteration between two assimilation, the window cross each other
n_assimil = 3 # number of assimilation windows
n_simul = (1+n_assimil)*n_step # number of iteration of the simulation

#########################
# assimilation parameter
#########################

# new parametrs for assimilation
# delta for each parameters
d_param = [0.5, 0.2, -1.] # d sigma, d rho, d beta
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
# True state
#########################

M_true = Model(dt,parameters,X0,n_simul)
M_true.forward(n_simul)

#########################
# observation parameter
#########################

# one observation every n_sub iteration
n_sub = 5 # number of iteration between two observation

##################################
# create full set of observation #
##################################

X_ana = np.zeros((n_simul,3)) # array to store analysed trajectory


# observatios time for first window
T_obs = [i*n_sub*dt for i in range(1,int(n_window/n_sub))]

M_obs = Model(dt,parameters,X0,n_window) # model to generate observations
M_obs.forward(n_window) # run model

Obs = Observation(T_obs,n_window)
Obs.gen_obs(M_obs)

# first assimilation
x_res = ana_4Dvar(dt,par_assimil,n_window,Xb,Pb,R,Obs) # result of assimilation

M_ana = Model(dt,par_assimil,x_res,n_simul) # result model
M_ana.forward(n_step)
x_start = M_ana.step(M_ana.xvar_series[n_step-1])

for k in range(1,n_assimil) :
    T_obs = [i*n_sub*dt for i in range(0,int(n_window/n_sub))]
    M_obs = Model(dt, parameters, M_true.xvar_series[k*n_step], n_window) # model to generate observation
    M_obs.forward(n_window) # run model
    Obs = Observation(T_obs,n_window)
    Obs.gen_obs(M_obs)
    # result of assimilation
    M_ana.xvar = ana_4Dvar(dt,parameters,n_window,x_start,Pb,R,Obs) # result of assimilation
    M_ana.xvar_series[k*n_step] = M_ana.xvar
    if k < n_assimil-1 :
        M_ana.forward(n_step+1,start=k*n_step+1) # solution on new assimilation window
        x_start = M_ana.step(M_ana.xvar_series[(k+1)*n_step]) # starting point for next assimilation window
    else :
        M_ana.forward(n_window,start=k*n_step+1)

del M_obs, T_obs


# plt.figure()

Time_plot = [dt*i for i in range(n_simul)]
Time_obs = [dt*i for i in range(n_sub,n_simul,n_sub)]
Obs_values = [M_true.xvar_series[i][0] for i in range(n_sub,n_simul,n_sub)]

plt.plot(Time_plot,M_true.xvar_series[:,0],label='true state')
plt.plot(Time_plot,M_ana.xvar_series[:,0],label='analysed')
plt.plot(Time_obs,Obs_values,'o',color='red',label='observations')
plt.plot(0,X0[0],'o',color='black')
for i in range(0,n_assimil) :
    plt.axvline(x=dt*n_step*i, color='green')
plt.legend()
plt.show()






    