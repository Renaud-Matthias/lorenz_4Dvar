#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthias

run a lorenz model without data assimilation
"""

import sys
sys.path.append('../src/')

from lorenz import Model
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np

##### parameter of simulation ####

dt = 0.01 # temporal discretisation
parameters = [20.,28,5.] # true parameters of the model
t_simu = 30. # time of simulation

condi_ini = np.array([10.,5.,10.]) # initial condition

# true model, reference
Lor = Model(dt,t_simu,parameters,condi_ini,scheme='euler')
Lor.forward(10000)

# fig, axs = plt.subplots(3,1,figsize=(15,5))
# i = -1
# for ax,char in zip(axs,'xyz') :
#     i+=1
#     ax.plot(Lor.xvar_series[:,i])
#     ax.set_ylabel(char)

fig, ax = plt.subplots()
ax.plot(Lor.xvar_series[:,0],Lor.xvar_series[:,2])