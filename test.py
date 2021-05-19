#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:46:57 2021

@author: Matthias
"""

from lorenz import Model
from ana import *
from obs import *
import numpy as np


#########################
# parameter of simulation
#########################

dt = 0.01 # temporal discretisation
parameters = [10.,28.,8/3] # true parameters of the model
n_simul = 30 # number of iteration of the simulation
scheme = 'RK4'
X0 = np.array([1.,3.,2.]) # initial condition

# create model, test tan and test adj
Lor = Model(dt,parameters,X0,n_simul,scheme=scheme,test=True)
